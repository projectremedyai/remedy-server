"""Build LAMC-domain heading_hierarchy training records (heading-v2 data).

Why: heading-v1 was trained ONLY on heading-rich synthetic/gov pages (1,280
records, 50/50 pass/fail, zero LAMC pages) — a pure domain shift from the
sparse scanned district forms it meets at inference, where it hallucinates
"title should be H1" flags that differ run-to-run (58%% of its residual flags
contradict the human-certified delivered files).

Three record families, all rendered/prompted EXACTLY as production
(``heading_hierarchy_quality_prompt`` + ``_get_page_structure_order`` at
150 dpi — the inference dpi, not the corpus's 200):

1. ``lamc_false_flag_pass`` — a page the production model FLAGGED but the
   delivered (human-certified) file has no heading on: the model's own
   hallucination becomes a pass-negative with real gold.
2. ``lamc_true_fail`` — a flagged page where delivered HAS the heading: the
   target retags to the human's level (indexed into the same numbered list the
   model sees when the text matches a safe line; index-omitted otherwise).
3. ``lamc_delivered_pass`` — bulk domain pass records from fidelity-clean
   delivered pages without headings ("a sane form page is fine").

Usage (repo root):
    uv run python tools/finetune/build_lamc_heading_negatives.py \
        --cohort <heading_only_residual.jsonl> \
        --delivered ~/code/lamc_district_forms/lamc_remediated/remediated_pdfs \
        --out tools/finetune/data_heading_lamc \
        [--bulk-pass-pages 300] [--dpi 150]

Output: <out>/records.jsonl (conversation format compatible with
build_heading_corpus.to_conversation) + <out>/renders/*.png + manifest.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

PASS_TARGET = {"status": "pass", "findings": []}

_HASH_PREFIX = re.compile(r"^[0-9a-f]{12}_")
_PAGE_RE = re.compile(r"^Page\s+(\d+):")
_ORDER_LINE = re.compile(r"^\s*(\d+)\.\s+/(\w+)(?:\s+\(text:\s+\"(.*?)\"\))?")
# Tags that may legitimately carry a page/section title. A TD/TH/LI line whose
# text happens to match the heading is never the node to retag.
_TITLE_SAFE_TAGS = {"P", "Span", "H1", "H2", "H3", "H4", "H5", "H6", "Figure", "NonStruct"}


@dataclass(frozen=True)
class OrderLine:
    index: int
    tag: str
    text: str


def parse_structure_order_lines(order_text: str) -> list[OrderLine]:
    """Parse ``_get_page_structure_order`` output into (index, tag, text) rows.

    The extractor's placeholder strings ("(no structure elements...)",
    "(invalid page number)") parse to an empty list.
    """
    lines: list[OrderLine] = []
    for raw in (order_text or "").splitlines():
        m = _ORDER_LINE.match(raw)
        if not m:
            continue
        lines.append(OrderLine(int(m.group(1)), m.group(2), (m.group(3) or "").strip()))
    return lines


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def build_fail_target(order_lines: list[OrderLine], delivered_headings) -> dict:
    """Target JSON for a true-fail page.

    ``delivered_headings`` = [(level_tag, text), ...] from the human-certified
    delivered page. When a heading's text matches a numbered line with a
    title-safe tag, the finding is indexed into the SAME list the model sees;
    otherwise the finding omits element_index and carries visible_text only
    (the schema's "missing heading" form — our fixer text-matches from there).
    """
    findings: list[dict] = []
    for level, text in delivered_headings:
        norm = _norm(text)
        finding: dict = {
            "severity": "error",
            "visible_text": text,
            "message": "title/section heading is tagged as body text",
            "correct_tag": level,
            "suggested_fix": f"Retag as {level}",
        }
        matched = None
        if norm:
            for line in order_lines:
                if line.tag not in _TITLE_SAFE_TAGS:
                    continue
                lnorm = _norm(line.text)
                if not lnorm:
                    continue
                if lnorm == norm or lnorm.startswith(norm) or norm.startswith(lnorm):
                    matched = line
                    break
        if matched is not None:
            finding["element_index"] = matched.index
            finding["current_tag"] = matched.tag
            if matched.tag == level:
                # already correct in the current file — nothing to teach
                continue
        findings.append(finding)
    if not findings:
        return dict(PASS_TARGET)
    return {"status": "fail", "findings": findings}


def flagged_pages(cohort_row: dict) -> list[int]:
    """1-based pages the checker's vision details flagged in a manifest row."""
    pages: set[int] = set()
    for failure in cohort_row.get("checker_failures") or []:
        if failure.get("rule_id") != "headings-nesting":
            continue
        for detail in failure.get("details") or []:
            text = str(detail)
            m = _PAGE_RE.match(text)
            if m and "->" in text:
                pages.add(int(m.group(1)))
    return sorted(pages)


def to_conversation(rec: dict) -> dict:
    """Wrap a record into the training conversation format (matches the
    heading corpus builder's shape exactly)."""
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": rec["image"]},
                    {"type": "text", "text": rec["prompt"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": json.dumps(rec["target"], ensure_ascii=False)}
                ],
            },
        ],
        "meta": {
            "doc_id": rec["doc_id"],
            "page": rec["page"],
            "task": "heading_hierarchy",
            "variant": rec["variant"],
            "source_family": "lamc",
        },
    }


# --------------------------------------------------------------------------- #
# IO helpers (production prompt / render / delivered inspection)
# --------------------------------------------------------------------------- #


def _render_page(pdf_path: Path, page_num: int, out_png: Path, dpi: int) -> None:
    import fitz

    with fitz.open(str(pdf_path)) as doc:
        page = doc[page_num - 1]
        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=matrix)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        pix.save(str(out_png))


def _delivered_headings_by_page(delivered_path: Path) -> dict[int, list[tuple[str, str]]]:
    """1-based page -> [(level, text)] for every H1-H6 in the delivered file."""
    import pikepdf

    import project_remedy.pdf_fixer as PF

    by_page: dict[int, list[tuple[str, str]]] = {}
    with pikepdf.open(delivered_path) as pdf:
        mcid_cache: dict[int, dict] = {}
        for node, _depth, _parent in PF.walk_structure_tree(pdf):
            stype = PF._get_struct_type(node)
            if not re.match(r"^H[1-6]$", stype):
                continue
            page_idx = PF._shared_find_node_page(node, pdf)
            if page_idx is None or page_idx < 0:
                continue
            text = PF._structure_node_text(node)
            if not text:
                if page_idx not in mcid_cache:
                    try:
                        mcid_cache[page_idx] = PF._extract_mcid_text(pdf.pages[page_idx])
                    except Exception:
                        mcid_cache[page_idx] = {}
                text = " ".join(
                    str(mcid_cache[page_idx].get(m, ""))
                    for m in PF._get_node_mcids(node)
                ).strip()
            by_page.setdefault(page_idx + 1, []).append((stype, text))
    return by_page


def _page_record(
    pdf_path: Path,
    page: int,
    *,
    doc_id: str,
    variant: str,
    target: dict,
    renders_dir: Path,
    dpi: int,
) -> dict | None:
    from project_remedy.pdf_vision import _get_page_structure_order
    from project_remedy.vision_prompts import heading_hierarchy_quality_prompt

    # include_mcid_text matches the (enriched) production heading prompt.
    order = _get_page_structure_order(pdf_path, page, include_mcid_text=True)
    png_name = f"lamc_{doc_id}_p{page}_{dpi}dpi.png"
    try:
        _render_page(pdf_path, page, renders_dir / png_name, dpi)
    except Exception as exc:  # noqa: BLE001
        print(f"  render failed {pdf_path.name} p{page}: {exc}", file=sys.stderr)
        return None
    return to_conversation({
        "image": f"renders/{png_name}",
        "prompt": heading_hierarchy_quality_prompt(logical_order=order),
        "target": target,
        "doc_id": f"lamc_{doc_id}",
        "page": page,
        "variant": variant,
    })


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cohort", type=Path, required=True,
                    help="heading-only residual jsonl (source/output/checker_failures)")
    ap.add_argument("--delivered", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bulk-pass-pages", type=int, default=300,
                    help="max lamc_delivered_pass records (0 = skip)")
    ap.add_argument("--dpi", type=int, default=150,
                    help="render dpi — 150 matches production inference")
    args = ap.parse_args(argv)

    renders = args.out / "renders"
    renders.mkdir(parents=True, exist_ok=True)
    delivered_by_name = {p.name: p for p in args.delivered.glob("*.pdf")}

    records: list[dict] = []
    counts = {"lamc_false_flag_pass": 0, "lamc_true_fail": 0, "lamc_delivered_pass": 0}
    used_delivered: set[str] = set()

    rows = [json.loads(l) for l in args.cohort.read_text().splitlines() if l.strip()]
    for row in rows:
        source = Path(row["source"])
        output = Path(row["output"])
        base = _HASH_PREFIX.sub("", source.name)
        delivered = delivered_by_name.get(base)
        if delivered is None or not output.exists():
            continue
        used_delivered.add(base)
        doc_id = source.stem[:12]
        try:
            heads = _delivered_headings_by_page(delivered)
        except Exception as exc:  # noqa: BLE001
            print(f"  delivered unreadable {delivered.name}: {exc}", file=sys.stderr)
            continue
        for page in flagged_pages(row):
            if page in heads:
                from project_remedy.pdf_vision import _get_page_structure_order

                order_lines = parse_structure_order_lines(
                    _get_page_structure_order(output, page, include_mcid_text=True))
                target = build_fail_target(order_lines, heads[page])
                variant = "lamc_true_fail" if target["status"] == "fail" \
                    else "lamc_false_flag_pass"
            else:
                target = dict(PASS_TARGET)
                variant = "lamc_false_flag_pass"
            rec = _page_record(output, page, doc_id=doc_id, variant=variant,
                               target=target, renders_dir=renders, dpi=args.dpi)
            if rec:
                records.append(rec)
                counts[variant] += 1
        print(f"[{len(records)} rec] {base[:50]}", flush=True)

    # Bulk domain pass records from delivered pages without headings. The pass
    # label needs no fidelity gate: "the human-certified remediation put no
    # heading on this page" is gold regardless of the file's text fidelity.
    if args.bulk_pass_pages > 0:
        emitted = 0
        for name, dpath in sorted(delivered_by_name.items()):
            if emitted >= args.bulk_pass_pages:
                break
            if name in used_delivered:
                continue  # already contributing hard examples
            try:
                heads = _delivered_headings_by_page(dpath)
                import pikepdf

                with pikepdf.open(dpath) as pdf:
                    n_pages = len(pdf.pages)
            except Exception:
                continue
            for page in range(1, min(n_pages, 3) + 1):  # up to 3 pages per doc
                if emitted >= args.bulk_pass_pages:
                    break
                if page in heads:
                    continue  # only no-heading pages are unambiguous pass gold
                rec = _page_record(dpath, page, doc_id=Path(name).stem[:12],
                                   variant="lamc_delivered_pass",
                                   target=dict(PASS_TARGET),
                                   renders_dir=renders, dpi=args.dpi)
                if rec:
                    records.append(rec)
                    counts["lamc_delivered_pass"] += 1
                    emitted += 1

    out_jsonl = args.out / "records.jsonl"
    with out_jsonl.open("w") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    manifest = {
        "total": len(records), "counts": counts, "dpi": args.dpi,
        "cohort": str(args.cohort), "delivered": str(args.delivered),
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
