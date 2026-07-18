#!/usr/bin/env python3
"""Build training data from our DELIVERED remediations — no human labeling.

The delivered `remediated_pdfs/` are human-certified remediations of the source
PDFs. That makes them GOLD: what the remediation CHANGED (source -> delivered) is
exactly the "findings" our verification-format prompts ask the model to produce.
So we can manufacture (image + production prompt -> gold JSON) examples with zero
new labeling.

v1 covers ALT-TEXT QUALITY — the cleanest signal (delivered figure /Alt text is
unambiguous human gold, figures match source<->delivered by visual bbox). The
target is built in the EXACT production schema (page_alt_text_quality_prompt), so
a tuned adapter drops into pdf_vision unchanged.
  - source figure alt unchanged in delivered      -> status=pass
  - delivered replaced missing/placeholder alt     -> status=fail, suggested_alt_text=<delivered alt>
  - delivered merely reworded meaningful source alt -> status=pass
  - delivered artifacted it (no delivered Figure)  -> status=fail, decorative=true, suggested_alt_text=""

heading_hierarchy and reading_order use the same source<->delivered diff idea but
need element/tag alignment; left as TODO below (approach documented).

CRITICAL: only CLEAN delivered files are gold. Many delivered files lost content
(see the content-loss work) — a lossy delivered file is BAD gold, so we gate on
word-fidelity vs source and skip the tail. VALIDATE a sample of targets before
training.

Output: drafts-format JSONL with target PRE-FILLED (reviewed=true, provenance
'delivered-derived'), consumable by finalize_dataset.py.

Usage:
    uv run python tools/finetune/build_delivered_dataset.py \
        --delivered ~/code/lamc_district_forms/lamc_remediated/remediated_pdfs \
        --sources   ~/code/lamc_district_forms/data/visual_match/downloads/lamc \
        --tasks alt_text_quality --max-loss 0.02 \
        --pages-per-doc 4 --render-dpi 200 \
        --out tools/finetune/data/delivered_alt.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

import pikepdf  # noqa: E402
from project_remedy.pdf_vision import (  # noqa: E402
    _get_page_figure_alt_entries,
    _get_page_figure_alt_list,
    _get_page_structure_order,
    render_page_to_image,
)
from project_remedy.vision_prompts import (  # noqa: E402
    heading_hierarchy_quality_prompt,
    page_alt_text_quality_prompt,
    reading_order_prompt,
    wcag_table_verify_prompt,
)

HASH_PREFIX = re.compile(r"^[0-9a-f]{12}_")


def _pair_delivered_to_source(delivered_dir: Path, sources_dir: Path) -> list[tuple[Path, Path]]:
    """Match each delivered PDF to its source by stripping the 12-hex prefix."""
    by_stripped: dict[str, Path] = {}
    for src in sorted(sources_dir.glob("*.pdf")):
        by_stripped.setdefault(HASH_PREFIX.sub("", src.name), src)
    pairs = []
    for deliv in sorted(delivered_dir.glob("*.pdf")):
        src = by_stripped.get(deliv.name)
        if src is not None:
            pairs.append((src, deliv))
    return pairs


def _iou(a, b) -> float:
    if not a or not b:
        return 0.0
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def _match_delivered(src_entry, delivered_entries, used: set[int]):
    """Best delivered figure for a source figure: bbox IoU, else same index."""
    best_i, best_iou = -1, 0.0
    for i, d in enumerate(delivered_entries):
        if i in used:
            continue
        s = _iou(src_entry.bbox, d.bbox)
        if s > best_iou:
            best_iou, best_i = s, i
    if best_i >= 0 and best_iou >= 0.3:
        return best_i, delivered_entries[best_i]
    # fallback: same 1-based figure_index (page order preserved by remediation)
    for i, d in enumerate(delivered_entries):
        if i not in used and d.figure_index == src_entry.figure_index:
            return i, d
    return -1, None


# Objective "the alt is deficient" signal. A present, non-trivial source alt that
# the human merely REWORDED in delivery is NOT a defect — training on those
# subjective "improved" fails taught the model to flag adequate alt (every eval
# real_pass false-positive was this `quality/improved` case). Fail only when the
# source alt is objectively missing / placeholder / generic.
_PLACEHOLDER_ALTS = {
    "image", "photo", "photograph", "figure", "fig", "graphic", "picture", "pic",
    "img", "logo", "icon", "chart", "diagram", "screenshot", "banner", "decorative",
    "alt text", "placeholder", "untitled",
}
_FILENAME_ALT_RE = re.compile(
    r"(.*\.(png|jpe?g|gif|svg|bmp|tiff?|webp)$)|(^(image|img|figure|fig|pic|graphic)[\s_-]?\d+$)"
)


def _alt_is_placeholder(alt: str) -> bool:
    """True when a source alt string is objectively deficient (missing/generic)."""
    a = (alt or "").strip().lower()
    if len(a) < 5:
        return True
    if a in _PLACEHOLDER_ALTS:
        return True
    return bool(_FILENAME_ALT_RE.match(a))


def _alt_target(src_entries, delivered_entries) -> dict:
    """Gold {figures:[...]} from the source->delivered alt diff (objective labels)."""
    figs = []
    used: set[int] = set()
    for s in src_entries:
        di, d = _match_delivered(s, delivered_entries, used)
        if d is None:
            # delivered dropped/artifacted this figure -> mark decorative
            figs.append({"figure_index": s.figure_index, "status": "fail",
                         "severity": "info", "decorative": True,
                         "issue_type": "decorative", "message": "",
                         "suggested_alt_text": "", "confidence": 1.0})
            continue
        used.add(di)
        d_alt = (d.current_alt_text or "").strip()
        s_alt = (s.current_alt_text or "").strip()
        if not d_alt:
            figs.append({"figure_index": s.figure_index, "status": "fail",
                         "severity": "info", "decorative": True,
                         "issue_type": "decorative", "message": "",
                         "suggested_alt_text": "", "confidence": 1.0})
        elif d_alt == s_alt:
            figs.append({"figure_index": s.figure_index, "status": "pass",
                         "severity": "info", "decorative": False,
                         "issue_type": "", "message": "",
                         "suggested_alt_text": "", "confidence": 1.0})
        elif _alt_is_placeholder(s_alt):
            # Source alt is objectively deficient (empty/generic/filename) and the
            # human wrote a real one -> a genuine, non-subjective fail.
            figs.append({"figure_index": s.figure_index, "status": "fail",
                         "severity": "warning", "decorative": False,
                         "issue_type": "missing_or_placeholder",
                         "message": "source alt text is missing or a placeholder",
                         "suggested_alt_text": d_alt, "confidence": 1.0})
        else:
            # Source alt is present and non-trivial; a human rewording is not a
            # defect. Pass (this is the discrimination signal the old "improved"
            # fail destroyed).
            figs.append({"figure_index": s.figure_index, "status": "pass",
                         "severity": "info", "decorative": False,
                         "issue_type": "", "message": "",
                         "suggested_alt_text": "", "confidence": 1.0})
    return {"figures": figs}


def _pass_target(delivered_entries) -> dict:
    """Gold {figures:[...]} where EVERY figure with real delivered alt is `pass`.

    Used for the v2 "already-good" negative: when the current alt shown to the
    model IS the human-approved delivered alt, the correct finding is `pass`
    (nothing to improve). This is the discrimination signal the source->delivered
    'fail/improve' examples lack — without it the model learns to always flag.
    """
    figs = []
    for d in delivered_entries:
        d_alt = (d.current_alt_text or "").strip()
        if not d_alt:
            continue  # no delivered alt = decorative/artifacted; not a pass case
        figs.append({"figure_index": d.figure_index, "status": "pass",
                     "severity": "info", "decorative": False,
                     "issue_type": "", "message": "",
                     "suggested_alt_text": "", "confidence": 1.0})
    return {"figures": figs}


_HEADING_TAGS = {"H1", "H2", "H3", "H4", "H5", "H6"}
_STRUCT_LINE = re.compile(r'^\s*(\d+)\.\s+/(\w+)(?:\s+\(text:\s*"(.*?)"\))?', re.S)
# Cap the structure-order text fed to the prompt. Big tables produce hundreds of
# TD lines -> huge sequences that blow up training memory (OOM/thrash on the 32B)
# and add nothing the page image doesn't already show. 60 lines keeps the header
# rows + enough context to judge structure.
_MAX_ORDER_LINES = 60


def _cap_order(order_str: str, max_lines: int = _MAX_ORDER_LINES) -> str:
    lines = (order_str or "").splitlines()
    if len(lines) <= max_lines:
        return order_str
    return "\n".join(lines[:max_lines] + [f"    ... (+{len(lines) - max_lines} more elements)"])


def _parse_structure_order(order_str: str) -> list[tuple[int, str, str]]:
    """Parse `_get_page_structure_order` output → [(element_index, tag, text)]."""
    out = []
    for line in (order_str or "").splitlines():
        m = _STRUCT_LINE.match(line)
        if not m:
            continue
        idx, tag, text = int(m.group(1)), m.group(2), (m.group(3) or "").strip()
        out.append((idx, tag, text))
    return out


def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "")).strip().lower()[:120]


def _heading_target(src_order_str: str, deliv_order_str: str) -> dict:
    """Gold heading corrections from the source->delivered tag diff.

    Align elements by visible text; every element whose tag CHANGED in a way that
    involves a heading (H-level change, or P/Span<->H promotion/demotion) is a
    correction, keyed by the SOURCE element_index (what the model is shown).
    """
    src = _parse_structure_order(src_order_str)
    deliv = _parse_structure_order(deliv_order_str)
    # text -> delivered tag (first occurrence wins; skip empty text)
    deliv_by_text: dict[str, str] = {}
    for _i, tag, text in deliv:
        key = _norm(text)
        if key and key not in deliv_by_text:
            deliv_by_text[key] = tag
    findings = []
    for idx, s_tag, text in src:
        key = _norm(text)
        if not key or key not in deliv_by_text:
            continue
        d_tag = deliv_by_text[key]
        if d_tag == s_tag:
            continue
        involves_heading = (s_tag in _HEADING_TAGS) or (d_tag in _HEADING_TAGS)
        if not involves_heading:
            continue
        findings.append({
            "severity": "warning" if {s_tag, d_tag} <= (_HEADING_TAGS | {"P", "Span"}) else "error",
            "element_index": idx, "current_tag": s_tag,
            "visible_text": text[:120],
            "message": f"heading level corrected {s_tag}->{d_tag} in remediation",
            "correct_tag": d_tag, "suggested_fix": f"Retag as {d_tag}",
        })
    return {"status": "fail" if findings else "pass", "findings": findings}


def _reemit_with_tag_changes(order_str: str, changes: dict[int, str]) -> str:
    """Rewrite specific elements' /TAG in a structure-order string (by index)."""
    out = []
    for line in (order_str or "").splitlines():
        m = re.match(r"^(\s*)(\d+)(\.\s+/)(\w+)(.*)$", line)
        if m and int(m.group(2)) in changes:
            out.append(f"{m.group(1)}{m.group(2)}{m.group(3)}{changes[int(m.group(2))]}{m.group(5)}")
        else:
            out.append(line)
    return "\n".join(out)


def _heading_corruption_examples(deliv_order: str, parsed, page: int, emit_pass: bool):
    """CORRUPTION-SYNTHESIS heading examples (sources are untagged, so we can't
    diff; instead corrupt the DELIVERED gold structure into the 'current' input
    and target the fix back). Returns [(prompt, target, provenance)].

    Emits 1 fail + 1 pass per page (balanced, applying the v1->v2 lesson):
      - fail: flatten headings to H1 (the real 'default to H1' trap); for all-H1
        pages instead demote the top heading (title) to P (title-as-body trap).
      - pass: the delivered (correct) order -> status=pass.
    The corrected `correct_tag` in every finding is the human-gold delivered tag.
    """
    headings = [(i, t, x) for (i, t, x) in parsed if t in _HEADING_TAGS]
    if not headings:
        return []
    examples = []
    non_h1 = [(i, t, x) for (i, t, x) in headings if t != "H1"]
    if non_h1:  # flatten trap
        changes = {i: "H1" for (i, t, x) in non_h1}
        findings = [{"severity": "warning", "element_index": i, "current_tag": "H1",
                     "visible_text": x[:120],
                     "message": f"heading flattened to H1; visual hierarchy is {t}",
                     "correct_tag": t, "suggested_fix": f"Retag as {t}"}
                    for (i, t, x) in non_h1]
    else:  # title-as-body trap (all headings already H1)
        i, t, x = headings[0]
        changes = {i: "P"}
        findings = [{"severity": "error", "element_index": i, "current_tag": "P",
                     "visible_text": x[:120],
                     "message": "title/section heading tagged as body P",
                     "correct_tag": t, "suggested_fix": f"Retag as {t}"}]
    corrupted = _reemit_with_tag_changes(deliv_order, changes)
    examples.append((heading_hierarchy_quality_prompt(logical_order=corrupted),
                     {"status": "fail", "findings": findings}, "delivered-derived-corrupt"))
    if emit_pass:
        examples.append((heading_hierarchy_quality_prompt(logical_order=deliv_order),
                         {"status": "pass", "findings": []}, "delivered-derived-pass"))
    return examples


def _table_corruption_examples(deliv_order: str, parsed, emit_pass: bool):
    """CORRUPTION-SYNTHESIS table-structure examples (WCAG 1.3.1).

    The delivered tables are human-tagged gold (TH header cells, regular TR/TD).
    Corrupt by demoting every /TH -> /TD (the classic "data table with no header
    cells" error) as the 'current' structure; target = flag missing headers.
    Pass = the correct delivered structure. Renders the src image (identical).
    Returns [(prompt, target, provenance)].
    """
    deliv_order = _cap_order(deliv_order)  # bound sequence length (big tables OOM)
    parsed = _parse_structure_order(deliv_order)
    has_table = any(t in {"Table", "TH", "TD"} for (_i, t, _x) in parsed)
    th_cells = [(i, t, x) for (i, t, x) in parsed if t == "TH"]
    if not has_table or not th_cells:
        return []  # need a table WITH header cells (gold) to corrupt
    changes = {i: "TD" for (i, t, x) in th_cells}
    corrupted = _reemit_with_tag_changes(deliv_order, changes)
    findings = [{
        "issue_id": "missing_table_headers", "severity": "error",
        "message": (f"Data table has no header cells; {len(th_cells)} cell(s) "
                    "that are column/row headers are tagged TD instead of TH"),
        "fixer": "fix_table_headers",
    }]
    out = [(wcag_table_verify_prompt(corrupted),
            {"status": "fail", "confidence": 0.9, "findings": findings},
            "delivered-derived-corrupt")]
    if emit_pass:
        out.append((wcag_table_verify_prompt(deliv_order),
                    {"status": "pass", "confidence": 0.9, "findings": []},
                    "delivered-derived-pass"))
    return out


def _reading_order_corruption_examples(deliv_order: str, parsed, emit_pass: bool):
    """CORRUPTION-SYNTHESIS reading-order examples.

    Delivered files are the gold order. Corrupt by moving the latter content
    region before the first region, mirroring the common multi-column/sidebar
    failure mode where the structure tree starts in the wrong visual region.
    """
    deliv_order = _cap_order(deliv_order)
    lines = [line for line in deliv_order.splitlines() if line.strip()]
    parsed = _parse_structure_order(deliv_order)
    content = [(idx, tag, text) for idx, tag, text in parsed if tag not in {"Document", "Sect"}]
    if len(lines) < 6 or len(content) < 4:
        return []

    pivot = max(2, len(lines) // 2)
    corrupted_lines = lines[pivot:] + lines[:pivot]
    corrupted = "\n".join(corrupted_lines)
    if corrupted == deliv_order:
        return []

    has_table = any(tag in {"Table", "TR", "TH", "TD"} for _idx, tag, _text in parsed)
    page_layout = "table_directory" if has_table else "unknown_complex"
    correct_order = [idx for idx, _tag, _text in content]
    findings = [{
        "severity": "error",
        "description": (
            "The tagged reading order starts in a later visual region before "
            "earlier body content, which can make columns, sidebars, or tables "
            "read out of sequence."
        ),
        "suggestion": "Restore the delivered gold reading order: "
                      + ", ".join(map(str, correct_order[:40])),
    }]
    out = [(
        reading_order_prompt(structure_order=corrupted),
        {"page_layout": page_layout, "issues": findings,
         "summary": "Reading order is corrupted; restore the delivered structure-tree order."},
        "delivered-derived-reading-order-corrupt",
    )]
    if emit_pass:
        out.append((
            reading_order_prompt(structure_order=deliv_order),
            {"page_layout": page_layout, "issues": [],
             "summary": "Reading order matches the delivered gold structure."},
            "delivered-derived-reading-order-pass",
        ))
    return out


def _page_examples(task: str, src: Path, deliv: Path, page: int, emit_pass: bool):
    """Return [(prompt, target_dict, provenance)] for one page & task (may be empty)."""
    out = []
    if task == "alt_text_quality":
        src_entries = _get_page_figure_alt_entries(src, page)
        if not src_entries:
            return out  # production skips figureless pages
        deliv_entries = _get_page_figure_alt_entries(deliv, page)
        try:
            prompt = page_alt_text_quality_prompt(figure_list=_get_page_figure_alt_list(src, page))
        except Exception:
            return out
        out.append((prompt, _alt_target(src_entries, deliv_entries), "delivered-derived"))
        if emit_pass:
            pt = _pass_target(deliv_entries)
            if pt["figures"]:
                try:
                    pp = page_alt_text_quality_prompt(figure_list=_get_page_figure_alt_list(deliv, page))
                    out.append((pp, pt, "delivered-derived-pass"))
                except Exception:
                    pass
    elif task == "heading_hierarchy":
        # Sources are UNTAGGED (no structure tree), so source<->delivered diff
        # yields nothing. Use corruption-synthesis from the DELIVERED gold instead
        # (the source & delivered pages render identically, so the src image is
        # correct). If the source ever IS tagged with heading structure, the diff
        # path (_heading_target) is the higher-fidelity signal — prefer it.
        src_order = _get_page_structure_order(src, page)
        src_parsed = _parse_structure_order(src_order)
        if any(t in _HEADING_TAGS for _i, t, _x in src_parsed):
            deliv_order = _get_page_structure_order(deliv, page)
            tgt = _heading_target(src_order, deliv_order)
            out.append((heading_hierarchy_quality_prompt(logical_order=src_order),
                        tgt, "delivered-derived"))
            if emit_pass:
                out.append((heading_hierarchy_quality_prompt(logical_order=deliv_order),
                            {"status": "pass", "findings": []}, "delivered-derived-pass"))
            return out
        deliv_order = _get_page_structure_order(deliv, page)
        deliv_parsed = _parse_structure_order(deliv_order)
        out.extend(_heading_corruption_examples(deliv_order, deliv_parsed, page, emit_pass))
    elif task == "table_structure":
        # Corruption-synthesis from the delivered table gold (abundant in this
        # forms/tables corpus). Src & delivered render identically -> src image.
        deliv_order = _get_page_structure_order(deliv, page)
        deliv_parsed = _parse_structure_order(deliv_order)
        out.extend(_table_corruption_examples(deliv_order, deliv_parsed, emit_pass))
    elif task == "reading_order":
        deliv_order = _get_page_structure_order(deliv, page)
        deliv_parsed = _parse_structure_order(deliv_order)
        out.extend(_reading_order_corruption_examples(deliv_order, deliv_parsed, emit_pass))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delivered", type=Path, required=True)
    ap.add_argument("--sources", type=Path, required=True)
    ap.add_argument("--tasks", default="alt_text_quality")
    ap.add_argument("--max-loss", type=float, default=0.02,
                    help="Skip a delivered file if its word-loss vs source exceeds this "
                         "(lossy delivered = bad gold).")
    ap.add_argument("--pages-per-doc", type=int, default=4)
    ap.add_argument("--render-dpi", type=int, default=200)
    ap.add_argument("--max-docs", type=int, default=0, help="0 = all")
    ap.add_argument("--emit-pass-negatives", action="store_true",
                    help="v2: also emit an 'already-good -> pass' example per page "
                         "(prompt shows the delivered/good alt as current). Balances "
                         "pass vs fail so the model learns to discriminate, not "
                         "always-flag.")
    ap.add_argument("--out", type=Path, default=Path("tools/finetune/data/delivered_alt.jsonl"))
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    supported = {"alt_text_quality", "heading_hierarchy", "table_structure", "reading_order"}
    bad = [t for t in tasks if t not in supported]
    if bad:
        print(f"unsupported task(s) {bad}; supported: {sorted(supported)} "
              f"(contrast still TODO)", file=sys.stderr)
        tasks = [t for t in tasks if t in supported]

    import run_font_residue_batch as R  # word_fidelity gate

    pairs = _pair_delivered_to_source(args.delivered, args.sources)
    if args.max_docs:
        pairs = pairs[: args.max_docs]
    renders = args.out.parent / "renders"
    renders.mkdir(parents=True, exist_ok=True)

    n = kept_docs = skipped_lossy = n_pass = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for src, deliv in pairs:
            # gate on fidelity: lossy delivered files are not gold
            try:
                hl, tot = R.word_fidelity(str(src), str(deliv))
                loss = hl / tot if tot else 1.0
            except Exception:
                loss = 1.0
            if loss > args.max_loss:
                skipped_lossy += 1
                continue
            kept_docs += 1
            doc_id = deliv.stem
            try:
                with pikepdf.open(src) as p:
                    npages = len(p.pages)
            except Exception:
                continue
            for page in range(1, min(args.pages_per_doc, npages) + 1):
                png = renders / f"{doc_id}_p{page}_{args.render_dpi}dpi.png"
                rendered = png.exists()
                for task in tasks:
                    try:
                        examples = _page_examples(task, src, deliv, page, args.emit_pass_negatives)
                    except Exception as e:
                        print(f"  {doc_id} p{page} {task}: {e}", file=sys.stderr)
                        continue
                    if not examples:
                        continue
                    if not rendered:
                        try:
                            tmp = render_page_to_image(src, page, dpi=args.render_dpi)
                            Path(tmp).replace(png)
                            rendered = True
                        except Exception as e:
                            print(f"  {doc_id} p{page}: render {e}", file=sys.stderr)
                            break
                    for prompt, target, prov in examples:
                        fh.write(json.dumps({
                            "doc_id": doc_id, "page": page, "task": task,
                            "image": str(png.resolve()), "prompt": prompt,
                            "draft_target": "", "target": json.dumps(target, ensure_ascii=False),
                            "reviewed": True, "provenance": prov,
                            "source_pdf": str(src.resolve()), "delivered_pdf": str(deliv.resolve()),
                        }, ensure_ascii=False) + "\n")
                        n += 1
                        if prov.endswith("-pass"):
                            n_pass += 1

    print(f"paired={len(pairs)} kept_clean={kept_docs} skipped_lossy={skipped_lossy} tasks={tasks} "
          f"-> {n} training records ({n_pass} 'already-good' pass negatives) at {args.out}")
    print("VALIDATE a sample of `target` before training. Then finalize_dataset.py "
          "(no --use-drafts-as-target; these are already reviewed=true gold).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
