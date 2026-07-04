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
  - delivered gave it a real (different) alt       -> status=fail, suggested_alt_text=<delivered alt>
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
    render_page_to_image,
)
from project_remedy.vision_prompts import page_alt_text_quality_prompt  # noqa: E402

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


def _alt_target(src_entries, delivered_entries) -> dict:
    """Gold {figures:[...]} from the source->delivered alt diff."""
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
        else:
            figs.append({"figure_index": s.figure_index, "status": "fail",
                         "severity": "warning", "decorative": False,
                         "issue_type": "quality", "message": "alt text improved in remediation",
                         "suggested_alt_text": d_alt, "confidence": 1.0})
    return {"figures": figs}


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
    ap.add_argument("--out", type=Path, default=Path("tools/finetune/data/delivered_alt.jsonl"))
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    if tasks != ["alt_text_quality"]:
        print("v1 implements alt_text_quality only; heading_hierarchy / reading_order are TODO.",
              file=sys.stderr)

    import run_font_residue_batch as R  # word_fidelity gate

    pairs = _pair_delivered_to_source(args.delivered, args.sources)
    if args.max_docs:
        pairs = pairs[: args.max_docs]
    renders = args.out.parent / "renders"
    renders.mkdir(parents=True, exist_ok=True)

    n = kept_docs = skipped_lossy = 0
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
                src_entries = _get_page_figure_alt_entries(src, page)
                if not src_entries:
                    continue  # production skips figureless pages
                deliv_entries = _get_page_figure_alt_entries(deliv, page)
                target = _alt_target(src_entries, deliv_entries)
                png = renders / f"{doc_id}_p{page}_{args.render_dpi}dpi.png"
                try:
                    tmp = render_page_to_image(src, page, dpi=args.render_dpi)
                    Path(tmp).replace(png)
                    prompt = page_alt_text_quality_prompt(
                        figure_list=_get_page_figure_alt_list(src, page))
                except Exception as e:
                    print(f"  {doc_id} p{page}: {e}", file=sys.stderr)
                    continue
                fh.write(json.dumps({
                    "doc_id": doc_id, "page": page, "task": "alt_text_quality",
                    "image": str(png.resolve()), "prompt": prompt,
                    "draft_target": "", "target": json.dumps(target, ensure_ascii=False),
                    "reviewed": True, "provenance": "delivered-derived",
                    "source_pdf": str(src.resolve()), "delivered_pdf": str(deliv.resolve()),
                }, ensure_ascii=False) + "\n")
                n += 1

    print(f"paired={len(pairs)} kept_clean={kept_docs} skipped_lossy={skipped_lossy} "
          f"-> {n} alt-text training records at {args.out}")
    print("VALIDATE a sample of `target` before training. Then finalize_dataset.py "
          "(no --use-drafts-as-target; these are already reviewed=true gold).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
