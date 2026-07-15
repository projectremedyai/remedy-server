#!/usr/bin/env python3
"""Build a CORRECTABLE starter training set from real corpus pages.

Emits one record per (page x task) using the EXACT production prompt + render
path (so training matches inference), optionally pre-filled with a draft answer
from a served base model that a human then corrects. This is the human-in-the-loop
label pipeline the strategy doc calls for.

Output (one JSON per line, correctable):
    {"doc_id","page","task","image","prompt","draft_target","target":"",
     "reviewed":false,"source_pdf": "..."}

Workflow:
    1. build_starter_dataset.py  -> drafts.jsonl   (this script)
    2. a human edits each `target` (copy+fix `draft_target`) and sets reviewed=true
    3. finalize_dataset.py       -> train.jsonl / val.jsonl  (Unsloth format)

Do NOT point --pdfs at the v1 eval corpus (tools/corpus_annotations/v1) — that is
the held-out eval set. Use the large source files, delivered pairs, or crawl pages.

Usage:
    uv run python tools/finetune/build_starter_dataset.py \
        --pdfs ~/code/lamc_district_forms/data/visual_match/downloads/lamc \
        --tasks reading_order,heading_hierarchy,alt_text_quality \
        --max-docs 6 --pages-per-doc 3 --render-dpi 200 \
        --draft-endpoint http://192.168.68.64:11434 --draft-model qwen2.5vl:7b \
        --out tools/finetune/data/drafts.jsonl
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from project_remedy.pdf_vision import (  # noqa: E402
    _get_page_figure_alt_entries,
    _get_page_figure_alt_list,
    _get_page_structure_order,
    render_page_to_image,
)
from project_remedy.vision_prompts import (  # noqa: E402
    contrast_detection_prompt,
    heading_hierarchy_quality_prompt,
    page_alt_text_quality_prompt,
    reading_order_prompt,
)


def build_prompt(task: str, pdf: Path, page: int) -> str | None:
    """Exact production prompt (mirrors export_corpus_jsonl._build_prompt)."""
    if task == "reading_order":
        return reading_order_prompt(structure_order=_get_page_structure_order(pdf, page))
    if task == "heading_hierarchy":
        return heading_hierarchy_quality_prompt(logical_order=_get_page_structure_order(pdf, page))
    if task == "alt_text_quality":
        if not _get_page_figure_alt_entries(pdf, page):
            return None  # production skips figureless pages
        return page_alt_text_quality_prompt(figure_list=_get_page_figure_alt_list(pdf, page))
    if task == "contrast":
        return contrast_detection_prompt("AA")
    raise ValueError(f"unknown task: {task}")


def draft_from_endpoint(url: str, model: str, png: Path, prompt: str, timeout: int) -> str:
    """Ask a served Ollama vision model for a draft answer (best-effort)."""
    import urllib.request

    img_b64 = base64.b64encode(png.read_bytes()).decode()
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [img_b64]}],
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        url.rstrip("/") + "/api/chat", data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read()).get("message", {}).get("content", "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdfs", type=Path, required=True,
                    help="A PDF file or a directory of PDFs (NOT the v1 eval corpus).")
    ap.add_argument("--tasks", default="reading_order,heading_hierarchy,alt_text_quality")
    ap.add_argument("--max-docs", type=int, default=6)
    ap.add_argument("--pages-per-doc", type=int, default=3)
    ap.add_argument("--render-dpi", type=int, default=200)
    ap.add_argument("--draft-endpoint", default=None,
                    help="Optional Ollama vision URL to pre-fill draft answers.")
    ap.add_argument("--draft-model", default="qwen2.5vl:7b")
    ap.add_argument("--draft-timeout", type=int, default=120)
    ap.add_argument("--out", type=Path, default=Path("tools/finetune/data/drafts.jsonl"))
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    pdfs = sorted(args.pdfs.glob("*.pdf")) if args.pdfs.is_dir() else [args.pdfs]
    pdfs = pdfs[: args.max_docs]
    renders = args.out.parent / "renders"
    renders.mkdir(parents=True, exist_ok=True)

    import pikepdf

    n = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for pdf in pdfs:
            doc_id = pdf.stem
            try:
                with pikepdf.open(pdf) as p:
                    npages = len(p.pages)
            except Exception as e:
                print(f"  skip {doc_id}: {e}", file=sys.stderr)
                continue
            # production render/structure paths are 1-indexed (doc[page_num - 1])
            for page in range(1, min(args.pages_per_doc, npages) + 1):
                png = renders / f"{doc_id}_p{page}_{args.render_dpi}dpi.png"
                try:
                    tmp = render_page_to_image(pdf, page, dpi=args.render_dpi)
                    Path(tmp).replace(png)
                except Exception as e:
                    print(f"  render fail {doc_id} p{page}: {e}", file=sys.stderr)
                    continue
                for task in tasks:
                    try:
                        prompt = build_prompt(task, pdf, page)
                    except Exception as e:
                        print(f"  prompt fail {doc_id} p{page} {task}: {e}", file=sys.stderr)
                        continue
                    if prompt is None:
                        continue
                    draft = ""
                    if args.draft_endpoint:
                        try:
                            draft = draft_from_endpoint(
                                args.draft_endpoint, args.draft_model, png, prompt,
                                args.draft_timeout)
                        except Exception as e:
                            draft = ""
                            print(f"  draft fail {doc_id} p{page} {task}: {e}", file=sys.stderr)
                    fh.write(json.dumps({
                        "doc_id": doc_id, "page": page, "task": task,
                        "image": str(png.resolve()),
                        "prompt": prompt,
                        "draft_target": draft,
                        "target": "",          # <- a human fills this in (fix the draft)
                        "reviewed": False,
                        "source_pdf": str(pdf.resolve()),
                    }, ensure_ascii=False) + "\n")
                    n += 1
    print(f"wrote {n} correctable records -> {args.out}")
    print("NEXT: a human edits each `target` (correct the `draft_target`) and sets "
          "reviewed=true, then run finalize_dataset.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
