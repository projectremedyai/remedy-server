#!/usr/bin/env python3
"""Generate task-eval predictions from a base or PEFT-adapted Qwen-VL model.

This is the companion producer for ``eval_task_metrics.py``. It reads the
conversation JSONL val files produced by ``finalize_dataset.py``, runs the model
against each user prompt/image, and writes same-order JSONL rows with a
``prediction`` field that the scorer can consume.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_records(val_path: Path, limit: int = 0) -> list[dict[str, Any]]:
    """Load val JSONL and resolve image paths relative to the JSONL directory."""
    base_dir = val_path.resolve().parent
    rows = [
        json.loads(line)
        for line in val_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit > 0:
        rows = rows[:limit]
    for rec in rows:
        for msg in rec.get("messages", []):
            for part in msg.get("content", []):
                if part.get("type") == "image" and isinstance(part.get("image"), str):
                    image_path = Path(part["image"])
                    part["image"] = str(image_path if image_path.is_absolute() else base_dir / image_path)
    return rows


def record_key(rec: dict[str, Any], index: int) -> str:
    """Match eval_task_metrics.record_key for prediction/gold alignment."""
    meta = rec.get("meta") or {}
    if meta.get("example_id"):
        return str(meta["example_id"])
    parts = [
        str(meta.get("doc_id") or rec.get("doc_id") or ""),
        str(meta.get("page") or rec.get("page") or rec.get("page_index") or ""),
        str(meta.get("task") or rec.get("task") or ""),
        str(meta.get("variant") or rec.get("variant") or ""),
    ]
    if any(parts):
        return "|".join(parts)
    return str(index)


def task_name(rec: dict[str, Any]) -> str:
    meta = rec.get("meta") or {}
    return str(meta.get("task") or rec.get("task") or "")


def prediction_row(rec: dict[str, Any], index: int, prediction: str) -> dict[str, Any]:
    meta = dict(rec.get("meta") or {})
    key = record_key(rec, index)
    return {
        "example_id": key,
        "task": task_name(rec),
        "prediction": prediction,
        "meta": meta,
    }


def _generate(model, processor, rec: dict[str, Any], max_new_tokens: int) -> str:
    from PIL import Image

    user_msgs = rec["messages"][:-1]
    images = [
        Image.open(part["image"]).convert("RGB")
        for msg in user_msgs
        for part in msg.get("content", [])
        if part.get("type") == "image"
    ]
    text = processor.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_ids = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="base HF model id or local model dir")
    ap.add_argument("--adapter", type=Path, default=None, help="optional PEFT adapter dir")
    ap.add_argument("--val", type=Path, required=True, help="conversation val JSONL")
    ap.add_argument("--out", type=Path, required=True, help="prediction JSONL output")
    ap.add_argument("--limit", type=int, default=0, help="0 means all records")
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--attn-implementation", default="sdpa")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    rows = load_records(args.val, args.limit)
    print(f"[generate-predictions] records={len(rows)} model={args.model}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=args.attn_implementation,
    )
    if args.adapter is not None:
        from peft import PeftModel

        print(f"[generate-predictions] attaching adapter={args.adapter}", flush=True)
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.config.use_cache = True

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for i, rec in enumerate(rows):
            pred = _generate(model, processor, rec, args.max_new_tokens)
            fh.write(json.dumps(prediction_row(rec, i, pred), ensure_ascii=False) + "\n")
            fh.flush()
            print(f"[generate-predictions] {i + 1}/{len(rows)} done", flush=True)
    print(f"[generate-predictions] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
