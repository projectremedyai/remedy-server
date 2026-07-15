#!/usr/bin/env python3
"""Quick HF/peft sanity eval: base vs LoRA-tuned Qwen-VL on the val set.

For the bf16 adapters produced by train_lora_vision_hf.py (the DGX Spark path).
The Unsloth eval_adapter.py can't load these — this uses plain transformers+peft.

Generates the answer for each val page with (a) the base model and (b) base+LoRA,
reports valid-JSON rate for each, and prints gold / base / tuned side by side so a
human can see whether the tuning actually improved the alt-text quality (the thing
valid-JSON can't measure).

Usage (Spark venv):
    python tools/finetune/eval_adapter_hf.py \
        --model Qwen/Qwen3-VL-32B-Instruct \
        --adapter outputs/lamc-qwen3vl-32b-lora \
        --val data/val.jsonl --limit 10
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def _valid_json(text: str) -> bool:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?", "", t).rstrip("`").strip()
    try:
        json.loads(t)
        return True
    except Exception:
        return False


def _load(val: Path, limit: int) -> list[dict]:
    base_dir = val.resolve().parent
    rows = [json.loads(l) for l in val.read_text(encoding="utf-8").splitlines() if l.strip()]
    for rec in rows:
        for msg in rec["messages"]:
            for part in msg.get("content", []):
                if part.get("type") == "image" and isinstance(part.get("image"), str):
                    p = Path(part["image"])
                    part["image"] = str(p if p.is_absolute() else base_dir / p)
    return rows[:limit] if limit > 0 else rows


def _generate(model, processor, rec, max_new_tokens=384) -> str:
    from PIL import Image

    user_msgs = rec["messages"][:-1]  # drop the assistant target
    images = [Image.open(p["image"]).convert("RGB")
              for m in user_msgs for p in m.get("content", []) if p.get("type") == "image"]
    text = processor.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_ids = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


def _score(gens: list[str]) -> float:
    n = max(1, len(gens))
    return round(sum(_valid_json(g) for g in gens) / n, 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=10,
                    help="How many val examples to eval (32B generation is slow).")
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--show-chars", type=int, default=400)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel

    rows = _load(args.val, args.limit)
    print(f"[eval-hf] {len(rows)} val examples | model={args.model}")

    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels, use_fast=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    model.config.use_cache = True

    print("[eval-hf] generating BASE ...", flush=True)
    base_gens = []
    for i, r in enumerate(rows):
        base_gens.append(_generate(model, processor, r, args.max_new_tokens))
        print(f"[eval-hf]   base {i+1}/{len(rows)} done", flush=True)

    print("[eval-hf] attaching adapter + generating TUNED ...", flush=True)
    tuned_model = PeftModel.from_pretrained(model, str(args.adapter))
    tuned_model.config.use_cache = True
    tuned_gens = []
    for i, r in enumerate(rows):
        tuned_gens.append(_generate(tuned_model, processor, r, args.max_new_tokens))
        print(f"[eval-hf]   tuned {i+1}/{len(rows)} done", flush=True)

    print(f"\n[eval-hf] valid_json  BASE={_score(base_gens)}  TUNED={_score(tuned_gens)}")
    changed = sum(1 for b, t in zip(base_gens, tuned_gens) if b.strip() != t.strip())
    print(f"[eval-hf] outputs changed by tuning: {changed}/{len(rows)}\n")

    sc = args.show_chars
    for i, (rec, b, t) in enumerate(zip(rows, base_gens, tuned_gens)):
        gold = rec["messages"][-1]["content"][0]["text"].strip()
        img = next((p["image"] for m in rec["messages"] for p in m.get("content", [])
                    if p.get("type") == "image"), "?")
        print("=" * 88)
        print(f"[{i}] {Path(img).name}")
        print(f"--- GOLD  : {gold[:sc]}")
        print(f"--- BASE  : {b[:sc]}")
        print(f"--- TUNED : {t[:sc]}")
    print("=" * 88)
    print("[eval-hf] NOTE: this is a coarse/qualitative sanity check. The real gate "
          "is merge+serve + tools/run_vision_eval.py (gold-vs-bad discrimination).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
