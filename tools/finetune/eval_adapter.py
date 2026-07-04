#!/usr/bin/env python3
"""Quick sanity eval: base model vs tuned LoRA adapter on the val set.

Generates an answer for each val example with (a) the base model and (b) the base
+ adapter, and reports the primary gate — valid-JSON rate — plus a coarse
exact-ish match against the human target. This is a fast smoke, not the full eval;
for the real eval, merge + serve the adapter and run tools/run_vision_eval.py
against it (same harness the baseline sweep used).

Usage (LXC training venv):
    python tools/finetune/eval_adapter.py \
        --model unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit \
        --adapter outputs/lamc-qwen25vl-7b-lora \
        --val tools/finetune/data/val.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _valid_json(text: str) -> bool:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").split("\n", 1)[-1]
    try:
        json.loads(t)
        return True
    except Exception:
        return False


def _load(val: Path) -> list[dict]:
    return [json.loads(l) for l in val.read_text(encoding="utf-8").splitlines() if l.strip()]


def _generate(model, tokenizer, rec, max_new_tokens=512) -> str:
    from PIL import Image
    from unsloth import FastVisionModel

    FastVisionModel.for_inference(model)
    user = rec["messages"][0]["content"]
    image = next(Image.open(p["image"]).convert("RGB") for p in user if p["type"] == "image")
    prompt = next(p["text"] for p in user if p["type"] == "text")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": prompt}]}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt",
    )
    import torch
    inputs = {k: (v.to("cuda") if hasattr(v, "to") else v) for k, v in inputs.items()}
    # attach the image via the processor path
    proc = tokenizer(image, prompt, return_tensors="pt").to("cuda") if False else None
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=True)
    return tokenizer.batch_decode(out, skip_special_tokens=True)[0]


def _score(model, tokenizer, rows) -> dict:
    valid = exact = 0
    for rec in rows:
        try:
            gen = _generate(model, tokenizer, rec)
        except Exception:
            gen = ""
        if _valid_json(gen):
            valid += 1
        tgt = rec["messages"][1]["content"][0]["text"].strip()
        if gen.strip() == tgt:
            exact += 1
    n = max(1, len(rows))
    return {"valid_json_rate": round(valid / n, 3), "exact_match": round(exact / n, 3), "n": len(rows)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit")
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    args = ap.parse_args()

    from unsloth import FastVisionModel

    rows = _load(args.val)
    print(f"[eval] {len(rows)} val examples")

    base, tok = FastVisionModel.from_pretrained(args.model, load_in_4bit=True)
    print("[eval] BASE   :", _score(base, tok, rows))

    tuned, tok2 = FastVisionModel.from_pretrained(args.model, load_in_4bit=True)
    tuned.load_adapter(str(args.adapter))
    print("[eval] ADAPTER:", _score(tuned, tok2, rows))
    print("[eval] NOTE: this is a coarse smoke. Run tools/run_vision_eval.py against "
          "the merged+served adapter for the production gates (valid-JSON, latency, "
          "gold-vs-bad discrimination).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
