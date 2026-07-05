#!/usr/bin/env python3
"""Merge a LoRA adapter into its base Qwen-VL model → a standalone bf16 model.

Produces a plain HF model directory that any server (vLLM, TGI, or HF) can load
with no PEFT/LoRA machinery — the robust path for serving the tuned model.

Usage (Spark venv):
    python tools/finetune/merge_adapter.py \
        --base Qwen/Qwen3-VL-32B-Instruct \
        --adapter outputs/lamc-qwen3vl-32b-lora \
        --out merged/lamc-qwen3vl-32b
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda", help="cuda or cpu for the merge")
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel

    print(f"[merge] base={args.base} adapter={args.adapter} -> {args.out}", flush=True)
    base = AutoModelForImageTextToText.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map=args.device)
    print("[merge] base loaded; attaching adapter ...", flush=True)
    model = PeftModel.from_pretrained(base, str(args.adapter))
    print("[merge] merging weights (merge_and_unload) ...", flush=True)
    merged = model.merge_and_unload()

    args.out.mkdir(parents=True, exist_ok=True)
    print("[merge] saving merged model (bf16 safetensors) ...", flush=True)
    merged.save_pretrained(str(args.out), safe_serialization=True)
    AutoProcessor.from_pretrained(args.base).save_pretrained(str(args.out))
    print(f"[merge] done -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
