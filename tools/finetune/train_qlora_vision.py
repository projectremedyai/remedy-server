#!/usr/bin/env python3
"""QLoRA fine-tune of a Qwen-VL model on the LAMC remediation tasks (Unsloth).

This is the DRY-RUN trainer: prove the pipeline on Qwen2.5-VL-7B on the RTX 4080
(16 GB, 4-bit), then scale to Qwen3-VL-32B on a rented H100 by changing --model
and the LoRA rank — the script is otherwise identical.

Input: a conversation JSONL produced by finalize_dataset.py, where each line is
    {"messages": [
        {"role":"user","content":[{"type":"image","image":"<abs png path>"},
                                    {"type":"text","text":"<production prompt>"}]},
        {"role":"assistant","content":[{"type":"text","text":"<target JSON>"}]}
    ]}
The prompts are the EXACT production prompts (vision_prompts.py), so a tuned
adapter drops straight into pdf_vision's OllamaVisionProvider path after merge.

Usage (in the LXC training venv, GPU available):
    python tools/finetune/train_qlora_vision.py \
        --model unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit \
        --train data/train.jsonl --val data/val.jsonl \
        --out outputs/lamc-qwen25vl-7b-lora \
        --epochs 1 --rank 8 --batch 1 --grad-accum 4

Scale to 32B (rented H100, bf16 4-bit):
    python tools/finetune/train_qlora_vision.py \
        --model unsloth/Qwen2.5-VL-32B-Instruct-bnb-4bit \
        --rank 16 --batch 1 --grad-accum 8 ...

NOTE: Unsloth's vision API moves fast. Pin versions per the handoff doc and, if
an import/signature fails, check the current Unsloth vision-finetuning notebook.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_conversations(path: Path) -> list[dict]:
    """Load conversation JSONL and materialize image paths as PIL images."""
    from PIL import Image

    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        # Replace {"type":"image","image":"<path>"} with a loaded PIL image so the
        # Unsloth vision collator can process it.
        for msg in rec["messages"]:
            for part in msg.get("content", []):
                if part.get("type") == "image" and isinstance(part.get("image"), str):
                    part["image"] = Image.open(part["image"]).convert("RGB")
        rows.append(rec)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
                    help="Unsloth 4-bit vision model id (7B for the 4080 dry-run; "
                         "swap to the 32B id on a rented H100).")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/lamc-vlora"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1,
                    help="Override epochs with a fixed step count (use ~30 for a "
                         "smoke dry-run that just proves the loop runs).")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-len", type=int, default=2048)
    ap.add_argument("--tune-vision-layers", action="store_true",
                    help="Also LoRA the vision tower (default: language layers only, "
                         "cheaper and usually enough for JSON-structure tasks).")
    args = ap.parse_args()

    import torch
    from unsloth import FastVisionModel
    from trl import SFTTrainer, SFTConfig
    from unsloth.trainer import UnslothVisionDataCollator

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    print(f"[train] model={args.model} rank={args.rank} bf16={bf16} "
          f"cuda={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU!'}")

    model, tokenizer = FastVisionModel.from_pretrained(
        args.model,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.tune_vision_layers,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.0,
        bias="none",
        random_state=3407,
    )

    train_ds = _load_conversations(args.train)
    val_ds = _load_conversations(args.val) if args.val else None
    print(f"[train] {len(train_ds)} train examples"
          + (f", {len(val_ds)} val" if val_ds else ""))

    FastVisionModel.for_training(model)
    cfg = SFTConfig(
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        warmup_steps=5,
        num_train_epochs=args.epochs if args.max_steps < 0 else 1,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        fp16=not bf16,
        bf16=bf16,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir=str(args.out),
        report_to="none",
        # vision SFT specifics
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=args.max_seq_len,
    )
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=cfg,
    )
    trainer.train()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))          # LoRA adapter
    tokenizer.save_pretrained(str(args.out))
    print(f"[train] saved LoRA adapter -> {args.out}")
    print("[train] to serve: merge with the base model then convert to GGUF for "
          "Ollama, or serve the merged model via vLLM (see the handoff doc).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
