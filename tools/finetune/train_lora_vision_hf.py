#!/usr/bin/env python3
"""HF-native bf16 LoRA fine-tune of a Qwen-VL model (no bitsandbytes / no Unsloth).

Purpose: the DGX Spark (GB10 / Grace-Blackwell, 121 GB unified memory, ARM64,
CUDA 13) has enough memory to LoRA-tune Qwen3-VL-32B in **bf16** — so we skip
4-bit quantization entirely and avoid the one fragile dependency on ARM+Blackwell
(bitsandbytes). This trainer uses only transformers + peft + trl-free HF Trainer,
which install cleanly from the cu13 aarch64 wheels.

Same input format as train_qlora_vision.py (finalize_dataset.py output):
    {"messages": [
        {"role":"user","content":[{"type":"image","image":"renders/x.png"},
                                    {"type":"text","text":"<production prompt>"}]},
        {"role":"assistant","content":[{"type":"text","text":"<target JSON>"}]}
    ]}
so a tuned adapter drops straight into the pdf_vision path after merge.

Usage (Spark venv, GPU available):
    # 7B smoke to validate the ARM/Blackwell stack:
    python tools/finetune/train_lora_vision_hf.py \
        --model Qwen/Qwen2.5-VL-7B-Instruct \
        --train tools/finetune/data/train.jsonl \
        --val   tools/finetune/data/val.jsonl \
        --out   outputs/hf-qwen25vl-7b-smoke --max-steps 30

    # 32B real run (bf16 LoRA, 121 GB unified fits it):
    python tools/finetune/train_lora_vision_hf.py \
        --model Qwen/Qwen3-VL-32B-Instruct \
        --train tools/finetune/data/train.jsonl \
        --val   tools/finetune/data/val.jsonl \
        --out   outputs/lamc-qwen3vl-32b-lora \
        --epochs 1 --rank 16 --alpha 32
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


PROJ_KEYS = ("q_proj", "k_proj", "v_proj", "o_proj",
             "gate_proj", "up_proj", "down_proj")


def _load_rows(path: Path) -> list[dict]:
    """Load conversation JSONL; resolve relative image paths against its dir."""
    base_dir = path.resolve().parent
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        for msg in rec["messages"]:
            for part in msg.get("content", []):
                if part.get("type") == "image" and isinstance(part.get("image"), str):
                    p = Path(part["image"])
                    part["image"] = str(p if p.is_absolute() else base_dir / p)
        rows.append(rec)
    return rows


def _bnb_config(qlora: bool):
    """4-bit NF4 quantization config for QLoRA, or None for bf16.

    Returning None keeps the bf16 base load unchanged. The QLoRA A/B should
    differ only in base-weight precision: 4-bit NF4 with double quantization,
    while compute and LoRA stay bf16.
    """
    if not qlora:
        return None
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def _lora_targets(model, tune_vision: bool) -> list[str]:
    """Exact Linear module names to LoRA — language tower only by default."""
    import torch.nn as nn
    targets = []
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if not any(k in name for k in PROJ_KEYS):
            continue
        is_vision = ("visual" in name) or ("vision" in name)
        if is_vision and not tune_vision:
            continue
        targets.append(name)
    return targets


def _attach_or_create_lora(
    model,
    *,
    init_adapter: Path | None,
    rank: int,
    alpha: int,
    tune_vision: bool,
):
    if init_adapter is not None:
        from peft import PeftModel

        print(f"[train-hf] continuing from adapter={init_adapter}")
        return PeftModel.from_pretrained(model, str(init_adapter), is_trainable=True)

    from peft import LoraConfig, get_peft_model

    targets = _lora_targets(model, tune_vision)
    print(f"[train-hf] LoRA on {len(targets)} Linear modules "
          f"({'incl' if tune_vision else 'excl'} vision tower)")
    lora = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0,
                      bias="none", task_type="CAUSAL_LM", target_modules=targets)
    return get_peft_model(model, lora)


class VisionCollator:
    """Build processor inputs + labels; mask the prompt so we train only on the
    assistant JSON. Relies on the chat-template prompt (add_generation_prompt=True)
    being a token-prefix of the full conversation, which holds for Qwen-VL."""

    def __init__(self, processor):
        self.processor = processor
        from PIL import Image
        self._Image = Image

    def _images(self, messages):
        imgs = []
        for msg in messages:
            for part in msg.get("content", []):
                if part.get("type") == "image":
                    imgs.append(self._Image.open(part["image"]).convert("RGB"))
        return imgs

    def __call__(self, batch):
        import torch

        input_ids_list, labels_list, other = [], [], []
        proc = self.processor
        for rec in batch:
            messages = rec["messages"]
            images = self._images(messages)
            full_text = proc.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False)
            prompt_text = proc.apply_chat_template(
                messages[:-1], tokenize=False, add_generation_prompt=True)

            full = proc(text=[full_text], images=images,
                        return_tensors="pt", padding=False)
            # token length of the prompt (with the same images expanded)
            prompt = proc(text=[prompt_text], images=images,
                          return_tensors="pt", padding=False)
            plen = prompt["input_ids"].shape[1]

            ids = full["input_ids"][0]
            labels = ids.clone()
            labels[:plen] = -100  # mask prompt (image tokens live here too)
            input_ids_list.append(ids)
            labels_list.append(labels)
            # Qwen-VL vision tensors (pixel_values [P,feat], image_grid_thw
            # [n_img,3]) are NOT batched by a leading dim — keep them whole and
            # concatenate across samples along dim 0 below. (Slicing v[0] here
            # corrupts image_grid_thw to 1-D and breaks the vision forward.)
            other.append({k: v for k, v in full.items()
                          if k not in ("input_ids", "attention_mask")})

        pad_id = proc.tokenizer.pad_token_id or proc.tokenizer.eos_token_id
        maxlen = max(x.shape[0] for x in input_ids_list)
        input_ids, attn, labels = [], [], []
        for ids, lab in zip(input_ids_list, labels_list):
            padn = maxlen - ids.shape[0]
            input_ids.append(torch.cat([ids, torch.full((padn,), pad_id, dtype=ids.dtype)]))
            attn.append(torch.cat([torch.ones(ids.shape[0], dtype=torch.long),
                                   torch.zeros(padn, dtype=torch.long)]))
            labels.append(torch.cat([lab, torch.full((padn,), -100, dtype=lab.dtype)]))

        out = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attn),
            "labels": torch.stack(labels),
        }
        # stack vision tensors (pixel_values / image_grid_thw); concat along dim 0
        for key in other[0].keys():
            vals = [o[key] for o in other]
            try:
                out[key] = torch.cat(vals, dim=0)
            except Exception:
                out[key] = torch.stack(vals)
        return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                    help="HF model id (bf16). 7B to validate the stack; "
                         "Qwen/Qwen3-VL-32B-Instruct for the real run.")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("outputs/lamc-vlora-hf"))
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28,
                    help="Cap image tokens to bound sequence length / memory.")
    ap.add_argument("--tune-vision-layers", action="store_true")
    ap.add_argument("--qlora", action="store_true",
                    help="Load the base in 4-bit NF4 with bitsandbytes. "
                         "Use only for QLoRA experiments on smaller GPUs; the "
                         "ship path remains bf16 unless the A/B fully matches.")
    ap.add_argument("--init-adapter", type=Path, default=None,
                    help="Continue training from an existing PEFT adapter.")
    args = ap.parse_args()

    import torch
    from transformers import (AutoProcessor, AutoModelForImageTextToText,
                              Trainer, TrainingArguments)

    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU!"
    mode = "qlora-4bit" if args.qlora else "bf16"
    print(f"[train-hf] model={args.model} rank={args.rank} base={mode} device={dev}")

    processor = AutoProcessor.from_pretrained(
        args.model, max_pixels=args.max_pixels, use_fast=True)
    quant = _bnb_config(args.qlora)
    if quant is not None:
        from peft import prepare_model_for_kbit_training

        model = AutoModelForImageTextToText.from_pretrained(
            args.model, quantization_config=quant, device_map="auto",
            attn_implementation="sdpa")
        model.config.use_cache = False
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=True)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation="sdpa")
        model.config.use_cache = False

    model = _attach_or_create_lora(
        model,
        init_adapter=args.init_adapter,
        rank=args.rank,
        alpha=args.alpha,
        tune_vision=args.tune_vision_layers,
    )
    model.enable_input_require_grads()  # needed for gradient checkpointing + PEFT
    model.print_trainable_parameters()

    train_rows = _load_rows(args.train)
    val_rows = _load_rows(args.val) if args.val else None
    print(f"[train-hf] {len(train_rows)} train"
          + (f", {len(val_rows)} val" if val_rows else ""))

    targs = TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs if args.max_steps < 0 else 1,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        warmup_steps=5,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="none",
        remove_unused_columns=False,
        save_strategy="no",
    )
    trainer = Trainer(
        model=model, args=targs,
        train_dataset=train_rows, eval_dataset=val_rows,
        data_collator=VisionCollator(processor),
    )
    trainer.train()

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    processor.save_pretrained(str(args.out))
    print(f"[train-hf] saved LoRA adapter -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
