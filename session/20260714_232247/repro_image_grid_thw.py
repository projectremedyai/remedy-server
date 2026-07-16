#!/usr/bin/env python3
"""Local CPU repro of the NeMo RL VLM SFT image_grid_thw IndexError.

Runs the REAL pinned NeMo RL v0.6.0 data path (OpenAIFormatDataset ->
sft_processor -> Qwen2_5_VLProcessor) against the first real row of our
campaign train.jsonl, on this Mac, no GPU, no paid box.

Probes:
  0. sanity        - raw row has exactly 1 image part and the image exists
  1. phantom_keys  - datasets==4.4.1 load_dataset("json") injects image:None
                     into text parts (the root cause)
  2. default_path  - standard OpenAIFormatDataset + sft_processor crashes with
                     the exact IndexError from the paid smoke  (RED)
  3. preserving    - use_preserving_dataset=True: does sft_processor succeed,
                     and does run_sft.py's concatenate_datasets([ds]) survive?
  4. strip_none    - read-time None-strip sanitizer (candidate NeMo patch):
                     does sft_processor succeed on a sanitized standard row?

Exit 0 iff probes 1+2 confirm the root cause (repro achieved). Probes 3/4
report fix viability.
"""

import json
import os
import sys
import traceback
import types
from copy import deepcopy
from pathlib import Path

SCRATCH = Path(__file__).resolve().parent
NEMO_SRC = SCRATCH.parent / "nemo-rl-src"
REPO = Path("/Users/laccd/code/lamc_district_forms/remedy-server-nemo-rl-brev")
TRAIN = REPO / "tools/finetune/generated/nemo_campaign_dataset/sft/train.jsonl"
MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"

# Keep HF caches inside the scratch dir.
os.environ.setdefault("HF_HOME", str(SCRATCH / "hf_home"))

sys.path.insert(0, str(NEMO_SRC))
# decord is a video-only dependency of nemo_rl.data.multimodal_utils; we only
# process images here, so a stub module satisfies the import.
sys.modules.setdefault("decord", types.ModuleType("decord"))


def first_row() -> dict:
    with TRAIN.open() as fh:
        return json.loads(fh.readline())


def absolutize(row: dict) -> dict:
    """Resolve image paths relative to the JSONL's directory (matches training)."""
    row = deepcopy(row)
    for message in row["messages"]:
        for part in message["content"]:
            if isinstance(part, dict) and part.get("type") == "image":
                p = Path(part["image"])
                if not p.is_absolute():
                    part["image"] = str((TRAIN.parent / p).resolve())
    return row


def write_row_jsonl(row: dict, path: Path) -> Path:
    path.write_text(json.dumps(row) + "\n")
    return path


def content_shapes(messages):
    return [
        [(part.get("type"), sorted(part.keys())) for part in m["content"]]
        for m in messages
    ]


def run_sft_processor(entry: dict, processor) -> "object":
    """Exactly what AllTaskProcessedDataset.__getitem__ does at line 129."""
    from nemo_rl.data.interfaces import TaskDataSpec
    from nemo_rl.data.processors import sft_processor

    spec = TaskDataSpec(task_name="sft")
    return sft_processor(entry, spec, processor, 4096, 0)


def main() -> int:
    results: dict[str, str] = {}

    row = first_row()

    # ---- probe 0: sanity -------------------------------------------------
    image_parts = [
        p
        for m in row["messages"]
        for p in m["content"]
        if isinstance(p, dict) and p.get("type") == "image"
    ]
    abs_row = absolutize(row)
    abs_image = [
        p
        for m in abs_row["messages"]
        for p in m["content"]
        if p.get("type") == "image"
    ][0]["image"]
    assert len(image_parts) == 1, f"expected 1 image part, got {len(image_parts)}"
    assert Path(abs_image).is_file(), f"missing media file {abs_image}"
    results["0 sanity"] = f"OK - 1 image part, media exists ({Path(abs_image).name})"

    row_jsonl = write_row_jsonl(abs_row, SCRATCH / "row0.jsonl")

    # ---- probe 1: phantom keys from datasets==4.4.1 ----------------------
    import datasets as hf_datasets
    from datasets import load_dataset

    ds = load_dataset("json", data_files=str(row_jsonl))["train"]
    loaded = ds[0]["messages"]
    phantom = [
        part
        for m in loaded
        for part in m["content"]
        if part.get("type") == "text" and "image" in part
    ]
    if phantom:
        results["1 phantom_keys"] = (
            f"CONFIRMED (datasets=={hf_datasets.__version__}) - text parts now carry "
            f"'image': {phantom[0]['image']!r}; shapes={content_shapes(loaded)}"
        )
    else:
        results["1 phantom_keys"] = (
            f"NOT REPRODUCED on datasets=={hf_datasets.__version__} - "
            f"shapes={content_shapes(loaded)}"
        )

    # ---- shared: the real HF processor ------------------------------------
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, use_fast=True)
    # Replicate nemo_rl.algorithms.utils.get_tokenizer's decoration (utils.py:355-361):
    # it copies the special tokens from processor.tokenizer onto the processor,
    # which get_formatted_message_log accesses via the "tokenizer" argument.
    tok = processor.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    for attr in (
        "pad_token",
        "eos_token",
        "bos_token",
        "pad_token_id",
        "eos_token_id",
        "bos_token_id",
    ):
        setattr(processor, attr, getattr(tok, attr))

    from nemo_rl.data.datasets.response_datasets.oai_format_dataset import (
        OpenAIFormatDataset,
    )

    # ---- probe 2: default (standard HF) path ------------------------------
    # Unpatched pinned NeMo: expect the IndexError (RED).
    # With tools/finetune/patches/nemo_rl_strip_none_multimodal_content.patch
    # applied to the clone (REPRO_EXPECT=fixed): expect a clean single-vision-
    # block row with the prompt text intact (GREEN).
    fixed_ok = False
    standard = OpenAIFormatDataset(str(row_jsonl))
    try:
        out = run_sft_processor(standard.dataset[0], processor)
        text = processor.tokenizer.decode(
            [t for m in out["message_log"] for t in m["token_ids"].tolist()]
        )
        pad_ok = text.count("<|vision_start|>") == 1
        prompt_ok = "verifying image alt text quality" in text
        fixed_ok = pad_ok and prompt_ok
        results["2 default_path"] = (
            f"NO CRASH - single vision block={pad_ok}, prompt text survives={prompt_ok}"
        )
        red = False
    except IndexError as exc:
        tb = traceback.format_exc()
        matches = "out of bounds" in str(exc) and "processing_qwen2_5_vl" in tb
        results["2 default_path"] = (
            f"CRASH REPRODUCED - IndexError: {exc} "
            f"({'matches paid-smoke signature' if matches else 'DIFFERENT signature'})"
        )
        red = matches
    except Exception as exc:  # noqa: BLE001
        results["2 default_path"] = f"DIFFERENT FAILURE - {type(exc).__name__}: {exc}"
        red = False

    # ---- probe 3: use_preserving_dataset=True ----------------------------
    preserving = OpenAIFormatDataset(str(row_jsonl), use_preserving_dataset=True)
    try:
        out = run_sft_processor(preserving.dataset[0], processor)
        n_msgs = len(out["message_log"])
        text = processor.tokenizer.decode(
            [t for m in out["message_log"] for t in m["token_ids"].tolist()]
        )
        pad_ok = text.count("<|vision_start|>") == 1
        prompt_ok = "verifying image alt text quality" in text
        results["3 preserving/process"] = (
            f"PROCESSES OK - {n_msgs} turns, single vision block={pad_ok}, "
            f"prompt text survives={prompt_ok}"
        )
    except Exception as exc:  # noqa: BLE001
        results["3 preserving/process"] = f"FAILS - {type(exc).__name__}: {exc}"

    from datasets import concatenate_datasets

    try:
        merged = concatenate_datasets([preserving.dataset])
        results["3 preserving/concat"] = (
            f"concatenate_datasets SURVIVES (returns {type(merged).__name__})"
        )
    except Exception as exc:  # noqa: BLE001
        results["3 preserving/concat"] = (
            f"concatenate_datasets FAILS - {type(exc).__name__}: {exc}"
        )

    # ---- probe 4: read-time None-strip sanitizer (candidate patch) -------
    def strip_none(entry: dict) -> dict:
        entry = deepcopy(entry)
        for message in entry["messages"]:
            content = message.get("content")
            if isinstance(content, list):
                message["content"] = [
                    {k: v for k, v in part.items() if v is not None}
                    if isinstance(part, dict)
                    else part
                    for part in content
                ]
        return entry

    try:
        out = run_sft_processor(strip_none(standard.dataset[0]), processor)
        text = processor.tokenizer.decode(
            [t for m in out["message_log"] for t in m["token_ids"].tolist()]
        )
        pad_ok = text.count("<|vision_start|>") == 1
        prompt_ok = "verifying image alt text quality" in text
        results["4 strip_none"] = (
            f"PROCESSES OK - single vision block={pad_ok}, prompt text survives={prompt_ok}"
        )
    except Exception as exc:  # noqa: BLE001
        results["4 strip_none"] = f"FAILS - {type(exc).__name__}: {exc}"

    print("\n===== RESULTS =====")
    for name, verdict in results.items():
        print(f"[{name}] {verdict}")

    phantom = "CONFIRMED" in results["1 phantom_keys"]
    if os.environ.get("REPRO_EXPECT") == "fixed":
        # Patched tree: loader still injects Nones (phantom stays CONFIRMED),
        # but the read-time strip must yield a clean row.
        return 0 if phantom and fixed_ok else 1
    return 0 if phantom and red else 1


if __name__ == "__main__":
    raise SystemExit(main())
