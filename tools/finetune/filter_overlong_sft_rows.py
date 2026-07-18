"""Drop SFT rows whose exact token length exceeds the training context.

Why build-time filtering (2026-07-16): rows longer than
``policy.max_total_sequence_length`` cannot be trained on at all. NeMo's
runtime truncation stubs the tokens but the pipeline's uniform-multimodal-
batch invariants then fail one layer at a time (model forward feature
mismatch -> collation NoneType -> BatchedDataDict [8,7] size assertion).
An overlong row is wasted signal even when handled "gracefully", so the
correct place to enforce the limit is the dataset itself: every shipped row
must fit, making the runtime truncation path dead code (the NeMo patches
stay as defense in depth).

Measures EXACT lengths (chat template + expanded image tokens) with the real
HF processor. Run locally on CPU:

    PYTHONPATH=. <venv>/bin/python tools/finetune/filter_overlong_sft_rows.py \
        --dataset-root tools/finetune/generated/nemo_campaign_dataset \
        --max-tokens 8128 --apply
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.finetune.remedy_nemo_rl.dataset import _sha256
from tools.finetune.remedy_nemo_rl.reward import normalized_status

# Runtime length = get_formatted_message_log's per-turn assembly, which adds
# BOS/EOS and role scaffolding beyond a plain apply_chat_template render.
# Keep a safety margin under policy.max_total_sequence_length (8192).
DEFAULT_MAX_TOKENS = 8128

_PROCESSOR = None


def partition_rows(
    rows: list[dict[str, Any]], lengths: list[int], max_tokens: int
) -> tuple[list[dict[str, Any]], list[tuple[int, int]]]:
    """Split rows into (kept, dropped) by exact token length.

    Returns:
        kept rows, and (row_index, token_length) for every dropped row.
    """
    kept: list[dict[str, Any]] = []
    dropped: list[tuple[int, int]] = []
    for idx, (row, length) in enumerate(zip(rows, lengths)):
        if length > max_tokens:
            dropped.append((idx, length))
        else:
            kept.append(row)
    return kept, dropped


def recount(rows: list[dict[str, Any]], task: str) -> dict[str, Any]:
    """Recompute the manifest's pass/fail/total counts for one task split."""
    counts: dict[str, Any] = {"total": len(rows), "pass": 0, "fail": 0}
    source_types: Counter[str] = Counter()
    for row in rows:
        status = normalized_status(row.get("verifier_target") or {}, task)
        if status in ("pass", "fail"):
            counts[status] += 1
        source_types[str((row.get("meta") or {}).get("source_type") or "unknown")] += 1
    counts["source_types"] = dict(sorted(source_types.items()))
    return counts


def update_manifest_after_filter(
    manifest: dict[str, Any],
    dataset_root: Path,
    jsonl: Path,
    rows: list[dict[str, Any]],
    task: str,
    split: str,
) -> None:
    """Refresh counts and the content hash after rewriting one SFT split.

    Args:
        manifest: Mutable campaign manifest for the dataset.
        dataset_root: Root used for manifest-relative dataset paths.
        jsonl: Rewritten task-specific SFT JSONL.
        rows: Rows retained after exact token-length filtering.
        task: Campaign task name.
        split: Dataset split name.
    """

    manifest["counts"][split][task] = recount(rows, task)
    relative = jsonl.relative_to(dataset_root).as_posix()
    manifest.setdefault("dataset_hashes", {})[relative] = _sha256(jsonl)


def _get_processor():
    global _PROCESSOR
    if _PROCESSOR is None:
        from transformers import AutoProcessor

        _PROCESSOR = AutoProcessor.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", use_fast=True
        )
    return _PROCESSOR


def measure_line(args: tuple[str, str]) -> int:
    """Exact token count for one JSONL line (text + expanded image tokens)."""
    line, jsonl_dir = args
    from PIL import Image

    processor = _get_processor()
    row = json.loads(line)
    messages, images = [], []
    for message in row["messages"]:
        content = []
        for part in message["content"]:
            if isinstance(part, dict) and part.get("type") == "image":
                path = Path(part["image"])
                if not path.is_absolute():
                    path = (Path(jsonl_dir) / path).resolve()
                images.append(Image.open(path).convert("RGB"))
                content.append({"type": "image"})
            else:
                content.append({"type": "text", "text": part.get("text", "")})
        messages.append({"role": message["role"], "content": content})
    text = processor.apply_chat_template(messages, tokenize=False)
    out = processor(text=[text], images=images or None)
    return len(out["input_ids"][0])


def filter_split(jsonl: Path, max_tokens: int, workers: int, apply: bool) -> dict[str, Any]:
    lines = [l for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        lengths = list(pool.map(measure_line, [(l, str(jsonl.parent)) for l in lines], chunksize=8))
    rows = [json.loads(l) for l in lines]
    kept, dropped = partition_rows(rows, lengths, max_tokens)
    if apply and dropped:
        payload = "\n".join(json.dumps(r, sort_keys=True) for r in kept)
        jsonl.write_text(payload + ("\n" if payload else ""), encoding="utf-8")
    return {
        "file": str(jsonl),
        "rows": len(rows),
        "dropped": [{"index": i, "tokens": t} for i, t in dropped],
        "kept": len(kept),
        "max_kept_tokens": max((l for l in lengths if l <= max_tokens), default=0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--apply", action="store_true", help="rewrite files (default: dry run)")
    args = parser.parse_args()

    manifest_path = args.dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = []
    for task_dir in sorted((args.dataset_root / "sft").iterdir()):
        if not task_dir.is_dir():
            continue
        task = task_dir.name
        for split in ("train", "validation"):
            jsonl = task_dir / f"{split}.jsonl"
            if not jsonl.exists():
                continue
            result = filter_split(jsonl, args.max_tokens, args.workers, args.apply)
            report.append(result)
            if args.apply:
                rows = [
                    json.loads(l)
                    for l in jsonl.read_text(encoding="utf-8").splitlines()
                    if l.strip()
                ]
                update_manifest_after_filter(
                    manifest, args.dataset_root, jsonl, rows, task, split
                )
            print(
                f"{task}/{split}: rows={result['rows']} dropped={len(result['dropped'])} "
                f"max_kept={result['max_kept_tokens']}"
            )
            for d in result["dropped"]:
                print(f"    dropped row {d['index']}: {d['tokens']} tokens")
    for split in ("train", "validation"):
        jsonl = args.dataset_root / "sft" / f"{split}.jsonl"
        if not jsonl.exists():
            continue
        result = filter_split(jsonl, args.max_tokens, args.workers, args.apply)
        report.append(result)
        if args.apply:
            relative = jsonl.relative_to(args.dataset_root).as_posix()
            manifest.setdefault("dataset_hashes", {})[relative] = _sha256(jsonl)
        print(
            f"aggregate/{split}: rows={result['rows']} dropped={len(result['dropped'])} "
            f"max_kept={result['max_kept_tokens']}"
        )
        for d in result["dropped"]:
            print(f"    dropped row {d['index']}: {d['tokens']} tokens")
    if args.apply:
        manifest["length_filter"] = {"max_tokens": args.max_tokens}
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print("manifest updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
