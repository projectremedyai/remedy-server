"""Dataloader preflight for NeMo RL VLM SFT: prove the data path before paying.

The 2026-07-15 paid SFT smoke died on the FIRST dataloader batch with an
``IndexError`` in Qwen2.5-VL ``image_grid_thw`` indexing, after the VM, image
pull, and setup had already been paid for. Root cause: ``datasets==4.4.1``
None-pads heterogeneous multimodal ``content`` lists at load time, making
Qwen chat templates render text parts as extra image placeholders.

This gate pushes real rows through the REAL NeMo RL processing path
(``OpenAIFormatDataset`` -> ``sft_processor`` -> HF processor) inside the
training container, in seconds, before any training run starts. It fails
loudly when a row crashes, when the rendered text carries a different number
of vision blocks than the row has images, or when a row's prompt text was
silently dropped.

Run inside the NeMo RL container (needs nemo_rl importable):

    python -m tools.finetune.remedy_nemo_rl.dataloader_preflight \
        --task-root /ephemeral/nemo-rl/datasets/sft/contrast \
        --config /home/ubuntu/RL/examples/configs/remedy/sft_qwen25_vl_3b_h200.yaml \
        --rows 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

VISION_START = "<|vision_start|>"


def row_failures(
    decoded_text: str,
    image_count: int,
    *,
    must_contain: str | None = None,
) -> list[str]:
    """Check one processed row's decoded token text for dataloader corruption.

    Args:
        decoded_text: The detokenized full message log for the row.
        image_count: Number of image content parts in the RAW (pre-datasets)
            row; the rendered text must carry exactly this many vision blocks.
        must_contain: A distinctive snippet of the row's prompt text. Phantom
            image keys make chat templates silently REPLACE text parts with
            image placeholders, so text survival must be asserted explicitly.

    Returns:
        Human-readable failure strings; empty when the row is healthy.
    """
    failures: list[str] = []
    vision_blocks = decoded_text.count(VISION_START)
    if vision_blocks != image_count:
        failures.append(
            f"vision block count {vision_blocks} != image count {image_count} "
            "(placeholder desync: the image_grid_thw crash class)"
        )
    if must_contain is not None and must_contain not in decoded_text:
        failures.append(
            f"prompt text missing from rendered tokens: {must_contain[:60]!r} "
            "(text part was silently dropped)"
        )
    return failures


def select_probe_rows(lines: list[str], rows: int) -> list[int]:
    """Indices to probe: the first ``rows`` rows PLUS each split's longest row.

    The first-N-only strategy missed the truncation crash (overlong
    table_structure rows stub their tokens but kept media attached, killing
    the forward). Raw line length is a cheap, reliable proxy for token
    length, so the longest row is always probed too.
    """
    head = list(range(min(rows, len(lines))))
    if not lines:
        return head
    longest = max(range(len(lines)), key=lambda i: len(lines[i]))
    if longest not in head:
        head.append(longest)
    return head


def _raw_row_facts(row: dict[str, Any]) -> tuple[int, str | None]:
    """Image count and a distinctive text snippet from a raw JSONL row."""
    image_count = 0
    snippet: str | None = None
    for message in row.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image":
                image_count += 1
            elif part.get("type") == "text" and snippet is None:
                text = str(part.get("text") or "").strip()
                if len(text) >= 20:
                    snippet = text[:48]
    return image_count, snippet


def _model_name_from_config(config_path: Path) -> str:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_name = config.get("policy", {}).get("model_name")
    if not model_name:
        raise SystemExit(f"policy.model_name not found in {config_path}")
    return str(model_name)


def _max_seq_length_from_config(config_path: Path) -> int | None:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    value = config.get("policy", {}).get("max_total_sequence_length")
    return int(value) if value else None


def _truncation_failures(out: dict[str, Any]) -> list[str]:
    """When a row is truncated (masked), NO media may survive on any message.

    Unpatched NeMo stubs token_ids but keeps media attached; the batch then
    dies in the model forward with 'Image features and image tokens do not
    match'. This asserts the patched invariant on the real processing output.
    """
    from nemo_rl.data.multimodal_utils import PackedTensor

    if out.get("loss_multiplier", 1.0) != 0.0:
        return []
    leftovers = [
        key
        for message in out["message_log"]
        for key, value in message.items()
        if isinstance(value, PackedTensor)
    ]
    if leftovers:
        return [
            "truncated row still carries media keys "
            f"{sorted(set(leftovers))} (would crash the model forward)"
        ]
    return []


def run_preflight(task_root: Path, config_path: Path, rows: int) -> dict[str, Any]:
    """Process the first ``rows`` rows of each split through the real path."""
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.datasets.response_datasets.oai_format_dataset import (
        OpenAIFormatDataset,
    )
    from nemo_rl.data.interfaces import TaskDataSpec
    from nemo_rl.data.processors import sft_processor

    model_name = _model_name_from_config(config_path)
    max_seq_length = _max_seq_length_from_config(config_path)
    processor = get_tokenizer({"name": model_name}, get_processor=True)
    spec = TaskDataSpec(task_name="dataloader_preflight")

    report: dict[str, Any] = {
        "model": model_name,
        "task_root": str(task_root),
        "rows_requested_per_split": rows,
        "max_seq_length_probed": max_seq_length,
        "splits": {},
        "passed": True,
    }

    for split in ("train", "validation"):
        jsonl = task_root / f"{split}.jsonl"
        lines = [
            line for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        probe_indices = select_probe_rows(lines, rows)
        dataset = OpenAIFormatDataset(str(jsonl))
        split_report: dict[str, Any] = {"rows_checked": 0, "failures": []}
        for idx in probe_indices:
            raw = json.loads(lines[idx])
            image_count, snippet = _raw_row_facts(raw)
            try:
                out = sft_processor(dataset.dataset[idx], spec, processor, 1 << 30, idx)
                decoded = processor.tokenizer.decode(
                    [t for m in out["message_log"] for t in m["token_ids"].tolist()]
                )
                failures = row_failures(
                    decoded, image_count, must_contain=snippet
                )
                # Re-process at the REAL training sequence limit so
                # truncation behavior is exercised on the rows most likely
                # to trigger it (notably each split's longest row).
                if max_seq_length is not None:
                    truncated_out = sft_processor(
                        dataset.dataset[idx], spec, processor, max_seq_length, idx
                    )
                    failures.extend(_truncation_failures(truncated_out))
            except Exception as exc:  # noqa: BLE001
                failures = [f"processing crashed: {type(exc).__name__}: {exc}"]
            split_report["rows_checked"] += 1
            for failure in failures:
                split_report["failures"].append(f"row {idx}: {failure}")
        report["splits"][split] = split_report
        if split_report["failures"]:
            report["passed"] = False

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=4)
    args = parser.parse_args()

    report = run_preflight(args.task_root, args.config, args.rows)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
