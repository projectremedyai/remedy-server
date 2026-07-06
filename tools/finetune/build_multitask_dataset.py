#!/usr/bin/env python3
"""Union per-task train/val JSONL files into one multitask dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from copy import deepcopy
from pathlib import Path


DEFAULT_DATASETS = [
    "tools/finetune/data_v2",
    "tools/finetune/data_table",
    "tools/finetune/data_reading_order",
    "tools/finetune/data_contrast",
    "tools/finetune/data/heading_hierarchy",
]


def load_rows(jsonl: Path, out_dir: Path) -> list[dict]:
    base = jsonl.resolve().parent
    rows = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    for row in rows:
        for message in row.get("messages", []):
            for part in message.get("content", []):
                if part.get("type") != "image" or not isinstance(part.get("image"), str):
                    continue
                image = Path(part["image"])
                absolute = image if image.is_absolute() else base / image
                if not absolute.exists():
                    raise FileNotFoundError(f"{jsonl}: missing image {part['image']}")
                part["image"] = os.path.relpath(absolute, out_dir.resolve())
    return rows


def task_of(row: dict) -> str:
    meta = row.get("meta") or {}
    return str(meta.get("task") or "unknown")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def parse_task_weight(raw: str) -> tuple[str, int]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("task weights must use task=integer")
    task, value = raw.split("=", 1)
    task = task.strip()
    if not task:
        raise argparse.ArgumentTypeError("task weights require a task name")
    try:
        weight = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("task weights must be integers") from exc
    if weight < 1:
        raise argparse.ArgumentTypeError("task weights must be >= 1")
    return task, weight


def weighted_train_rows(rows: list[dict], task_weights: dict[str, int]) -> list[dict]:
    weighted: list[dict] = []
    for row in rows:
        weight = task_weights.get(task_of(row), 1)
        for _ in range(weight):
            weighted.append(deepcopy(row))
    return weighted


def build(
    out_dir: Path,
    dataset_dirs: list[Path],
    seed: int,
    task_weights: dict[str, int] | None = None,
) -> dict:
    task_weights = task_weights or {}
    train: list[dict] = []
    val: list[dict] = []
    included = []
    for data_dir in dataset_dirs:
        if not data_dir.exists():
            continue
        train_path = data_dir / "train.jsonl"
        val_path = data_dir / "val.jsonl"
        if not train_path.exists() or not val_path.exists():
            continue
        train_rows = load_rows(train_path, out_dir)
        val_rows = load_rows(val_path, out_dir)
        train.extend(train_rows)
        val.extend(val_rows)
        included.append({
            "dir": str(data_dir),
            "train": len(train_rows),
            "val": len(val_rows),
            "tasks": dict(Counter(task_of(row) for row in train_rows + val_rows)),
        })

    if not included:
        raise SystemExit("No dataset dirs with train.jsonl and val.jsonl were found")

    original_train_counts = Counter(task_of(row) for row in train)
    train = weighted_train_rows(train, task_weights)

    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(val)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    manifest = {
        "seed": seed,
        "train": len(train),
        "val": len(val),
        "task_weights": task_weights,
        "tasks_train_unweighted": dict(original_train_counts),
        "tasks_train": dict(Counter(task_of(row) for row in train)),
        "tasks_val": dict(Counter(task_of(row) for row in val)),
        "included": included,
        "note": "LoRA adapters are not merged; this is dataset union for one multitask adapter.",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("tools/finetune/data_multitask"))
    ap.add_argument("--datasets", nargs="*", type=Path,
                    default=[Path(p) for p in DEFAULT_DATASETS])
    ap.add_argument("--seed", type=int, default=20260705)
    ap.add_argument("--task-weight", action="append", type=parse_task_weight, default=[],
                    help="Duplicate train rows for a task, e.g. contrast=6. "
                         "Validation rows are never weighted.")
    args = ap.parse_args()
    manifest = build(args.out_dir, args.datasets, args.seed, dict(args.task_weight))
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
