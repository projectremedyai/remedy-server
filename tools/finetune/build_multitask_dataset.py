#!/usr/bin/env python3
"""Union per-task train/val JSONL files into one multitask dataset."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
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


def build(out_dir: Path, dataset_dirs: list[Path], seed: int) -> dict:
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

    rng = random.Random(seed)
    rng.shuffle(train)
    rng.shuffle(val)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "val.jsonl", val)
    manifest = {
        "seed": seed,
        "train": len(train),
        "val": len(val),
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
    args = ap.parse_args()
    manifest = build(args.out_dir, args.datasets, args.seed)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
