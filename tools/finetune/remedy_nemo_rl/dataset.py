"""Rebuild grouped SFT and NeMo Gym datasets for five Remedy tasks."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

from .reward import TASKS, normalized_status, parse_response


DEFAULT_HOLDOUT_PATTERNS = (
    "ad142f824d25",
    "0312e204645b",
    "1d3a9d09c6f7",
    "2025 26 catalog",
    "2024 25 catalog",
    "2023 24 catalog",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalized_words(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _is_holdout(doc_id: str, patterns: Sequence[str]) -> bool:
    normalized = _normalized_words(doc_id)
    return any(_normalized_words(pattern) in normalized for pattern in patterns)


def _assistant_target(row: dict[str, Any]) -> dict[str, Any] | None:
    for message in reversed(row.get("messages", [])):
        if message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return parse_response(content)
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in {"text", "output_text"}:
                    return parse_response(str(part.get("text", "")))
    return None


def _source_type(row: dict[str, Any]) -> str:
    meta = row.get("meta") or {}
    explicit = str(meta.get("source_type") or meta.get("source_family") or "").lower()
    doc_id = str(meta.get("doc_id") or "").lower()
    if "synthetic" in explicit or "synthetic" in doc_id:
        return "synthetic"
    if "delivered" in explicit or "delivered" in doc_id:
        return "delivered"
    return "real"


def _image_parts(row: dict[str, Any]) -> list[dict[str, Any]]:
    parts = []
    for message in row.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        parts.extend(part for part in content if isinstance(part, dict) and part.get("type") == "image")
    return parts


def _load_sources(
    source_paths: Sequence[Path],
    holdout_patterns: Sequence[str],
) -> tuple[list[dict[str, Any]], int, dict[str, str]]:
    rows = []
    excluded = 0
    source_hashes = {}
    seen = set()
    for source_path in sorted(path.resolve() for path in source_paths):
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_hashes[str(source_path)] = _sha256(source_path)
        for line_number, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            meta = row.get("meta") or {}
            task = str(meta.get("task") or row.get("task") or "")
            doc_id = str(meta.get("doc_id") or row.get("doc_id") or "")
            if task not in TASKS or not doc_id:
                raise ValueError(f"{source_path}:{line_number}: missing supported task or doc_id")
            if _is_holdout(doc_id, holdout_patterns):
                excluded += 1
                continue
            target = _assistant_target(row)
            if target is None or normalized_status(target, task) is None:
                raise ValueError(f"{source_path}:{line_number}: invalid assistant target for {task}")

            row = deepcopy(row)
            row["meta"] = {**meta, "task": task, "doc_id": doc_id, "source_type": _source_type(row)}
            for part in _image_parts(row):
                image_path = Path(str(part.get("image", "")))
                resolved = image_path if image_path.is_absolute() else source_path.parent / image_path
                resolved = resolved.resolve()
                if not resolved.is_file():
                    raise FileNotFoundError(f"{source_path}:{line_number}: missing image {image_path}")
                part["image"] = str(resolved)
            dedupe_key = _canonical_json(row)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            identity = hashlib.sha256(f"{dedupe_key}|{source_path.name}".encode()).hexdigest()[:20]
            row["meta"]["example_id"] = f"{task}-{identity}"
            row["verifier_target"] = target
            rows.append(row)
    return rows, excluded, source_hashes


def _split_documents(rows: Sequence[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    doc_tasks: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        doc_tasks[str(row["meta"]["doc_id"])].add(str(row["meta"]["task"]))

    assignments: dict[str, str] = {}
    split_names = ("train", "validation", "test")
    for task in sorted(TASKS):
        task_docs = sorted(doc_id for doc_id, tasks in doc_tasks.items() if task in tasks)
        if not task_docs:
            continue
        desired = {
            "train": int(len(task_docs) * 0.70),
            "validation": int(len(task_docs) * 0.15),
        }
        desired["test"] = len(task_docs) - desired["train"] - desired["validation"]
        current = Counter(assignments[doc_id] for doc_id in task_docs if doc_id in assignments)
        unassigned = [doc_id for doc_id in task_docs if doc_id not in assignments]
        random.Random(f"{seed}:{task}").shuffle(unassigned)
        for doc_id in unassigned:
            split = max(
                split_names,
                key=lambda name: (
                    desired[name] - current[name],
                    -split_names.index(name),
                ),
            )
            assignments[doc_id] = split
            current[split] += 1

    splits = {"train": [], "validation": [], "test": []}
    for row in rows:
        splits[assignments[str(row["meta"]["doc_id"])]].append(deepcopy(row))
    return splits


def _balance_train(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: {"pass": [], "fail": []})
    for row in rows:
        task = str(row["meta"]["task"])
        status = normalized_status(row["verifier_target"], task)
        assert status in {"pass", "fail"}
        grouped[task][status].append(deepcopy(row))

    balanced = []
    for task in sorted(grouped):
        statuses = grouped[task]
        if not statuses["pass"] or not statuses["fail"]:
            raise ValueError(f"training task {task} cannot be balanced without pass and fail examples")
        target_count = max(len(statuses["pass"]), len(statuses["fail"]))
        for status in ("pass", "fail"):
            originals = sorted(statuses[status], key=lambda row: row["meta"]["example_id"])
            for index in range(target_count):
                row = deepcopy(originals[index % len(originals)])
                repeat = index // len(originals)
                if repeat:
                    row["meta"]["balance_repeat"] = repeat
                    row["meta"]["example_id"] = f"{row['meta']['example_id']}-r{repeat}"
                balanced.append(row)
    return sorted(balanced, key=lambda row: row["meta"]["example_id"])


def _image_block(path: Path) -> dict[str, Any]:
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "input_image", "image_url": f"data:{mime};base64,{payload}", "detail": "high"}


def _materialize_media(rows: Sequence[dict[str, Any]], media_dir: Path) -> None:
    """Copy referenced images into a content-addressed portable media directory."""

    media_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        for part in _image_parts(row):
            source = Path(str(part["image"]))
            suffix = source.suffix.lower() or ".png"
            destination = media_dir / f"{_sha256(source)}{suffix}"
            if not destination.exists():
                shutil.copy2(source, destination)
            part["image"] = str(destination.resolve())


def _input_message(message: dict[str, Any]) -> dict[str, Any]:
    content = message.get("content")
    blocks = []
    if isinstance(content, str):
        blocks.append({"type": "input_text", "text": content})
    elif isinstance(content, list):
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "image":
                blocks.append(_image_block(Path(str(part["image"]))))
            elif part.get("type") in {"text", "input_text"}:
                blocks.append({"type": "input_text", "text": str(part.get("text", ""))})
    return {"role": str(message.get("role") or "user"), "content": blocks}


def _gym_row(row: dict[str, Any]) -> dict[str, Any]:
    model_messages = [
        _input_message(message)
        for message in row.get("messages", [])
        if message.get("role") != "assistant"
    ]
    meta = row["meta"]
    return {
        "example_id": meta["example_id"],
        "task": meta["task"],
        "responses_create_params": {"input": model_messages, "max_output_tokens": 512},
        "verifier_target": row["verifier_target"],
        "verifier_metadata": {
            "doc_id": meta["doc_id"],
            "page": meta.get("page"),
            "variant": meta.get("variant"),
            "source_type": meta["source_type"],
        },
    }


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(_canonical_json(row) for row in rows)
    path.write_text(payload + ("\n" if payload else ""), encoding="utf-8")


def _write_sft_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    """Write SFT rows with image paths relative to that JSONL's directory."""

    portable_rows = deepcopy(list(rows))
    for row in portable_rows:
        for part in _image_parts(row):
            part["image"] = os.path.relpath(Path(str(part["image"])), path.parent)
    _write_jsonl(path, portable_rows)


def _counts(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_task[str(row["meta"]["task"])].append(row)
    result = {}
    for task, task_rows in sorted(by_task.items()):
        statuses = Counter(normalized_status(row["verifier_target"], task) for row in task_rows)
        sources = Counter(str(row["meta"]["source_type"]) for row in task_rows)
        result[task] = {
            "total": len(task_rows),
            "pass": statuses["pass"],
            "fail": statuses["fail"],
            "source_types": dict(sorted(sources.items())),
        }
    return result


def build_dataset(
    source_paths: Sequence[Path],
    output_dir: Path,
    *,
    seed: int = 20260714,
    holdout_patterns: Sequence[str] = DEFAULT_HOLDOUT_PATTERNS,
    holdout_files: Sequence[Path] = (),
) -> dict[str, Any]:
    """Build deterministic grouped SFT and Gym splits from task builder outputs.

    Args:
        source_paths: Task-specific JSONL files emitted by source builders.
        output_dir: Destination for SFT, Gym, and manifest files.
        seed: Document-level split seed.
        holdout_patterns: Normalized document ID substrings excluded entirely.
        holdout_files: Production holdout PDFs recorded by path, hash, and size.

    Returns:
        Reproducibility and class-balance manifest.
    """

    resolved_holdouts = sorted(path.resolve() for path in holdout_files)
    missing_holdouts = [path for path in resolved_holdouts if not path.is_file()]
    if missing_holdouts:
        raise FileNotFoundError(missing_holdouts[0])
    holdout_manifest = {
        str(path): {"sha256": _sha256(path), "size_bytes": path.stat().st_size}
        for path in resolved_holdouts
    }

    rows, excluded, source_hashes = _load_sources(source_paths, holdout_patterns)
    if not rows:
        raise ValueError("no eligible dataset rows were loaded")
    _materialize_media(rows, output_dir / "media")
    splits = _split_documents(rows, seed)
    splits["train"] = _balance_train(splits["train"])
    for split in splits:
        splits[split] = sorted(splits[split], key=lambda row: row["meta"]["example_id"])
        _write_sft_jsonl(output_dir / "sft" / f"{split}.jsonl", splits[split])
        _write_jsonl(output_dir / "gym" / f"{split}.jsonl", [_gym_row(row) for row in splits[split]])
        for task in sorted(TASKS):
            task_rows = [row for row in splits[split] if row["meta"]["task"] == task]
            _write_sft_jsonl(output_dir / "sft" / task / f"{split}.jsonl", task_rows)
            _write_jsonl(
                output_dir / "gym" / task / f"{split}.jsonl",
                [_gym_row(row) for row in task_rows],
            )

    dataset_hashes = {}
    for family in ("sft", "gym"):
        for split in splits:
            path = output_dir / family / f"{split}.jsonl"
            dataset_hashes[f"{family}/{split}.jsonl"] = _sha256(path)
            for task in sorted(TASKS):
                task_path = output_dir / family / task / f"{split}.jsonl"
                dataset_hashes[f"{family}/{task}/{split}.jsonl"] = _sha256(task_path)

    manifest = {
        "schema_version": 1,
        "seed": seed,
        "split_percentages": {"train": 70, "validation": 15, "test": 15},
        "holdout_patterns": list(holdout_patterns),
        "holdout_files": holdout_manifest,
        "excluded_rows": excluded,
        "source_hashes": source_hashes,
        "dataset_hashes": dataset_hashes,
        "counts": {split: _counts(split_rows) for split, split_rows in splits.items()},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    """Run the grouped dataset builder CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--holdout-pattern", action="append", default=[])
    parser.add_argument("--holdout-pdf", action="append", type=Path, default=[])
    args = parser.parse_args()
    patterns = tuple(args.holdout_pattern) or DEFAULT_HOLDOUT_PATTERNS
    manifest = build_dataset(
        args.source,
        args.output_dir,
        seed=args.seed,
        holdout_patterns=patterns,
        holdout_files=args.holdout_pdf,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
