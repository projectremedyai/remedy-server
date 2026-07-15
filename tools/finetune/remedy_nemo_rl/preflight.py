"""Validate the complete rebuilt campaign dataset before paid GPU work."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .dataset import DEFAULT_HOLDOUT_PATTERNS, _image_parts, _is_holdout, _sha256
from .reward import TASKS, normalized_status


REQUIRED_CATALOG_SHA256 = {
    "277bd584009593eeef7411aec67f28b10cdf7e0499cf6c4cbb957a33797278ac",
    "63929fb3950d6dfa96bcf4fa8c3a5224148d31d8fd68fdff2433a89d6b3263d7",
    "c28caab76a0a2826b313c94e15b8dfc088e23302964caee8ff004bbaad1f7144",
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate_campaign_dataset(output_dir: Path) -> dict[str, Any]:
    """Validate leakage, images, schemas, balance, holdouts, and all recorded hashes."""

    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_docs: dict[str, set[str]] = {}
    missing_images: list[str] = []
    schema_errors: list[str] = []
    holdout_leaks: list[str] = []
    balance_errors: list[str] = []

    for split in ("train", "validation", "test"):
        path = output_dir / "sft" / f"{split}.jsonl"
        rows = _jsonl(path)
        split_docs[split] = {str(row["meta"]["doc_id"]) for row in rows}
        status_counts = {task: {"pass": 0, "fail": 0} for task in TASKS}
        for row in rows:
            meta = row.get("meta") or {}
            task = str(meta.get("task") or "")
            doc_id = str(meta.get("doc_id") or "")
            if _is_holdout(doc_id, manifest.get("holdout_patterns", DEFAULT_HOLDOUT_PATTERNS)):
                holdout_leaks.append(doc_id)
            target = row.get("verifier_target")
            status = normalized_status(target, task) if isinstance(target, dict) else None
            if task not in TASKS or status is None:
                schema_errors.append(str(meta.get("example_id") or doc_id))
            elif task in status_counts:
                status_counts[task][status] += 1
            for part in _image_parts(row):
                image_path = Path(str(part.get("image", "")))
                resolved = image_path if image_path.is_absolute() else path.parent / image_path
                if not resolved.resolve().is_file():
                    missing_images.append(str(resolved))
        if split == "train":
            for task, counts in status_counts.items():
                total = counts["pass"] + counts["fail"]
                fail_fraction = counts["fail"] / total if total else 0.0
                if total == 0 or not 0.45 <= fail_fraction <= 0.55:
                    balance_errors.append(f"{task}:{counts}")

    document_leakage = sorted(
        (split_docs["train"] & split_docs["validation"])
        | (split_docs["train"] & split_docs["test"])
        | (split_docs["validation"] & split_docs["test"])
    )
    hash_mismatches = []
    for relative, expected in manifest.get("dataset_hashes", {}).items():
        path = output_dir / relative
        actual = _sha256(path) if path.is_file() else "missing"
        if actual != expected:
            hash_mismatches.append(relative)
    recorded_holdout_hashes = {
        entry.get("sha256")
        for entry in manifest.get("holdout_files", {}).values()
        if isinstance(entry, dict)
    }
    missing_holdout_hashes = sorted(REQUIRED_CATALOG_SHA256 - recorded_holdout_hashes)
    failures = (
        document_leakage
        + missing_images
        + schema_errors
        + holdout_leaks
        + balance_errors
        + hash_mismatches
        + missing_holdout_hashes
    )
    return {
        "passed": not failures,
        "document_leakage": document_leakage,
        "missing_images": sorted(set(missing_images)),
        "schema_errors": sorted(set(schema_errors)),
        "holdout_leaks": sorted(set(holdout_leaks)),
        "balance_errors": balance_errors,
        "hash_mismatches": hash_mismatches,
        "missing_holdout_hashes": missing_holdout_hashes,
        "split_documents": {name: len(doc_ids) for name, doc_ids in split_docs.items()},
    }


def main() -> int:
    """Validate a campaign dataset and optionally persist the report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate_campaign_dataset(args.dataset_root)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
