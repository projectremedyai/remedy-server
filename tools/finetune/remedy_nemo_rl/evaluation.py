"""Evaluate frozen or adapted model outputs with the NeMo Gym verifier."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .reward import normalized_status, parse_response, verify_response


STATUS_THRESHOLDS = {
    "alt_text_quality": 0.90,
    "table_structure": 1.00,
    "contrast": 0.90,
    "reading_order": 0.80,
    "heading_hierarchy": 0.95,
}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _near_contrast_threshold(target: dict[str, Any]) -> bool:
    ratios = [item.get("ratio") for item in target.get("issues", []) if isinstance(item, dict)]
    try:
        return any(abs(float(ratio) - 4.5) <= 0.25 for ratio in ratios)
    except (TypeError, ValueError):
        return False


def evaluate_predictions(dataset_path: Path, predictions_path: Path) -> dict[str, Any]:
    """Score predictions by example ID and calculate the approved promotion gates."""

    dataset = {row["example_id"]: row for row in _jsonl(dataset_path)}
    predictions = {
        row["example_id"]: row.get("response", row.get("prediction"))
        for row in _jsonl(predictions_path)
    }
    missing = sorted(set(dataset) - set(predictions))
    extra = sorted(set(predictions) - set(dataset))
    if missing or extra:
        raise ValueError(f"prediction IDs do not match dataset: missing={missing[:5]} extra={extra[:5]}")

    task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for example_id, row in dataset.items():
        task = row["task"]
        target = row["verifier_target"]
        response = predictions[example_id]
        parsed = parse_response(response)
        result = verify_response(task, response, target)
        gold_status = normalized_status(target, task)
        predicted_status = normalized_status(parsed, task) if parsed is not None else None
        metadata = row.get("verifier_metadata") or {}
        task_rows[task].append(
            {
                "valid_json": parsed is not None,
                "status_correct": result.components.get("status_accuracy", 0.0) == 1.0,
                "reward": result.reward,
                "real_pass_false_positive": (
                    metadata.get("source_type") != "synthetic"
                    and gold_status == "pass"
                    and predicted_status == "fail"
                ),
                "structured_exact": gold_status == "fail"
                and predicted_status == "fail"
                and all(value == 1.0 for name, value in result.components.items() if name != "status_accuracy"),
                "near_threshold": task == "contrast" and _near_contrast_threshold(target),
                "gold_fail": gold_status == "fail",
            }
        )

    metrics: dict[str, dict[str, Any]] = {}
    promotion: dict[str, dict[str, Any]] = {}
    for task, rows in sorted(task_rows.items()):
        near_rows = [row for row in rows if row["near_threshold"]]
        fail_rows = [row for row in rows if row["gold_fail"]]
        task_metrics = {
            "total": len(rows),
            "valid_json": mean(float(row["valid_json"]) for row in rows),
            "status_accuracy": mean(float(row["status_correct"]) for row in rows),
            "mean_reward": mean(float(row["reward"]) for row in rows),
            "real_pass_false_positives": sum(bool(row["real_pass_false_positive"]) for row in rows),
            "structured_exact_accuracy": (
                mean(float(row["structured_exact"]) for row in fail_rows) if fail_rows else 1.0
            ),
            "near_threshold_accuracy": (
                mean(float(row["status_correct"]) for row in near_rows) if near_rows else None
            ),
        }
        checks = {
            "valid_json": task_metrics["valid_json"] == 1.0,
            "real_pass_false_positives": task_metrics["real_pass_false_positives"] == 0,
            "status_accuracy": task_metrics["status_accuracy"] >= STATUS_THRESHOLDS[task],
        }
        if task == "contrast" and task_metrics["near_threshold_accuracy"] is not None:
            checks["near_threshold_accuracy"] = task_metrics["near_threshold_accuracy"] >= 0.85
        if task == "heading_hierarchy":
            checks["exact_correction_accuracy"] = task_metrics["structured_exact_accuracy"] >= 0.85
        metrics[task] = task_metrics
        promotion[task] = {"passed": all(checks.values()), "checks": checks}

    return {"dataset": str(dataset_path), "predictions": str(predictions_path), "tasks": metrics, "promotion": promotion}


def main() -> int:
    """Evaluate a prediction JSONL and write a machine-readable report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate_predictions(args.dataset, args.predictions)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
