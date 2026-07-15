"""Tests for offline promotion metrics using the shared verifier."""

from __future__ import annotations

import json
from pathlib import Path

from tools.finetune.remedy_nemo_rl.evaluation import evaluate_predictions


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_evaluation_counts_invalid_json_and_real_pass_false_positives(tmp_path: Path) -> None:
    dataset = tmp_path / "test.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write(
        dataset,
        [
            {
                "example_id": "pass",
                "task": "heading_hierarchy",
                "verifier_target": {"status": "pass", "findings": []},
                "verifier_metadata": {"source_type": "real"},
            },
            {
                "example_id": "fail",
                "task": "heading_hierarchy",
                "verifier_target": {
                    "status": "fail",
                    "findings": [{"element_index": 2, "correct_tag": "H2"}],
                },
                "verifier_metadata": {"source_type": "real"},
            },
        ],
    )
    _write(
        predictions,
        [
            {"example_id": "pass", "response": '{"status":"fail","findings":[{"element_index":9,"correct_tag":"H2"}]}'},
            {"example_id": "fail", "response": "not-json"},
        ],
    )

    report = evaluate_predictions(dataset, predictions)

    metrics = report["tasks"]["heading_hierarchy"]
    assert metrics["valid_json"] == 0.5
    assert metrics["status_accuracy"] == 0.0
    assert metrics["real_pass_false_positives"] == 1
    assert report["promotion"]["heading_hierarchy"]["passed"] is False


def test_exact_gold_predictions_satisfy_json_and_safety_gates(tmp_path: Path) -> None:
    target = {"status": "fail", "findings": [{"issue_id": "headers", "fixer": "fix_table_headers"}]}
    dataset = tmp_path / "test.jsonl"
    predictions = tmp_path / "predictions.jsonl"
    _write(
        dataset,
        [{"example_id": "table", "task": "table_structure", "verifier_target": target, "verifier_metadata": {"source_type": "real"}}],
    )
    _write(predictions, [{"example_id": "table", "response": target}])

    report = evaluate_predictions(dataset, predictions)

    assert report["tasks"]["table_structure"]["valid_json"] == 1.0
    assert report["tasks"]["table_structure"]["status_accuracy"] == 1.0
    assert report["tasks"]["table_structure"]["mean_reward"] == 1.0
