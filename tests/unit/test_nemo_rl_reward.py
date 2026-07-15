"""Tests for deterministic Remedy NeMo Gym rewards."""

from __future__ import annotations

import json

import pytest

from tools.finetune.remedy_nemo_rl.reward import verify_response


GOLD_BY_TASK = {
    "alt_text_quality": {
        "figures": [
            {
                "figure_index": 1,
                "status": "fail",
                "issue_type": "inaccurate",
                "suggested_alt_text": "Portrait of the college president.",
            }
        ]
    },
    "table_structure": {
        "status": "fail",
        "findings": [
            {
                "issue_id": "missing_table_headers",
                "table_id": "table-1",
                "cell_id": "r1c1",
                "fixer": "fix_table_headers",
            }
        ],
    },
    "contrast": {
        "issues": [
            {
                "issue_id": "body-text",
                "ratio": 3.9,
                "text_rgb": [129, 129, 129],
                "bg_rgb": [255, 255, 255],
                "fix_rgb": [0, 0, 0],
            }
        ]
    },
    "reading_order": {
        "page_layout": "table_directory",
        "issues": [
            {
                "issue_id": "later-region-first",
                "description": "Later visual region appears before earlier body content.",
                "suggestion": "Restore the delivered gold reading order: 1, 2, 3, 4",
            }
        ],
    },
    "heading_hierarchy": {
        "status": "fail",
        "findings": [
            {"element_index": 3, "current_tag": "H1", "correct_tag": "H2"},
            {"element_index": 5, "current_tag": "H1", "correct_tag": "H3"},
        ],
    },
}


@pytest.mark.parametrize("task", GOLD_BY_TASK)
def test_every_exact_gold_response_scores_one(task: str) -> None:
    result = verify_response(task, json.dumps(GOLD_BY_TASK[task]), GOLD_BY_TASK[task])

    assert result.reward == pytest.approx(1.0)
    assert result.passed is True
    assert result.error is None


def test_invalid_json_scores_negative_one() -> None:
    result = verify_response("heading_hierarchy", "not json", GOLD_BY_TASK["heading_hierarchy"])

    assert result.reward == -1.0
    assert result.passed is False
    assert result.error == "invalid_json"


def test_invalid_task_schema_scores_negative_one() -> None:
    result = verify_response("table_structure", "{}", GOLD_BY_TASK["table_structure"])

    assert result.reward == -1.0
    assert result.error == "invalid_schema"


def test_pass_page_false_positive_scores_negative_one() -> None:
    gold = {"status": "pass", "findings": []}
    prediction = {"status": "fail", "findings": [{"issue_id": "invented"}]}

    result = verify_response("heading_hierarchy", prediction, gold)

    assert result.reward == -1.0
    assert result.components["status_accuracy"] == 0.0


def test_false_negative_on_gold_fail_scores_zero() -> None:
    prediction = {"status": "pass", "findings": []}

    result = verify_response("heading_hierarchy", prediction, GOLD_BY_TASK["heading_hierarchy"])

    assert result.reward == 0.0
    assert result.components["status_accuracy"] == 0.0


def test_contrast_ratio_mutation_only_reduces_ratio_component() -> None:
    prediction = json.loads(json.dumps(GOLD_BY_TASK["contrast"]))
    prediction["issues"][0]["ratio"] = 4.2

    result = verify_response("contrast", prediction, GOLD_BY_TASK["contrast"])

    assert result.components["issue_match"] == 1.0
    assert result.components["rgb_accuracy"] == 1.0
    assert result.components["ratio_accuracy"] == 0.0
    assert 0.2 < result.reward < 1.0


def test_contrast_rgb_mutation_only_reduces_rgb_component() -> None:
    prediction = json.loads(json.dumps(GOLD_BY_TASK["contrast"]))
    prediction["issues"][0]["text_rgb"] = [128, 128, 128]

    result = verify_response("contrast", prediction, GOLD_BY_TASK["contrast"])

    assert result.components["issue_match"] == 1.0
    assert result.components["ratio_accuracy"] == 1.0
    assert result.components["rgb_accuracy"] < 1.0


def test_reading_order_is_parsed_from_correction_text() -> None:
    prediction = json.loads(json.dumps(GOLD_BY_TASK["reading_order"]))
    prediction["issues"][0]["suggestion"] = "Restore the delivered gold reading order: 1, 3, 2, 4"

    result = verify_response("reading_order", prediction, GOLD_BY_TASK["reading_order"])

    assert result.components["issue_match"] == 1.0
    assert result.components["ordered_element_ids"] == 0.0


def test_heading_uses_f1_over_exact_index_tag_pairs() -> None:
    prediction = json.loads(json.dumps(GOLD_BY_TASK["heading_hierarchy"]))
    prediction["findings"][1]["correct_tag"] = "H4"

    result = verify_response("heading_hierarchy", prediction, GOLD_BY_TASK["heading_hierarchy"])

    assert result.components["heading_pair_f1"] == pytest.approx(0.5)
    assert result.reward == pytest.approx(0.6)


def test_self_reported_confidence_does_not_affect_reward() -> None:
    high = json.loads(json.dumps(GOLD_BY_TASK["table_structure"]))
    low = json.loads(json.dumps(GOLD_BY_TASK["table_structure"]))
    high["confidence"] = 1.0
    low["confidence"] = 0.0

    assert verify_response("table_structure", high, GOLD_BY_TASK["table_structure"]).reward == verify_response(
        "table_structure", low, GOLD_BY_TASK["table_structure"]
    ).reward
