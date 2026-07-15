"""Deterministic task rewards shared by offline evaluation and NeMo Gym."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable


TASKS = {
    "alt_text_quality",
    "table_structure",
    "contrast",
    "reading_order",
    "heading_hierarchy",
}


@dataclass(frozen=True)
class VerificationResult:
    """Result returned by the deterministic verifier.

    Attributes:
        reward: Scalar GRPO reward in the range -1.0 to 1.0.
        passed: Whether the predicted page-level verdict matches the target.
        components: Named deterministic reward components.
        error: Machine-readable parse or schema error, when present.
    """

    reward: float
    passed: bool
    components: dict[str, float]
    error: str | None = None


def parse_response(response: str | dict[str, Any]) -> dict[str, Any] | None:
    """Parse a strict JSON object without accepting Markdown fences."""

    if isinstance(response, dict):
        return response
    if not isinstance(response, str):
        return None
    try:
        value = json.loads(response)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def normalized_status(value: dict[str, Any], task: str) -> str | None:
    """Return the normalized page verdict when the task schema is valid."""

    if task not in TASKS or not isinstance(value, dict):
        return None
    explicit = str(value.get("status", "")).strip().lower()
    if task in {"table_structure", "heading_hierarchy"}:
        key = "findings"
        if explicit not in {"pass", "fail"} or not isinstance(value.get(key), list):
            return None
        return explicit
    if task == "alt_text_quality":
        figures = value.get("figures")
        if not isinstance(figures, list) or any(not isinstance(item, dict) for item in figures):
            return None
        statuses = [str(item.get("status", "")).strip().lower() for item in figures]
        if any(status not in {"pass", "fail"} for status in statuses):
            return None
        return "fail" if "fail" in statuses else "pass"
    issues = value.get("issues")
    if not isinstance(issues, list) or any(not isinstance(item, dict) for item in issues):
        return None
    return "fail" if issues else "pass"


def _norm(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _f1(gold: set[Any], predicted: set[Any]) -> float:
    if not gold and not predicted:
        return 1.0
    if not gold or not predicted:
        return 0.0
    overlap = len(gold & predicted)
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _items(value: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [item for item in value.get(key, []) if isinstance(item, dict)]


def _alt_components(gold: dict[str, Any], predicted: dict[str, Any]) -> dict[str, float]:
    gold_items = _items(gold, "figures")
    predicted_items = _items(predicted, "figures")
    gold_failures = {
        (str(item.get("figure_index", "")), _norm(item.get("issue_type")))
        for item in gold_items
        if _norm(item.get("status")) == "fail"
    }
    predicted_failures = {
        (str(item.get("figure_index", "")), _norm(item.get("issue_type")))
        for item in predicted_items
        if _norm(item.get("status")) == "fail"
    }
    predicted_by_index = {str(item.get("figure_index", "")): _norm(item.get("status")) for item in predicted_items}
    verdicts = [
        predicted_by_index.get(str(item.get("figure_index", ""))) == _norm(item.get("status"))
        for item in gold_items
    ]
    return {
        "figure_issue_f1": _f1(gold_failures, predicted_failures),
        "verdict_accuracy": sum(verdicts) / len(verdicts) if verdicts else 1.0,
    }


def _table_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        _norm(item.get("issue_id") or item.get("issue")),
        _norm(item.get("table_id") or item.get("table_index")),
        _norm(item.get("cell_id") or item.get("cell")),
        _norm(item.get("fixer")),
    )


def _table_components(gold: dict[str, Any], predicted: dict[str, Any]) -> dict[str, float]:
    return {
        "finding_f1": _f1(
            {_table_key(item) for item in _items(gold, "findings")},
            {_table_key(item) for item in _items(predicted, "findings")},
        )
    }


def _issue_key(item: dict[str, Any]) -> str:
    return _norm(item.get("issue_id") or item.get("issue_type") or item.get("description") or item.get("message"))


def _contrast_components(gold: dict[str, Any], predicted: dict[str, Any]) -> dict[str, float]:
    gold_items = _items(gold, "issues")
    predicted_items = _items(predicted, "issues")
    gold_by_key = {_issue_key(item): item for item in gold_items}
    predicted_by_key = {_issue_key(item): item for item in predicted_items}
    matching_keys = sorted(set(gold_by_key) & set(predicted_by_key))

    ratio_scores = []
    rgb_scores = []
    for key in matching_keys:
        gold_item = gold_by_key[key]
        predicted_item = predicted_by_key[key]
        if "ratio" in gold_item:
            try:
                ratio_scores.append(abs(float(gold_item["ratio"]) - float(predicted_item.get("ratio"))) <= 0.05)
            except (TypeError, ValueError):
                ratio_scores.append(False)
        for field in ("text_rgb", "bg_rgb", "fix_rgb"):
            if field in gold_item:
                rgb_scores.append(predicted_item.get(field) == gold_item[field])

    return {
        "issue_match": _f1(set(gold_by_key), set(predicted_by_key)),
        "ratio_accuracy": sum(ratio_scores) / len(ratio_scores) if ratio_scores else 1.0,
        "rgb_accuracy": sum(rgb_scores) / len(rgb_scores) if rgb_scores else 1.0,
    }


def _ordered_ids(value: dict[str, Any]) -> tuple[int, ...] | None:
    for key in ("corrected_order", "reading_order", "ordered_element_ids"):
        explicit = value.get(key)
        if isinstance(explicit, list):
            try:
                return tuple(int(item) for item in explicit)
            except (TypeError, ValueError):
                return None
    for item in _items(value, "issues"):
        correction = str(item.get("suggestion") or item.get("correction") or "")
        if "order" not in correction.lower():
            continue
        tail = correction.split(":", 1)[-1]
        numbers = re.findall(r"\b\d+\b", tail)
        if numbers:
            return tuple(int(number) for number in numbers)
    return None


def _reading_components(gold: dict[str, Any], predicted: dict[str, Any]) -> dict[str, float]:
    gold_issues = {_issue_key(item) for item in _items(gold, "issues")}
    predicted_issues = {_issue_key(item) for item in _items(predicted, "issues")}
    return {
        "issue_match": _f1(gold_issues, predicted_issues),
        "ordered_element_ids": float(_ordered_ids(gold) == _ordered_ids(predicted)),
    }


def _heading_pairs(value: dict[str, Any]) -> set[tuple[int, str]]:
    pairs: set[tuple[int, str]] = set()
    for item in _items(value, "findings"):
        try:
            index = int(item.get("element_index"))
        except (TypeError, ValueError):
            continue
        tag = str(item.get("correct_tag", "")).strip().lstrip("/").upper()
        if re.fullmatch(r"H[1-6]|P|SPAN", tag):
            pairs.add((index, tag))
    return pairs


def _heading_components(gold: dict[str, Any], predicted: dict[str, Any]) -> dict[str, float]:
    return {"heading_pair_f1": _f1(_heading_pairs(gold), _heading_pairs(predicted))}


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def verify_response(
    task: str,
    response: str | dict[str, Any],
    gold_target: dict[str, Any],
) -> VerificationResult:
    """Score one model response using the approved asymmetric reward policy.

    Args:
        task: One of the five Remedy specialist task names.
        response: Strict JSON response text or an already parsed object.
        gold_target: Normalized deterministic target excluded from model input.

    Returns:
        Reward, verdict match, named components, and any parse/schema error.
    """

    predicted = parse_response(response)
    if predicted is None:
        return VerificationResult(-1.0, False, {"status_accuracy": 0.0}, "invalid_json")
    gold_status = normalized_status(gold_target, task)
    predicted_status = normalized_status(predicted, task)
    if gold_status is None:
        raise ValueError(f"invalid normalized gold target for task {task}")
    if predicted_status is None:
        return VerificationResult(-1.0, False, {"status_accuracy": 0.0}, "invalid_schema")

    status_match = gold_status == predicted_status
    status_components = {"status_accuracy": float(status_match)}
    if gold_status == "pass":
        reward = 1.0 if predicted_status == "pass" else -1.0
        return VerificationResult(reward, status_match, status_components)
    if predicted_status == "pass":
        return VerificationResult(0.0, False, status_components)

    component_fn = {
        "alt_text_quality": _alt_components,
        "table_structure": _table_components,
        "contrast": _contrast_components,
        "reading_order": _reading_components,
        "heading_hierarchy": _heading_components,
    }[task]
    structured = component_fn(gold_target, predicted)
    reward = 0.2 + 0.8 * _mean(structured.values())
    return VerificationResult(reward, True, {**status_components, **structured})
