"""Objective-label tests for the delivered alt-text data builder."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.finetune.build_delivered_dataset import _alt_is_placeholder, _alt_target


@pytest.mark.parametrize(
    ("alt", "expected"),
    [
        ("", True),
        ("image", True),
        ("IMG_123", True),
        ("photo.JPG", True),
        ("A student using a microscope in the biology lab", False),
        ("College logo reading Los Angeles Mission College", False),
    ],
)
def test_alt_placeholder_detection_uses_objective_signals(alt: str, expected: bool) -> None:
    assert _alt_is_placeholder(alt) is expected


def _entry(index: int, alt: str, bbox: tuple[float, float, float, float]):
    return SimpleNamespace(figure_index=index, current_alt_text=alt, bbox=bbox)


def test_alt_target_does_not_label_human_rewording_as_a_defect() -> None:
    source = [
        _entry(1, "Students studying together in the library", (0, 0, 10, 10)),
        _entry(2, "photo", (20, 0, 30, 10)),
        _entry(3, "Decorative flourish", (40, 0, 50, 10)),
    ]
    delivered = [
        _entry(1, "A study group meets in the college library", (0, 0, 10, 10)),
        _entry(2, "A nursing student practices taking blood pressure", (20, 0, 30, 10)),
    ]

    target = _alt_target(source, delivered)

    assert [figure["status"] for figure in target["figures"]] == ["pass", "fail", "fail"]
    assert target["figures"][0]["issue_type"] == ""
    assert target["figures"][1]["issue_type"] == "missing_or_placeholder"
    assert target["figures"][2]["issue_type"] == "decorative"
