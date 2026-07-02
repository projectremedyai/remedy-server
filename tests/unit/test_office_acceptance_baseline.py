"""Regression baseline for the pre-office-verify legacy checks (PRD §10, NFR5).

These tests pin the observable behavior of the 12 legacy rule ids so that the
Task-9 engine swap is provably non-regressive (AC7). Do not modify them when
swapping the docx engine — they must pass before AND after.
"""

from __future__ import annotations

import pytest

from project_remedy.models import FileType
from project_remedy.office_acceptance import evaluate_office_acceptance, run_office_checker
from tests.unit.office_fixtures import make_docx, make_pptx, make_xlsx


def _status(report, rule_id: str) -> str:
    matches = [r.status for r in report.results if r.rule_id == rule_id]
    assert matches, f"rule {rule_id!r} missing from report: {[r.rule_id for r in report.results]}"
    return matches[0]


GOOD_DOCX_KWARGS = dict(
    title="Good Doc",
    language="en-US",
    headings=[("Good Doc", 0), ("Section", 1)],
    body_paragraphs=["Body text long enough to be clearly a body paragraph of prose."],
    tables=1,
    mark_table_headers=True,
    inline_images=1,
    image_alt="A sample image",
)


@pytest.mark.parametrize(
    ("rule_id", "bad_kwargs"),
    [
        ("docx-title", {**GOOD_DOCX_KWARGS, "title": ""}),
        ("docx-language", {**GOOD_DOCX_KWARGS, "language": ""}),
        ("docx-headings", {**GOOD_DOCX_KWARGS, "headings": []}),
        ("docx-table-headers", {**GOOD_DOCX_KWARGS, "mark_table_headers": False}),
        ("docx-alt-text", {**GOOD_DOCX_KWARGS, "image_alt": None}),
    ],
)
def test_docx_legacy_rule_fails_on_bad_input(tmp_path, rule_id, bad_kwargs):
    path = make_docx(tmp_path / "bad.docx", **bad_kwargs)
    report = run_office_checker(path, FileType.DOCX)
    assert _status(report, rule_id) == "Failed"


def test_docx_legacy_rules_all_pass_on_good_input(tmp_path):
    path = make_docx(tmp_path / "good.docx", **GOOD_DOCX_KWARGS)
    report = run_office_checker(path, FileType.DOCX)
    for rule_id in ("docx-title", "docx-language", "docx-headings", "docx-table-headers", "docx-alt-text"):
        assert _status(report, rule_id) == "Passed"


def test_pptx_legacy_rules(tmp_path):
    good = make_pptx(tmp_path / "good.pptx", title="Deck", language="en-US",
                     slides=1, slide_titles=True, pictures=1, picture_alt="chart")
    report = run_office_checker(good, FileType.PPTX)
    for rule_id in ("pptx-title", "pptx-language", "pptx-slide-titles", "pptx-alt-text"):
        assert _status(report, rule_id) == "Passed"

    bad = make_pptx(tmp_path / "bad.pptx", slides=1, slide_titles=False, pictures=1, picture_alt=None)
    report = run_office_checker(bad, FileType.PPTX)
    assert _status(report, "pptx-title") == "Failed"
    assert _status(report, "pptx-language") == "Failed"
    assert _status(report, "pptx-alt-text") == "Failed"


def test_xlsx_legacy_rules(tmp_path):
    good = make_xlsx(tmp_path / "good.xlsx", title="Book", language="en-US", header_behaviors=True)
    report = run_office_checker(good, FileType.XLSX)
    for rule_id in ("xlsx-title", "xlsx-language", "xlsx-header-behaviors"):
        assert _status(report, rule_id) == "Passed"

    bad = make_xlsx(tmp_path / "bad.xlsx", header_behaviors=False)
    report = run_office_checker(bad, FileType.XLSX)
    assert _status(report, "xlsx-title") == "Failed"
    assert _status(report, "xlsx-header-behaviors") == "Failed"


def test_acceptance_passed_reflects_failures(tmp_path):
    bad = make_docx(tmp_path / "bad.docx")  # no title/language/headings
    result = evaluate_office_acceptance(bad)
    assert result.openable
    assert not result.passed
    assert result.checker_failures

    good = make_docx(tmp_path / "good.docx", **GOOD_DOCX_KWARGS)
    result = evaluate_office_acceptance(good)
    assert result.openable
    assert result.passed


def test_acceptance_screen_reader_issues_mirror_checker_failures(tmp_path):
    """De-dup regression (perf follow-up): screen_reader issues must still
    mirror the checker's Failed rule ids exactly, even though the engine now
    runs once and the screen-reader result is derived from the same report."""
    bad = make_docx(tmp_path / "bad.docx")  # no title/language/headings
    result = evaluate_office_acceptance(bad)

    failed_rule_ids = {r.rule_id for r in result.checker_failures}
    mirrored_rule_ids = {issue.rule_id for issue in result.screen_reader_result.issues}

    assert failed_rule_ids
    assert mirrored_rule_ids == failed_rule_ids
    assert all(issue.severity == "error" for issue in result.screen_reader_result.issues)


def test_acceptance_runs_rule_engine_exactly_once(tmp_path, monkeypatch):
    """evaluate_office_acceptance must run OfficeAccessibilityChecker.run_all
    exactly once (previously it ran twice: once for checker_report, once more
    inside run_office_screen_reader_checks)."""
    from project_remedy.office_checker import OfficeAccessibilityChecker

    call_count = 0
    original_run_all = OfficeAccessibilityChecker.run_all

    def counting_run_all(self):
        nonlocal call_count
        call_count += 1
        return original_run_all(self)

    monkeypatch.setattr(OfficeAccessibilityChecker, "run_all", counting_run_all)

    bad = make_docx(tmp_path / "bad.docx")
    evaluate_office_acceptance(bad)

    assert call_count == 1
