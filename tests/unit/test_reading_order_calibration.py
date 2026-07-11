"""P5 — reading-order judge calibration.

The deterministic reading-order proxy penalised every repeated transcript line
0.1, so multi-category evaluation forms and Likert grids (which legitimately
repeat rating labels, section prompts and ``[Form: ...]`` control
announcements) failed even though their reading order is perfectly monotonic.

The calibration:
- form-control scaffold lines and adjacent-duplicate lines are dropped before
  counting repeats;
- ``repeated_transcript_line`` / ``repeated_transcript_block`` become advisory
  (severity ``info``) — they no longer subtract from the score;
- genuine duplication is still caught: ``page_order_backtracking`` (multi-page,
  counted once) and ``duplicated_document_content`` (>=50% redundant content,
  for single-page full copies).
"""

from __future__ import annotations

from pathlib import Path

from project_remedy.behavioral_proxies.pdf.transcript_analyzer import (
    analyze_tag_tree_report,
)
from project_remedy.behavioral_proxies.pdf.reading_order_comprehension import (
    score_reading_order_report,
)
from project_remedy.behavioral_proxies.shared.transcript_analysis import (
    analyze_transcript_text,
    duplicated_content_ratio,
)
from project_remedy.tag_tree_reader import TagNode, TagTreeReport


def _report(nodes: list[TagNode]) -> TagTreeReport:
    return TagTreeReport(
        file_path=Path("mem.pdf"),
        page_count=(max((n.page for n in nodes), default=0) + 1),
        has_structure_tree=True,
        nodes=nodes,
    )


def _p(text: str, page: int = 0, tag: str = "P") -> TagNode:
    return TagNode(tag=tag, depth=1, page=page, text=text, alt_text="",
                   lang="", children_count=0, has_content=True)


def _form(alt: str, page: int = 0) -> TagNode:
    return TagNode(tag="Form", depth=1, page=page, text="", alt_text=alt,
                   lang="", children_count=0, has_content=True)


LONG_LABEL = "Meets or Exceeds Expectations on the Standard"   # >=20 chars


def _issues(findings, issue):
    return [f for f in findings if f.get("issue") == issue]


# --- form-scaffold + advisory behaviour -------------------------------------

def test_form_scaffold_lines_not_penalized():
    transcript = "\n".join(["[Form: Checkbox field]"] * 20 + ["Body paragraph text here that is unique."])
    findings = analyze_transcript_text(transcript)
    # No form-scaffold line should appear as a repeated_transcript_line finding.
    reps = _issues(findings, "repeated_transcript_line")
    assert all("[Form:" not in f.get("preview", "") for f in reps)


def test_repeated_line_is_advisory_info():
    transcript = "\n".join([LONG_LABEL, "spacer one two three", LONG_LABEL,
                            "spacer four five six", LONG_LABEL])
    findings = analyze_transcript_text(transcript)
    reps = _issues(findings, "repeated_transcript_line")
    assert reps, "legitimate repeat should still be reported"
    assert all(f["severity"] == "info" for f in reps)


def test_adjacent_duplicates_collapse():
    # Same label 4x in a row (a grid column) collapses to one -> not >=3 repeats.
    transcript = "\n".join([LONG_LABEL] * 4)
    findings = analyze_transcript_text(transcript)
    assert not _issues(findings, "repeated_transcript_line")


# --- legitimate form passes, duplication fails ------------------------------

def test_legitimate_repeated_form_passes():
    # Monotonic pages, repeated rating labels + checkbox scaffold: should pass.
    nodes = []
    for page in range(3):
        nodes.append(_p(f"Evaluation category {page} unique prompt text.", page))
        nodes.append(_p(LONG_LABEL, page))
        nodes.append(_form("Checkbox field", page))
        nodes.append(_p("No Basis for Judgment on this item.", page))
    result = score_reading_order_report(_report(nodes))
    assert result.passed
    assert result.score >= 0.9


def test_page_backtracking_still_fails_once():
    # Page 0,1 then back to 0 (a doubled tree). One backtracking error -> 0.5.
    nodes = [_p("Alpha content line one here.", 0), _p("Beta content line two.", 1),
             _p("Alpha content line one here.", 0), _p("Beta content line two.", 1)]
    findings = analyze_tag_tree_report(_report(nodes))
    bt = _issues(findings, "page_order_backtracking")
    assert len(bt) == 1
    assert bt[0]["severity"] == "error"
    result = score_reading_order_report(_report(nodes))
    assert not result.passed


def test_single_page_full_duplication_fails():
    # One page, content duplicated -> ratio >= 0.5 -> duplicated_document_content.
    unique = [f"Unique paragraph number {i} with enough length to count." for i in range(6)]
    nodes = [_p(t, 0) for t in unique] + [_p(t, 0) for t in unique]
    assert duplicated_content_ratio([n.text for n in nodes]) >= 0.5
    findings = analyze_tag_tree_report(_report(nodes))
    assert _issues(findings, "duplicated_document_content")
    assert not score_reading_order_report(_report(nodes)).passed


def test_low_redundancy_form_not_flagged_as_duplicated():
    nodes = [_p(f"Distinct instruction line {i} of the form here.", 0) for i in range(10)]
    nodes += [_p(LONG_LABEL, 0), _p(LONG_LABEL, 0)]  # small repeated bit
    assert duplicated_content_ratio([n.text for n in nodes]) < 0.5
    findings = analyze_tag_tree_report(_report(nodes))
    assert not _issues(findings, "duplicated_document_content")
