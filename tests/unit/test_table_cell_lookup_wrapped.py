"""Regression tests for descendant-aware table cell text extraction.

The tag-tree reader only attaches marked-content text to the struct node that
directly owns the MCID. Real-world (and Adobe/AI-authored) tables wrap each
cell's text in a child ``Span``/``P`` element, which is valid PDF/UA. The table
cell-lookup proxy must read the whole cell subtree, otherwise every wrapped
table is scored 0.0 ("table_not_lookup_ready") even though a screen reader
speaks the cell content correctly.
"""

from __future__ import annotations

from pathlib import Path

from project_remedy.behavioral_proxies.pdf.table_cell_lookup import (
    score_table_cell_lookup_report,
)
from project_remedy.tag_tree_reader import TagNode, TagTreeReport


def _node(tag: str, depth: int, text: str = "") -> TagNode:
    return TagNode(
        tag=tag,
        depth=depth,
        page=0,
        text=text,
        alt_text="",
        lang="",
        children_count=0,
        has_content=bool(text),
    )


def _report(nodes: list[TagNode]) -> TagTreeReport:
    return TagTreeReport(
        file_path=Path("synthetic.pdf"),
        page_count=1,
        has_structure_tree=True,
        nodes=nodes,
    )


def _wrapped_table() -> TagTreeReport:
    """Table whose cell text lives one level down in a Span (valid PDF/UA)."""
    return _report(
        [
            _node("Table", 1),
            _node("TR", 2),
            _node("TH", 3),
            _node("Span", 4, "Date"),
            _node("TH", 3),
            _node("Span", 4, "Amount"),
            _node("TR", 2),
            _node("TD", 3),
            _node("Span", 4, "2024-01-01"),
            _node("TD", 3),
            _node("Span", 4, "$50.00"),
        ]
    )


def test_wrapped_cells_are_lookup_ready() -> None:
    result = score_table_cell_lookup_report(_wrapped_table())
    assert result.score == 1.0, result.findings
    assert result.passed is True


def test_direct_text_cells_still_pass() -> None:
    """Cells that carry their own text must keep passing (no regression)."""
    report = _report(
        [
            _node("Table", 1),
            _node("TR", 2),
            _node("TH", 3, "Date"),
            _node("TH", 3, "Amount"),
            _node("TR", 2),
            _node("TD", 3, "2024-01-01"),
            _node("TD", 3, "$50.00"),
        ]
    )
    result = score_table_cell_lookup_report(report)
    assert result.score == 1.0
    assert result.passed is True


def test_labeled_headers_with_empty_cells_is_a_blank_grid() -> None:
    """Labeled headers + present-but-empty data cells is an unfilled form grid.

    Superseded semantics: this used to be asserted as a hard failure
    ("genuinely empty cells still fail"). But the *input* documents score 0.0
    the same way — remediation cannot invent data that the source form does not
    contain, so failing table_structure here punishes remediation for the
    document being blank rather than for a structural defect. The grid is
    navigable (a screen reader speaks each header and each empty field), so it
    is treated as not-applicable for cell lookup. Genuine failures — dropped
    cells, missing headers — are covered by the tests below.
    """
    report = _report(
        [
            _node("Table", 1),
            _node("TR", 2),
            _node("TH", 3),
            _node("Span", 4, "Date"),
            _node("TH", 3),
            _node("Span", 4, "Amount"),
            _node("TR", 2),
            _node("TD", 3),  # empty cell, no descendant text
            _node("TD", 3),  # empty cell, no descendant text
        ]
    )
    result = score_table_cell_lookup_report(report, threshold=0.75)
    assert result.metadata["blank_fillable_grids"] == 1
    assert result.score == 1.0
    assert result.passed is True
    assert any(f["issue"] == "blank_fillable_grid" for f in result.findings)


def _blank_rating_grid() -> TagTreeReport:
    """Labeled headers, data-cell nodes present but text-empty (unfilled form)."""
    return _report(
        [
            _node("Table", 1),
            _node("TR", 2),
            _node("TH", 3, "Excellent"),
            _node("TH", 3, "Good"),
            _node("TH", 3, "Fair"),
            _node("TR", 2),
            _node("TD", 3, ""),
            _node("TD", 3, ""),
            _node("TD", 3, ""),
        ]
    )


def test_blank_fillable_grid_is_not_a_failure() -> None:
    result = score_table_cell_lookup_report(_blank_rating_grid(), threshold=0.75)
    assert result.metadata["blank_fillable_grids"] == 1
    assert result.metadata["scorable_tables"] == 0
    assert result.score == 1.0
    assert result.passed
    assert not any(f["severity"] == "error" for f in result.findings)


def test_dropped_data_cells_still_fail() -> None:
    # Headers present but the data-cell nodes are gone (dropped by remediation)
    # -> not a blank form, must still fail. This is the guard that keeps the
    # blank-grid exemption from masking cell loss.
    report = _report(
        [
            _node("Table", 1),
            _node("TR", 2),
            _node("TH", 3, "Excellent"),
            _node("TH", 3, "Good"),
            _node("TH", 3, "Fair"),
        ]
    )
    result = score_table_cell_lookup_report(report, threshold=0.75)
    assert result.metadata["blank_fillable_grids"] == 0
    assert result.score == 0.0
    assert not result.passed


def test_blank_grid_does_not_mask_a_real_failing_table() -> None:
    # One blank grid (N/A) + one genuinely broken data table (data, no headers)
    # -> score reflects only the scorable table, which fails.
    nodes = _blank_rating_grid().nodes + [
        _node("Table", 1),
        _node("TR", 2),
        _node("TD", 3, "42"),
    ]
    result = score_table_cell_lookup_report(_report(nodes), threshold=0.75)
    assert result.metadata["blank_fillable_grids"] == 1
    assert result.metadata["scorable_tables"] == 1
    assert result.score == 0.0
    assert not result.passed
