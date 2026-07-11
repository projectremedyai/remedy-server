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


def test_genuinely_empty_cells_still_fail() -> None:
    """A blank-form table with empty data cells must not be scored as ready."""
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
    result = score_table_cell_lookup_report(report)
    assert result.score == 0.0
    assert result.passed is False
    finding = result.findings[0]
    assert finding["issue"] == "table_not_lookup_ready"
    assert finding["has_non_empty_headers"] is True
    assert finding["has_non_empty_data_cells"] is False
