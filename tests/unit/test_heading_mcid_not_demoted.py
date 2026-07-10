"""Regression tests for the empty-heading demotion (P2b).

Root cause (diagnosed 2026-07-10): the empty-heading demotion in
``_fix_subtitle_and_transitional_headings`` / ``_ensure_first_page_metadata_title_heading``
judged a heading "empty" via ``_structure_node_text``, which only reads
``/ActualText``, ``/Alt`` and ``/T``. Headings whose text lives in MCID marked
content (the common case for tagged PDFs) therefore looked blank and were
demoted to ``/P``, silently destroying valid document structure — this is what
made ``heading_semantics`` collapse to 0.0 once the duplicating visible-text
scaffold (which had been re-adding headings) stopped firing.

``_heading_has_renderable_text`` is the MCID-aware guard that fixes it: a
heading is only treated as empty when it has no ActualText/Alt/T, no marked
content of its own, and no descendant struct element that does.
"""

from __future__ import annotations

import pikepdf

from project_remedy.pdf_fixer import _heading_has_renderable_text


def _pdf() -> pikepdf.Pdf:
    return pikepdf.new()


def test_heading_with_direct_mcid_is_not_empty() -> None:
    """A heading whose text is MCID marked content must not read as empty."""
    node = pikepdf.Dictionary({"/S": pikepdf.Name("/H1"), "/K": 0})  # MCID 0
    assert _heading_has_renderable_text(node) is True


def test_heading_with_actual_text_is_not_empty() -> None:
    node = pikepdf.Dictionary(
        {"/S": pikepdf.Name("/H1"), "/ActualText": pikepdf.String("Section Title")}
    )
    assert _heading_has_renderable_text(node) is True


def test_heading_with_child_span_mcid_is_not_empty() -> None:
    """Cell/heading text wrapped in a child Span (valid PDF/UA) still counts."""
    pdf = _pdf()
    span = pdf.make_indirect(pikepdf.Dictionary({"/S": pikepdf.Name("/Span"), "/K": 7}))
    node = pikepdf.Dictionary({"/S": pikepdf.Name("/H1"), "/K": span})
    assert _heading_has_renderable_text(node) is True


def test_genuinely_empty_heading_is_empty() -> None:
    """A heading with no text, no MCID and no children is still demotable."""
    node = pikepdf.Dictionary({"/S": pikepdf.Name("/H1")})
    assert _heading_has_renderable_text(node) is False


def test_empty_heading_with_empty_child_is_empty() -> None:
    pdf = _pdf()
    empty_child = pdf.make_indirect(pikepdf.Dictionary({"/S": pikepdf.Name("/Span")}))
    node = pikepdf.Dictionary({"/S": pikepdf.Name("/H1"), "/K": empty_child})
    assert _heading_has_renderable_text(node) is False
