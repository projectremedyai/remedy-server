"""Fidelity regression: fix_untagged_content must never delete visible text.

`_remove_top_level_whitespace_actualtext_spans` strips whitespace-ActualText
placeholder spans, but the placeholder branch failed to check the span body for
visible content — so a Span tagged with a tab ActualText that ALSO drew real text
(common in tabular PDFs, e.g. course schedules) was deleted wholesale, silently
losing the words. See CONTENT_LOSS_ROOTCAUSE — schedule deletion bug.
"""

from __future__ import annotations

from project_remedy.pdf_fixer import _remove_top_level_whitespace_actualtext_spans


def test_actualtext_tab_span_with_visible_text_is_preserved():
    text = (
        "/Span <</ActualText <FEFF0009>>> BDC\n"
        "BT /F1 10 Tf (Real schedule cell text) Tj ET\n"
        "EMC\n"
    )
    cleaned, removed = _remove_top_level_whitespace_actualtext_spans(text)
    assert removed == 0, f"deleted a span containing visible text: {cleaned!r}"
    assert "Real schedule cell text" in cleaned


def test_truly_empty_actualtext_tab_span_is_still_removed():
    # A genuine placeholder — whitespace ActualText, no visible content — is removed.
    text = (
        "/Span <</ActualText <FEFF0009>>> BDC\n"
        "BT /F1 10 Tf ET\n"
        "EMC\n"
    )
    cleaned, removed = _remove_top_level_whitespace_actualtext_spans(text)
    assert removed == 1
    assert "/Span" not in cleaned
