"""The OCR preflight must never ship a rebuild that loses real text.

`_maybe_rebuild_broken_text_layer` rebuilds a page's text layer with Tesseract
when it looks broken (e.g. a math worksheet whose symbols are PUA-encoded). But
OCR of a page that already has extractable words is a net fidelity regression —
Tesseract mangles the words while "fixing" the symbols. The gate keeps the OCR
rebuild only when it preserves at least as many real (alphabetic) words as the
original; otherwise the original is kept (and the issue is flagged).
"""
from __future__ import annotations

from project_remedy.pdf_fixer import _ocr_preserves_real_words


def test_ocr_that_loses_real_words_is_rejected():
    # A math answer key: words extract fine; OCR drops most of them.
    original = "Math 240 Entry Skills Solutions the answers are shown here today"
    rebuilt = "Math Entry"
    assert _ocr_preserves_real_words(original, rebuilt) is False


def test_ocr_that_recovers_garbage_is_accepted():
    # A genuinely broken page: only PUA/symbol noise, no real words.
    original = "     3 4 5"
    rebuilt = "Recovered readable sentence from the scanned image"
    assert _ocr_preserves_real_words(original, rebuilt) is True


def test_equal_real_words_is_accepted():
    assert _ocr_preserves_real_words("alpha beta gamma", "alpha beta gamma") is True
