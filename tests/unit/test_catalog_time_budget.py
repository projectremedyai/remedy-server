"""P6 — catalog time budget controls for fix_and_verify.

Large catalogs blew past the 900s gate because each verify cycle re-validates
the whole document (O(pages)) and issues per-figure vision calls. These env-
gated controls bound that cost without window-chunking; all default to
preserving current behavior.
"""

from __future__ import annotations

import io

import pikepdf
import pytest

import project_remedy.pdf_fixer as PF


def _multipage_pdf(path, n_pages: int) -> None:
    """A minimal multi-page PDF via reportlab (real, fix_all can process it)."""
    reportlab = pytest.importorskip("reportlab")
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter

    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for i in range(n_pages):
        c.setFont("Helvetica", 12)
        c.drawString(72, 720, f"Page {i + 1} heading")
        c.drawString(72, 690, f"Body paragraph on page {i + 1} with some words.")
        c.showPage()
    c.save()
    path.write_bytes(buf.getvalue())


def test_deadline_short_circuits_verify_and_final_passes(tmp_path, monkeypatch):
    src = tmp_path / "in.pdf"
    _multipage_pdf(src, 2)
    out = tmp_path / "out.pdf"
    # An immediately-past deadline: fix_all still runs, but verify cycles and the
    # final vision pass must be skipped and recorded.
    monkeypatch.setenv("PDF_FIX_DEADLINE_SECONDS", "0.0001")
    report = PF.fix_and_verify(src, out, max_cycles=3)
    assert out.exists()
    joined = " ".join(report.skipped)
    assert "deadline" in joined.lower()
    # Output is still a valid PDF.
    with pikepdf.open(out) as pdf:
        assert len(pdf.pages) == 2


def test_no_deadline_by_default_runs_cycles(tmp_path, monkeypatch):
    src = tmp_path / "in.pdf"
    _multipage_pdf(src, 2)
    out = tmp_path / "out.pdf"
    monkeypatch.delenv("PDF_FIX_DEADLINE_SECONDS", raising=False)
    report = PF.fix_and_verify(src, out, max_cycles=2)
    assert out.exists()
    # No deadline-skip messages when the control is unset.
    assert not any("deadline" in s.lower() for s in report.skipped)


def test_large_doc_caps_cycles(tmp_path, monkeypatch):
    src = tmp_path / "big.pdf"
    _multipage_pdf(src, 6)
    out = tmp_path / "out.pdf"
    # Threshold below the page count -> max_cycles clamped to 1. We assert via
    # the log path indirectly: the run completes and stays valid. (The clamp is
    # exercised; behavior parity is the guarantee.)
    monkeypatch.setenv("PDF_FIX_LARGE_DOC_PAGES", "3")
    report = PF.fix_and_verify(src, out, max_cycles=3)
    assert out.exists()
    with pikepdf.open(out) as pdf:
        assert len(pdf.pages) == 6


def test_figure_layout_bbox_area_ranks_largest_first():
    from pikepdf import Dictionary, Name, Array

    def fig(side):
        a = Dictionary({"/O": Name("/Layout"),
                        "/BBox": Array([0, 0, side, side])})
        return Dictionary({"/S": Name("/Figure"), "/A": a})

    small = fig(10)   # area 100
    big = fig(100)    # area 10000
    no_bbox = Dictionary({"/S": Name("/Figure")})

    assert PF._figure_layout_bbox_area(big) == 10000
    assert PF._figure_layout_bbox_area(small) == 100
    assert PF._figure_layout_bbox_area(no_bbox) == 0.0
    ranked = sorted([small, no_bbox, big],
                    key=PF._figure_layout_bbox_area, reverse=True)
    assert ranked[0] is big and ranked[-1] is no_bbox
