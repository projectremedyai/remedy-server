"""Regression: table-regularity repair must not manufacture runaway /ColSpan.

`fix_table_regularity` derives ``target_width = max(row_widths)``, and a row's
width is the SUM of its cells' existing /ColSpan values. Nothing clamps that value
on write or on read, so once a table carries a bogus span, the next repair pass
reads it back, computes a larger target_width, and writes a larger span still.
Observed on the LAMC 2008 monthly calendars: source has NO spans at all, one
fix_all pass wrote /ColSpan 32, a second wrote 373, and the DELIVERED SEP-08.pdf
ships /ColSpan 7,208,595 on 157 cells.

Two harms, both covered here:
  1. Semantic: a cell claiming a multi-million-column span is broken table
     structure for a screen reader -- invisible to veraPDF, fatal to a human.
  2. Operational: pdf_checker._check_table_regularity does
     ``range(col_idx, col_idx + span)`` over that value, materialising a 60M-element
     set per cell, which hangs the acceptance gate indefinitely.

A /ColSpan can never legitimately exceed the table's column count.
"""
from __future__ import annotations

import signal
import time

import pikepdf
from pikepdf import Array, Dictionary, Name

import project_remedy.pdf_fixer as PF


def _table_pdf(rowspec, preset_spans=None):
    """Table -> TBody -> TR*, where rowspec[i] = cell count of row i.

    preset_spans: {(row_idx, cell_idx): span} seeds an EXISTING /ColSpan, i.e. the
    state a file is in after a previous (buggy) remediation pass.
    """
    preset_spans = preset_spans or {}
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    pg = pdf.pages[0].obj
    pdf.Root.MarkInfo = Dictionary(Marked=True)

    content = b""
    mcid = 0
    trs = []
    for r, ncells in enumerate(rowspec):
        cells = []
        for c in range(ncells):
            content += (
                b"/TD <</MCID %d>> BDC BT /F1 10 Tf 10 700 Td (x) Tj ET EMC\n" % mcid
            )
            cell = pdf.make_indirect(Dictionary(
                Type=Name("/StructElem"), S=Name("/TD"), Pg=pg, K=mcid))
            if (r, c) in preset_spans:
                cell["/ColSpan"] = preset_spans[(r, c)]
            cells.append(cell)
            mcid += 1
        tr = pdf.make_indirect(Dictionary(
            Type=Name("/StructElem"), S=Name("/TR"), Pg=pg, K=Array(cells)))
        for cell in cells:
            cell.P = tr
        trs.append(tr)

    pg.Contents = pdf.make_stream(content)
    tbody = pdf.make_indirect(Dictionary(
        Type=Name("/StructElem"), S=Name("/TBody"), Pg=pg, K=Array(trs)))
    for tr in trs:
        tr.P = tbody
    table = pdf.make_indirect(Dictionary(
        Type=Name("/StructElem"), S=Name("/Table"), Pg=pg, K=Array([tbody])))
    tbody.P = table
    doc = pdf.make_indirect(Dictionary(
        Type=Name("/StructElem"), S=Name("/Document"), K=Array([table])))
    table.P = doc
    pdf.Root.StructTreeRoot = pdf.make_indirect(
        Dictionary(Type=Name("/StructTreeRoot"), K=Array([doc])))
    return pdf


def _spans(pdf):
    out = []
    for obj in pdf.objects:
        if isinstance(obj, pikepdf.Dictionary) and str(obj.get("/S")) in ("/TD", "/TH"):
            v = obj.get("/ColSpan")
            attrs = obj.get("/A")
            if v is None and isinstance(attrs, pikepdf.Dictionary):
                v = attrs.get("/ColSpan")
            if v is not None:
                out.append(int(v))
    return out


def test_repair_sanitizes_a_preexisting_absurd_colspan():
    """A bogus span already in the file must be clamped, never propagated.

    This is the delivered-file state: SEP-08.pdf carries /ColSpan 7,208,595. Today
    the repair reads that back as the row's width, makes it target_width, and grows
    every other row to match -- so re-remediating a delivered file makes it worse.
    """
    pdf = _table_pdf([1, 3], preset_spans={(0, 0): 5_000_000})

    PF.fix_table_regularity(pdf)

    spans = _spans(pdf)
    assert spans, "expected the repair to touch the table"
    # The table has at most 3 columns. No cell may claim more.
    assert max(spans) <= 3, (
        f"repair propagated an absurd span instead of clamping it: {max(spans)}"
    )


def test_repair_does_not_compound_when_a_later_pass_adds_a_cell():
    """The real runaway: repair, then another fixer tags one more cell into a row,
    then repair runs again. Row width = SUM of existing spans, so the already-written
    span is read back and inflated. This is how 32 became 373 became 7.2 million.
    """
    pdf = _table_pdf([1, 3])
    PF.fix_table_regularity(pdf)
    first_max = max(_spans(pdf), default=0)

    # A subsequent fixer pass tags one more cell into the wide row (this happens:
    # untagged content gets pulled into the table on a later pass).
    table = next(o for o in pdf.objects
                 if isinstance(o, pikepdf.Dictionary) and str(o.get("/S")) == "/Table")
    tbody = table.K[0]
    wide_tr = tbody.K[1]
    extra = pdf.make_indirect(Dictionary(
        Type=Name("/StructElem"), S=Name("/TD"), Pg=wide_tr.Pg, K=99))
    extra.P = wide_tr
    wide_tr.K = Array([*list(wide_tr.K), extra])

    PF.fix_table_regularity(pdf)
    second_max = max(_spans(pdf), default=0)

    # The table now has 4 columns. Nothing may claim more than that, and certainly
    # the span must not balloon relative to the first pass.
    assert second_max <= 4, (
        f"repair compounded its own output: max span {first_max} -> {second_max}"
    )


def test_checker_does_not_hang_on_absurd_colspan(tmp_path_factory):
    """pdf_checker must not build range() over a multi-million span.

    The real file (AUG-08, /ColSpan 60,410,228 on 338 cells) wedges the acceptance
    gate forever. Bound the check: a table this small must finish effectively
    instantly.
    """
    from project_remedy.pdf_checker import PDFAccessibilityChecker

    pdf = _table_pdf([1, 3], preset_spans={(0, 0): 60_000_000})
    out = tmp_path_factory.mktemp("colspan") / "absurd.pdf"
    pdf.save(str(out))
    checker = PDFAccessibilityChecker(out)

    # No pytest-timeout here, and the bug is an effectively-infinite loop -- without
    # our own alarm this test would HANG the suite instead of failing it.
    def _boom(signum, frame):
        raise TimeoutError("table-regularity check hung on an absurd /ColSpan")

    old = signal.signal(signal.SIGALRM, _boom)
    signal.alarm(10)
    start = time.monotonic()
    try:
        with pikepdf.open(str(out)) as saved:
            checker._check_table_regularity(saved)  # type: ignore[attr-defined]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, f"table-regularity check took {elapsed:.1f}s on an absurd span"
