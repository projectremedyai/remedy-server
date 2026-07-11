"""MCID-text enrichment for the heading task's structure-order prompt.

``_get_page_structure_order`` skips P/Span/H* nodes without ActualText, so on
real remediated forms (whose text lives in content streams, not ActualText)
the model sees a nearly-empty numbered list — it can't name current tags
(the "? -> H1" flags) and the fixer gets no usable element_index. heading-v1
was trained on synthetic pages WITH populated lists, so the empty list is also
a train/test mismatch. ``include_mcid_text=True`` (heading agent only) pulls
each node's marked-content text so the list matches what the model saw in
training.
"""
from __future__ import annotations

import pikepdf
from pikepdf import Array, Dictionary, Name

from project_remedy.pdf_vision import _get_page_structure_order


def _form_pdf(path):
    """1-page tagged PDF whose P nodes have MCID content but NO ActualText —
    the shape of every remediated LAMC form."""
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    pg = pdf.pages[0].obj
    pg.Contents = pdf.make_stream(
        b"/P <</MCID 0>> BDC\n"
        b"BT /F1 24 Tf 72 720 Td (LAMC CRIME STATS - AUGUST 2012) Tj ET\nEMC\n"
        b"/P <</MCID 1>> BDC\n"
        b"BT /F1 11 Tf 72 690 Td (Body details of the report.) Tj ET\nEMC\n"
    )
    elems = [
        pdf.make_indirect(Dictionary(Type=Name("/StructElem"), S=Name("/P"), Pg=pg, K=0)),
        pdf.make_indirect(Dictionary(Type=Name("/StructElem"), S=Name("/P"), Pg=pg, K=1)),
    ]
    doc = pdf.make_indirect(Dictionary(
        Type=Name("/StructElem"), S=Name("/Document"), K=Array(elems)))
    for e in elems:
        e.P = doc
    pdf.Root.StructTreeRoot = pdf.make_indirect(
        Dictionary(Type=Name("/StructTreeRoot"), K=Array([doc])))
    pdf.Root.MarkInfo = Dictionary(Marked=True)
    pdf.save(path)


def test_default_behavior_unchanged_skips_untexted_nodes(tmp_path):
    path = tmp_path / "form.pdf"
    _form_pdf(path)
    order = _get_page_structure_order(path, 1)
    assert "CRIME STATS" not in order, \
        "default (no ActualText -> skipped) must stay unchanged for other tasks"


def test_mcid_text_enrichment_lists_content_stream_text(tmp_path):
    path = tmp_path / "form.pdf"
    _form_pdf(path)
    order = _get_page_structure_order(path, 1, include_mcid_text=True)
    assert "LAMC CRIME STATS - AUGUST 2012" in order, \
        "enriched list must carry the node's marked-content text"
    assert "/P" in order
    # both P nodes now enumerated with stable 1-based indexes
    lines = [l for l in order.splitlines() if l.strip()]
    assert lines[0].lstrip().startswith("1.")
    assert lines[1].lstrip().startswith("2.")


def test_mcid_enrichment_prefers_actual_text_when_present(tmp_path):
    path = tmp_path / "form.pdf"
    _form_pdf(path)
    with pikepdf.open(path, allow_overwriting_input=True) as pdf:
        for node in pdf.Root.StructTreeRoot.K[0].K:
            if int(node.K) == 0:
                node.ActualText = pikepdf.String("Official Title")
        pdf.save(path)
    order = _get_page_structure_order(path, 1, include_mcid_text=True)
    assert "Official Title" in order, "ActualText still wins when present"
