"""Regression: `_fix_empty_leaf_text_elements` must not orphan MCID markings.

When it removes a struct leaf that only marks whitespace, it also has to demote
that leaf's content-stream MCID marking to /Artifact — otherwise the marking is
left with no structure element, which veraPDF flags as 7.1-3 ("Content is neither
marked as Artifact nor tagged as real content"). This introduced 7.1-3 on ~38
delivered files that were otherwise fully content-recovered.

Second test guards correctness: if fitz text extraction returns empty for a leaf
that actually DRAWS real glyphs (e.g. a font with no ToUnicode), the leaf must be
kept — never removed or artifacted — so real text is never hidden from AT.
"""
from __future__ import annotations

import pikepdf
from pikepdf import Array, Dictionary, Name

import project_remedy.pdf_fixer as PF


def _doc(content_bytes: bytes, specs):
    """1-page PDF: StructTreeRoot -> Document -> [P(mcid)…]; page draws content_bytes."""
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    pg = pdf.pages[0].obj
    pg.Contents = pdf.make_stream(content_bytes)
    elems = []
    for s, mcid in specs:
        elems.append(pdf.make_indirect(Dictionary(
            Type=Name("/StructElem"), S=Name(s), Pg=pg, K=mcid)))
    doc = pdf.make_indirect(Dictionary(
        Type=Name("/StructElem"), S=Name("/Document"), K=Array(elems)))
    for e in elems:
        e.P = doc
    pdf.Root.StructTreeRoot = pdf.make_indirect(
        Dictionary(Type=Name("/StructTreeRoot"), K=Array([doc])))
    pdf.Root.MarkInfo = Dictionary(Marked=True)
    return pdf, doc, elems


def _k_mcids(container):
    kids = container.get("/K")
    if kids is None:
        return []
    items = kids if isinstance(kids, pikepdf.Array) else [kids]
    return [int(e.K) for e in items if isinstance(e, pikepdf.Dictionary) and "/K" in e]


CONTENT_WS = (
    b"/P <</MCID 0>> BDC\n"
    b"BT /F1 12 Tf 100 700 Td (Real heading text) Tj ET\n"
    b"EMC\n"
    b"/P <</MCID 1>> BDC\n"
    b"BT /F1 12 Tf 100 680 Td ( ) Tj ET\n"
    b"EMC\n"
)


def test_removed_whitespace_leaf_is_artifacted_not_orphaned(monkeypatch):
    pdf, doc, _ = _doc(CONTENT_WS, [("/P", 0), ("/P", 1)])
    # MCID 0 extracts real text; MCID 1 extracts empty (it draws only a space).
    monkeypatch.setattr(PF, "_extract_mcid_text",
                        lambda page: {0: "Real heading text", 1: ""})

    PF._fix_empty_leaf_text_elements(pdf)

    content = pdf.pages[0].obj.Contents.read_bytes()
    # the removed whitespace leaf's marking is demoted, not left orphaned
    assert b"/MCID 1>>" not in content, "orphan whitespace MCID left in content stream"
    assert b"/Artifact" in content, "whitespace marking was not demoted to /Artifact"
    # the real-text leaf is untouched
    assert b"/MCID 0>>" in content
    k = _k_mcids(doc)
    assert 0 in k and 1 not in k


CONTENT_REAL = (
    b"/P <</MCID 2>> BDC\n"
    b"BT /F1 12 Tf 100 700 Td (Actual words here) Tj ET\n"
    b"EMC\n"
)


def test_real_glyph_leaf_misjudged_empty_is_not_removed(monkeypatch):
    pdf, doc, _ = _doc(CONTENT_REAL, [("/P", 2)])
    # extraction FAILS (returns empty) even though the content draws real glyphs
    monkeypatch.setattr(PF, "_extract_mcid_text", lambda page: {2: ""})

    PF._fix_empty_leaf_text_elements(pdf)

    content = pdf.pages[0].obj.Contents.read_bytes()
    assert b"/MCID 2>>" in content, "real-text leaf was wrongly artifacted (content hidden)"
    assert b"/Artifact" not in content
    assert 2 in _k_mcids(doc), "real-text leaf was wrongly removed from the struct tree"
