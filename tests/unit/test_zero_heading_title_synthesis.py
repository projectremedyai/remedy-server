"""P4 — zero-heading title synthesis coverage.

The heading behavioral proxy returns a hard 0.0 for any document with no
headings.  Many already-tagged forms reach the end of remediation with a
visually-evident title but zero ``/H*`` nodes because the metadata-title path's
matcher only inspected ``/Alt``/``/ActualText`` (not MCID marked content) and a
fragile content-stream regex.

``_ensure_document_has_title_heading`` closes that gap: on a zero-heading
document it promotes a confident first-page title node (metadata title, largest
first-page text, or first bookmark) to ``/H1`` using the SAME MCID text reader
the judge uses — and only ever promotes an existing node, never fabricates one.
"""

from __future__ import annotations

import pikepdf
from pikepdf import Array, Dictionary, Name

from project_remedy.pdf_fixer import (
    _confident_title_candidates,
    _ensure_document_has_title_heading,
    _get_struct_type,
    _match_first_page_node_by_text,
    walk_structure_tree,
)
from project_remedy.behavioral_proxies.pdf.heading_navigation import (
    score_heading_navigation_report,
)
from project_remedy.tag_tree_reader import read_tag_tree


def _doc(content_bytes: bytes, specs, *, title: str | None = None):
    """1-page tagged PDF: StructTreeRoot -> Document -> leaf elements."""
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    pg = pdf.pages[0].obj
    pg.Contents = pdf.make_stream(content_bytes)
    elems = []
    for s, mcid in specs:
        elems.append(
            pdf.make_indirect(
                Dictionary(Type=Name("/StructElem"), S=Name(s), Pg=pg, K=mcid)
            )
        )
    doc = pdf.make_indirect(
        Dictionary(Type=Name("/StructElem"), S=Name("/Document"), K=Array(elems))
    )
    for e in elems:
        e.P = doc
    pdf.Root.StructTreeRoot = pdf.make_indirect(
        Dictionary(Type=Name("/StructTreeRoot"), K=Array([doc]))
    )
    pdf.Root.MarkInfo = Dictionary(Marked=True)
    if title is not None:
        with pdf.open_metadata() as meta:
            meta["dc:title"] = title
    return pdf, elems


CONTENT = (
    b"/P <</MCID 0>> BDC\n"
    b"BT /F1 24 Tf 72 720 Td (Revolving Fund Reimbursement Request) Tj ET\n"
    b"EMC\n"
    b"/P <</MCID 1>> BDC\n"
    b"BT /F1 11 Tf 72 690 Td (Requested by:) Tj ET\n"
    b"EMC\n"
)


def _heading_tags(pdf):
    return [
        _get_struct_type(n)
        for n, _d, _p in walk_structure_tree(pdf)
        if _get_struct_type(n).startswith("H") and _get_struct_type(n) != "StructTreeRoot"
    ]


def test_metadata_title_promotes_matching_first_page_node():
    pdf, elems = _doc(CONTENT, [("/P", 0), ("/P", 1)],
                      title="Revolving Fund Reimbursement Request")
    assert _heading_tags(pdf) == []
    assert _ensure_document_has_title_heading(pdf) == 1
    assert _get_struct_type(elems[0]) == "H1"  # the title node
    assert _get_struct_type(elems[1]) == "P"    # body untouched


def test_promoted_heading_passes_the_judge(tmp_path):
    pdf, elems = _doc(CONTENT, [("/P", 0), ("/P", 1)],
                      title="Revolving Fund Reimbursement Request")
    _ensure_document_has_title_heading(pdf)
    out = tmp_path / "out.pdf"
    pdf.save(out)
    report = read_tag_tree(out)
    result = score_heading_navigation_report(report)
    assert result.passed
    assert result.score >= 0.85


def test_no_promotion_when_title_matches_no_page_content():
    # Metadata title that appears nowhere on the page must not force a heading
    # onto unrelated content (never fabricate). The synthetic /F1 font is not
    # fitz-extractable, so the visible-text fallback contributes nothing here —
    # leaving only the (non-matching) metadata title candidate.
    pdf, elems = _doc(CONTENT, [("/P", 0), ("/P", 1)],
                      title="Completely Unrelated Metadata String")
    assert _ensure_document_has_title_heading(pdf) == 0
    assert _heading_tags(pdf) == []


def test_metadata_title_is_a_candidate():
    pdf, elems = _doc(CONTENT, [("/P", 0), ("/P", 1)],
                      title="Revolving Fund Reimbursement Request")
    cands = _confident_title_candidates(pdf)
    assert "revolving fund reimbursement request" in " ".join(cands).lower()


def test_does_not_touch_document_that_already_has_heading():
    content = (
        b"/H1 <</MCID 0>> BDC\n"
        b"BT /F1 24 Tf 72 720 Td (Existing Title) Tj ET\nEMC\n"
        b"/P <</MCID 1>> BDC\n"
        b"BT /F1 11 Tf 72 690 Td (Body copy here.) Tj ET\nEMC\n"
    )
    pdf, elems = _doc(content, [("/H1", 0), ("/P", 1)], title="Existing Title")
    assert _ensure_document_has_title_heading(pdf) == 0
    assert _get_struct_type(elems[0]) == "H1"


def test_matcher_returns_none_for_absent_text():
    pdf, elems = _doc(CONTENT, [("/P", 0), ("/P", 1)])
    assert _match_first_page_node_by_text(pdf, "nonexistent phrase xyz") is None


def test_matcher_finds_node_by_mcid_text():
    pdf, elems = _doc(CONTENT, [("/P", 0), ("/P", 1)])
    node = _match_first_page_node_by_text(pdf, "Revolving Fund Reimbursement Request")
    assert node is not None
    assert node.objgen == elems[0].objgen
