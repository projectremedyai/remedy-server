"""Terminal sweep: artifact whitespace MCID markings left orphaned by any pass.

Some passes (notably fix_page_retag) mark content with MCIDs that never get
wired into the structure tree. veraPDF flags each as 7.1-3 ("Content is neither
marked as Artifact nor tagged as real content"). `_artifact_orphan_whitespace_mcids`
sweeps these up at the end of remediation: provably-whitespace orphans are
demoted to /Artifact; real-content orphans are left untouched so no visible text
is ever hidden.
"""
from __future__ import annotations

import pikepdf
from pikepdf import Array, Dictionary, Name

import project_remedy.pdf_fixer as PF


CONTENT = (
    b"/P <</MCID 0>> BDC BT /F1 10 Tf 10 700 Td (real referenced) Tj ET EMC\n"
    b"/P <</MCID 1>> BDC BT /F1 10 Tf 10 680 Td ( ) Tj ET EMC\n"
    b"/P <</MCID 2>> BDC BT /F1 10 Tf 10 660 Td (real orphan words) Tj ET EMC\n"
)


def _doc():
    """1-page PDF where struct references only MCID 0; MCID 1 (whitespace) and
    MCID 2 (real) are orphaned in the content stream."""
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    pg = pdf.pages[0].obj
    pg.Contents = pdf.make_stream(CONTENT)
    p0 = pdf.make_indirect(Dictionary(Type=Name("/StructElem"), S=Name("/P"), Pg=pg, K=0))
    doc = pdf.make_indirect(Dictionary(Type=Name("/StructElem"), S=Name("/Document"),
                                       K=Array([p0])))
    p0.P = doc
    pdf.Root.StructTreeRoot = pdf.make_indirect(
        Dictionary(Type=Name("/StructTreeRoot"), K=Array([doc])))
    pdf.Root.MarkInfo = Dictionary(Marked=True)
    return pdf


def test_whitespace_orphan_artifacted_real_orphan_kept():
    pdf = _doc()

    PF._artifact_orphan_whitespace_mcids(pdf)

    content = pdf.pages[0].obj.Contents.read_bytes()
    # whitespace orphan (MCID 1) -> demoted to /Artifact
    assert b"/MCID 1>>" not in content, "whitespace orphan not artifacted"
    assert b"/Artifact" in content
    # real orphan (MCID 2) -> kept, never hidden
    assert b"/MCID 2>>" in content, "real-content orphan was wrongly artifacted"
    # referenced content (MCID 0) -> untouched
    assert b"/MCID 0>>" in content
