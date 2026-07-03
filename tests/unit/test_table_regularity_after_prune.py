"""Regression: `_prune_dead_and_empty_nodes` must keep tables regular.

Pruning dead/empty table cells (dangling MCRs) leaves rows with unequal column
counts, which veraPDF flags as 7.2-42/43 ("Table rows shall have the same number
of columns"). The pruner already reruns header/scope repair when it removes table
nodes; it must also re-enforce row regularity, or the pruned tables ship
irregular. This introduced 7.2-42/43 on delivered ISS reports / catalog addenda
whose tables were otherwise fully content-recovered.
"""
from __future__ import annotations

import pikepdf
from pikepdf import Array, Dictionary, Name

import project_remedy.pdf_fixer as PF


# One live cell per drawn MCID; MCID 3 is intentionally absent (dead cell).
CONTENT = (
    b"/TD <</MCID 0>> BDC BT /F1 10 Tf 10 700 Td (a) Tj ET EMC\n"
    b"/TD <</MCID 1>> BDC BT /F1 10 Tf 40 700 Td (b) Tj ET EMC\n"
    b"/TD <</MCID 2>> BDC BT /F1 10 Tf 10 680 Td (c) Tj ET EMC\n"
)


def _table_pdf():
    """1-page PDF: Document -> Table -> [TR(TD0,TD1), TR(TD2,TD3)]; TD3 is dead."""
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    pg = pdf.pages[0].obj
    pg.Contents = pdf.make_stream(CONTENT)

    def td(mcid):
        return pdf.make_indirect(Dictionary(
            Type=Name("/StructElem"), S=Name("/TD"), Pg=pg, K=mcid))

    tr1_cells = [td(0), td(1)]
    tr2_cells = [td(2), td(3)]  # td(3) references absent MCID 3 -> dead
    tr1 = pdf.make_indirect(Dictionary(Type=Name("/StructElem"), S=Name("/TR"),
                                       Pg=pg, K=Array(tr1_cells)))
    tr2 = pdf.make_indirect(Dictionary(Type=Name("/StructElem"), S=Name("/TR"),
                                       Pg=pg, K=Array(tr2_cells)))
    for c in tr1_cells: c.P = tr1
    for c in tr2_cells: c.P = tr2
    table = pdf.make_indirect(Dictionary(Type=Name("/StructElem"), S=Name("/Table"),
                                         Pg=pg, K=Array([tr1, tr2])))
    tr1.P = table; tr2.P = table
    doc = pdf.make_indirect(Dictionary(Type=Name("/StructElem"), S=Name("/Document"),
                                       K=Array([table])))
    table.P = doc
    pdf.Root.StructTreeRoot = pdf.make_indirect(
        Dictionary(Type=Name("/StructTreeRoot"), K=Array([doc])))
    pdf.Root.MarkInfo = Dictionary(Marked=True)
    return pdf, table


def _table_cells(table):
    return sum(
        1
        for tr in table.K
        for c in tr.K
        if isinstance(c, pikepdf.Dictionary) and str(c.get("/S")) in ("/TD", "/TH")
    )


def test_pruning_table_cell_reenforces_regularity(monkeypatch):
    pdf, table = _table_pdf()
    assert _table_cells(table) == 4  # sanity: the table starts with 4 cells

    calls = []
    real = PF.fix_table_regularity

    def spy(p, **k):
        calls.append(True)
        return real(p, **k)

    monkeypatch.setattr(PF, "fix_table_regularity", spy)

    PF._prune_dead_and_empty_nodes(pdf)

    # A table cell was pruned (dead MCR), which can leave rows with unequal
    # column counts (veraPDF 7.2-42/43). The pruner must re-enforce row
    # regularity — not just header/scope repair — after removing table nodes.
    assert calls, "table-cell pruning did not re-run fix_table_regularity"
    assert _table_cells(table) < 4, "expected the dead table cell to be pruned"
