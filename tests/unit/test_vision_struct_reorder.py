"""Tests for vision-driven struct-tree reorder (vision_struct_reorder.py).

Uses a FAKE ``vision_fn(image_path, prompt) -> str`` returning canned JSON and
monkeypatches ``_render_page`` so no rendering / no model is needed. Covers the
pure helpers and one end-to-end reorder, and — as the fix-#1 regression guard —
asserts the rebuilt tree keeps the /P<->/K parent/child bijection and the leaf
count (the invariants that guarantee the reorder does not corrupt the tree).
"""
from __future__ import annotations

import json

import pikepdf
import pytest
from pikepdf import Array, Dictionary, Name

import project_remedy.vision_struct_reorder as VSR
from project_remedy.vision_struct_reorder import (
    _ask_order,
    _collect_units,
    _count_leaves,
    _descend_to_container,
    _renormalize_headings,
    reorder_struct_vision,
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _doc_with_blocks(specs):
    """Build a 1-page PDF: StructTreeRoot -> Document -> [elem…].

    specs: list of (S_tag, mcid). Each elem gets K=<mcid int> and Pg=page.
    Returns (pdf, container_document, [elems]).
    """
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    pg = pdf.pages[0].obj
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
    return [int(e.K) for e in container.K]


def _bijection_ok(container) -> bool:
    """Every child lists container as its /P (the invariant a flatten would break)."""
    for child in container.K:
        if not isinstance(child, pikepdf.Dictionary):
            continue
        p = child.get("/P")
        if p is None or p.objgen != container.objgen:
            return False
    return True


@pytest.fixture(autouse=True)
def _no_render(monkeypatch):
    # never shell out to a renderer; the fake vision_fn ignores the image path.
    monkeypatch.setattr(VSR, "_render_page", lambda *a, **k: "/tmp/_stub.png")


def reverse_fn(image_path, prompt):
    import re
    seg = prompt.split("BLOCKS:", 1)[-1].split("\n\n", 1)[0]
    nums = [int(m) for m in re.findall(r"(?m)^\s*(\d+)\.", seg)]
    return json.dumps({"order": list(range(max(nums), 0, -1))}) if nums else ""


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def test_ask_order_parses_permutation():
    labeled = [(1, "title"), (2, "body"), (3, "footer")]
    assert _ask_order(lambda ip, pr: '{"order": [3, 1, 2]}', "img", labeled) == [3, 1, 2]


def test_ask_order_returns_none_on_garbage():
    assert _ask_order(lambda ip, pr: "no json here", "img", [(1, "x")]) is None
    assert _ask_order(lambda ip, pr: "", "img", [(1, "x")]) is None


def test_count_leaves_counts_mcid_bearing_elems():
    pdf, doc, _ = _doc_with_blocks([("/H1", 0), ("/P", 1), ("/P", 2)])
    assert _count_leaves(pdf.Root.StructTreeRoot) == 3
    pdf.close()


def test_descend_to_container_finds_document():
    pdf, doc, _ = _doc_with_blocks([("/H1", 0), ("/P", 1), ("/P", 2)])
    assert _descend_to_container(pdf).objgen == doc.objgen
    pdf.close()


def test_collect_units_returns_blocks_with_page():
    pdf, doc, _ = _doc_with_blocks([("/H1", 0), ("/P", 1), ("/P", 2)])
    pidx = {p.obj.objgen: i for i, p in enumerate(pdf.pages)}
    units = _collect_units(doc, pdf, pidx)
    assert len(units) == 3
    assert all(u["kind"] == "block" and u["page"] == 0 for u in units)
    pdf.close()


def test_renormalize_headings_clamps_skips():
    # H1 then H4 (skip) -> H4 clamped to H2
    pdf, doc, elems = _doc_with_blocks([("/H1", 0), ("/H4", 1)])
    units = [{"elem": elems[0]}, {"elem": elems[1]}]
    changed = _renormalize_headings({0: units})
    assert changed == 1
    assert str(elems[1].S) == "/H2"
    pdf.close()


# --------------------------------------------------------------------------- #
# end-to-end reorder + fix-#1 structural regression guard
# --------------------------------------------------------------------------- #
def test_reorder_reverses_order_and_preserves_invariants():
    pdf, doc, _ = _doc_with_blocks([("/P", 0), ("/P", 1), ("/P", 2), ("/P", 3)])
    assert _k_mcids(doc) == [0, 1, 2, 3]
    before_leaves = _count_leaves(pdf.Root.StructTreeRoot)

    rep = reorder_struct_vision(pdf, reverse_fn, pdf_path="ignored")
    assert rep["changed"] is True

    container = _descend_to_container(pdf)
    assert _k_mcids(container) == [3, 2, 1, 0]                 # reordered
    assert _count_leaves(pdf.Root.StructTreeRoot) == before_leaves  # leaf gate
    assert _bijection_ok(container)                            # /P<->/K intact
    pdf.close()


def test_reorder_skips_small_pages():
    # fewer than 3 units -> engine declines (no change)
    pdf, doc, _ = _doc_with_blocks([("/P", 0), ("/P", 1)])
    rep = reorder_struct_vision(pdf, reverse_fn, pdf_path="ignored")
    assert rep["changed"] is False
    assert _k_mcids(_descend_to_container(pdf)) == [0, 1]
    pdf.close()


def _doc_with_sections(section_specs):
    """Build Document -> [Sect -> [P(mcid)…], …] on one page, with /P set.

    section_specs: list of lists of mcids. Returns (pdf, doc, [sect…]).
    """
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    pg = pdf.pages[0].obj
    sects = []
    for mcids in section_specs:
        ps = [pdf.make_indirect(Dictionary(
            Type=Name("/StructElem"), S=Name("/P"), Pg=pg, K=m)) for m in mcids]
        sect = pdf.make_indirect(Dictionary(
            Type=Name("/StructElem"), S=Name("/Sect"), K=Array(ps)))
        for p in ps:
            p.P = sect
        sects.append(sect)
    doc = pdf.make_indirect(Dictionary(
        Type=Name("/StructElem"), S=Name("/Document"), K=Array(sects)))
    for s in sects:
        s.P = doc
    pdf.Root.StructTreeRoot = pdf.make_indirect(
        Dictionary(Type=Name("/StructTreeRoot"), K=Array([doc])))
    pdf.Root.MarkInfo = Dictionary(Marked=True)
    return pdf, doc, sects


def test_reorder_preserves_grouping_subtrees_no_flatten():
    """Fix-#1 regression: reordering must NOT flatten /Sect grouping — leaves
    stay under their real parent (the flatten broke veraPDF 7.2-26/7.4.2-1)."""
    pdf, doc, sects = _doc_with_sections([[0, 1, 2], [3, 4, 5]])
    rep = reorder_struct_vision(pdf, reverse_fn, pdf_path="ignored")
    assert rep["changed"] is True

    doc = _descend_to_container(pdf)
    # Document still holds exactly two /Sect children (NOT flattened to leaves)
    child_types = [str(k.S) for k in doc.K]
    assert child_types == ["/Sect", "/Sect"]
    # reverse order -> section containing mcids 3,4,5 now comes first
    first, second = doc.K[0], doc.K[1]
    assert [int(p.K) for p in first.K] == [5, 4, 3]
    assert [int(p.K) for p in second.K] == [2, 1, 0]
    # every leaf still parented to ITS section (bijection preserved, no orphan)
    for sect in doc.K:
        for p in sect.K:
            assert p.P.objgen == sect.objgen
    pdf.close()


def test_verapdf_delta_gate_reverts_on_new_violation(monkeypatch):
    """verify_verapdf reverts when the reorder introduces a NEW veraPDF rule."""
    pdf, doc, _ = _doc_with_blocks([("/P", 0), ("/P", 1), ("/P", 2)])
    calls = {"n": 0}

    def fake_rules(_pdf):
        calls["n"] += 1
        return set() if calls["n"] == 1 else {"ISO 14289-1:2014-7.4.2-1"}

    monkeypatch.setattr(VSR, "_verapdf_rule_ids", fake_rules)
    rep = reorder_struct_vision(pdf, reverse_fn, pdf_path="x", verify_verapdf=True)
    assert rep["changed"] is False
    assert any("ABORTED" in n for n in rep["notes"])
    assert _k_mcids(_descend_to_container(pdf)) == [0, 1, 2]  # fully restored
    pdf.close()


def test_verapdf_delta_gate_keeps_preexisting_failure(monkeypatch):
    """A rule failing BOTH before and after (pre-existing font residue) is not
    blamed on the reorder — the reorder is kept."""
    pdf, doc, _ = _doc_with_blocks([("/P", 0), ("/P", 1), ("/P", 2)])
    monkeypatch.setattr(
        VSR, "_verapdf_rule_ids", lambda _pdf: {"ISO 14289-1:2014-7.21.7-2"})
    rep = reorder_struct_vision(pdf, reverse_fn, pdf_path="x", verify_verapdf=True)
    assert rep["changed"] is True
    assert _k_mcids(_descend_to_container(pdf)) == [2, 1, 0]  # kept reordered
    pdf.close()


def test_reorder_target_pages_leaves_other_pages_untouched():
    # two pages, 3 blocks each; only reorder page 2 -> page 1 struct unchanged.
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    pdf.add_blank_page(page_size=(612, 792))
    pg1, pg2 = pdf.pages[0].obj, pdf.pages[1].obj
    mk = lambda mcid, pg: pdf.make_indirect(Dictionary(
        Type=Name("/StructElem"), S=Name("/P"), Pg=pg, K=mcid))
    p1 = [mk(i, pg1) for i in range(3)]
    p2 = [mk(i + 3, pg2) for i in range(3)]
    doc = pdf.make_indirect(Dictionary(Type=Name("/StructElem"), S=Name("/Document"),
                                       K=Array(p1 + p2)))
    for e in p1 + p2:
        e.P = doc
    pdf.Root.StructTreeRoot = pdf.make_indirect(
        Dictionary(Type=Name("/StructTreeRoot"), K=Array([doc])))
    pdf.Root.MarkInfo = Dictionary(Marked=True)

    rep = reorder_struct_vision(pdf, reverse_fn, pdf_path="ignored", target_pages={2})
    assert rep["changed"] is True
    order = _k_mcids(_descend_to_container(pdf))
    # page-1 block mcids (0,1,2) keep their original relative order at the front
    assert order[:3] == [0, 1, 2]
    # page-2 block mcids (3,4,5) are reversed
    assert order[3:] == [5, 4, 3]
    pdf.close()
