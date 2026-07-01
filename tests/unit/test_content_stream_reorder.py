"""Tests for content-stream reordering (content_stream_reorder.py).

Pure-pikepdf: builds synthetic tagged content streams and asserts the physical
marked-content order is re-sequenced to a target order WITHOUT renumbering
MCIDs, dropping content, or touching the struct tree — and that the pass is
idempotent and bails safely on unsafe pages. No rendering / no model here (the
render pixel-diff gate is the caller's job and is out of scope for unit tests).
"""
from __future__ import annotations

from pathlib import Path

import pikepdf
import pytest
from pikepdf import Array, ContentStreamInstruction as CSI, Dictionary, Name, Operator, String

from project_remedy.content_stream_reorder import (
    _IDENTITY,
    _collect_slice,
    _mat_inverse,
    _mat_mul,
    _mcid_of,
    _segment,
    reorder_page_to_order,
    struct_reading_order,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def op(name, *operands):
    return CSI(list(operands), Operator(name))


def text_block(mcid: int, text: str, tag: str = "/P"):
    """A movable tagged text block: /Tag <</MCID n>> BDC BT Tf Tj ET EMC."""
    return [
        op("BDC", Name(tag), Dictionary(MCID=mcid)),
        op("BT"),
        op("Tf", Name("/F1"), 12),
        op("Td", 72, 700 - mcid * 20),
        op("Tj", String(text)),
        op("ET"),
        op("EMC"),
    ]


def paint_block(mcid: int):
    """A NON-movable tagged block (contains a paint op) — must stay in place."""
    return [
        op("BDC", Name("/Figure"), Dictionary(MCID=mcid)),
        op("re", 0, 0, 10, 10),
        op("f"),
        op("EMC"),
    ]


def _page_with(pdf, instrs):
    pdf.add_blank_page(page_size=(612, 792))
    page = pdf.pages[-1]
    page.Resources = Dictionary(Font=Dictionary(F1=pdf.make_indirect(Dictionary(
        Type=Name("/Font"), Subtype=Name("/Type1"), BaseFont=Name("/Helvetica")))))
    page.Contents = pdf.make_stream(pikepdf.unparse_content_stream(instrs))
    return page


def mcid_sequence(page) -> list[int]:
    """Physical BDC-order MCID sequence of a page's content stream."""
    seq = []
    for instr in pikepdf.parse_content_stream(page):
        if str(instr.operator) == "BDC" and len(instr.operands) > 1:
            props = instr.operands[1]
            if isinstance(props, pikepdf.Dictionary) and "/MCID" in props:
                seq.append(int(props["/MCID"]))
    return seq


# --------------------------------------------------------------------------- #
# matrix helpers
# --------------------------------------------------------------------------- #
def test_mat_mul_identity_is_noop():
    m = (2.0, 0.0, 0.0, 3.0, 5.0, 7.0)
    assert _mat_mul(_IDENTITY, m) == m
    assert _mat_mul(m, _IDENTITY) == m


def test_mat_inverse_roundtrip_and_singular():
    m = (2.0, 0.0, 0.0, 4.0, 10.0, 20.0)
    inv = _mat_inverse(m)
    assert inv is not None
    prod = _mat_mul(m, inv)
    for got, want in zip(prod, _IDENTITY):
        assert abs(got - want) < 1e-9
    assert _mat_inverse((0.0, 0.0, 0.0, 0.0, 0.0, 0.0)) is None


# --------------------------------------------------------------------------- #
# _mcid_of
# --------------------------------------------------------------------------- #
def test_mcid_of_inline_dict():
    assert _mcid_of([Name("/P"), Dictionary(MCID=3)], None) == 3


def test_mcid_of_named_property_resolved():
    props = Dictionary(MC0=Dictionary(MCID=5))
    assert _mcid_of([Name("/P"), Name("/MC0")], props) == 5


def test_mcid_of_missing_returns_none():
    assert _mcid_of([Name("/P")], None) is None               # too few operands
    assert _mcid_of([Name("/P"), Dictionary(Foo=1)], None) is None  # no /MCID


# --------------------------------------------------------------------------- #
# _collect_slice / _segment safety
# --------------------------------------------------------------------------- #
def test_collect_slice_detects_unbalanced():
    instrs = [op("BDC", Name("/P"), Dictionary(MCID=0)), op("BT"), op("Tj", String("x"))]
    _slice, end, has_text, has_paint, nested, bt_idx = _collect_slice(instrs, 0)
    assert end == -1  # no matching EMC


def test_segment_bails_on_unbalanced_marked_content():
    instrs = [op("BDC", Name("/P"), Dictionary(MCID=0)), op("BT"), op("Tj", String("x"))]
    _bg, blocks, _ctm, bail = _segment(instrs, None)
    assert bail == "unbalanced marked content"


# --------------------------------------------------------------------------- #
# reorder_page_to_order
# --------------------------------------------------------------------------- #
def test_reorder_moves_blocks_to_target_order():
    pdf = pikepdf.Pdf.new()
    page = _page_with(pdf, text_block(0, "a") + text_block(1, "b") + text_block(2, "c"))
    assert mcid_sequence(page) == [0, 1, 2]

    moved = reorder_page_to_order(pdf, page, [2, 0, 1])
    assert moved == 3
    assert mcid_sequence(page) == [2, 0, 1]
    pdf.close()


def test_reorder_preserves_mcids_and_is_idempotent():
    pdf = pikepdf.Pdf.new()
    page = _page_with(pdf, text_block(0, "a") + text_block(1, "b") + text_block(2, "c"))

    reorder_page_to_order(pdf, page, [2, 1, 0])
    assert sorted(mcid_sequence(page)) == [0, 1, 2]  # no MCID dropped/renumbered
    assert mcid_sequence(page) == [2, 1, 0]

    # second call with the same target is a no-op (idempotent)
    assert reorder_page_to_order(pdf, page, [2, 1, 0]) == 0
    assert mcid_sequence(page) == [2, 1, 0]
    pdf.close()


def test_reorder_already_in_order_is_noop():
    pdf = pikepdf.Pdf.new()
    page = _page_with(pdf, text_block(0, "a") + text_block(1, "b"))
    assert reorder_page_to_order(pdf, page, [0, 1]) == 0
    pdf.close()


def test_non_movable_paint_block_not_reordered():
    pdf = pikepdf.Pdf.new()
    # movable text MCID 0, non-movable paint MCID 1
    page = _page_with(pdf, text_block(0, "a") + paint_block(1))
    # ask to move paint (1) before text (0): paint stays in place, only the
    # movable text block is repositioned, so no crash and both MCIDs survive.
    reorder_page_to_order(pdf, page, [1, 0])
    assert sorted(mcid_sequence(page)) == [0, 1]
    pdf.close()


def test_unsafe_page_returns_zero():
    pdf = pikepdf.Pdf.new()
    # unbalanced marked content -> segment bails -> no rewrite
    page = _page_with(pdf, [op("BDC", Name("/P"), Dictionary(MCID=0)),
                            op("BT"), op("Tj", String("x"))])
    assert reorder_page_to_order(pdf, page, [0]) == 0
    pdf.close()


# --------------------------------------------------------------------------- #
# struct_reading_order
# --------------------------------------------------------------------------- #
def test_struct_reading_order_maps_pages_in_struct_order():
    pdf = pikepdf.Pdf.new()
    page = _page_with(pdf, text_block(0, "a") + text_block(1, "b"))
    pg = page.obj
    # Document -> [P(mcid 1), P(mcid 0)]  (struct order deliberately 1 then 0)
    p1 = pdf.make_indirect(Dictionary(Type=Name("/StructElem"), S=Name("/P"), Pg=pg, K=1))
    p0 = pdf.make_indirect(Dictionary(Type=Name("/StructElem"), S=Name("/P"), Pg=pg, K=0))
    doc = pdf.make_indirect(Dictionary(Type=Name("/StructElem"), S=Name("/Document"),
                                       K=Array([p1, p0])))
    pdf.Root.StructTreeRoot = pdf.make_indirect(
        Dictionary(Type=Name("/StructTreeRoot"), K=Array([doc])))
    pdf.Root.MarkInfo = Dictionary(Marked=True)

    order = struct_reading_order(pdf)
    assert order == {0: [1, 0]}  # follows struct order, not MCID number
    pdf.close()


def test_struct_reading_order_dedups_repeated_mcid():
    pdf = pikepdf.Pdf.new()
    page = _page_with(pdf, text_block(0, "a"))
    pg = page.obj
    # two struct elems referencing the SAME mcid 0 -> deduped to one entry
    e1 = pdf.make_indirect(Dictionary(Type=Name("/StructElem"), S=Name("/Span"), Pg=pg, K=0))
    e2 = pdf.make_indirect(Dictionary(Type=Name("/StructElem"), S=Name("/Span"), Pg=pg, K=0))
    pdf.Root.StructTreeRoot = pdf.make_indirect(
        Dictionary(Type=Name("/StructTreeRoot"), K=Array([e1, e2])))
    order = struct_reading_order(pdf)
    assert order == {0: [0]}
    pdf.close()
