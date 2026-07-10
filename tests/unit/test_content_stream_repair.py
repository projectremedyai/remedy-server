"""Tests for the content-stream BT/ET repair (content_stream_repair.py).

Reproduces the engine's corruption shapes (dropped ET, dropped intermediate BT,
orphaned ET) and asserts the repair yields balanced, well-formed text objects
while preserving text, marked content (MCIDs), and the struct tree.
"""

from __future__ import annotations

import collections
from pathlib import Path

import pikepdf
import pytest
from pikepdf import Array, ContentStreamInstruction as CSI, Dictionary, Name, Operator, String

from project_remedy.content_stream_repair import _renormalize, repair_pdf


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def op(name, *operands):
    return CSI(list(operands), Operator(name))


def ops_of(instructions):
    return [str(i.operator) for i in instructions]


def balanced(instructions) -> bool:
    bt = et = 0
    depth = 0
    for i in instructions:
        s = str(i.operator)
        if s == "BT":
            bt += 1; depth += 1
        elif s == "ET":
            et += 1; depth -= 1
            if depth < 0:
                return False  # ET before BT
    return bt == et and depth == 0


def text_ops_all_inside(instructions) -> bool:
    """Every Tj/TJ/Td/Tm sits inside a BT/ET and no q/path op does."""
    in_text = False
    TEXT = {"Tj", "TJ", "'", '"', "Td", "TD", "Tm", "T*"}
    ILLEGAL_IN_TEXT = {"q", "Q", "cm", "re", "m", "l", "c", "Do", "n", "W", "W*", "f", "h"}
    for i in instructions:
        s = str(i.operator)
        if s == "BT":
            in_text = True
        elif s == "ET":
            in_text = False
        elif s in TEXT and not in_text:
            return False
        elif s in ILLEGAL_IN_TEXT and in_text:
            return False
    return True


# --------------------------------------------------------------------------- #
# pure _renormalize unit tests                                                 #
# --------------------------------------------------------------------------- #
def test_missing_et_before_emc_is_inserted():
    # engine shape: BDC/tag BT <text> EMC   (ET dropped before EMC)
    instrs = [
        op("BDC", Name("/P"), Dictionary(MCID=0)),
        op("BT"), op("Tf", Name("/F1"), 12), op("Tj", String("hi")),
        op("EMC"),
    ]
    out, changes = _renormalize(instrs)
    assert changes == 1
    assert balanced(out) and text_ops_all_inside(out)
    # ET must come before EMC (proper nesting: BDC opened outside the text object)
    seq = ops_of(out)
    assert seq[seq.index("EMC") - 1] == "ET"


def test_dropped_intermediate_bt_is_reopened():
    # engine shape: BT text1 <graphics> text2  (middle ET+BT dropped)
    instrs = [
        op("BT"), op("Tm", 1, 0, 0, 1, 0, 0), op("Tj", String("a")),
        op("q"), op("cm", 1, 0, 0, 1, 5, 5), op("re", 0, 0, 9, 9), op("f"), op("Q"),
        op("Tj", String("b")),  # second run lost its BT
    ]
    out, changes = _renormalize(instrs)
    assert balanced(out) and text_ops_all_inside(out)
    # both text runs are inside text objects, the q/cm/re/f are not
    assert ops_of(out).count("BT") == 2 and ops_of(out).count("ET") == 2


def test_reopened_text_object_restores_text_matrix():
    """Regression for the invisible-body-text loss: when repair re-opens a dropped
    text object, the fresh BT resets the text matrix to identity. A run positioned
    by a RELATIVE Td (Acrobat/Word norm: one Tm, then relative moves) then collapses
    to the origin at ~1pt and vanishes. Repair must emit an explicit Tm restoring the
    accumulated text-line matrix at the re-opened BT."""
    instrs = [
        op("BT"),
        op("Tm", 1, 0, 0, 1, 100, 700),
        op("Tj", String("line1")),
        # illegal-in-text graphics force an ET; the next run lost its BT
        op("q"), op("re", 0, 0, 9, 9), op("f"), op("Q"),
        op("Td", 0, -14),            # RELATIVE move — this run has no Tm of its own
        op("Tj", String("line2")),
        op("ET"),
    ]
    out, changes = _renormalize(instrs)
    assert balanced(out) and text_ops_all_inside(out)
    seq = ops_of(out)
    bt_positions = [i for i, s in enumerate(seq) if s == "BT"]
    assert len(bt_positions) == 2
    second_bt = bt_positions[1]
    # the re-opened BT must be immediately followed by a Tm restoring the matrix
    assert seq[second_bt + 1] == "Tm", f"re-opened BT not followed by Tm: {seq}"
    # and that Tm must be the accumulated line matrix (100,700) so the following
    # Td 0 -14 lands line2 at (100,686), not (0,-14)
    assert [float(x) for x in out[second_bt + 1].operands] == [1, 0, 0, 1, 100, 700]


def test_mid_line_continuation_restores_advanced_text_matrix():
    """A run split mid-line (continuation is a show op with NO leading Td) must be
    restored to the ADVANCED text matrix (line origin + the glyph advances already
    shown), not the line origin. Requires a font-width map so advances are known."""
    widths = {"/F1": ((lambda code: 500.0), False)}   # every glyph is 500/1000 wide
    instrs = [
        op("BT"),
        op("Tm", 1, 0, 0, 1, 100, 700),
        op("Tf", Name("/F1"), 10),
        op("Tj", String("AB")),                 # advances 2 * (0.5 * 10) = 10 units
        op("q"), op("re", 0, 0, 9, 9), op("f"), op("Q"),   # forces ET; next run lost BT
        op("Tj", String("CD")),                 # mid-line continuation, no Td
        op("ET"),
    ]
    out, changes = _renormalize(instrs, widths)
    assert balanced(out) and text_ops_all_inside(out)
    seq = ops_of(out)
    bt_positions = [i for i, s in enumerate(seq) if s == "BT"]
    assert len(bt_positions) == 2
    second_bt = bt_positions[1]
    assert seq[second_bt + 1] == "Tm", f"continuation BT not followed by Tm: {seq}"
    # restored to the advanced position: e = 100 + 10 = 110 (NOT the 100 line origin)
    assert [float(x) for x in out[second_bt + 1].operands] == [1, 0, 0, 1, 110, 700]


def test_orphaned_et_is_dropped():
    instrs = [op("ET"), op("q"), op("Q")]  # ET with no BT
    out, changes = _renormalize(instrs)
    assert "ET" not in ops_of(out)
    assert changes == 1 and balanced(out)


def test_redundant_nested_bt_is_dropped():
    instrs = [op("BT"), op("BT"), op("Tj", String("x")), op("ET")]
    out, changes = _renormalize(instrs)
    assert ops_of(out).count("BT") == 1 and ops_of(out).count("ET") == 1
    assert balanced(out)


def test_wellformed_stream_is_unchanged():
    instrs = [
        op("q"), op("cm", 1, 0, 0, 1, 0, 0), op("Q"),
        op("BT"), op("Tf", Name("/F1"), 10), op("Tj", String("ok")), op("ET"),
    ]
    out, changes = _renormalize(instrs)
    assert changes == 0
    assert ops_of(out) == ops_of(instrs)


def test_marked_content_inside_text_is_kept_inside():
    # BDC opened INSIDE the text object -> EMC stays inside, no early ET
    instrs = [
        op("BT"), op("BDC", Name("/Span"), Dictionary(MCID=1)),
        op("Tj", String("x")), op("EMC"), op("ET"),
    ]
    out, changes = _renormalize(instrs)
    assert changes == 0  # already well-formed
    assert balanced(out)


# --------------------------------------------------------------------------- #
# end-to-end PDF tests                                                         #
# --------------------------------------------------------------------------- #
def _write_pdf(path: Path, page_ops: list) -> None:
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(300, 300))
    page = pdf.pages[0]
    res = Dictionary(Font=Dictionary(F1=pdf.make_indirect(Dictionary(
        Type=Name("/Font"), Subtype=Name("/Type1"), BaseFont=Name("/Helvetica")))))
    page.Resources = res
    page.Contents = pdf.make_stream(pikepdf.unparse_content_stream(page_ops))
    pdf.save(str(path))
    pdf.close()


def _page_counts(path: Path):
    pdf = pikepdf.open(path)
    c = collections.Counter()
    for o, operator in pikepdf.parse_content_stream(pdf.pages[0]):
        c[str(operator)] += 1
    pdf.close()
    return c


def test_repair_pdf_balances_and_is_idempotent(tmp_path: Path):
    p = tmp_path / "broken.pdf"
    # BDC BT text EMC (no ET) + a second text run that lost its BT after graphics
    _write_pdf(p, [
        op("BDC", Name("/P"), Dictionary(MCID=0)),
        op("BT"), op("Tf", Name("/F1"), 12), op("Tj", String("hello")),
        op("EMC"),
        op("q"), op("re", 0, 0, 10, 10), op("f"), op("Q"),
        op("Tf", Name("/F1"), 12), op("Tj", String("world")),
    ])

    before = _page_counts(p)
    assert before["BT"] != before["ET"]  # starts unbalanced

    changed = repair_pdf(p)
    assert changed > 0
    after = _page_counts(p)
    assert after["BT"] == after["ET"] and after["BT"] > 0
    assert after["EMC"] == before["EMC"]      # marked content preserved
    assert after["Tj"] == before["Tj"]        # text runs preserved

    # second pass = no-op
    assert repair_pdf(p) == 0


def test_repair_preserves_text_marked_content_and_struct_tree(tmp_path: Path):
    p = tmp_path / "struct.pdf"
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(300, 300))
    page = pdf.pages[0]
    page.Resources = Dictionary(Font=Dictionary(F1=pdf.make_indirect(Dictionary(
        Type=Name("/Font"), Subtype=Name("/Type1"), BaseFont=Name("/Helvetica")))))
    page.Contents = pdf.make_stream(pikepdf.unparse_content_stream([
        op("BDC", Name("/P"), Dictionary(MCID=0)),
        op("BT"), op("Tf", Name("/F1"), 12), op("Tj", String("tagged")),
        op("EMC"),  # ET dropped
    ]))
    # a struct tree referencing MCID 0
    elem = pdf.make_indirect(Dictionary(Type=Name("/StructElem"), S=Name("/P"), K=0))
    pdf.Root.StructTreeRoot = pdf.make_indirect(
        Dictionary(Type=Name("/StructTreeRoot"), K=Array([elem])))
    pdf.Root.MarkInfo = Dictionary(Marked=True)
    pdf.save(str(p))
    pdf.close()

    repair_pdf(p)

    with pikepdf.open(p) as out:
        assert "/StructTreeRoot" in out.Root           # struct tree intact
        st = out.Root.StructTreeRoot.K[0]
        assert str(st.S) == "/P" and int(st.K) == 0    # MCID mapping intact
        c = _page_counts(p)
        assert c["BT"] == c["ET"] == 1
        assert c["EMC"] == 1 and c["Tj"] == 1


def test_process_directory_survives_bad_files(tmp_path: Path):
    _write_pdf(tmp_path / "a.pdf", [op("BDC", Name("/P"), Dictionary(MCID=0)),
                                    op("BT"), op("Tj", String("x")), op("EMC")])
    (tmp_path / "bad.pdf").write_bytes(b"%PDF-1.7 garbage")

    from project_remedy.content_stream_repair import process_directory
    r = process_directory(tmp_path)
    assert r.files == 1 and r.errors == 1
    assert r.error_files == ["bad.pdf"]
    assert r.files_changed == 1


# --------------------------------------------------------------------------- #
# integration: fix_all must re-balance before saving (engine-wiring regression) #
# --------------------------------------------------------------------------- #
def test_fix_all_emits_balanced_bt_et(tmp_path: Path):
    """Regression for the engine bug: the marked-content injectors are not
    BT/ET-aware, so fix_all must run the content-stream repair before saving.
    A crafted PDF with an unbalanced stream must come out of fix_all balanced."""
    from project_remedy import pdf_fixer

    src = tmp_path / "in.pdf"
    out = tmp_path / "out.pdf"
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(300, 300))
    page = pdf.pages[0]
    page.Resources = Dictionary(Font=Dictionary(F1=pdf.make_indirect(Dictionary(
        Type=Name("/Font"), Subtype=Name("/Type1"), BaseFont=Name("/Helvetica")))))
    # corruption shape: BDC/tag BT <text> EMC  (ET dropped)
    page.Contents = pdf.make_stream(pikepdf.unparse_content_stream([
        op("BDC", Name("/P"), Dictionary(MCID=0)),
        op("BT"), op("Tf", Name("/F1"), 12), op("Tj", String("hello world")),
        op("EMC"),
    ]))
    elem = pdf.make_indirect(Dictionary(Type=Name("/StructElem"), S=Name("/P"), K=0))
    pdf.Root.StructTreeRoot = pdf.make_indirect(
        Dictionary(Type=Name("/StructTreeRoot"), K=Array([elem])))
    pdf.Root.MarkInfo = Dictionary(Marked=True)
    pdf.save(str(src))
    pdf.close()

    # only="__none__" skips every fixer; gs_was_used=True skips OCR preflight —
    # so this isolates the repair step that fix_all now runs before saving.
    report = pdf_fixer.fix_all(src, out, only="__none__", gs_was_used=True)

    assert out.exists()
    counts = _page_counts(out)
    assert counts["BT"] == counts["ET"] == 1     # balanced
    assert counts["EMC"] == 1 and counts["Tj"] == 1
    assert any("BT/ET" in c for c in report.changes)
