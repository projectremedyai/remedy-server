"""FR-10/11: independent AST<->struct-tree round-trip verification."""

from __future__ import annotations

import shutil

import pytest

from project_remedy.rebuild.ast import FigureBlock, HeadingBlock, ListBlock, ListItem, ParagraphBlock, Run
from project_remedy.rebuild.struct_assert import StructAssertReport, verify
from project_remedy.rebuild.typst_renderer import TypstRenderer, resolve_typst_binary
from tests.unit.rebuild_fixtures import make_request

needs_typst = pytest.mark.skipif(shutil.which("typst") is None, reason="typst CLI not installed")


@needs_typst
async def test_full_fixture_round_trips(tmp_path):
    request = make_request(asset_dir=tmp_path)
    pdf = await TypstRenderer(binary_path=resolve_typst_binary()).render(request)
    report = verify(request, pdf)
    assert report.passed, report.mismatches


@needs_typst
async def test_mismatch_detected_when_ast_expects_more(tmp_path):
    """Compile a doc with ONE heading, then verify against a request claiming TWO —
    the report must fail with an H-count mismatch (proves the checker is not a rubber stamp)."""
    one = make_request(asset_dir=tmp_path,
                       content=[HeadingBlock(level=1, runs=[Run(text="Only")])], assets={})
    pdf = await TypstRenderer(binary_path=resolve_typst_binary()).render(one)
    two = make_request(asset_dir=tmp_path,
                       content=[HeadingBlock(level=1, runs=[Run(text="Only")]),
                                HeadingBlock(level=2, runs=[Run(text="Ghost")])], assets={})
    report = verify(two, pdf)
    assert not report.passed
    assert any("H2" in m for m in report.mismatches)


@needs_typst
async def test_alt_text_byte_exact(tmp_path):
    from project_remedy.rebuild.ast import AssetRef

    request = make_request(
        asset_dir=tmp_path,
        content=[FigureBlock(asset_ref="img-1", alt="Exact alt 42")],
        assets={"img-1": AssetRef(path=str(tmp_path / "img-1.png"), mime="image/png")},
    )
    pdf = await TypstRenderer(binary_path=resolve_typst_binary()).render(request)
    assert verify(request, pdf).passed
    wrong = make_request(
        asset_dir=tmp_path,
        content=[FigureBlock(asset_ref="img-1", alt="Different alt")],
        assets={"img-1": AssetRef(path=str(tmp_path / "img-1.png"), mime="image/png")},
    )
    report = verify(wrong, pdf)
    assert not report.passed
    assert any("Alt" in m for m in report.mismatches)


@needs_typst
async def test_nested_list_round_trips(tmp_path):
    """A list item whose body contains a paragraph AND a nested list must
    round-trip cleanly — exercises the nested-ListBlock recursion in
    _expected end-to-end (previously only asserted by code inspection)."""
    nested = ListBlock(
        ordered=False,
        items=[
            ListItem(
                label_runs=[Run(text="•")],
                body=[
                    ParagraphBlock(runs=[Run(text="Outer item text")]),
                    ListBlock(
                        ordered=False,
                        items=[
                            ListItem(
                                label_runs=[Run(text="•")],
                                body=[ParagraphBlock(runs=[Run(text="Inner item text")])],
                            )
                        ],
                    ),
                ],
            )
        ],
    )
    request = make_request(asset_dir=tmp_path, content=[nested], assets={})
    pdf = await TypstRenderer(binary_path=resolve_typst_binary()).render(request)
    report = verify(request, pdf)
    assert report.passed, report.mismatches


def test_verify_handles_untagged_pdf(tmp_path):
    """A PDF with no struct tree must fail cleanly, not crash."""
    import pikepdf
    from io import BytesIO

    buf = BytesIO()
    with pikepdf.new() as pdf:
        pdf.add_blank_page()
        pdf.save(buf)
    request = make_request(asset_dir=tmp_path, content=[HeadingBlock(level=1, runs=[Run(text="X")])], assets={})
    report = verify(request, buf.getvalue())
    assert not report.passed
    assert any("StructTreeRoot" in m for m in report.mismatches)
