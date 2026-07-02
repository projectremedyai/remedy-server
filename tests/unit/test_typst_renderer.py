"""FR-4/5: TypstRenderer subprocess wrapper. Binary-dependent tests skip when
typst is not installed; the negative alt-text test is AC #4 of the PRD."""

from __future__ import annotations

import pathlib
import shutil

import pytest

from project_remedy.rebuild.typst_renderer import (
    TypstCompileError,
    TypstNotAvailable,
    TypstRenderer,
    TypstTimeout,
    TypstUnsupportedConstruct,
    resolve_typst_binary,
)
from tests.unit.rebuild_fixtures import make_request

needs_typst = pytest.mark.skipif(shutil.which("typst") is None, reason="typst CLI not installed")


def test_resolve_typst_binary_type():
    binary = resolve_typst_binary()
    assert binary is None or binary.name == "typst"


@needs_typst
async def test_render_produces_pdf(tmp_path):
    renderer = TypstRenderer(binary_path=resolve_typst_binary())
    pdf = await renderer.render(make_request(asset_dir=tmp_path))
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000


@needs_typst
async def test_missing_alt_hard_fails_compile(tmp_path):
    """AC #4: the ua-1 compiler gate. Bypass the generator (which cannot emit
    alt-less images) with a handcrafted bad source via the renderer's internals."""
    renderer = TypstRenderer(binary_path=resolve_typst_binary())
    from tests.unit.rebuild_fixtures import TINY_PNG

    (tmp_path / "p.png").write_bytes(TINY_PNG)
    (tmp_path / "main.typ").write_text('#image("p.png")\n')
    with pytest.raises(TypstCompileError) as excinfo:
        await renderer._compile(tmp_path)  # noqa: SLF001 - deliberate internal test
    assert "alt" in str(excinfo.value).lower()


@needs_typst
async def test_timeout_raises(tmp_path):
    renderer = TypstRenderer(binary_path=resolve_typst_binary(), timeout_s=0.0001)
    with pytest.raises(TypstTimeout):
        await renderer.render(make_request(asset_dir=tmp_path))


async def test_whitespace_alt_becomes_unsupported_construct(tmp_path):
    """Whitespace-only alt passes Pydantic (min_length=1) but the generator
    refuses it; that GeneratorError must surface as TypstUnsupportedConstruct,
    not leak out untranslated. Fires before any subprocess call, so this runs
    with or without typst installed."""
    from project_remedy.rebuild.ast import AssetRef, FigureBlock

    request = make_request(
        asset_dir=tmp_path,
        content=[FigureBlock(asset_ref="img-1", alt=" ")],
        assets={"img-1": AssetRef(path=str(tmp_path / "img-1.png"), mime="image/png")},
    )
    renderer = TypstRenderer(binary_path=resolve_typst_binary() or pathlib.Path("/usr/bin/true"))
    with pytest.raises(TypstUnsupportedConstruct):
        await renderer.render(request)


async def test_zero_row_table_becomes_unsupported_construct(tmp_path):
    """Zero-row table passes Pydantic (no min_length on rows) but the
    generator refuses it; same translation as above."""
    from project_remedy.rebuild.ast import SimpleTableBlock

    request = make_request(asset_dir=tmp_path, content=[SimpleTableBlock(rows=[])], assets={})
    renderer = TypstRenderer(binary_path=resolve_typst_binary() or pathlib.Path("/usr/bin/true"))
    with pytest.raises(TypstUnsupportedConstruct):
        await renderer.render(request)
