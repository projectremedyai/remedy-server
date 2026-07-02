"""FR-4/5: TypstRenderer subprocess wrapper. Binary-dependent tests skip when
typst is not installed; the negative alt-text test is AC #4 of the PRD."""

from __future__ import annotations

import shutil

import pytest

from project_remedy.rebuild.typst_renderer import (
    TypstCompileError,
    TypstNotAvailable,
    TypstRenderer,
    TypstTimeout,
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
