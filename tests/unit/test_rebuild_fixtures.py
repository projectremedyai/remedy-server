"""Smoke tests for the shared RebuildRequest test builders."""

from __future__ import annotations

from project_remedy.rebuild.ast import FigureBlock, HeadingBlock, RebuildRequest
from tests.unit.rebuild_fixtures import make_request, write_assets


def test_make_request_default_is_valid_and_rich(tmp_path):
    assets = write_assets(tmp_path)
    request = make_request(asset_dir=tmp_path)
    assert isinstance(request, RebuildRequest)
    kinds = [b.kind for b in request.content]
    assert kinds == ["heading", "paragraph", "list", "simple_table", "figure", "artifact"]
    assert request.conformance.pdfua == "PDFUA_1"
    assert set(request.assets) == {"img-1", "img-2"}
    assert set(assets) == {"img-1", "img-2"}


def test_make_request_overrides_content(tmp_path):
    request = make_request(
        asset_dir=tmp_path,
        content=[HeadingBlock(level=1, runs=[{"text": "Only"}])],
        assets={},
    )
    assert len(request.content) == 1
