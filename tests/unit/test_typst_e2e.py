"""Phase T0 E2E: RebuildRequest -> generate -> typst compile (ua-1) ->
struct_assert -> (veraPDF when available). PRD §8 metrics 1-4 on the fixture doc."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from project_remedy.rebuild.struct_assert import verify
from project_remedy.rebuild.typst_renderer import TypstRenderer, resolve_typst_binary
from tests.unit.rebuild_fixtures import make_request

needs_typst = pytest.mark.skipif(shutil.which("typst") is None, reason="typst CLI not installed")
needs_verapdf = pytest.mark.skipif(shutil.which("verapdf") is None, reason="verapdf not installed")


@needs_typst
async def test_e2e_fixture_document_round_trips(tmp_path):
    request = make_request(asset_dir=tmp_path)
    pdf = await TypstRenderer(binary_path=resolve_typst_binary()).render(request)
    report = verify(request, pdf)
    assert report.passed, report.mismatches


@needs_typst
@needs_verapdf
async def test_e2e_verapdf_ua1_clean(tmp_path):
    request = make_request(asset_dir=tmp_path)
    pdf = await TypstRenderer(binary_path=resolve_typst_binary()).render(request)
    out = tmp_path / "typst_e2e.pdf"
    out.write_bytes(pdf)
    result = subprocess.run(
        ["verapdf", "-f", "ua1", str(out)], capture_output=True, text=True, timeout=300,
    )
    assert 'isCompliant="true"' in result.stdout, result.stdout[-2000:]
