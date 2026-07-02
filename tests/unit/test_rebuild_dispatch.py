"""AC #9: dispatch through the real orchestration function.

_rebuild_from_semantics has heavy upstream deps (extractor/ollama/vision).
These tests monkeypatch exactly that upstream boundary — extraction returns
fixed markdown, vision returns no images — and let everything from
ast_builder onward run for real, with the render step exercised per-backend.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pikepdf
import pytest

import backend.app.engine_service as engine_service
from backend.app.jobs import Job

needs_typst = pytest.mark.skipif(shutil.which("typst") is None, reason="typst CLI not installed")


class _FakeStore:
    def __init__(self):
        self.updates: list[dict] = []

    async def update(self, job_id, **kwargs):
        self.updates.append({"job_id": job_id, **kwargs})

    def last(self):
        return self.updates[-1]


def _blank_pdf(path: Path) -> Path:
    with pikepdf.new() as pdf:
        pdf.add_blank_page()
        pdf.save(path)
    return path


def _job(input_path: Path, metadata: dict | None = None) -> Job:
    return Job(
        id="job-typst-1", kind="remediate_pdf", status="running", stage="",
        progress=0.5, input_path=str(input_path), output_path="", report_path="",
        error="", created_at="", updated_at="",
        metadata_json=json.dumps(metadata or {}),
    )


@pytest.fixture
def patched_upstream(monkeypatch, tmp_path):
    """Stub extractor/vision so the function runs to the render step."""

    class _FakeExtractor:
        def __init__(self, *a, **k): ...
        async def extract(self, doc_job):
            return "# Title\n\nBody paragraph text.\n"

    class _FakeOllama:
        def __init__(self, *a, **k): ...
        async def start(self): ...
        async def close(self): ...

    class _FakeDB:
        def __init__(self, *a, **k): ...
        async def create_job(self, doc_job): ...

    monkeypatch.setattr("project_remedy.extractor.ContentExtractor", _FakeExtractor)
    monkeypatch.setattr("project_remedy.ollama_client.OllamaClient", _FakeOllama)
    monkeypatch.setattr(engine_service, "DatabaseManager", _FakeDB)
    monkeypatch.setattr(
        "project_remedy.pdf_vision.create_provider_from_config", lambda cfg: object()
    )

    async def fake_vision(*args, **kwargs):
        return {}

    monkeypatch.setattr(engine_service, "_vision_enrich", fake_vision)
    # DocumentJob.get_extracted_images must return [] — patch at use site:
    monkeypatch.setattr(
        engine_service.DocumentJob, "get_extracted_images", lambda self: [], raising=False
    )
    return tmp_path


def _cfg(tmp_path, backend="questpdf"):
    import dataclasses

    from project_remedy.config import load_config

    # Config dataclasses are frozen; rebuild the nested pieces via replace().
    cfg = load_config()
    cfg = dataclasses.replace(
        cfg,
        rebuild=dataclasses.replace(cfg.rebuild, backend=backend),
        output=dataclasses.replace(cfg.output, output_dir=tmp_path / "out"),
    )
    return cfg


@needs_typst
async def test_typst_backend_produces_pdf_via_orchestrator(patched_upstream, tmp_path, monkeypatch):
    store = _FakeStore()
    input_path = _blank_pdf(tmp_path / "in.pdf")
    output_path = tmp_path / "remediated.pdf"

    # Acceptance always passes for this dispatch test — acceptance itself is
    # covered by its own suite; here we test routing + struct-assert wiring.
    monkeypatch.setattr(
        engine_service, "evaluate_pdf_acceptance",
        lambda *a, **k: SimpleNamespace(passed=True, warning_reasons=[]),
    )

    await engine_service._rebuild_from_semantics(
        input_path, output_path, _cfg(tmp_path, backend="typst"),
        _job(input_path), store, SimpleNamespace(job_dir=tmp_path / "jobs"),
    )
    final = store.last()
    assert final.get("status") == "done", store.updates
    assert output_path.exists() and output_path.read_bytes().startswith(b"%PDF")


async def test_unknown_backend_fails_closed(patched_upstream, tmp_path):
    store = _FakeStore()
    input_path = _blank_pdf(tmp_path / "in.pdf")
    await engine_service._rebuild_from_semantics(
        input_path, tmp_path / "out.pdf", _cfg(tmp_path, backend="nonsense"),
        _job(input_path), store, SimpleNamespace(),
    )
    final = store.last()
    assert final.get("status") == "failed"
    assert "unknown backend" in final.get("error", "")


async def test_acroform_source_routed_away(patched_upstream, tmp_path):
    store = _FakeStore()
    form_path = tmp_path / "form.pdf"
    with pikepdf.new() as pdf:
        pdf.add_blank_page()
        f = pdf.make_indirect(pikepdf.Dictionary(FT=pikepdf.Name("/Tx"), T=pikepdf.String("x")))
        pdf.Root.AcroForm = pdf.make_indirect(pikepdf.Dictionary(Fields=pikepdf.Array([f])))
        pdf.save(form_path)
    await engine_service._rebuild_from_semantics(
        form_path, tmp_path / "out.pdf", _cfg(tmp_path, backend="typst"),
        _job(form_path), store, SimpleNamespace(),
    )
    final = store.last()
    assert final.get("status") == "failed"
    assert final.get("error") == "rebuild_acroform_present"


async def test_job_override_beats_config(patched_upstream, tmp_path, monkeypatch):
    """metadata rebuild_backend overrides cfg.rebuild.backend (FR-2)."""
    captured = {}

    class _SpyRenderer:
        def __init__(self, *a, **k): ...
        async def render(self, request):
            captured["backend"] = "typst"
            raise engine_service._TypstCompileError("spy stop")

    monkeypatch.setattr(engine_service, "_TypstRenderer", _SpyRenderer)
    monkeypatch.setattr(engine_service, "_resolve_typst_binary", lambda: Path("/usr/bin/true"))
    store = _FakeStore()
    input_path = _blank_pdf(tmp_path / "in.pdf")
    await engine_service._rebuild_from_semantics(
        input_path, tmp_path / "out.pdf", _cfg(tmp_path, backend="questpdf"),
        _job(input_path, {"rebuild_backend": "typst"}), store, SimpleNamespace(),
        backend_override="typst",
    )
    final = store.last()
    assert captured.get("backend") == "typst"
    assert final.get("status") == "failed"
    assert final.get("error", "").startswith("rebuild_typst_compile_failed:")
