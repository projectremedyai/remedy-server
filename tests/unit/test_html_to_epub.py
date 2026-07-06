from __future__ import annotations

import shutil
import zipfile
from pathlib import Path

import pytest
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.app.config import Settings
from backend.app.jobs import Job, JobStore, JobWorker
from backend.app.routes import build_router
from project_remedy.epub_verifier import (
    EPUBValidationResult,
    EPUBVerifyReport,
    _extract_ace_violations,
    verify_epub,
)
from project_remedy.html_to_epub import EPUB_A11Y_CONFORMS_TO, HTMLToEPUBConverter


_PIXEL_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
)

ACCESSIBLE_HTML = f"""\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Accessible Form</title></head>
<body>
  <main>
    <section aria-labelledby="overview">
      <h1 id="overview">Overview</h1>
      <p>This is the overview section.</p>
      <img src="data:image/png;base64,{_PIXEL_PNG_B64}" alt="District logo" />
    </section>
    <section aria-labelledby="instructions">
      <h2 id="instructions">Instructions</h2>
      <ol>
        <li>Read the policy.</li>
        <li>Sign the form.</li>
      </ol>
    </section>
  </main>
</body>
</html>
"""

TEXT_ONLY_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Text only</title></head>
<body>
  <main>
    <section><h1>Title</h1><p>Plain text body.</p></section>
  </main>
</body>
</html>
"""

PARTIAL_IMAGE_ALT_HTML = f"""\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Partial alt</title></head>
<body>
  <main>
    <section>
      <h1>Images</h1>
      <img src="data:image/png;base64,{_PIXEL_PNG_B64}" alt="District logo" />
      <img src="data:image/png;base64,{_PIXEL_PNG_B64}" />
    </section>
  </main>
</body>
</html>
"""

NESTED_SECTION_HTML = """\
<!DOCTYPE html>
<html lang="en">
<head><meta charset="utf-8"><title>Nested sections</title></head>
<body>
  <main>
    <article>
      <section>
        <h1>Parent</h1>
        <p>Parent body.</p>
        <section>
          <h2>Child</h2>
          <p>Child body.</p>
        </section>
      </section>
    </article>
  </main>
</body>
</html>
"""


def _read_opf(epub_path: Path) -> str:
    with zipfile.ZipFile(epub_path) as zf:
        opf_names = [name for name in zf.namelist() if name.endswith(".opf")]
        assert opf_names
        return zf.read(opf_names[0]).decode("utf-8")


def _list_epub(epub_path: Path) -> list[str]:
    with zipfile.ZipFile(epub_path) as zf:
        return zf.namelist()


async def _noop_runner(job: Job) -> None:
    return None


def test_router_exposes_html_to_epub_endpoint(tmp_path: Path) -> None:
    settings = Settings(
        job_store_path=tmp_path / "jobs.db",
        job_dir=tmp_path / "jobs",
    )
    store = JobStore(settings.job_store_path)
    worker = JobWorker(store, _noop_runner)
    router = build_router(
        settings,
        store,
        worker,
        Limiter(key_func=get_remote_address),
        "10/minute",
    )

    routes = [
        (route.path, getattr(route, "methods", set()))
        for route in router.routes
    ]
    assert any(
        path == "/v1/convert/html-to-epub" and "POST" in methods
        for path, methods in routes
    )


async def test_html_to_epub_emits_epub_accessibility_metadata(tmp_path: Path) -> None:
    out = tmp_path / "out.epub"
    converter = HTMLToEPUBConverter(max_concurrent=1)
    await converter.start()
    try:
        result = await converter.convert(ACCESSIBLE_HTML, out, title="Accessible Form")
    finally:
        await converter.close()

    assert result.success, result.error_message
    assert result.chapters == 2

    opf = _read_opf(out)
    assert 'property="dcterms:conformsTo"' in opf
    assert EPUB_A11Y_CONFORMS_TO in opf
    assert 'property="schema:accessMode"' in opf
    assert 'property="schema:accessModeSufficient"' in opf
    assert 'property="schema:accessibilityFeature"' in opf
    assert 'property="schema:accessibilityHazard"' in opf
    assert ">alternativeText<" in opf
    assert ">structuralNavigation<" in opf

    names = _list_epub(out)
    assert any(name.endswith("nav.xhtml") for name in names)
    assert any(name.endswith("toc.ncx") for name in names)


async def test_html_to_epub_access_modes_reflect_source_encoding(tmp_path: Path) -> None:
    text_epub = tmp_path / "text.epub"
    visual_epub = tmp_path / "visual.epub"
    converter = HTMLToEPUBConverter(max_concurrent=1)
    await converter.start()
    try:
        text_result = await converter.convert(TEXT_ONLY_HTML, text_epub, title="Text")
        visual_result = await converter.convert(ACCESSIBLE_HTML, visual_epub, title="Visual")
    finally:
        await converter.close()

    assert text_result.success, text_result.error_message
    assert visual_result.success, visual_result.error_message

    text_opf = _read_opf(text_epub)
    visual_opf = _read_opf(visual_epub)

    assert ">textual<" in text_opf
    assert ">visual<" not in text_opf
    assert ">alternativeText<" not in text_opf

    assert ">textual<" in visual_opf
    assert ">visual<" in visual_opf
    assert ">alternativeText<" in visual_opf


async def test_html_to_epub_does_not_overstate_partial_alt_coverage(tmp_path: Path) -> None:
    epub_path = tmp_path / "partial-alt.epub"
    converter = HTMLToEPUBConverter(max_concurrent=1)
    await converter.start()
    try:
        result = await converter.convert(PARTIAL_IMAGE_ALT_HTML, epub_path, title="Partial alt")
    finally:
        await converter.close()

    assert result.success, result.error_message
    assert ">alternativeText<" not in _read_opf(epub_path)


async def test_html_to_epub_uses_explicit_language_when_provided(tmp_path: Path) -> None:
    epub_path = tmp_path / "language.epub"
    converter = HTMLToEPUBConverter(max_concurrent=1)
    await converter.start()
    try:
        result = await converter.convert(TEXT_ONLY_HTML, epub_path, title="Text", language="fr")
    finally:
        await converter.close()

    assert result.success, result.error_message
    assert "<dc:language>fr</dc:language>" in _read_opf(epub_path)


async def test_html_to_epub_does_not_duplicate_nested_sections(tmp_path: Path) -> None:
    epub_path = tmp_path / "nested.epub"
    converter = HTMLToEPUBConverter(max_concurrent=1)
    await converter.start()
    try:
        result = await converter.convert(NESTED_SECTION_HTML, epub_path, title="Nested sections")
    finally:
        await converter.close()

    assert result.success, result.error_message
    assert result.chapters == 1
    chapter_files = [
        name for name in _list_epub(epub_path)
        if name.startswith("OEBPS/chap_") and name.endswith(".xhtml")
    ]
    assert chapter_files == ["OEBPS/chap_1.xhtml"]


async def test_epub_verifier_fails_soft_when_validators_are_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EPUBCHECK_PATH", raising=False)
    monkeypatch.delenv("ACE_PATH", raising=False)
    monkeypatch.setattr("project_remedy.epub_verifier.shutil.which", lambda _name: None)

    epub_path = tmp_path / "out.epub"
    converter = HTMLToEPUBConverter(max_concurrent=1)
    await converter.start()
    try:
        result = await converter.convert(ACCESSIBLE_HTML, epub_path, title="Accessible Form")
    finally:
        await converter.close()

    assert result.success, result.error_message
    report = await verify_epub(epub_path)

    assert isinstance(report, EPUBVerifyReport)
    assert report.epubcheck.skipped
    assert report.ace.skipped
    assert not report.passed

    body = report.to_dict()
    assert body["epubcheck"]["skipped"] is True
    assert body["ace"]["skipped"] is True
    assert "manual SMART-style review" in body["caveat"]


def test_epub_verify_report_requires_both_validators_to_run(tmp_path: Path) -> None:
    report = EPUBVerifyReport(
        epub_path=tmp_path / "out.epub",
        epubcheck=EPUBValidationResult(tool="epubcheck", passed=True, skipped=True),
        ace=EPUBValidationResult(tool="ace", passed=True, skipped=False),
    )

    assert not report.passed


def test_ace_violation_extraction_accepts_cfi_list() -> None:
    payload = {
        "assertions": [
            {
                "earl:testSubject": {"url": "chap_1.xhtml"},
                "assertions": [
                    {
                        "earl:result": {
                            "earl:outcome": "earl:failed",
                            "earl:pointer": {"cfi": ["/4/2", "/4/4"]},
                        },
                        "earl:test": {
                            "dct:title": "image-alt",
                            "earl:impact": "serious",
                            "dct:description": "Image missing alt text",
                        },
                    }
                ],
            }
        ]
    }

    violations = _extract_ace_violations(payload)

    assert violations[0]["location"] == "chap_1.xhtml /4/2"


@pytest.mark.skipif(
    shutil.which("epubcheck") is None or shutil.which("ace") is None,
    reason="epubcheck and ace are not both installed",
)
async def test_epubcheck_and_ace_pass_on_text_fixture(tmp_path: Path) -> None:
    epub_path = tmp_path / "out.epub"
    converter = HTMLToEPUBConverter(max_concurrent=1)
    await converter.start()
    try:
        result = await converter.convert(TEXT_ONLY_HTML, epub_path, title="Text")
    finally:
        await converter.close()

    assert result.success, result.error_message
    report = await verify_epub(epub_path)

    assert not report.epubcheck.skipped
    assert not report.ace.skipped
    assert report.passed
