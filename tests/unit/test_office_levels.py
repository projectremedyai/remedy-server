"""Office L0-L4 gates (PRD §3.1) + the AC3 never-L5 invariant."""

from __future__ import annotations

import random
from pathlib import Path

from project_remedy.levels import LevelResult
from project_remedy.models import FileType
from project_remedy.office_acceptance import (
    OfficeAcceptanceResult,
    OfficeCheckReport,
    OfficeCheckResult,
    OfficePackageResult,
    OfficeScreenReaderResult,
    evaluate_office_acceptance,
)
from project_remedy.office_levels import (
    OFFICE_PROFILE_NAME,
    OfficeStructureProbe,
    classify_level,
    probe_office_structure,
)
from project_remedy.quality_judges.shared.base import QualityResult
from tests.unit.office_fixtures import make_docx


def _acceptance(*, openable=True, statuses=(), quality=None) -> OfficeAcceptanceResult:
    path = Path("synthetic.docx")
    results = [
        OfficeCheckResult(rule_id=f"r{i}", description="d", status=s)
        for i, s in enumerate(statuses)
    ]
    return OfficeAcceptanceResult(
        file_path=path,
        file_type=FileType.DOCX,
        checker_report=OfficeCheckReport(file_path=path, file_type=FileType.DOCX, results=results),
        screen_reader_result=OfficeScreenReaderResult(file_path=path, file_type=FileType.DOCX, issues=[]),
        package_result=OfficePackageResult(checked=True, passed=openable),
        quality_result=quality,
    )


def _probe(**overrides) -> OfficeStructureProbe:
    base = dict(has_text=True, has_heading_structure=True, has_table_header_marks=False,
                has_alt_text_signal=False, paragraph_count=3, table_count=0, image_count=0)
    base.update(overrides)
    return OfficeStructureProbe(**base)


def test_gate_ladder():
    assert classify_level(None, _probe()).level == "L0"
    assert classify_level(_acceptance(openable=False), _probe()).level == "L0"
    assert classify_level(_acceptance(), _probe(has_text=False)).level == "L0"
    r = classify_level(_acceptance(), _probe(has_heading_structure=False))
    assert r.level == "L1" and r.blocking_conditions == ["no_structural_signal"]
    assert classify_level(_acceptance(statuses=["Failed"]), _probe()).level == "L2"
    r = classify_level(_acceptance(statuses=["Passed"]), _probe())
    assert r.level == "L3" and "quality_layer_not_run" in r.blocking_conditions
    bad_q = QualityResult(format="docx", overall_pass=False)
    r = classify_level(_acceptance(statuses=["Passed"], quality=bad_q), _probe())
    assert r.level == "L3"
    assert "quality_failed" in r.blocking_conditions
    good_q = QualityResult(format="docx", overall_pass=True)
    r = classify_level(_acceptance(statuses=["Passed"], quality=good_q), _probe())
    assert r.level == "L4" and r.profile == OFFICE_PROFILE_NAME
    assert isinstance(r, LevelResult)  # FR3: shared dataclass, not a fork


def test_structural_signal_via_table_or_alt_alone():
    r = classify_level(_acceptance(statuses=["Failed"]),
                       _probe(has_heading_structure=False, has_table_header_marks=True))
    assert r.level == "L2"  # table header marks alone are a structural signal
    r = classify_level(_acceptance(statuses=["Failed"]),
                       _probe(has_heading_structure=False, has_alt_text_signal=True))
    assert r.level == "L2"  # alt text alone is a structural signal


def test_probe_detects_accessibility_fallback_heading_styles(tmp_path):
    # remediator fallback styles have IDs like "AccessibilityTitle" — the probe
    # must count them as heading structure just like the checker rule does
    from docx import Document
    from docx.enum.style import WD_STYLE_TYPE

    path = tmp_path / "fallback.docx"
    doc = Document()
    style = doc.styles.add_style("Accessibility Title", WD_STYLE_TYPE.PARAGRAPH)
    doc.add_paragraph("Fallback heading", style=style)
    doc.save(str(path))
    probe = probe_office_structure(path, FileType.DOCX)
    assert probe.has_heading_structure


def test_manual_check_routes_to_needs_human_not_blocking():
    r = classify_level(_acceptance(statuses=["Passed", "Manual Check Needed"]), _probe())
    assert r.level == "L3"
    assert "r1" in r.needs_human


def test_never_l5_fuzz():
    """AC3: randomized inputs can never produce L5 (seeded — deterministic)."""
    rng = random.Random(20260701)
    statuses = ["Passed", "Failed", "Manual Check Needed"]
    for _ in range(500):
        acceptance = None
        if rng.random() > 0.1:
            quality = None
            if rng.random() > 0.5:
                quality = QualityResult(format="docx", overall_pass=rng.random() > 0.5)
            acceptance = _acceptance(
                openable=rng.random() > 0.2,
                statuses=[rng.choice(statuses) for _ in range(rng.randrange(0, 6))],
                quality=quality,
            )
        probe = _probe(
            has_text=rng.random() > 0.3,
            has_heading_structure=rng.random() > 0.5,
            has_table_header_marks=rng.random() > 0.5,
            has_alt_text_signal=rng.random() > 0.5,
        )
        result = classify_level(acceptance, probe)
        assert result.level in {"L0", "L1", "L2", "L3", "L4"}


def test_probe_reads_real_docx(tmp_path):
    rich = make_docx(tmp_path / "rich.docx", headings=[("T", 0)], tables=1,
                     inline_images=1, image_alt="chart",
                     body_paragraphs=["Some body text."])
    probe = probe_office_structure(rich, FileType.DOCX)
    assert probe.has_text and probe.has_heading_structure
    assert probe.has_table_header_marks and probe.has_alt_text_signal
    assert probe.table_count == 1 and probe.image_count == 1

    empty = make_docx(tmp_path / "empty.docx")
    probe = probe_office_structure(empty, FileType.DOCX)
    assert not probe.has_text and not probe.has_heading_structure

    broken = tmp_path / "broken.docx"
    broken.write_bytes(b"not a zip at all")
    probe = probe_office_structure(broken, FileType.DOCX)  # must never raise
    assert probe == OfficeStructureProbe(False, False, False, False, 0, 0, 0)


def test_end_to_end_classification(tmp_path):
    path = make_docx(tmp_path / "l2.docx", title="T", headings=[("T", 0)],
                     inline_images=1, image_alt=None)  # structure present, alt fails
    acceptance = evaluate_office_acceptance(path)
    result = classify_level(acceptance, probe_office_structure(path, FileType.DOCX))
    assert result.level == "L2"
    assert "docx-alt-text" in result.blocking_conditions


# --- Branch-coverage gap closers (AC1) --------------------------------------

import pytest


def test_probe_rejects_non_docx():
    with pytest.raises(ValueError, match="DOCX only"):
        probe_office_structure(Path("x.pptx"), FileType.PPTX)


def test_probe_counts_missing_alt_and_non_heading_styled_paragraph(tmp_path):
    path = make_docx(
        tmp_path / "probe_mix.docx",
        inline_images=1,
        image_alt=None,
        real_list_items=["item"],
    )
    probe = probe_office_structure(path, FileType.DOCX)
    assert probe.image_count == 1
    assert probe.has_alt_text_signal is False


def test_probe_skips_drawing_container_without_docpr(tmp_path):
    """A wp:inline with no docPr child at all (malformed but well-formed XML)
    must not be counted as an image — exercises the `doc_pr is None: continue`
    branch in probe_office_structure's image-scan loop.
    """
    import re
    import shutil
    import zipfile

    path = make_docx(tmp_path / "probe_nodocpr.docx", inline_images=1, image_alt="alt text")
    tmp = path.with_suffix(".tmp.docx")
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "word/document.xml":
                text = data.decode("utf-8")
                text = re.sub(r"<wp:docPr\b[^>]*/>", "", text, count=1)
                data = text.encode("utf-8")
            dst.writestr(item, data)
    shutil.move(str(tmp), str(path))

    probe = probe_office_structure(path, FileType.DOCX)
    assert probe.image_count == 0
    assert probe.has_alt_text_signal is False
