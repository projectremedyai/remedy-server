"""CLI tests: exit codes, JSON output, FR8 fail-closed legacy guard."""

from __future__ import annotations

import json

from click.testing import CliRunner

from project_remedy.cli_office import office_group
from tests.unit.office_fixtures import make_docx, make_fake_ole2


def test_check_passes_on_good_docx(tmp_path):
    path = make_docx(
        tmp_path / "good.docx", title="Good", language="en-US",
        headings=[("Good", 0)],
        body_paragraphs=["Body text following the title paragraph."],
    )
    result = CliRunner().invoke(office_group, ["check", str(path), "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["passed"] is True
    assert len(payload["checks"]) == 12


def test_check_fails_on_bad_docx(tmp_path):
    path = make_docx(tmp_path / "bad.docx")
    result = CliRunner().invoke(office_group, ["check", str(path), "--json"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["passed"] is False
    assert "docx-title" in payload["failed_rule_ids"]


def test_classify_level_outputs_level(tmp_path):
    path = make_docx(tmp_path / "doc.docx", title="T", headings=[("T", 0)])
    result = CliRunner().invoke(office_group, ["classify-level", str(path)])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["level"] in {"L0", "L1", "L2", "L3", "L4"}
    assert payload["profile"] == "LACCD-DistrictUA1-Office"


def test_fr8_legacy_ole2_fails_closed(tmp_path):
    doc = make_fake_ole2(tmp_path / "legacy.doc")
    result = CliRunner().invoke(office_group, ["check", str(doc)])
    assert result.exit_code != 0
    assert "OOXML conversion" in result.output

    # OLE2 bytes hiding behind a modern suffix must also fail closed
    disguised = tmp_path / "disguised.docx"
    disguised.write_bytes(doc.read_bytes())
    result = CliRunner().invoke(office_group, ["check", str(disguised)])
    assert result.exit_code != 0
    assert "OOXML conversion" in result.output


def test_non_zip_garbage_fails_closed(tmp_path):
    junk = tmp_path / "junk.docx"
    junk.write_bytes(b"\x00\x01\x02 not a package")
    result = CliRunner().invoke(office_group, ["check", str(junk)])
    assert result.exit_code != 0
