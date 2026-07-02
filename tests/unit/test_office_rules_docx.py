"""Rule-catalog invariants + per-rule Pass/Fail tests (grows through Task 8)."""

from __future__ import annotations

from project_remedy.office_rules import NS, RULE_CATALOG, RULE_SPECS_BY_ID, RuleSpec


def test_catalog_has_twelve_docx_rules_with_unique_ids():
    docx = [s for s in RULE_CATALOG if s.format == "docx"]
    assert len(docx) == 12
    assert len({s.rule_id for s in docx}) == 12
    assert len({s.emitted_id for s in docx}) == 12


def test_every_rule_declares_xml_refs_and_wcag_ref():
    for spec in RULE_CATALOG:
        assert spec.rule_id.startswith("OOXML-DOCX-")
        assert spec.xml_refs, spec.rule_id           # FR2: self-documenting XML refs
        assert spec.wcag_ref, spec.rule_id
        assert spec.flag_status in ("Failed", "Manual Check Needed")


def test_legacy_alias_mapping_is_exact():
    aliases = {s.rule_id: s.emitted_id for s in RULE_CATALOG if s.emitted_id != s.rule_id}
    assert aliases == {
        "OOXML-DOCX-1.1": "docx-title",
        "OOXML-DOCX-1.2": "docx-language",
        "OOXML-DOCX-2.1": "docx-headings",
        "OOXML-DOCX-3.1": "docx-alt-text",
        "OOXML-DOCX-4.1": "docx-table-headers",
    }


def test_pattern_rules_route_to_manual_check_needed():
    assert RULE_SPECS_BY_ID["OOXML-DOCX-5.1"].flag_status == "Manual Check Needed"
    assert RULE_SPECS_BY_ID["OOXML-DOCX-7.1"].flag_status == "Manual Check Needed"


def test_new_rules_without_remediator_support_ship_fixable_false():
    # FR6: never fixable=True without a real office_remediator code path
    for rule_id in ("OOXML-DOCX-2.2", "OOXML-DOCX-2.3", "OOXML-DOCX-3.2",
                    "OOXML-DOCX-4.2", "OOXML-DOCX-5.1", "OOXML-DOCX-6.1", "OOXML-DOCX-7.1"):
        assert RULE_SPECS_BY_ID[rule_id].fixable is False, rule_id


def test_namespace_map_covers_wordprocessing_drawing():
    assert set(NS) >= {"w", "wp", "a", "r"}


from pathlib import Path

from project_remedy.office_checker import DOCX_RULES, DocxContext
from tests.unit.office_fixtures import make_docx


def _run(rule_id: str, path: Path):
    return DOCX_RULES[rule_id](DocxContext.load(path))


def test_rule_1_1_title(tmp_path):
    good = make_docx(tmp_path / "g.docx", title="Has Title")
    bad = make_docx(tmp_path / "b.docx", title="")
    ok = _run("OOXML-DOCX-1.1", good)
    fail = _run("OOXML-DOCX-1.1", bad)
    assert ok.status == "Passed" and ok.rule_id == "docx-title"
    assert fail.status == "Failed" and fail.fixable is True
    assert fail.checkpoint == "Document metadata" and fail.wcag_ref == "2.4.2"


def test_rule_1_2_language(tmp_path):
    good = make_docx(tmp_path / "g.docx", language="en-US")
    bad = make_docx(tmp_path / "b.docx", language="")
    assert _run("OOXML-DOCX-1.2", good).status == "Passed"
    result = _run("OOXML-DOCX-1.2", bad)
    assert result.status == "Failed" and result.rule_id == "docx-language"
