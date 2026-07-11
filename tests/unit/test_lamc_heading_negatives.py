"""LAMC-domain heading training data: delivered-arbitrated hard negatives.

heading-v1 was trained only on heading-rich synthetic/gov pages (50/50
pass/fail) — zero LAMC pages — so on sparse scanned forms it hallucinates
"title should be H1" flags that differ run-to-run. Delivered (human-certified)
files arbitrate the model's own production flags: no heading in delivered on a
flagged page -> the flag was a false positive -> a pass record on the EXACT
production prompt/image; delivered has the heading -> a true-fail record whose
target retags to the human's level. These tests drive the pure logic of
tools/finetune/build_lamc_heading_negatives.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "finetune"))

import build_lamc_heading_negatives as B  # noqa: E402


ORDER_TEXT = (
    '  1. /P  (text: "LOS ANGELES MISSION COLLEGE")\n'
    '  2.   /P  (text: "Annual Program Review")\n'
    '  3.   /TD  (text: "cell value")\n'
    "  4.   /Figure  (alt: \"Campus logo\")"
)


def test_parse_structure_order_lines():
    lines = B.parse_structure_order_lines(ORDER_TEXT)
    assert [(l.index, l.tag) for l in lines] == [(1, "P"), (2, "P"), (3, "TD"), (4, "Figure")]
    assert lines[1].text == "Annual Program Review"
    assert lines[3].text == ""  # alt-only line carries no visible text


def test_parse_structure_order_handles_empty_marker():
    assert B.parse_structure_order_lines("(no structure elements found on this page)") == []
    assert B.parse_structure_order_lines("(invalid page number)") == []


def test_build_fail_target_with_index_match():
    """Delivered heading text matches a numbered line -> indexed finding."""
    lines = B.parse_structure_order_lines(ORDER_TEXT)
    target = B.build_fail_target(lines, [("H1", "Annual Program Review")])
    assert target["status"] == "fail"
    (f,) = target["findings"]
    assert f["element_index"] == 2
    assert f["current_tag"] == "P"
    assert f["correct_tag"] == "H1"
    assert f["visible_text"] == "Annual Program Review"
    assert f["severity"] == "error"
    assert f["suggested_fix"] == "Retag as H1"


def test_build_fail_target_without_index_match():
    """Delivered heading not in the (ActualText-only) list -> index omitted,
    visible_text carries identity — the schema's 'missing heading' form."""
    lines = B.parse_structure_order_lines(ORDER_TEXT)
    target = B.build_fail_target(lines, [("H2", "Enrollment Summary")])
    (f,) = target["findings"]
    assert "element_index" not in f or f["element_index"] is None
    assert f["visible_text"] == "Enrollment Summary"
    assert f["correct_tag"] == "H2"


def test_build_fail_target_never_indexes_unsafe_tags():
    """A TD line matching the heading text must not be indexed (a table cell
    is never the heading node) — fall back to the index-omitted form."""
    lines = B.parse_structure_order_lines('  1. /TD  (text: "Annual Report")')
    target = B.build_fail_target(lines, [("H1", "Annual Report")])
    (f,) = target["findings"]
    assert f.get("element_index") is None or "element_index" not in f
    assert f["correct_tag"] == "H1"


def test_pass_target_shape():
    assert B.PASS_TARGET == {"status": "pass", "findings": []}


def test_to_conversation_matches_corpus_format(tmp_path):
    rec = {
        "image": "renders/x.png",
        "prompt": "PROMPT TEXT",
        "target": B.PASS_TARGET,
        "doc_id": "lamc_abc",
        "page": 3,
        "variant": "lamc_false_flag_pass",
    }
    conv = B.to_conversation(rec)
    assert conv["messages"][0]["role"] == "user"
    assert conv["messages"][0]["content"][0] == {"type": "image", "image": "renders/x.png"}
    assert conv["messages"][0]["content"][1]["text"] == "PROMPT TEXT"
    assert json.loads(conv["messages"][1]["content"][0]["text"]) == B.PASS_TARGET
    meta = conv["meta"]
    assert meta["task"] == "heading_hierarchy"
    assert meta["source_family"] == "lamc"
    assert meta["variant"] == "lamc_false_flag_pass"


def test_flagged_pages_from_cohort_row():
    row = {
        "checker_failures": [
            {"rule_id": "headings-nesting", "details": [
                "Page 2: title/section heading is tagged as body text (? -> H1) (Retag as H1)",
                "First heading is H2, expected H1",
            ]},
            {"rule_id": "page-char-encoding", "details": ["Page 9: junk"]},
        ]
    }
    assert B.flagged_pages(row) == [2]
