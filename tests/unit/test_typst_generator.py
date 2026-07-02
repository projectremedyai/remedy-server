"""FR-7/8/9: pure AST->Typst generation. Escape set pinned by the spike
(docs/typst_backend_decisions.md): \\ # $ * _ @ [ ] < > ~ ` and / (as \\/);
smartquotes disabled in the preamble."""

from __future__ import annotations

import pytest

from project_remedy.rebuild.ast import HeadingBlock, ParagraphBlock, Run
from project_remedy.rebuild.typst_generator import escape_markup, escape_string, generate
from tests.unit.rebuild_fixtures import make_request


NASTY = 'Chars: \\ # $ * _ @ [ ] < > ~ ` / // = + - " quotes'


def test_escape_markup_neutralizes_all_special_chars():
    out = escape_markup(NASTY)
    for ch in ("\\\\", "\\#", "\\$", "\\*", "\\_", "\\@", "\\[", "\\]", "\\<", "\\>", "\\~", "\\`", "\\/"):
        assert ch in out
    assert "//" not in out.replace("\\/", "")  # no live comment marker survives


def test_escape_string_for_code_context():
    assert escape_string('say "hi" \\ done') == 'say \\"hi\\" \\\\ done'


def test_preamble_sets_language_title_page_and_smartquote(tmp_path):
    src = generate(make_request(asset_dir=tmp_path, content=[], assets={}), asset_paths={})
    assert '#set text(lang: "en-US")' in src or '#set text(lang: "en")' in src
    assert '#set document(title: "Sample Form")' in src
    assert '#set smartquote(enabled: false)' in src
    assert 'paper: "us-letter"' in src
    assert "margin:" in src and "0.75in" in src


def test_heading_and_paragraph_emit_semantic_markup(tmp_path):
    request = make_request(
        asset_dir=tmp_path,
        content=[
            HeadingBlock(level=2, runs=[Run(text="Section "), Run(text="Two", bold=True)]),
            ParagraphBlock(runs=[Run(text="Plain "), Run(text="ital", italic=True), Run(text=" end.")]),
        ],
        assets={},
    )
    src = generate(request, asset_paths={})
    assert "== Section *Two*" in src
    assert "Plain _ital_ end." in src
    # FR-9: no styled-text heading fallback anywhere
    assert "#text(" not in src
