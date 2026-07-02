"""FR-7/8/9: pure AST->Typst generation. Escape set pinned by the spike
(docs/typst_backend_decisions.md): \\ # $ * _ @ [ ] < > ~ ` and / (as \\/);
smartquotes disabled in the preamble."""

from __future__ import annotations

import pytest

from project_remedy.rebuild.ast import (
    ArtifactBlock,
    AssetRef,
    FigureBlock,
    HeadingBlock,
    ListBlock,
    ListItem,
    ParagraphBlock,
    Run,
    SimpleTableBlock,
    TableCell,
    TableRow,
)
from project_remedy.rebuild.typst_generator import GeneratorError, escape_markup, escape_string, generate
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
    # content=[] avoids firing Task-4/5 stubs
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


def _para(text: str) -> ParagraphBlock:
    return ParagraphBlock(runs=[Run(text=text)])


def test_list_emits_markup_never_label_prose(tmp_path):
    block = ListBlock(
        ordered=True,
        items=[
            ListItem(label_runs=[Run(text="1.")], body=[_para("alpha")]),
            ListItem(label_runs=[Run(text="2.")], body=[_para("beta")]),
        ],
    )
    request = make_request(asset_dir=tmp_path, content=[block], assets={})
    src = generate(request, asset_paths={})
    assert "+ alpha" in src and "+ beta" in src
    assert "1." not in src and "2." not in src  # Caveat-2: labels never leak as prose


def test_unordered_and_nested_lists(tmp_path):
    inner = ListBlock(ordered=False, items=[ListItem(label_runs=[], body=[_para("sub")])])
    block = ListBlock(
        ordered=False,
        items=[ListItem(label_runs=[Run(text="•")], body=[_para("outer"), inner])],
    )
    request = make_request(asset_dir=tmp_path, content=[block], assets={})
    src = generate(request, asset_paths={})
    assert "- outer" in src
    assert "  - sub" in src  # nested item indented under its parent


def test_table_header_row_and_row_header_degradation(tmp_path):
    block = SimpleTableBlock(
        rows=[
            TableRow(cells=[TableCell(text="Name", header="col"), TableCell(text="Age", header="both")]),
            TableRow(cells=[TableCell(text="Alice", header="row"), TableCell(text="30")]),
        ]
    )
    request = make_request(asset_dir=tmp_path, content=[block], assets={})
    src = generate(request, asset_paths={})
    assert "#table(" in src and "columns: 2" in src
    assert 'table.header([Name], [Age])' in src
    assert "[Alice], [30]" in src  # row-header cell degrades to a plain cell (spike decision)


def test_table_cell_text_is_escaped(tmp_path):
    block = SimpleTableBlock(rows=[TableRow(cells=[TableCell(text="a#b"), TableCell(text="c[d]")])])
    request = make_request(asset_dir=tmp_path, content=[block], assets={})
    src = generate(request, asset_paths={})
    assert "[a\\#b], [c\\[d\\]]" in src


def _fig_request(tmp_path, block):
    return make_request(
        asset_dir=tmp_path,
        content=[block],
        assets={"img-1": AssetRef(path=str(tmp_path / "img-1.png"), mime="image/png")},
    )


def test_figure_with_caption_and_alt(tmp_path):
    block = FigureBlock(asset_ref="img-1", alt='A "quoted" dot', caption=[Run(text="Cap")])
    src = generate(_fig_request(tmp_path, block), asset_paths={"img-1": "img-1.png"})
    assert '#figure(image("img-1.png", alt: "A \\"quoted\\" dot"), caption: [Cap])' in src


def test_figure_without_caption_is_bare_image(tmp_path):
    block = FigureBlock(asset_ref="img-1", alt="Bare")
    src = generate(_fig_request(tmp_path, block), asset_paths={"img-1": "img-1.png"})
    assert '#image("img-1.png", alt: "Bare")' in src
    assert "#figure(" not in src


def test_artifact_wraps_in_pdf_artifact(tmp_path):
    block = ArtifactBlock(asset_ref="img-1")
    src = generate(_fig_request(tmp_path, block), asset_paths={"img-1": "img-1.png"})
    assert '#pdf.artifact[#image("img-1.png")]' in src
    assert "alt:" not in src


def test_missing_asset_path_raises(tmp_path):
    block = FigureBlock(asset_ref="img-1", alt="x")
    with pytest.raises(GeneratorError, match="img-1"):
        generate(_fig_request(tmp_path, block), asset_paths={})


def test_ragged_table_rows_are_padded(tmp_path):
    block = SimpleTableBlock(
        rows=[
            TableRow(cells=[TableCell(text="A"), TableCell(text="B"), TableCell(text="C")]),
            TableRow(cells=[TableCell(text="only-one")]),
        ]
    )
    request = make_request(asset_dir=tmp_path, content=[block], assets={})
    src = generate(request, asset_paths={})
    assert "columns: 3" in src
    assert "[only-one], [], []," in src  # short row padded — no cell-stream shift


def test_whitespace_alt_raises_at_generation_time(tmp_path):
    from project_remedy.rebuild.ast import AssetRef, FigureBlock

    block = FigureBlock(asset_ref="img-1", alt=" ")  # min_length=1 passes; generator must still refuse
    request = make_request(
        asset_dir=tmp_path,
        content=[block],
        assets={"img-1": AssetRef(path=str(tmp_path / "img-1.png"), mime="image/png")},
    )
    with pytest.raises(GeneratorError, match="empty alt"):
        generate(request, asset_paths={"img-1": "img-1.png"})


def test_lang_region_handles_bcp47_variants(tmp_path):
    for language, expected_lang, expected_region in [
        ("en-US", '"en"', '"US"'),
        ("en", '"en"', None),
        ("zh-Hans-CN", '"zh"', '"CN"'),
        ("zh-Hans", '"zh"', None),
    ]:
        src = generate(make_request(asset_dir=tmp_path, content=[], assets={}, language=language), asset_paths={})
        assert f"#set text(lang: {expected_lang})" in src
        if expected_region:
            assert f"#set text(region: {expected_region})" in src
        else:
            assert "region:" not in src
