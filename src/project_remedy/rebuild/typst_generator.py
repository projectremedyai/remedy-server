"""AST -> Typst source generator (PRD_typst_backend.md §5.3).

Pure function of the RebuildRequest (FR-7). Emits ONLY real Typst semantic
constructs — heading markup, list markup, table()/figure()/image() calls —
never manually-styled text (FR-9, the Caveat-2 mitigation). Escape set and
construct mappings are pinned empirically in docs/typst_backend_decisions.md.
"""

from __future__ import annotations

from project_remedy.rebuild.ast import (
    ArtifactBlock,
    FigureBlock,
    HeadingBlock,
    ListBlock,
    ParagraphBlock,
    RebuildRequest,
    Run,
    SimpleTableBlock,
)


class GeneratorError(ValueError):
    """The AST contains a value the generator refuses to render silently."""


# Spike-pinned markup-active characters. '/' is escaped so '//' can never
# become a line comment; smartquote is disabled in the preamble instead of
# escaping quotes.
_MARKUP_SPECIALS = "\\#$*_@[]<>~`/"

_PAPER = {"Letter": "us-letter", "A4": "a4"}


def escape_markup(text: str) -> str:
    out: list[str] = []
    for ch in text:
        if ch in _MARKUP_SPECIALS:
            out.append("\\" + ch)
        else:
            out.append(ch)
    return "".join(out)


def escape_string(text: str) -> str:
    """Escape for a double-quoted Typst code-context string literal."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _emit_runs(runs: list[Run]) -> str:
    parts: list[str] = []
    for run in runs:
        text = escape_markup(run.text)
        if run.bold:
            text = f"*{text}*"
        if run.italic:
            text = f"_{text}_"
        parts.append(text)
    return "".join(parts)


def _preamble(request: RebuildRequest) -> str:
    margin = request.page.margin
    unit = margin.unit
    return "\n".join(
        [
            f'#set document(title: "{escape_string(request.metadata.title)}")',
            f'#set text(lang: "{escape_string(request.metadata.language)}")',
            "#set smartquote(enabled: false)",
            (
                f'#set page(paper: "{_PAPER[request.page.size]}", margin: ('
                f"top: {margin.top}{unit}, right: {margin.right}{unit}, "
                f"bottom: {margin.bottom}{unit}, left: {margin.left}{unit}))"
            ),
        ]
    )


def _emit_heading(block: HeadingBlock) -> str:
    return f"{'=' * block.level} {_emit_runs(block.runs)}"


def _emit_paragraph(block: ParagraphBlock) -> str:
    return _emit_runs(block.runs)


def _emit_list(block, asset_paths):  # implemented in Task 4
    raise GeneratorError("list emission not yet implemented")


def _emit_table(block):  # implemented in Task 4
    raise GeneratorError("table emission not yet implemented")


def _emit_figure(block, asset_paths):  # implemented in Task 5
    raise GeneratorError("figure emission not yet implemented")


def _emit_artifact(block, asset_paths):  # implemented in Task 5
    raise GeneratorError("artifact emission not yet implemented")


def _emit_block(block, asset_paths: dict[str, str]) -> str:
    if isinstance(block, HeadingBlock):
        return _emit_heading(block)
    if isinstance(block, ParagraphBlock):
        return _emit_paragraph(block)
    if isinstance(block, ListBlock):
        return _emit_list(block, asset_paths)
    if isinstance(block, SimpleTableBlock):
        return _emit_table(block)
    if isinstance(block, FigureBlock):
        return _emit_figure(block, asset_paths)
    if isinstance(block, ArtifactBlock):
        return _emit_artifact(block, asset_paths)
    raise GeneratorError(f"unmapped block kind: {getattr(block, 'kind', type(block))}")


def generate(request: RebuildRequest, *, asset_paths: dict[str, str]) -> str:
    """RebuildRequest -> complete .typ source text (FR-7)."""
    pieces = [_preamble(request), ""]
    for block in request.content:
        pieces.append(_emit_block(block, asset_paths))
        pieces.append("")
    return "\n".join(pieces)
