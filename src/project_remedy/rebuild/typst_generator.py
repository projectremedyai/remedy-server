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


def _emit_list(block: ListBlock, asset_paths: dict[str, str], indent: int = 0) -> str:
    """Markup lists: '-' unordered / '+' ordered. label_runs are NEVER emitted
    as prose (Caveat 2 — Typst's markers are the semantic list construct)."""
    marker = "+" if block.ordered else "-"
    pad = "  " * indent
    lines: list[str] = []
    for item in block.items:
        first = True
        for child in item.body:
            if isinstance(child, ListBlock):
                lines.append(_emit_list(child, asset_paths, indent + 1))
            elif first:
                lines.append(f"{pad}{marker} {_emit_block(child, asset_paths)}")
                first = False
            else:
                # continuation content of the same item, indented under it
                lines.append(f"{pad}  {_emit_block(child, asset_paths)}")
        if first:  # item had no body at all
            lines.append(f"{pad}{marker} ")
    return "\n".join(lines)


def _emit_table(block: SimpleTableBlock) -> str:
    if not block.rows:
        raise GeneratorError("SimpleTableBlock with zero rows")
    columns = max(len(row.cells) for row in block.rows)

    def cell(c) -> str:
        return f"[{escape_markup(c.text)}]"

    lines = [f"#table(", f"  columns: {columns},"]
    rows = list(block.rows)
    first = rows[0]
    if first.cells and all(c.header in ("col", "both") for c in first.cells):
        header_cells = ", ".join(cell(c) for c in first.cells)
        lines.append(f"  table.header({header_cells}),")
        rows = rows[1:]
    for row in rows:
        # 'row'/'both' header cells outside a full header row degrade to plain
        # cells: Typst 0.15's row-header construct is behind an unstable flag
        # (docs/typst_backend_decisions.md).
        lines.append("  " + ", ".join(cell(c) for c in row.cells) + ",")
    lines.append(")")
    return "\n".join(lines)


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
