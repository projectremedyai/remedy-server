# Typst Rebuild Backend — Phase T0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Typst as an opt-in second implementation of the `RebuildRequest → PDF bytes` contract — AST→Typst generator, `TypstRenderer`, struct-tree assertion pass, AcroForm pre-flight gate, and the dispatch branch in `_rebuild_from_semantics` — per `PRD_typst_backend.md` Phase T0.

**Architecture:** A pure generator (`rebuild/typst_generator.py`) turns a `RebuildRequest` into `.typ` source using only real Typst semantic constructs (headings/lists/tables/figures — never styled text, per PRD Caveat 2). `TypstRenderer` (`rebuild/typst_renderer.py`) writes source + assets to a temp dir and invokes `typst compile --pdf-standard ua-1` (the compiler itself is a hard alt-text gate). `rebuild/struct_assert.py` then verifies the output struct tree round-trips the input AST before the existing veraPDF acceptance gate runs. Backend selection is a config field + per-job override threaded through the existing `metadata_json` path into a branch inside `_rebuild_from_semantics`.

**Tech Stack:** Python 3.13, Pydantic AST (`rebuild/ast.py` — UNCHANGED), Typst CLI 0.15.0 (`/opt/homebrew/bin/typst` locally; Docker install in Task 10), pikepdf, pytest. Run via `uv run`.

## Global Constraints

- **Spec:** `PRD_typst_backend.md` (committed in Task 1). FR/NFR numbers below refer to it.
- **Zero changes** to `rebuild/ast.py`, `rebuild/markdown_parser.py`, `rebuild/vision_enricher.py`, `rebuild/ast_builder.py`, `rebuild/sidecar.py`, `pdf_acceptance.py` (PRD §7 blast-radius contract). `faithful_rebuild/` untouched (Non-goal 2).
- **FR-9 (Caveat-2 mitigation):** the generator has NO code path that stringifies a heading/list/table into styled text. Every block maps to exactly one construct from the FR-8 table.
- **Error convention (§1.1a):** every new failure writes `store.update(job.id, status="failed", error=f"rebuild_typst_*: ...")` — exact strings: `rebuild_typst_not_available`, `rebuild_typst_compile_failed: {exc}`, `rebuild_typst_timeout: {exc}`, `rebuild_typst_unsupported_construct: {exc}`, `rebuild_struct_assert_failed: {mismatches}`, `rebuild_acroform_present`.
- **NFR-5:** subprocess via `asyncio.create_subprocess_exec` argv list — never shell strings, never untrusted content in argv.
- **Spike-pinned decisions (from `.frugal-fable/typst-spike/FINDINGS.md`, distilled into `docs/typst_backend_decisions.md` in Task 1):**
  - Row-header (`TableCell.header in {"row"}`) cells **degrade to plain TD** in Phase T0 — the real mechanism (`pdf.header-cell(scope:)`) requires the unstable `--features a11y-extras` flag and is off-limits. `"col"`/`"both"` cells in the FIRST row map to `table.header(...)`. A `"col"`/`"both"` cell NOT in the first row also degrades to TD (Typst's `table.header` is row-scoped).
  - `ArtifactBlock` maps to `#pdf.artifact[#image("<path>")]` — wraps in `/Artifact` BMC/EMC, zero struct-tree entries, ua-1 passes. Empty-string alt is NOT a substitute (hard compile failure).
  - Markup escape set: backslash-escape `\ # $ * _ @ [ ] < > ~ ` (backtick) and `/` (as `\/` — unescaped `//` silently truncates the line as a comment). Preamble must include `#set smartquote(enabled: false)` (else quotes are silently curled).
  - `/Alt` is byte-exact in the output — struct_assert uses strict equality.
  - Nested-list struct shape: a nested `L` is a SIBLING of the parent `LI`, not a child of its `LBody` — struct_assert counts, it does not assume HTML nesting.
- Tests live in `tests/unit/`, run `uv run pytest tests/unit -q`. Tests that invoke the typst binary are marked `@pytest.mark.skipif(shutil.which("typst") is None, reason="typst CLI not installed")` so CI without typst stays green; the same applies to `verapdf` where used.
- The repo `.gitignore` blocks `*.md`/`docs/` — commit spec/docs with `git add -f`.
- All commits end with: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`

## Deferred (explicitly NOT in this plan)

- Corpus-representative pilots on real District forms and multi-column reading-order pilots (Phase T0 items a–c and open question 8) — need corpus access + human review; separate effort.
- The unstable `pdf.header-cell` row-header path (revisit when Typst stabilizes it).
- `FormFieldBlock` AST extension (Non-goal 5); README documentation of the whole rebuild tier beyond the new flags (open Q9 gets a minimal note only).
- Multi-validator (PAC/Acrobat) reconciliation (open Q7); promotion criteria (open Q4).

## File Structure

| File | Responsibility |
|---|---|
| Create `src/project_remedy/rebuild/typst_generator.py` | pure `generate(request, asset_paths) -> str` + `escape_markup()`/`escape_string()` |
| Create `src/project_remedy/rebuild/typst_renderer.py` | `TypstRenderer.render()`, `TypstCompileError`/`TypstTimeout`/`TypstNotAvailable`/`TypstUnsupportedConstruct`, binary resolution |
| Create `src/project_remedy/rebuild/struct_assert.py` | `verify(request, pdf_bytes) -> StructAssertReport` |
| Create `src/project_remedy/rebuild/acroform_gate.py` | `has_acroform(path) -> bool` |
| Modify `src/project_remedy/config.py:148-155,360-374` | `RebuildConfig.backend`, `typst_timeout_s` + `_env` loader lines |
| Modify `backend/app/engine_service.py:489-` | AcroForm gate + backend dispatch branch + new error codes |
| Modify `backend/app/routes.py:192,205` | `rebuild_backend` Form field → metadata |
| Modify `Dockerfile`, `README.md` | typst install + flag documentation |
| Create `tests/unit/test_typst_generator.py`, `test_typst_renderer.py`, `test_struct_assert.py`, `test_rebuild_dispatch.py`, `tests/unit/rebuild_fixtures.py` | tests + shared RebuildRequest builders |

---

### Task 1: Commit spec + decisions doc + shared RebuildRequest fixtures

**Files:**
- Create: `PRD_typst_backend.md` (copy from main checkout), `docs/typst_backend_decisions.md`
- Create: `tests/unit/rebuild_fixtures.py`
- Test: `tests/unit/test_rebuild_fixtures.py`

**Interfaces:**
- Produces: `make_request(**overrides) -> RebuildRequest` and `TINY_PNG: bytes` used by every later test task; `write_assets(tmp_path) -> dict[str, str]` returning `{ref: absolute_path}`.

- [ ] **Step 1: Copy + commit the spec and decisions doc**

```bash
cp /Users/laccd/code/lamc_district_forms/remedy-server/PRD_typst_backend.md .
cp docs/superpowers/plans/2026-07-02-typst-backend-phase-t0.md docs/superpowers/plans/ 2>/dev/null || true
```

Create `docs/typst_backend_decisions.md` by distilling `.frugal-fable/typst-spike/FINDINGS.md` (read it) into the five spike-pinned decisions listed in this plan's Global Constraints, each with its experiment evidence (exit codes / struct dumps) summarized in 2-4 lines. Then:

```bash
git add -f PRD_typst_backend.md docs/typst_backend_decisions.md docs/superpowers/plans/2026-07-02-typst-backend-phase-t0.md
git commit -m "docs: typst backend PRD + spike-pinned design decisions + Phase T0 plan

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 2: Write the failing fixture smoke test**

Create `tests/unit/test_rebuild_fixtures.py`:

```python
"""Smoke tests for the shared RebuildRequest test builders."""

from __future__ import annotations

from project_remedy.rebuild.ast import FigureBlock, HeadingBlock, RebuildRequest
from tests.unit.rebuild_fixtures import make_request, write_assets


def test_make_request_default_is_valid_and_rich(tmp_path):
    assets = write_assets(tmp_path)
    request = make_request(asset_dir=tmp_path)
    assert isinstance(request, RebuildRequest)
    kinds = [b.kind for b in request.content]
    assert kinds == ["heading", "paragraph", "list", "simple_table", "figure", "artifact"]
    assert request.conformance.pdfua == "PDFUA_1"
    assert set(request.assets) == {"img-1", "img-2"}
    assert set(assets) == {"img-1", "img-2"}


def test_make_request_overrides_content(tmp_path):
    request = make_request(
        asset_dir=tmp_path,
        content=[HeadingBlock(level=1, runs=[{"text": "Only"}])],
        assets={},
    )
    assert len(request.content) == 1
```

- [ ] **Step 3: Run to verify failure**

Run: `uv run pytest tests/unit/test_rebuild_fixtures.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.unit.rebuild_fixtures'` (create empty `tests/__init__.py`/`tests/unit/__init__.py` if imports fail that way — this worktree branched before the office-verify branch added them).

- [ ] **Step 4: Implement the fixtures**

Create `tests/unit/rebuild_fixtures.py`:

```python
"""Shared RebuildRequest builders for typst-backend tests (no binary blobs)."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from project_remedy.rebuild.ast import (
    ArtifactBlock,
    AssetRef,
    Conformance,
    FigureBlock,
    HeadingBlock,
    ListBlock,
    ListItem,
    Margin,
    Metadata,
    PageSettings,
    ParagraphBlock,
    RebuildRequest,
    Run,
    SimpleTableBlock,
    TableCell,
    TableRow,
)

# 1x1 transparent PNG
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def write_assets(asset_dir: Path) -> dict[str, str]:
    """Write the two standard fixture images; return {ref: absolute path}."""
    asset_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for ref in ("img-1", "img-2"):
        p = asset_dir / f"{ref}.png"
        p.write_bytes(TINY_PNG)
        paths[ref] = str(p)
    return paths


def default_content() -> list[Any]:
    return [
        HeadingBlock(level=1, runs=[Run(text="Sample Form")]),
        ParagraphBlock(runs=[Run(text="Intro text with "), Run(text="bold", bold=True), Run(text=" words.")]),
        ListBlock(
            ordered=False,
            items=[
                ListItem(label_runs=[Run(text="•")], body=[ParagraphBlock(runs=[Run(text="first item")])]),
                ListItem(label_runs=[Run(text="•")], body=[ParagraphBlock(runs=[Run(text="second item")])]),
            ],
        ),
        SimpleTableBlock(
            rows=[
                TableRow(cells=[TableCell(text="Name", header="col"), TableCell(text="Age", header="col")]),
                TableRow(cells=[TableCell(text="Alice"), TableCell(text="30")]),
            ]
        ),
        FigureBlock(asset_ref="img-1", alt="A tiny dot", caption=[Run(text="The caption")]),
        ArtifactBlock(asset_ref="img-2"),
    ]


def make_request(
    *,
    asset_dir: Path,
    content: list[Any] | None = None,
    assets: dict[str, AssetRef] | None = None,
    title: str = "Sample Form",
    language: str = "en-US",
) -> RebuildRequest:
    write_assets(asset_dir)
    if assets is None:
        assets = {
            "img-1": AssetRef(path=str(asset_dir / "img-1.png"), mime="image/png"),
            "img-2": AssetRef(path=str(asset_dir / "img-2.png"), mime="image/png"),
        }
    return RebuildRequest(
        metadata=Metadata(title=title, language=language),
        page=PageSettings(size="Letter", margin=Margin(top=0.75, right=0.75, bottom=0.75, left=0.75, unit="in")),
        conformance=Conformance(pdfua="PDFUA_1", pdfa=None),
        content=default_content() if content is None else content,
        assets=assets,
    )
```

- [ ] **Step 5: Run to verify pass, commit**

Run: `uv run pytest tests/unit/test_rebuild_fixtures.py -q` — expected PASS (2 tests). Then full suite: `uv run pytest tests/unit -q` — 87 pre-existing + 2 pass.

```bash
git add tests/
git commit -m "test: shared RebuildRequest fixture builders for typst backend

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: Config — `backend` + `typst_timeout_s`

**Files:**
- Modify: `src/project_remedy/config.py:148-155` (RebuildConfig) and `:360-374` (loader)
- Test: `tests/unit/test_rebuild_config.py`

**Interfaces:**
- Produces: `cfg.rebuild.backend: str` (default `"questpdf"`), `cfg.rebuild.typst_timeout_s: float` (default `120.0`), env vars `REBUILD_BACKEND` / `REBUILD_TYPST_TIMEOUT_S` (FR-1; two-part field+loader pattern per PRD §5.1).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_rebuild_config.py`:

```python
"""FR-1: backend selector + typst timeout follow the two-part config pattern."""

from __future__ import annotations

from project_remedy.config import RebuildConfig, load_config


def test_rebuild_config_defaults():
    cfg = RebuildConfig()
    assert cfg.backend == "questpdf"
    assert cfg.typst_timeout_s == 120.0


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("REBUILD_BACKEND", "typst")
    monkeypatch.setenv("REBUILD_TYPST_TIMEOUT_S", "45.5")
    cfg = load_config()
    assert cfg.rebuild.backend == "typst"
    assert cfg.rebuild.typst_timeout_s == 45.5
```

(If `load_config()` needs a config file/env scaffold to run in tests, check how existing config tests call it — `grep -rn "load_config" tests/` — and mirror that setup; if no test does, call it bare and fix only if it errors, mirroring whatever `_env` machinery needs.)

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_rebuild_config.py -q`
Expected: FAIL — `AttributeError: ... no attribute 'backend'`

- [ ] **Step 3: Implement**

In `RebuildConfig` (config.py:148) add fields after `markdown_parser`:

```python
    markdown_parser: str = "markdown-it-py"
    backend: str = "questpdf"          # "questpdf" | "typst" (FR-1)
    typst_timeout_s: float = 120.0     # NFR-2: budgeted like sidecar_timeout_s
```

In the loader (near `config.py:374`, inside the same `RebuildConfig(...)` construction that has the `REBUILD_MARKDOWN_PARSER` line), add:

```python
        backend=_env("REBUILD_BACKEND", rebuild_yml.get("backend", "questpdf")),
        typst_timeout_s=_env_float(
            "REBUILD_TYPST_TIMEOUT_S",
            rebuild_yml.get("typst_timeout_s", 120.0),
        ),
```

(Match the exact `_env`/`_env_float` helper names used by the neighboring lines — read them first; `REBUILD_SIDECAR_TIMEOUT_S` at :370 shows the float pattern.)

- [ ] **Step 4: Run to verify pass, full suite, commit**

`uv run pytest tests/unit/test_rebuild_config.py -q` → PASS; `uv run pytest tests/unit -q` → all green.

```bash
git add src/project_remedy/config.py tests/unit/test_rebuild_config.py
git commit -m "feat: RebuildConfig.backend + typst_timeout_s (FR-1)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Generator — escaping + preamble + heading/paragraph

**Files:**
- Create: `src/project_remedy/rebuild/typst_generator.py`
- Test: `tests/unit/test_typst_generator.py`

**Interfaces:**
- Produces: `generate(request: RebuildRequest, *, asset_paths: dict[str, str]) -> str` (FR-7 pure function; `asset_paths` maps `asset_ref` → path string to embed in `image()` calls); `escape_markup(text: str) -> str`; `escape_string(text: str) -> str` (for code-context string literals like `alt:`); internal `_emit_block(block) -> str` dispatch that later tasks extend. `GeneratorError(ValueError)` raised on empty `FigureBlock.alt` at generation time (FR-8 defense-in-depth).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_typst_generator.py`:

```python
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
    src = generate(make_request(asset_dir=tmp_path), asset_paths={"img-1": "img-1.png", "img-2": "img-2.png"})
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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_typst_generator.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'project_remedy.rebuild.typst_generator'`

- [ ] **Step 3: Implement**

Create `src/project_remedy/rebuild/typst_generator.py`:

```python
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


def _emit_block(block, asset_paths: dict[str, str]) -> str:
    if isinstance(block, HeadingBlock):
        return _emit_heading(block)
    if isinstance(block, ParagraphBlock):
        return _emit_paragraph(block)
    if isinstance(block, ListBlock):
        return _emit_list(block, asset_paths)          # Task 4
    if isinstance(block, SimpleTableBlock):
        return _emit_table(block)                      # Task 4
    if isinstance(block, FigureBlock):
        return _emit_figure(block, asset_paths)        # Task 5
    if isinstance(block, ArtifactBlock):
        return _emit_artifact(block, asset_paths)      # Task 5
    raise GeneratorError(f"unmapped block kind: {getattr(block, 'kind', type(block))}")


def generate(request: RebuildRequest, *, asset_paths: dict[str, str]) -> str:
    """RebuildRequest -> complete .typ source text (FR-7)."""
    pieces = [_preamble(request), ""]
    for block in request.content:
        pieces.append(_emit_block(block, asset_paths))
        pieces.append("")
    return "\n".join(pieces)
```

Also add stub raisers so the module imports before Tasks 4-5 land:

```python
def _emit_list(block, asset_paths):  # implemented in Task 4
    raise GeneratorError("list emission not yet implemented")


def _emit_table(block):  # implemented in Task 4
    raise GeneratorError("table emission not yet implemented")


def _emit_figure(block, asset_paths):  # implemented in Task 5
    raise GeneratorError("figure emission not yet implemented")


def _emit_artifact(block, asset_paths):  # implemented in Task 5
    raise GeneratorError("artifact emission not yet implemented")
```

(Define the four stubs ABOVE `_emit_block` or rely on module-level name resolution at call time — Python resolves at call time, so order is fine; keep them at the bottom with their Task-N comments so Tasks 4-5 replace them in place.)

- [ ] **Step 4: Run to verify pass, full suite, commit**

The `test_preamble...` and `test_heading...` tests use content WITHOUT lists/tables/figures, so the stubs never fire. `uv run pytest tests/unit/test_typst_generator.py -q` → PASS; full suite green.

```bash
git add src/project_remedy/rebuild/typst_generator.py tests/unit/test_typst_generator.py
git commit -m "feat: typst generator core — escaping, preamble, heading/paragraph (FR-7/9)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Generator — lists + tables

**Files:**
- Modify: `src/project_remedy/rebuild/typst_generator.py` (replace the two stubs)
- Test: `tests/unit/test_typst_generator.py` (append)

**Interfaces:**
- Consumes: `_emit_runs`, `escape_string`, `GeneratorError` from Task 3.
- Produces: `_emit_list(block: ListBlock, asset_paths) -> str` (markup `-`/`+` lists, recursive via `_emit_block` for item bodies, `label_runs` NEVER emitted as prose — spike/Caveat-2), `_emit_table(block: SimpleTableBlock) -> str` (`table()` call; first row all-`col`/`both` → `table.header(...)`; `row`-header cells degrade to TD per the spike decision).

- [ ] **Step 1: Append failing tests**

```python
from project_remedy.rebuild.ast import (
    ListBlock,
    ListItem,
    SimpleTableBlock,
    TableCell,
    TableRow,
)


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
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/unit/test_typst_generator.py -q`
Expected: the four new tests FAIL with `GeneratorError: list emission not yet implemented` / `table emission not yet implemented`.

- [ ] **Step 3: Implement (replace the two stubs)**

```python
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
```

- [ ] **Step 4: Run to verify pass, full suite, commit**

```bash
uv run pytest tests/unit/test_typst_generator.py -q   # PASS
uv run pytest tests/unit -q                            # green
git add src/project_remedy/rebuild/typst_generator.py tests/unit/test_typst_generator.py
git commit -m "feat: typst generator — semantic lists and tables with header-row mapping

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Generator — figure + artifact

**Files:**
- Modify: `src/project_remedy/rebuild/typst_generator.py` (replace the two remaining stubs)
- Test: `tests/unit/test_typst_generator.py` (append)

**Interfaces:**
- Consumes: `escape_string`, `escape_markup`, `GeneratorError`.
- Produces: `_emit_figure` → `#figure(image("<path>", alt: "<alt>"), caption: [..])` (or bare `#image(..., alt: ...)` without caption); `_emit_artifact` → `#pdf.artifact[#image("<path>")]` (spike decision). Paths come from `asset_paths[block.asset_ref]`; a missing ref or empty alt raises `GeneratorError`.

- [ ] **Step 1: Append failing tests**

```python
from project_remedy.rebuild.ast import ArtifactBlock, AssetRef, FigureBlock

from project_remedy.rebuild.typst_generator import GeneratorError


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
```

- [ ] **Step 2: Run to verify failure** — the new tests fail with `GeneratorError: figure emission not yet implemented`.

- [ ] **Step 3: Implement (replace the two stubs)**

```python
def _asset_path(asset_ref: str, asset_paths: dict[str, str]) -> str:
    try:
        return asset_paths[asset_ref]
    except KeyError as exc:
        raise GeneratorError(f"no asset path provided for asset_ref {asset_ref!r}") from exc


def _emit_figure(block: FigureBlock, asset_paths: dict[str, str]) -> str:
    alt = (block.alt or "").strip()
    if not alt:
        # Defense-in-depth: Pydantic min_length=1 should make this unreachable
        # (FR-8) — fail loudly, never substitute a placeholder.
        raise GeneratorError(f"FigureBlock {block.asset_ref!r} reached the generator with empty alt")
    path = _asset_path(block.asset_ref, asset_paths)
    image = f'image("{escape_string(path)}", alt: "{escape_string(alt)}")'
    if block.caption:
        return f"#figure({image}, caption: [{_emit_runs(block.caption)}])"
    return f"#{image}"


def _emit_artifact(block: ArtifactBlock, asset_paths: dict[str, str]) -> str:
    """Decorative image: /Artifact marked-content, zero struct-tree entries
    (spike-verified). Never an alt-less Figure — that hard-fails ua-1."""
    path = _asset_path(block.asset_ref, asset_paths)
    return f'#pdf.artifact[#image("{escape_string(path)}")]'
```

- [ ] **Step 4: Run to verify pass, full suite, commit**

```bash
uv run pytest tests/unit/test_typst_generator.py -q && uv run pytest tests/unit -q
git add src/project_remedy/rebuild/typst_generator.py tests/unit/test_typst_generator.py
git commit -m "feat: typst generator — figure alt/caption + pdf.artifact decorative mapping

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: `TypstRenderer` — subprocess wrapper + compile-gate tests

**Files:**
- Create: `src/project_remedy/rebuild/typst_renderer.py`
- Test: `tests/unit/test_typst_renderer.py`

**Interfaces:**
- Consumes: `generate()` from Tasks 3-5.
- Produces (FR-4/5/6): module `typst_renderer` with:
  - `class TypstError(RuntimeError)`; `class TypstNotAvailable(TypstError)`; `class TypstCompileError(TypstError)` (carries stderr verbatim, NFR-4); `class TypstTimeout(TypstError)`; `class TypstUnsupportedConstruct(TypstError)`.
  - `resolve_typst_binary() -> pathlib.Path | None` (uses `shutil.which("typst")`).
  - `@dataclass TypstRenderer: binary_path: pathlib.Path; timeout_s: float = 120.0` with `async def render(self, request: RebuildRequest) -> bytes`.
  - render(): create `tempfile.TemporaryDirectory`; copy each `request.assets[ref].path` into it as `{ref}.png`/`.jpg` (extension from mime); call `generate(request, asset_paths={ref: filename})` (RELATIVE filenames — typst resolves against the .typ's directory); write `main.typ`; run `typst compile main.typ out.pdf --pdf-standard ua-1` with `cwd=tmpdir` via `asyncio.create_subprocess_exec` (argv list, NFR-5); non-zero exit → `TypstCompileError(stderr)`; timeout → kill + `TypstTimeout`; read `out.pdf`, check `%PDF` magic, return bytes. Temp dir always cleaned (context manager).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_typst_renderer.py`:

```python
"""FR-4/5: TypstRenderer subprocess wrapper. Binary-dependent tests skip when
typst is not installed; the negative alt-text test is AC #4 of the PRD."""

from __future__ import annotations

import shutil

import pytest

from project_remedy.rebuild.typst_renderer import (
    TypstCompileError,
    TypstNotAvailable,
    TypstRenderer,
    TypstTimeout,
    resolve_typst_binary,
)
from tests.unit.rebuild_fixtures import make_request

needs_typst = pytest.mark.skipif(shutil.which("typst") is None, reason="typst CLI not installed")


def test_resolve_typst_binary_type():
    binary = resolve_typst_binary()
    assert binary is None or binary.name == "typst"


@needs_typst
async def test_render_produces_pdf(tmp_path):
    renderer = TypstRenderer(binary_path=resolve_typst_binary())
    pdf = await renderer.render(make_request(asset_dir=tmp_path))
    assert pdf.startswith(b"%PDF")
    assert len(pdf) > 1000


@needs_typst
async def test_missing_alt_hard_fails_compile(tmp_path):
    """AC #4: the ua-1 compiler gate. Bypass the generator (which cannot emit
    alt-less images) with a handcrafted bad source via the renderer's internals."""
    renderer = TypstRenderer(binary_path=resolve_typst_binary())
    from tests.unit.rebuild_fixtures import TINY_PNG

    (tmp_path / "p.png").write_bytes(TINY_PNG)
    (tmp_path / "main.typ").write_text('#image("p.png")\n')
    with pytest.raises(TypstCompileError) as excinfo:
        await renderer._compile(tmp_path)  # noqa: SLF001 - deliberate internal test
    assert "alt" in str(excinfo.value).lower()


@needs_typst
async def test_timeout_raises(tmp_path):
    renderer = TypstRenderer(binary_path=resolve_typst_binary(), timeout_s=0.0001)
    with pytest.raises(TypstTimeout):
        await renderer.render(make_request(asset_dir=tmp_path))
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: ... typst_renderer`.

- [ ] **Step 3: Implement**

Create `src/project_remedy/rebuild/typst_renderer.py`:

```python
"""Typst rebuild renderer (PRD_typst_backend.md §5.2).

Same effective contract as QuestPdfSidecar.render(): RebuildRequest -> PDF
bytes. Unlike the sidecar's stdin/stdout protocol, Typst compiles generated
source from a job-scoped temp directory. `typst compile --pdf-standard ua-1`
is itself a hard accessibility gate (missing image/equation alt text fails
the build) — surfaced verbatim via TypstCompileError (NFR-4).
"""

from __future__ import annotations

import asyncio
import pathlib
import shutil
import tempfile
from dataclasses import dataclass

from project_remedy.rebuild.ast import RebuildRequest
from project_remedy.rebuild.typst_generator import generate


class TypstError(RuntimeError):
    """Base class for Typst renderer failures."""


class TypstNotAvailable(TypstError):
    """No typst binary on PATH / configured."""


class TypstCompileError(TypstError):
    """typst compile exited non-zero; message carries stderr verbatim."""


class TypstTimeout(TypstError):
    """typst compile exceeded the configured timeout."""


class TypstUnsupportedConstruct(TypstError):
    """The RebuildRequest needs constructs the Typst backend cannot yet render safely (FR-6)."""


_EXT_BY_MIME = {"image/png": ".png", "image/jpeg": ".jpg"}


def resolve_typst_binary() -> pathlib.Path | None:
    found = shutil.which("typst")
    return pathlib.Path(found) if found else None


@dataclass
class TypstRenderer:
    binary_path: pathlib.Path
    timeout_s: float = 120.0

    async def render(self, request: RebuildRequest) -> bytes:
        with tempfile.TemporaryDirectory(prefix="typst-rebuild-") as tmp:
            tmpdir = pathlib.Path(tmp)
            asset_paths: dict[str, str] = {}
            for ref, asset in request.assets.items():
                filename = f"{ref}{_EXT_BY_MIME[asset.mime]}"
                shutil.copyfile(asset.path, tmpdir / filename)
                asset_paths[ref] = filename
            source = generate(request, asset_paths=asset_paths)
            (tmpdir / "main.typ").write_text(source, encoding="utf-8")
            return await self._compile(tmpdir)

    async def _compile(self, tmpdir: pathlib.Path) -> bytes:
        argv = [str(self.binary_path), "compile", "main.typ", "out.pdf", "--pdf-standard", "ua-1"]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(tmpdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_s)
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise TypstTimeout(f"typst compile timed out after {self.timeout_s}s") from exc
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace").strip()
            raise TypstCompileError(f"typst exited {proc.returncode}: {err or '<no stderr>'}")
        out = tmpdir / "out.pdf"
        if not out.exists():
            raise TypstCompileError("typst exited 0 but produced no out.pdf")
        pdf = out.read_bytes()
        if not pdf.startswith(b"%PDF"):
            raise TypstCompileError("typst output is not a PDF (missing %PDF magic bytes)")
        return pdf
```

- [ ] **Step 4: Run to verify pass (typst installed locally), full suite, commit**

```bash
uv run pytest tests/unit/test_typst_renderer.py -q   # 4 pass locally
uv run pytest tests/unit -q
git add src/project_remedy/rebuild/typst_renderer.py tests/unit/test_typst_renderer.py
git commit -m "feat: TypstRenderer — ua-1 compile gate subprocess wrapper (FR-4/5, AC4)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: `struct_assert` — AST↔struct-tree round-trip verification

**Files:**
- Create: `src/project_remedy/rebuild/struct_assert.py`
- Test: `tests/unit/test_struct_assert.py`

**Interfaces:**
- Consumes: `TypstRenderer` (tests compile real PDFs, `@needs_typst`); `RebuildRequest` blocks.
- Produces (FR-10/11): `@dataclass StructAssertReport: passed: bool; mismatches: list[str]` and `verify(request: RebuildRequest, pdf_bytes: bytes) -> StructAssertReport`. Checks, all count/value-based against the INPUT AST (never "an H1 exists somewhere"):
  - per level N: count of `H{N}` struct elements == count of `HeadingBlock(level=N)` (a heading emitted as `P` shows up as a missing-H mismatch — the Caveat-2 signature);
  - `Table` count == `SimpleTableBlock` count; every table with a full header row in the AST has ≥1 `TH` descendant;
  - `L` count == TOTAL ListBlock count including nested (spike: nested L is a sibling of the parent LI); per-list `LI` totals == total item count;
  - `Figure` count == `FigureBlock` count; the multiset of `/Alt` strings == the multiset of `FigureBlock.alt` values (byte-exact — spike-verified);
  - `ArtifactBlock`s produce NO struct elements (artifact images must not appear as `Figure`).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_struct_assert.py`:

```python
"""FR-10/11: independent AST<->struct-tree round-trip verification."""

from __future__ import annotations

import shutil

import pytest

from project_remedy.rebuild.ast import FigureBlock, HeadingBlock, Run
from project_remedy.rebuild.struct_assert import StructAssertReport, verify
from project_remedy.rebuild.typst_renderer import TypstRenderer, resolve_typst_binary
from tests.unit.rebuild_fixtures import make_request

needs_typst = pytest.mark.skipif(shutil.which("typst") is None, reason="typst CLI not installed")


@needs_typst
async def test_full_fixture_round_trips(tmp_path):
    request = make_request(asset_dir=tmp_path)
    pdf = await TypstRenderer(binary_path=resolve_typst_binary()).render(request)
    report = verify(request, pdf)
    assert report.passed, report.mismatches


@needs_typst
async def test_mismatch_detected_when_ast_expects_more(tmp_path):
    """Compile a doc with ONE heading, then verify against a request claiming TWO —
    the report must fail with an H-count mismatch (proves the checker is not a rubber stamp)."""
    one = make_request(asset_dir=tmp_path,
                       content=[HeadingBlock(level=1, runs=[Run(text="Only")])], assets={})
    pdf = await TypstRenderer(binary_path=resolve_typst_binary()).render(one)
    two = make_request(asset_dir=tmp_path,
                       content=[HeadingBlock(level=1, runs=[Run(text="Only")]),
                                HeadingBlock(level=2, runs=[Run(text="Ghost")])], assets={})
    report = verify(two, pdf)
    assert not report.passed
    assert any("H2" in m for m in report.mismatches)


@needs_typst
async def test_alt_text_byte_exact(tmp_path):
    from project_remedy.rebuild.ast import AssetRef

    request = make_request(
        asset_dir=tmp_path,
        content=[FigureBlock(asset_ref="img-1", alt="Exact alt 42")],
        assets={"img-1": AssetRef(path=str(tmp_path / "img-1.png"), mime="image/png")},
    )
    pdf = await TypstRenderer(binary_path=resolve_typst_binary()).render(request)
    assert verify(request, pdf).passed
    wrong = make_request(
        asset_dir=tmp_path,
        content=[FigureBlock(asset_ref="img-1", alt="Different alt")],
        assets={"img-1": AssetRef(path=str(tmp_path / "img-1.png"), mime="image/png")},
    )
    report = verify(wrong, pdf)
    assert not report.passed
    assert any("Alt" in m for m in report.mismatches)


def test_verify_handles_untagged_pdf():
    """A PDF with no struct tree must fail cleanly, not crash."""
    import pikepdf
    from io import BytesIO

    buf = BytesIO()
    with pikepdf.new() as pdf:
        pdf.add_blank_page()
        pdf.save(buf)
    request_stub = make_request.__wrapped__ if hasattr(make_request, "__wrapped__") else None
    # Build a minimal request inline (no assets needed):
    from tests.unit.rebuild_fixtures import make_request as mk
    import tempfile, pathlib
    with tempfile.TemporaryDirectory() as td:
        request = mk(asset_dir=pathlib.Path(td),
                     content=[HeadingBlock(level=1, runs=[Run(text="X")])], assets={})
    report = verify(request, buf.getvalue())
    assert not report.passed
    assert any("StructTreeRoot" in m for m in report.mismatches)
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError: ... struct_assert`.

- [ ] **Step 3: Implement**

Create `src/project_remedy/rebuild/struct_assert.py`:

```python
"""Post-generation struct-tree assertion pass (PRD_typst_backend.md §5.4).

Independently verifies that the compiled PDF's struct tree round-trips the
input RebuildRequest — the gate veraPDF cannot provide (Caveat 2: veraPDF
validates tag well-formedness, not semantic correctness of authoring). A
failure here is a GENERATOR BUG by definition (FR-11): the caller must
hard-fail the job, never degrade silently.
"""

from __future__ import annotations

import io
from collections import Counter
from dataclasses import dataclass, field

import pikepdf

from project_remedy.rebuild.ast import (
    Block,
    FigureBlock,
    HeadingBlock,
    ListBlock,
    RebuildRequest,
    SimpleTableBlock,
)


@dataclass
class StructAssertReport:
    passed: bool
    mismatches: list[str] = field(default_factory=list)


def _walk_struct(elem, tags: Counter, alts: list[str]) -> None:
    s = elem.get("/S")
    if s is not None:
        name = str(s)[1:] if str(s).startswith("/") else str(s)
        tags[name] += 1
        if name == "Figure":
            alt = elem.get("/Alt")
            alts.append(str(alt) if alt is not None else "")
    kids = elem.get("/K")
    if kids is None:
        return
    if not isinstance(kids, pikepdf.Array):
        kids = [kids]
    for kid in kids:
        if isinstance(kid, pikepdf.Dictionary):
            _walk_struct(kid, tags, alts)


def _expected(blocks: list[Block], exp: Counter, alts: list[str], tables: list[SimpleTableBlock]) -> None:
    for b in blocks:
        if isinstance(b, HeadingBlock):
            exp[f"H{b.level}"] += 1
        elif isinstance(b, SimpleTableBlock):
            exp["Table"] += 1
            tables.append(b)
        elif isinstance(b, ListBlock):
            exp["L"] += 1
            exp["LI"] += len(b.items)
            for item in b.items:
                _expected(item.body, exp, alts, tables)
        elif isinstance(b, FigureBlock):
            exp["Figure"] += 1
            alts.append(b.alt)


def verify(request: RebuildRequest, pdf_bytes: bytes) -> StructAssertReport:
    mismatches: list[str] = []
    tags: Counter = Counter()
    found_alts: list[str] = []
    try:
        with pikepdf.open(io.BytesIO(pdf_bytes)) as pdf:
            root = pdf.Root.get("/StructTreeRoot")
            if root is None:
                return StructAssertReport(False, ["no /StructTreeRoot in output PDF"])
            _walk_struct(root, tags, found_alts)
    except Exception as exc:  # noqa: BLE001 - unreadable output is a hard mismatch
        return StructAssertReport(False, [f"could not read struct tree: {exc}"])

    expected: Counter = Counter()
    expected_alts: list[str] = []
    expected_tables: list[SimpleTableBlock] = []
    _expected(request.content, expected, expected_alts, expected_tables)

    for level in range(1, 7):
        tag = f"H{level}"
        if tags.get(tag, 0) != expected.get(tag, 0):
            mismatches.append(
                f"{tag}: expected {expected.get(tag, 0)} from AST, found {tags.get(tag, 0)}"
            )
    for tag in ("Table", "L", "LI", "Figure"):
        if tags.get(tag, 0) != expected.get(tag, 0):
            mismatches.append(
                f"{tag}: expected {expected.get(tag, 0)} from AST, found {tags.get(tag, 0)}"
            )
    if any(
        row.cells and all(c.header in ("col", "both") for c in row.cells)
        for t in expected_tables
        for row in t.rows[:1]
    ) and tags.get("TH", 0) == 0:
        mismatches.append("AST has header rows but output has zero TH elements")
    if Counter(expected_alts) != Counter(found_alts):
        mismatches.append(
            f"Figure /Alt mismatch: expected {sorted(expected_alts)}, found {sorted(found_alts)}"
        )
    return StructAssertReport(passed=not mismatches, mismatches=mismatches)
```

- [ ] **Step 4: Run to verify pass, full suite, commit**

If `test_full_fixture_round_trips` fails on the LI count or THead/TBody shape, compare against `.frugal-fable/typst-spike/FINDINGS.md`'s recorded struct dumps and adjust the EXPECTATION model in `_expected` (with a comment citing the findings), never loosen an assertion to `>=`. Clean up the messy inline-request construction in `test_verify_handles_untagged_pdf` if you find a tidier equivalent — its assertion must stay.

```bash
uv run pytest tests/unit/test_struct_assert.py -q && uv run pytest tests/unit -q
git add src/project_remedy/rebuild/struct_assert.py tests/unit/test_struct_assert.py
git commit -m "feat: struct-tree assertion pass — AST round-trip gate (FR-10/11)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: AcroForm pre-flight gate (FR-13)

**Files:**
- Create: `src/project_remedy/rebuild/acroform_gate.py`
- Test: `tests/unit/test_acroform_gate.py`

**Interfaces:**
- Produces: `has_acroform(path: Path) -> bool` — True iff the source PDF's catalog has a non-empty `/AcroForm` with `/Fields`. Never raises (unreadable → False with a debug log; the rebuild path will fail later anyway on a truly broken file). Task 9 wires it into `_rebuild_from_semantics` for BOTH backends.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_acroform_gate.py`:

```python
"""FR-13: fillable-form sources must be detected before any AST rebuild."""

from __future__ import annotations

import pikepdf

from project_remedy.rebuild.acroform_gate import has_acroform


def _blank_pdf(path):
    with pikepdf.new() as pdf:
        pdf.add_blank_page()
        pdf.save(path)


def test_plain_pdf_has_no_acroform(tmp_path):
    p = tmp_path / "plain.pdf"
    _blank_pdf(p)
    assert has_acroform(p) is False


def test_acroform_pdf_detected(tmp_path):
    p = tmp_path / "form.pdf"
    with pikepdf.new() as pdf:
        pdf.add_blank_page()
        field = pdf.make_indirect(pikepdf.Dictionary(FT=pikepdf.Name("/Tx"), T=pikepdf.String("name")))
        pdf.Root.AcroForm = pdf.make_indirect(
            pikepdf.Dictionary(Fields=pikepdf.Array([field]))
        )
        pdf.save(p)
    assert has_acroform(p) is True


def test_empty_acroform_fields_is_not_a_form(tmp_path):
    p = tmp_path / "emptyform.pdf"
    with pikepdf.new() as pdf:
        pdf.add_blank_page()
        pdf.Root.AcroForm = pdf.make_indirect(pikepdf.Dictionary(Fields=pikepdf.Array([])))
        pdf.save(p)
    assert has_acroform(p) is False


def test_unreadable_file_is_false_not_raise(tmp_path):
    p = tmp_path / "junk.pdf"
    p.write_bytes(b"not a pdf")
    assert has_acroform(p) is False
```

- [ ] **Step 2: Run to verify failure** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
"""AcroForm pre-flight detection (PRD_typst_backend.md FR-13).

Neither rebuild backend regenerates form fields from the AST — a fillable
source routed into the AST rebuild tier would silently lose its fields.
Detection is backend-agnostic and runs before extraction.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pikepdf

logger = logging.getLogger(__name__)


def has_acroform(path: Path) -> bool:
    """True iff the PDF catalog carries an AcroForm with at least one field."""
    try:
        with pikepdf.open(path) as pdf:
            acroform = pdf.Root.get("/AcroForm")
            if acroform is None:
                return False
            fields = acroform.get("/Fields")
            return bool(fields is not None and len(fields) > 0)
    except Exception as exc:  # noqa: BLE001 - pre-flight must never raise
        logger.debug("AcroForm probe failed for %s: %s", path, exc)
        return False
```

- [ ] **Step 4: Run to verify pass, full suite, commit**

```bash
uv run pytest tests/unit/test_acroform_gate.py -q && uv run pytest tests/unit -q
git add src/project_remedy/rebuild/acroform_gate.py tests/unit/test_acroform_gate.py
git commit -m "feat: AcroForm pre-flight detection for the rebuild tier (FR-13)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Dispatch — backend branch in `_rebuild_from_semantics` + route override

**Files:**
- Modify: `backend/app/engine_service.py` (`_rebuild_from_semantics`, around :489-614; and its caller at :225)
- Modify: `backend/app/routes.py:192,205`
- Test: `tests/unit/test_rebuild_dispatch.py`

**Interfaces:**
- Consumes: `TypstRenderer`, `resolve_typst_binary`, `TypstCompileError`, `TypstTimeout`, `TypstNotAvailable`, `TypstUnsupportedConstruct` (Task 6); `verify`/`StructAssertReport` (Task 7); `has_acroform` (Task 8); `cfg.rebuild.backend`/`typst_timeout_s` (Task 2).
- Produces: `_rebuild_from_semantics(..., backend_override: str | None = None)`; route flag `rebuild_backend: str | None = Form(None)` carried in `metadata_json` exactly like `allow_semantic_rebuild` (routes.py:205 pattern); caller reads `meta.get("rebuild_backend")` where it already reads `allow_semantic_rebuild` (engine_service.py:207) and passes it through. Effective backend = `backend_override or cfg.rebuild.backend`; unknown value → `rebuild_typst_unsupported_construct: unknown backend '<x>'` failure (fail closed).

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_rebuild_dispatch.py`:

```python
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

    async def fake_extract_stage(*args, **kwargs):
        return "# Title\n\nBody paragraph text.\n"

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
    from project_remedy.config import load_config

    cfg = load_config()
    cfg.rebuild.backend = backend
    cfg.output.output_dir = tmp_path / "out"
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
        _job(input_path), store, SimpleNamespace(),
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
```

- [ ] **Step 2: Run to verify failure** — `TypeError: _rebuild_from_semantics() got an unexpected keyword argument 'backend_override'` (and missing `_TypstRenderer` attr).

- [ ] **Step 3: Implement the dispatch**

In `backend/app/engine_service.py`:

1. Add to the aliased rebuild imports (near :35):

```python
from project_remedy.rebuild.acroform_gate import has_acroform as _has_acroform
from project_remedy.rebuild.struct_assert import verify as _struct_verify
from project_remedy.rebuild.typst_renderer import (
    TypstCompileError as _TypstCompileError,
    TypstNotAvailable as _TypstNotAvailable,
    TypstRenderer as _TypstRenderer,
    TypstTimeout as _TypstTimeout,
    TypstUnsupportedConstruct as _TypstUnsupported,
    resolve_typst_binary as _resolve_typst_binary,
)
```

2. Change the signature (`:489`):

```python
async def _rebuild_from_semantics(
    input_path: Path,
    output_path: Path,
    cfg,  # PipelineConfig
    job: Job,
    store: JobStore,
    settings: Settings,
    backend_override: str | None = None,
) -> None:
```

3. FIRST thing in the body (before the extractor, FR-13 — backend-agnostic):

```python
    # FR-13 pre-flight: fillable forms must never be silently flattened by an
    # AST rebuild (neither backend regenerates form fields).
    if await asyncio.to_thread(_has_acroform, input_path):
        await store.update(job.id, status="failed", error="rebuild_acroform_present")
        return
```

4. Replace step 3 (":591 --- step 3: render via sidecar ---" through the two `except _Sidecar*` blocks) with the dispatch:

```python
    # --- step 3: render via the selected backend ---
    backend = (backend_override or getattr(cfg.rebuild, "backend", "questpdf")).strip().lower()
    if backend == "questpdf":
        binary = _resolve_sidecar_binary()
        if binary is None:
            await store.update(job.id, status="failed", error="rebuild_sidecar_not_available")
            return
        sidecar = _Sidecar(binary_path=binary, timeout_s=cfg.rebuild.sidecar_timeout_s)
        try:
            pdf_bytes = await sidecar.render(request)
        except _SidecarTimeout as exc:
            await store.update(job.id, status="failed", error=f"rebuild_sidecar_timeout: {exc}")
            return
        except _SidecarError as exc:
            await store.update(job.id, status="failed", error=f"rebuild_sidecar_failed: {exc}")
            return
    elif backend == "typst":
        typst_binary = _resolve_typst_binary()
        if typst_binary is None:
            await store.update(job.id, status="failed", error="rebuild_typst_not_available")
            return
        renderer = _TypstRenderer(
            binary_path=typst_binary,
            timeout_s=getattr(cfg.rebuild, "typst_timeout_s", 120.0),
        )
        try:
            pdf_bytes = await renderer.render(request)
        except _TypstTimeout as exc:
            await store.update(job.id, status="failed", error=f"rebuild_typst_timeout: {exc}")
            return
        except _TypstUnsupported as exc:
            await store.update(
                job.id, status="failed", error=f"rebuild_typst_unsupported_construct: {exc}"
            )
            return
        except (_TypstCompileError, _TypstNotAvailable) as exc:
            await store.update(
                job.id, status="failed", error=f"rebuild_typst_compile_failed: {exc}"
            )
            return
        # FR-10/11: a struct-assert failure is a generator bug — hard fail.
        assert_report = await asyncio.to_thread(_struct_verify, request, pdf_bytes)
        if not assert_report.passed:
            await store.update(
                job.id, status="failed",
                error=f"rebuild_struct_assert_failed: {'; '.join(assert_report.mismatches)[:500]}",
            )
            return
    else:
        await store.update(
            job.id, status="failed",
            error=f"rebuild_typst_unsupported_construct: unknown backend {backend!r}",
        )
        return
```

5. In the caller (around :207/:225): read the override next to the existing metadata read and pass it through:

```python
    allow_rebuild = bool(meta.get("allow_semantic_rebuild", False))
    rebuild_backend_override = meta.get("rebuild_backend") or None
```
```python
            await _rebuild_from_semantics(
                input_path, output_path, cfg, job, store, settings,
                backend_override=rebuild_backend_override,
            )
```

6. In `backend/app/routes.py` (:192 area): add the Form field and metadata entry:

```python
        allow_semantic_rebuild: bool = Form(False),
        rebuild_backend: str | None = Form(None),
```
```python
        metadata: dict[str, object] = {"allow_semantic_rebuild": allow_semantic_rebuild}
        if rebuild_backend:
            metadata["rebuild_backend"] = rebuild_backend
```

- [ ] **Step 4: Run to verify pass**

`uv run pytest tests/unit/test_rebuild_dispatch.py -q` — expected PASS (4 tests; 1 skipped without typst). The fixture monkeypatching may need adjustment against the real import structure (e.g. if `ContentExtractor` is imported inside the function body, patch `project_remedy.extractor.ContentExtractor` as shown — that IS the lazy-import target). If `DocumentJob.get_extracted_images` already exists as a real method returning stored rows, prefer monkeypatching it on the class as shown. If a stub turns out to miss a required attribute, extend the stub — do NOT weaken the assertions.

- [ ] **Step 5: Full suite, commit**

```bash
uv run pytest tests/unit -q
git add backend/app/engine_service.py backend/app/routes.py tests/unit/test_rebuild_dispatch.py
git commit -m "feat: backend dispatch in _rebuild_from_semantics + AcroForm gate + rebuild_backend override (FR-2/13, AC9)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: E2E acceptance + Dockerfile/README + sweep

**Files:**
- Modify: `Dockerfile`, `README.md`
- Test: `tests/unit/test_typst_e2e.py`

**Interfaces:**
- Consumes: everything above.
- Produces: the Phase T0 exit evidence (PRD §8 metrics 2/3/4 automated; metric 1 on the fixture document + veraPDF when available).

- [ ] **Step 1: Write the E2E test**

Create `tests/unit/test_typst_e2e.py`:

```python
"""Phase T0 E2E: RebuildRequest -> generate -> typst compile (ua-1) ->
struct_assert -> (veraPDF when available). PRD §8 metrics 1-4 on the fixture doc."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from project_remedy.rebuild.struct_assert import verify
from project_remedy.rebuild.typst_renderer import TypstRenderer, resolve_typst_binary
from tests.unit.rebuild_fixtures import make_request

needs_typst = pytest.mark.skipif(shutil.which("typst") is None, reason="typst CLI not installed")
needs_verapdf = pytest.mark.skipif(shutil.which("verapdf") is None, reason="verapdf not installed")


@needs_typst
async def test_e2e_fixture_document_round_trips(tmp_path):
    request = make_request(asset_dir=tmp_path)
    pdf = await TypstRenderer(binary_path=resolve_typst_binary()).render(request)
    report = verify(request, pdf)
    assert report.passed, report.mismatches


@needs_typst
@needs_verapdf
async def test_e2e_verapdf_ua1_clean(tmp_path):
    request = make_request(asset_dir=tmp_path)
    pdf = await TypstRenderer(binary_path=resolve_typst_binary()).render(request)
    out = tmp_path / "typst_e2e.pdf"
    out.write_bytes(pdf)
    result = subprocess.run(
        ["verapdf", "-f", "ua1", str(out)], capture_output=True, text=True, timeout=300,
    )
    assert 'isCompliant="true"' in result.stdout, result.stdout[-2000:]
```

- [ ] **Step 2: Run to verify pass**

`uv run pytest tests/unit/test_typst_e2e.py -q` — both should PASS locally (typst + verapdf are installed on this machine). If veraPDF reports non-compliance, read its rule failures — that is a REAL generator bug to fix (most likely candidates per the spike: table struct shape or missing document title) — fix the generator, not the test.

- [ ] **Step 3: Dockerfile + README**

In `Dockerfile`, locate the stage that installs veraPDF and add the typst binary alongside (pin the version):

```dockerfile
# Typst CLI (rebuild backend, PRD_typst_backend.md NFR-3)
ARG TYPST_VERSION=0.15.0
RUN curl -fsSL "https://github.com/typst/typst/releases/download/v${TYPST_VERSION}/typst-x86_64-unknown-linux-musl.tar.xz" \
    | tar -xJ --strip-components=1 -C /usr/local/bin typst-x86_64-unknown-linux-musl/typst
```

(Adapt to the Dockerfile's existing download/install idiom — read it first; if the base image is arm64, use the matching artifact or `TARGETARCH` logic consistent with how the Dockerfile already handles architecture, if it does.)

In `README.md`, in the section documenting remediation flags/env vars, add 4-6 lines: `REBUILD_BACKEND=questpdf|typst` (default questpdf), `REBUILD_TYPST_TIMEOUT_S`, the per-job `rebuild_backend` form field alongside `allow_semantic_rebuild`, and one sentence that the Typst backend adds a compile-time PDF/UA-1 gate plus a struct-tree assertion pass before the usual veraPDF acceptance gate.

- [ ] **Step 4: Sweep + commit**

```bash
uv run pytest tests/unit -q && uv run pytest tests/unit -q   # deterministic double-run
git diff 50da047 -- pyproject.toml                           # expect: NO changes (zero new deps)
git add -f Dockerfile README.md tests/unit/test_typst_e2e.py
git commit -m "feat: typst E2E acceptance tests + Docker/README install docs (Phase T0 exit)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Self-Review Notes (performed at plan-writing time)

- **Spec coverage:** FR-1/2→Tasks 2+9; FR-3→no schema changes anywhere (verified: no task touches ast.py); FR-4/5→Task 6; FR-6→typed `TypstUnsupportedConstruct` exists (Task 6) and gates unknown backends (Task 9) — the multi-column opt-out flag from FR-6 is NOT implementable (the AST carries no column signal; deferred with open Q8, noted in Deferred); FR-7/8/9→Tasks 3-5 (row-header + artifact rows pinned by spike, superseding the PRD's open Q1/Q2); FR-10/11→Task 7 + hard-fail wiring in Task 9; FR-12 (QuestPDF retrofit)→deferred, PRD marks it "evaluate"; FR-13→Tasks 8-9. NFR-1→determinism double-run (Task 10; full byte-stability testing deferred to T1 comparison); NFR-2→`typst_timeout_s` (Task 2); NFR-3→Task 10 Dockerfile; NFR-4→stderr-verbatim errors (Task 6/9); NFR-5→argv-list subprocess (Task 6); NFR-6 (audit ledger)→**deferred**: the §7.1-item-G audit ledger does not exist yet in this repo; the backend identity IS recoverable from job error strings and config, and full ledger integration belongs to the corpus-orchestrator effort — recorded here so it isn't silently lost. §8 metrics: 2/3→Task 7; 4→Task 6 negative test; 5→Tasks 8-9; 9→Task 9; 1/7→fixture-scale only (corpus + latency benchmarks deferred with T0 items a-c); 6/8→human-gated, deferred.
- **Placeholder scan:** the Task 3 stubs are deliberate compile-order scaffolding replaced by Tasks 4-5 within this same plan (each has its full code in its own task) — not TBDs. Task 9's fixture patching includes explicit adjust-don't-weaken guidance keyed to the real import structure.
- **Type consistency:** `generate(request, *, asset_paths: dict[str, str])` used identically in Tasks 3/5/6; `TypstRenderer(binary_path, timeout_s).render(request) -> bytes` identical in Tasks 6/7/9/10; `verify(request, pdf_bytes) -> StructAssertReport(passed, mismatches)` identical in Tasks 7/9/10; error class names in Task 6 match the aliases imported in Task 9; config field names in Task 2 match `getattr` reads in Task 9.
