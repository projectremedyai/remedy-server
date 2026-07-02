# Typst backend: spike-pinned design decisions

Distilled from `.frugal-fable/typst-spike/FINDINGS.md` (empirical spike against
Typst 0.15.0, `--pdf-standard ua-1`). Each decision below is binding for Phase
T0 codegen; see the FINDINGS section referenced for full struct dumps.

## 1. Row-header cells degrade to plain TD (FINDINGS §2)

`table.cell(kind: "rowhead")` does not exist (exit 1: "unexpected argument:
kind"). The real mechanism, `pdf.header-cell(scope: "row"|"column"|"both")`,
compiles successfully (exit 0, produces `TH Scope=/Row`) but requires the
unstable `--features a11y-extras` flag ("temporary API... may be changed or
removed at any time" — Typst's own docs string). Plain first-column cells
with no marking tag as ordinary `TD` (exp2c), confirming there is no implicit
row-header inference. **Decision:** `TableCell.header == "row"` degrades to
plain TD in Phase T0. `"col"`/`"both"` cells in the first row map to
`table.header(...)`; the same header value on a non-first row also degrades
to TD, since `table.header` is row-scoped.

## 2. `ArtifactBlock` maps to `pdf.artifact[#image(...)]` (FINDINGS §3)

`#pdf.artifact(image("p.png"))` and `#pdf.artifact[#image("p.png")]` both
compile (exit 0) and produce **no** `/Figure` struct entry — verified via raw
content-stream dump showing `/Artifact BMC ... /x0 Do ... EMC` wrapping the
draw ops, exactly PDF/UA's artifact semantics. Bare `#image(...)` with no alt
text fails compile (exit 1, "missing alt text"), and critically
`alt: ""` (empty string) **also** fails — Typst treats empty string as
absent, not present-but-empty. **Decision:** `ArtifactBlock` emits
`#pdf.artifact[#image("<path>")]`; there is no alt-text escape hatch for
non-artifact images — every `FigureBlock` needs a real, non-empty alt.

## 3. Escape set + forced `smartquote(enabled: false)` (FINDINGS §4)

Escaping all markup-special chars except `=`, `+`, `-`, `"`, `'` round-tripped
correctly except quotes: default smartquote conversion silently curls `"`/`'`
even with correct escaping elsewhere. Fix: `#set smartquote(enabled: false)`
in the preamble, confirmed byte-exact after. Separately, unescaped `//` is a
**silent** line-comment truncation (not a compile error) that drops
everything after it — `\/` is the valid, necessary escape and should be
applied unconditionally rather than only at detected adjacency.
**Decision:** minimal escape set = `\ # $ * _ @ [ ] < > ~ ` ` (backtick) and
`/` (always as `\/`); preamble always sets
`#set smartquote(enabled: false)`.

## 4. `/Alt` is byte-exact (FINDINGS §7)

`figure(image(..., alt: "A tiny dot"), caption: [...])` produces
`/Figure Alt="A tiny dot"`; reading the raw `/Alt` bytes via pikepdf gives
`b'A tiny dot'` — plain ASCII, no PDFDocEncoding BOM or escaping artifacts,
an exact match to the source string. (Struct order note, not part of this
decision: `/Caption` is emitted before `/Figure` regardless of argument
order in source.) **Decision:** `struct_assert` compares `/Alt` with strict
byte equality against `FigureBlock.alt` — no normalization, no fuzzy match.

## 5. Nested lists: sibling `/L`, not child of `/LBody` (FINDINGS §6)

For a bulleted list with a nested sub-list under one item, the struct tree
puts the nested `/L` as a direct **sibling** of the parent `/L`'s `/LI`
elements (immediately after the `/LI` whose text precedes the sub-items) —
it is NOT nested inside that `/LI`'s `/LBody`, contrary to the naive
HTML-nesting mental model. Confirmed for both unordered (`-`) and ordered
(`+`) lists, which both produce `L/LI/Lbl/LBody`. **Decision:**
`struct_assert` mirrors the flat-sibling shape (counts `L`/`LI` at the
correct nesting sibling position) rather than assuming
`L > LI > LBody > L`.
