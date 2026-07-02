"""Post-generation struct-tree assertion pass (PRD_typst_backend.md §5.4).

Independently verifies that the compiled PDF's struct tree round-trips the
input RebuildRequest — the gate veraPDF cannot provide (Caveat 2: veraPDF
validates tag well-formedness, not semantic correctness of authoring). A
failure here is a GENERATOR BUG by definition (FR-11): the caller must
hard-fail the job, never degrade silently.

Struct shapes below are pinned against .frugal-fable/typst-spike/FINDINGS.md:
- §6 Lists: a nested list's /L is a SIBLING of the parent /LI (not inside
  /LBody). The walk below is tag-count based and traverses whatever /K shape
  is actually present, so it does not need to model that positioning itself
  — it only needs to recurse into nested ListBlocks in the AST (which it
  does) so the expected L/LI counts include nested lists too.
- §1/§7: table.header(...) rows wrap in /THead//TR//TH, body rows in
  /TBody//TR//TD — TH count is checked, not the THead/TBody wrapper shape
  itself (the wrapper is not part of FR-10's contract).
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
