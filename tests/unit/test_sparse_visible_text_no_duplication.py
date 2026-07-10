"""Regression guard: fix_sparse_visible_text_structure must not duplicate a page.

Root cause (diagnosed 2026-07-10, structural gate e2e_gate_v1_full): on
already-tagged forms whose pages trip the sparse/garbled-text heuristic
(``has_form_cues`` + >=3 table cells), ``fix_sparse_visible_text_structure``
builds a *parallel* visible-text scaffold under new ``/Sect`` elements while only
downgrading the original ``P``/``H`` nodes to ``/Span`` (which a screen reader
still voices) and leaving the original tables/lists intact. The page's text
content therefore ends up in the tag tree twice, which the reading-order judge
correctly flags as ``page_order_backtracking`` + repeated-content, and the
heading judge flags as ``duplicate_heading``.

This test is marked ``xfail`` because the fix (either narrowing the candidate
trigger so it does not fire on well-tagged pages, or making the scaffold truly
*replace* rather than *append* the original semantic content) is a substantial
change to a 250-line function with real regression risk on the genuinely-garbled
pages the function exists to serve, and was intentionally deferred.

It is gated on a fixture PDF supplied via ``REMEDY_SPARSE_DUP_FIXTURE`` (an
already-tagged multi-page form input, e.g. the ``e49049…`` gate input) so no
binary is committed. When the fixture is present the test demonstrates the bug
(xfail); when the fix lands it will xpass and the marker should be removed.

Run locally with::

    REMEDY_SPARSE_DUP_FIXTURE=/path/to/input.pdf \
        .venv/bin/python -m pytest tests/unit/test_sparse_visible_text_no_duplication.py -q -rx
"""

from __future__ import annotations

import os
import shutil
from itertools import groupby
from pathlib import Path

import pikepdf
import pytest

from project_remedy.pdf_fixer import fix_sparse_visible_text_structure
from project_remedy.tag_tree_reader import read_tag_tree

_FIXTURE = os.environ.get("REMEDY_SPARSE_DUP_FIXTURE", "")


def _page_visits(nodes) -> list[int]:
    """Ordered list of distinct page runs a screen reader would traverse."""
    return [page for page, _group in groupby(node.page for node in nodes)]


@pytest.mark.skipif(not _FIXTURE, reason="set REMEDY_SPARSE_DUP_FIXTURE to an already-tagged form PDF")
@pytest.mark.xfail(reason="known duplication bug in fix_sparse_visible_text_structure; fix deferred", strict=False)
def test_sparse_scaffold_does_not_duplicate_pages(tmp_path: Path) -> None:
    src = Path(_FIXTURE)
    work = tmp_path / "work.pdf"
    shutil.copy(src, work)

    before = read_tag_tree(src)
    before_nodes = len(before.nodes)

    with pikepdf.open(work, allow_overwriting_input=True) as pdf:
        fix_sparse_visible_text_structure(pdf)
        pdf.save(work)

    after = read_tag_tree(work)

    # Invariant 1: node count must not roughly double.
    assert len(after.nodes) <= before_nodes * 1.25, (
        f"tag tree grew {before_nodes} -> {len(after.nodes)} nodes "
        "(content duplicated, not replaced)"
    )

    # Invariant 2: reading order must not revisit an earlier page after a later
    # one — i.e. each page appears in a single contiguous run.
    visits = _page_visits(after.nodes)
    assert len(visits) == len(set(visits)), (
        f"page reading order backtracks: {visits} (duplicated structure)"
    )
