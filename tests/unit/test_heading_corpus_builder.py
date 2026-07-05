from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import pytest


BUILDER_PATH = Path(__file__).resolve().parents[2] / "tools" / "finetune" / "build_heading_corpus.py"
SPEC = importlib.util.spec_from_file_location("build_heading_corpus", BUILDER_PATH)
builder = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules[SPEC.name] = builder
SPEC.loader.exec_module(builder)


@pytest.mark.skipif(
    not builder.PDF_ASSOCIATION_FIXTURE.exists(),
    reason="PDF Association fixture repo is not present in /tmp/remedy_heading_research",
)
def test_pdf_association_fixture_verifies_h1_through_h6_without_training_use():
    pages = builder.inspect_pdf_heading_pages(
        builder.PDF_ASSOCIATION_FIXTURE,
        doc_id="pdf_association_fixture",
        family="unit-fixture",
        provenance="unit-fixture-not-training",
        min_headings=1,
        extract_text=False,
    )

    counts = sum((Counter(page.heading_counts) for page in pages), Counter())
    for tag in builder.HEADING_TAGS:
        assert counts[tag] >= 1
    assert all(page.provenance == "unit-fixture-not-training" for page in pages)


@pytest.mark.skipif(
    not builder.VERAPDF_FIXTURE.exists(),
    reason="veraPDF corpus fixture repo is not present in /tmp/remedy_heading_research",
)
def test_verapdf_fixture_verifies_numbered_heading_tags_without_bulk_ingest():
    pages = builder.inspect_pdf_heading_pages(
        builder.VERAPDF_FIXTURE,
        doc_id="verapdf_fixture",
        family="unit-fixture",
        provenance="unit-fixture-not-training",
        min_headings=1,
        extract_text=False,
    )

    counts = sum((Counter(page.heading_counts) for page in pages), Counter())
    assert counts["H1"] >= 1
    assert counts["H2"] >= 1
    assert counts["H3"] >= 1
    assert counts["H4"] >= 1
    assert all(page.family == "unit-fixture" for page in pages)


def test_default_real_sources_exclude_fixture_corpora():
    urls = "\n".join(source["url"] for source in builder.DEFAULT_REAL_SOURCES)
    assert "pdf-association/techniques-for-accessible-pdf" not in urls
    assert "veraPDF-corpus" not in urls
