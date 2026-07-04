"""_MARKED_CONTENT_TOKEN_RE must tokenize named-property openers as one opener.

`/Tag /Name BDC` (a marked-content opener whose properties are a named resource,
e.g. `/PlacedPDF /MC0 BDC`) was mis-parsed as `/MC0 BDC` (tag=MC0), leaving the
real tag stranded. When such a token is artifacted this produced malformed
`/PlacedPDF /Artifact BMC`, which veraPDF reads as an Artifact scope (tag = last
operand) around tagged content = 7.1-2. The tokenizer must consume the whole
opener so downstream rewrites stay well-formed.
"""
from __future__ import annotations

from project_remedy.pdf_fixer import (
    _MARKED_CONTENT_TOKEN_RE,
    _artifactize_unlinked_marked_content_without_mcids,
)


def test_named_property_opener_is_one_token():
    m = _MARKED_CONTENT_TOKEN_RE.search("/PlacedPDF /MC0 BDC")
    assert m.group("tag") == "PlacedPDF"
    assert m.group("props") == "/MC0"
    assert m.group("op") == "BDC"


def test_inline_dict_and_bare_openers_unchanged():
    m = _MARKED_CONTENT_TOKEN_RE.search("/P <</MCID 5>> BDC")
    assert m.group("tag") == "P" and "/MCID 5" in m.group("props")
    m2 = _MARKED_CONTENT_TOKEN_RE.search("/Span BDC")
    assert m2.group("tag") == "Span" and m2.group("props") is None
    m3 = _MARKED_CONTENT_TOKEN_RE.search("/Artifact BMC")
    assert m3.group("tag") == "Artifact" and m3.group("op") == "BMC"


def test_artifactize_named_prop_leaf_is_wellformed():
    # a leaf placed-content group with no tagged parent -> clean /Artifact BMC.
    text = "/PlacedPDF /MC0 BDC (x) Tj EMC"
    out, n = _artifactize_unlinked_marked_content_without_mcids(text)
    assert n == 1
    assert "/Artifact BMC" in out
    assert "/PlacedPDF" not in out  # no stranded tag left behind
