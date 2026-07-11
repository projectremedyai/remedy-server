"""Feedback-driven heading retag: apply what the acceptance checker detected.

The acceptance checker's vision pass emits headings-nesting failures like
``Page 5: title/section heading is tagged as body text (P -> H1) (Retag as H1)``
but the fixer bound to that rule (fix_heading_nesting) only renumbers existing
H1-H6 nodes, and the real retag pass (fix_heading_hierarchy_quality) samples
its own pages and is skipped for large docs — so flagged files never get the
retag applied. These tests drive:

1. ``force_pages`` on fix_heading_hierarchy_quality — analyze exactly the
   checker-flagged pages instead of sampling.
2. A node-aware safe-retag guard: Figure -> H* is allowed only when the node
   carries speakable text (ActualText or extractable marked content), never
   for a pure image.
3. ``heading_retag_pages_from_failures`` — parse checker failures into the
   0-based page list the targeted refix should run on.
"""
from __future__ import annotations

from types import SimpleNamespace

import pikepdf
from pikepdf import Array, Dictionary, Name

import project_remedy.pdf_fixer as PF
from project_remedy.pdf_vision import HeadingIssue


def _doc(content_bytes: bytes, specs):
    """1-page PDF: StructTreeRoot -> Document -> leaf structure elements."""
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    pg = pdf.pages[0].obj
    pg.Contents = pdf.make_stream(content_bytes)
    elems = []
    for s, mcid in specs:
        elems.append(pdf.make_indirect(Dictionary(
            Type=Name("/StructElem"), S=Name(s), Pg=pg, K=mcid)))
    doc = pdf.make_indirect(Dictionary(
        Type=Name("/StructElem"), S=Name("/Document"), K=Array(elems)))
    for e in elems:
        e.P = doc
    pdf.Root.StructTreeRoot = pdf.make_indirect(
        Dictionary(Type=Name("/StructTreeRoot"), K=Array([doc])))
    pdf.Root.MarkInfo = Dictionary(Marked=True)
    return pdf, elems


CONTENT_TITLE_AND_BODY = (
    b"/P <</MCID 0>> BDC\n"
    b"BT /F1 24 Tf 72 720 Td (Annual Program Review) Tj ET\n"
    b"EMC\n"
    b"/P <</MCID 1>> BDC\n"
    b"BT /F1 11 Tf 72 690 Td (This report summarizes the year.) Tj ET\n"
    b"EMC\n"
)


def _issue_for(pdf, node, current_tag, correct_tag="H1"):
    """Build a vision HeadingIssue whose element_index points at ``node``."""
    nodes = PF._page_structure_nodes_for_vision_order(pdf, 0)
    idx = next(i for i, n in enumerate(nodes) if n.objgen == node.objgen)
    return HeadingIssue(
        page=1,
        description="title/section heading is tagged as body text",
        severity="error",
        suggestion=f"Retag as {correct_tag}",
        element_index=idx + 1,
        current_tag=current_tag,
        correct_tag=correct_tag,
        text="Annual Program Review",
    )


def _patch_vision(monkeypatch, issues, captured):
    """Stub the vision round-trip: capture the pages kwarg, return canned issues."""
    import project_remedy.pdf_vision as PV

    monkeypatch.setattr(
        PV, "VisionAnalyzer",
        lambda provider: SimpleNamespace(analyze_heading_hierarchy=lambda *a, **k: None),
    )

    def fake_blocking(func, *args, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(heading_issues=issues)

    monkeypatch.setattr(PF, "_run_async_callable_blocking", fake_blocking)


def test_force_pages_analyzes_flagged_pages_and_retags(monkeypatch):
    pdf, elems = _doc(CONTENT_TITLE_AND_BODY, [("/P", 0), ("/P", 1)])
    captured: dict = {}
    _patch_vision(monkeypatch, [_issue_for(pdf, elems[0], "P")], captured)

    changes = PF.fix_heading_hierarchy_quality(
        pdf, vision_provider=object(), force_pages=[0])

    # The analyzer API (render_page_to_image, _get_page_structure_order) is
    # 1-based; passing 0-based indexes makes it analyze the WRONG pages and
    # lose the struct-order context ("(invalid page number)" for 0).
    assert captured.get("pages") == [1], \
        "flagged 0-based pages must be converted to the analyzer's 1-based numbering"
    assert PF._get_struct_type(elems[0]) == "H1", "flagged P node must be retagged H1"
    assert PF._get_struct_type(elems[1]) == "P", "body text must be untouched"
    assert changes, "retag must be reported as a change"


CONTENT_FIGURE_TITLE = (
    b"/Figure <</MCID 0>> BDC\n"
    b"BT /F1 24 Tf 72 720 Td (2026 Course Catalog) Tj ET\n"
    b"EMC\n"
)


def test_textful_figure_is_retagged_to_heading(monkeypatch):
    pdf, elems = _doc(CONTENT_FIGURE_TITLE, [("/Figure", 0)])
    captured: dict = {}
    _patch_vision(monkeypatch, [_issue_for(pdf, elems[0], "Figure")], captured)
    monkeypatch.setattr(PF, "_extract_mcid_text",
                        lambda page: {0: "2026 Course Catalog"})

    PF.fix_heading_hierarchy_quality(pdf, vision_provider=object(), force_pages=[0])

    assert PF._get_struct_type(elems[0]) == "H1", \
        "Figure wrapping real title text must be retagged H1"


CONTENT_FIGURE_IMAGE = (
    b"/Figure <</MCID 0>> BDC\n"
    b"q 100 0 0 50 72 700 cm /Im0 Do Q\n"
    b"EMC\n"
)


def test_image_only_figure_is_not_retagged(monkeypatch):
    pdf, elems = _doc(CONTENT_FIGURE_IMAGE, [("/Figure", 0)])
    elems[0].Alt = pikepdf.String("Campus photo")  # Alt alone must NOT qualify
    captured: dict = {}
    _patch_vision(monkeypatch, [_issue_for(pdf, elems[0], "Figure")], captured)
    monkeypatch.setattr(PF, "_extract_mcid_text", lambda page: {0: ""})

    PF.fix_heading_hierarchy_quality(pdf, vision_provider=object(), force_pages=[0])

    assert PF._get_struct_type(elems[0]) == "Figure", \
        "pure-image Figure must never become a heading"


def test_heading_retag_pages_from_failures_parses_vision_details():
    failures = [
        {
            "rule_id": "headings-nesting",
            "details": [
                "Page 5: title/section heading is tagged as body text (P -> H1) (Retag as H1)",
                "Page 12: title/section heading is tagged as Figure (? -> H1) (Retag as H1)",
                "First heading is H2, expected H1",   # deterministic detail: no page info
                "Page 5: another issue on the same page (P -> H2) (Retag as H2)",
            ],
        },
        {"rule_id": "page-char-encoding",
         "details": ["Page 8: suspicious extracted text"]},
    ]
    assert PF.heading_retag_pages_from_failures(failures) == [4, 11]

    # Also accepts objects with attributes (CheckResult-style), and returns []
    # when only deterministic ordering details are present.
    objs = [SimpleNamespace(rule_id="headings-nesting",
                            details=["Skipped from H1 to H3"])]
    assert PF.heading_retag_pages_from_failures(objs) == []


def test_apply_heading_retag_refix_fixes_file_in_place(tmp_path, monkeypatch):
    pdf, elems = _doc(CONTENT_TITLE_AND_BODY, [("/P", 0), ("/P", 1)])
    issue = _issue_for(pdf, elems[0], "P")
    path = tmp_path / "flagged.pdf"
    pdf.save(path)

    captured: dict = {}
    _patch_vision(monkeypatch, [issue], captured)
    failures = [SimpleNamespace(
        rule_id="headings-nesting",
        details=["Page 1: title/section heading is tagged as body text"
                 " (P -> H1) (Retag as H1)"])]

    changes = PF.apply_heading_retag_refix(
        path, vision_provider=object(), checker_failures=failures)

    assert changes, "must report the applied retag"
    assert captured.get("pages") == [1], "analyzer receives 1-based page numbers"
    with pikepdf.open(path) as fixed:
        types = [PF._get_struct_type(n)
                 for n, _d, _p in PF.walk_structure_tree(fixed)]
    assert "H1" in types, "saved file must contain the retagged heading"


CONTENT_TABLE_AND_TITLE = (
    b"/P <</MCID 0>> BDC\n"
    b"BT /F1 20 Tf 72 730 Td (Federal Data) Tj ET\n"
    b"EMC\n"
    b"/TD <</MCID 1>> BDC\n"
    b"BT /F1 10 Tf 72 700 Td (cell value) Tj ET\n"
    b"EMC\n"
)


def test_misaligned_element_index_falls_back_to_text_match(monkeypatch):
    """The vision model numbers visual elements, not struct nodes — a wrong
    element_index must not silently drop (or worse, retag) the wrong node.
    When the indexed node doesn't match the issue's claimed tag/text, the
    fixer must locate the node by current_tag + text (MCID-aware)."""
    pdf, elems = _doc(CONTENT_TABLE_AND_TITLE, [("/P", 0), ("/TD", 1)])
    nodes = PF._page_structure_nodes_for_vision_order(pdf, 0)
    td_index = next(i for i, n in enumerate(nodes)
                    if n.objgen == elems[1].objgen) + 1  # 1-based, points at TD

    issue = HeadingIssue(
        page=1, description="title/section heading is tagged as body text",
        severity="error", suggestion="Retag as H1",
        element_index=td_index,          # WRONG: model's numbering, lands on TD
        current_tag="P", correct_tag="H1",
        text="Federal Data # Question Answer",  # model text is a superset
    )
    captured: dict = {}
    _patch_vision(monkeypatch, [issue], captured)
    monkeypatch.setattr(PF, "_extract_mcid_text",
                        lambda page: {0: "Federal Data", 1: "cell value"})

    PF.fix_heading_hierarchy_quality(pdf, vision_provider=object(), force_pages=[0])

    assert PF._get_struct_type(elems[0]) == "H1", \
        "must find the P node by text and retag it"
    assert PF._get_struct_type(elems[1]) == "TD", "the mis-indexed TD must be untouched"


def test_misaligned_index_without_text_match_is_skipped(monkeypatch):
    pdf, elems = _doc(CONTENT_TABLE_AND_TITLE, [("/P", 0), ("/TD", 1)])
    nodes = PF._page_structure_nodes_for_vision_order(pdf, 0)
    td_index = next(i for i, n in enumerate(nodes)
                    if n.objgen == elems[1].objgen) + 1

    issue = HeadingIssue(
        page=1, description="title/section heading is tagged as body text",
        severity="error", suggestion="Retag as H1",
        element_index=td_index, current_tag="P", correct_tag="H1",
        text="Completely Different Title",   # matches nothing on the page
    )
    captured: dict = {}
    _patch_vision(monkeypatch, [issue], captured)
    monkeypatch.setattr(PF, "_extract_mcid_text",
                        lambda page: {0: "Federal Data", 1: "cell value"})

    PF.fix_heading_hierarchy_quality(pdf, vision_provider=object(), force_pages=[0])

    assert PF._get_struct_type(elems[0]) == "P", "no verified target -> no retag"
    assert PF._get_struct_type(elems[1]) == "TD"


CONTENT_SECT_TITLE = (
    b"/P <</MCID 0>> BDC\n"
    b"BT /F1 20 Tf 72 730 Td (LAMC CRIME STATS- MAR 2010) Tj ET\n"
    b"EMC\n"
)


def test_container_indexed_issue_rescued_via_guard_passable_text_leaf(monkeypatch):
    """The model often indexes the CONTAINER (Sect) that holds a title, with
    an empty current_tag. Retagging a Sect would swallow its content — the
    guard refuses — but the fix must then rescue by finding the guard-passable
    text leaf (the P actually carrying the title) instead of giving up."""
    pdf = pikepdf.Pdf.new()
    pdf.add_blank_page(page_size=(612, 792))
    pg = pdf.pages[0].obj
    pg.Contents = pdf.make_stream(CONTENT_SECT_TITLE)
    p_node = pdf.make_indirect(Dictionary(
        Type=Name("/StructElem"), S=Name("/P"), Pg=pg, K=0))
    sect = pdf.make_indirect(Dictionary(
        Type=Name("/StructElem"), S=Name("/Sect"), Pg=pg, K=Array([p_node])))
    p_node.P = sect
    doc = pdf.make_indirect(Dictionary(
        Type=Name("/StructElem"), S=Name("/Document"), K=Array([sect])))
    sect.P = doc
    pdf.Root.StructTreeRoot = pdf.make_indirect(
        Dictionary(Type=Name("/StructTreeRoot"), K=Array([doc])))
    pdf.Root.MarkInfo = Dictionary(Marked=True)

    nodes = PF._page_structure_nodes_for_vision_order(pdf, 0)
    sect_index = next(i for i, n in enumerate(nodes)
                      if n.objgen == sect.objgen) + 1

    issue = HeadingIssue(
        page=1, description="page title not tagged as heading",
        severity="error", suggestion="Retag as H1",
        element_index=sect_index, current_tag="", correct_tag="H1",
        text="LAMC CRIME STATS- MAR 2010 WEEK ENDING: ",
    )
    captured: dict = {}
    _patch_vision(monkeypatch, [issue], captured)
    monkeypatch.setattr(PF, "_extract_mcid_text",
                        lambda page: {0: "LAMC CRIME STATS- MAR 2010"})

    PF.fix_heading_hierarchy_quality(pdf, vision_provider=object(), force_pages=[0])

    assert PF._get_struct_type(p_node) == "H1", \
        "the P text leaf must be promoted instead of the refused Sect"
    assert PF._get_struct_type(sect) == "Sect", "the Sect container must be untouched"


def test_apply_heading_retag_refix_no_pages_is_noop(tmp_path):
    pdf, _elems = _doc(CONTENT_TITLE_AND_BODY, [("/P", 0), ("/P", 1)])
    path = tmp_path / "ok.pdf"
    pdf.save(path)
    before = path.read_bytes()

    failures = [SimpleNamespace(rule_id="headings-nesting",
                                details=["First heading is H2, expected H1"])]
    changes = PF.apply_heading_retag_refix(
        path, vision_provider=object(), checker_failures=failures)

    assert changes == []
    assert path.read_bytes() == before, "file must be untouched when no vision-flagged pages"
