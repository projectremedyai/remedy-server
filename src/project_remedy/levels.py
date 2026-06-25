"""Remediation-Level classifier (L0–L5) — Phase 0 thin wrapper.

Assigns every document a machine-checkable remediation level by reading
objects the engine already produces (``PDFAcceptanceResult`` +
``compliance_report._determine_conformance``) plus a small catalog-level
structure probe. No engine code is modified.

See ``../../PHASE0_SPEC.md`` and ``../../RESEARCH_remedy_server_ADA_refactor.md`` §5.

Level ladder (machine-assignable L0–L4 only):

  L0  Inaccessible / image-only  — no extractable text layer
  L1  Text, untagged             — extractable text but no structure tree
  L2  Auto-tagged (unverified)   — tagged but fails the PDF/UA-1 machine gate
  L3  PDF/UA-1 machine-verified  — veraPDF clean + /Lang + DisplayDocTitle + UA id
  L4  WCAG 2.1 AA engine-complete — L3 + Conformant + quality layer passed
  L5  Human-validated            — NEVER assigned by this function (see invariant)

INVARIANT: ``classify_level`` must never return ``"L5"``. Per the research §4,
only human review can move a document to L5 / VPAT "Supports"; emitting an
"ADA compliant" verdict from automation is itself legal exposure (the *Payan*
"deliberate indifference" trap). L5 is assigned only by the future
human-review queue (Phase 4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pikepdf

from project_remedy.compliance_report import (
    Conformance,
    _LLM_HANDLED_CHECKS,
    _determine_conformance,
)
from project_remedy.pdf_acceptance import PDFAcceptanceResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConformanceProfile:
    """Pins the level gates + report vocabulary. Default = the District's bar."""

    name: str = "LACCD-DistrictUA1"
    require_uaid: bool = True
    require_display_doc_title: bool = True
    require_lang: bool = True
    wcag_target: str = "WCAG21AA"


LACCD_DISTRICT_UA1 = ConformanceProfile()


@dataclass(frozen=True)
class StructureProbe:
    """Catalog-level facts not already surfaced on ``PDFAcceptanceResult``."""

    has_text: bool
    has_struct_tree: bool       # /StructTreeRoot present
    is_marked: bool             # /MarkInfo /Marked == true
    has_lang: bool              # /Lang in catalog
    display_doc_title: bool     # /ViewerPreferences /DisplayDocTitle == true
    has_uaid: bool              # pdfuaid:part in XMP
    page_count: int


@dataclass(frozen=True)
class LevelResult:
    """Outcome of classifying one document."""

    level: str                       # "L0".."L4"  (NEVER "L5")
    machine_certifiable: bool        # True iff reached without human input
    sub_scores: dict[str, Any]
    blocking_conditions: list[str]   # why it isn't one level higher
    needs_human: list[str]           # conditions requiring human sign-off for L5
    profile: str


def probe_structure(pdf_path: Path) -> StructureProbe:
    """Read catalog-level accessibility facts from a PDF. Never raises."""
    has_text = False
    has_struct_tree = False
    is_marked = False
    has_lang = False
    display_doc_title = False
    has_uaid = False
    page_count = 0

    try:
        with pikepdf.open(pdf_path) as pdf:
            root = pdf.Root
            page_count = len(pdf.pages)
            has_struct_tree = "/StructTreeRoot" in root
            has_lang = "/Lang" in root

            mark_info = root.get("/MarkInfo")
            is_marked = bool(mark_info is not None and mark_info.get("/Marked"))

            view_prefs = root.get("/ViewerPreferences")
            if view_prefs is not None:
                display_doc_title = bool(view_prefs.get("/DisplayDocTitle"))

            try:
                meta = pdf.open_metadata()
                has_uaid = bool(meta.get("pdfuaid:part"))
            except Exception as exc:  # noqa: BLE001 - metadata is optional
                logger.debug("XMP read failed for %s: %s", pdf_path, exc)
    except Exception as exc:  # noqa: BLE001 - probe must never raise
        logger.warning("probe_structure failed for %s: %s", pdf_path, exc)
        return StructureProbe(
            has_text=False, has_struct_tree=False, is_marked=False,
            has_lang=False, display_doc_title=False, has_uaid=False, page_count=0,
        )

    has_text = _has_text_layer(pdf_path, page_count)

    return StructureProbe(
        has_text=has_text,
        has_struct_tree=has_struct_tree,
        is_marked=is_marked,
        has_lang=has_lang,
        display_doc_title=display_doc_title,
        has_uaid=has_uaid,
        page_count=page_count,
    )


def _has_text_layer(pdf_path: Path, page_count: int) -> bool:
    """True if any page has extractable text. Falls back conservatively."""
    try:
        import fitz  # pymupdf

        with fitz.open(pdf_path) as doc:
            return any(page.get_text().strip() for page in doc)
    except Exception as exc:  # noqa: BLE001
        logger.debug("text-layer probe via fitz failed for %s: %s", pdf_path, exc)
        # Conservative fallback: assume no text only when we truly can't tell.
        return False


def classify_level(
    acceptance: PDFAcceptanceResult | None,
    probe: StructureProbe,
    *,
    profile: ConformanceProfile = LACCD_DISTRICT_UA1,
) -> LevelResult:
    """Classify a document into L0–L4. Never returns L5 (invariant)."""

    sub_scores = _build_sub_scores(acceptance, probe)
    needs_human = _build_needs_human(acceptance)

    def result(level: str, blocking: list[str], *, machine: bool = True) -> LevelResult:
        # Defensive: the classifier must never emit L5.
        assert level in {"L0", "L1", "L2", "L3", "L4"}, f"illegal level {level!r}"
        return LevelResult(
            level=level,
            machine_certifiable=machine,
            sub_scores=sub_scores,
            blocking_conditions=blocking,
            needs_human=needs_human,
            profile=profile.name,
        )

    # --- Error / unopenable → L0 (not machine-certifiable) ----------------
    if acceptance is None:
        return result("L0", ["evaluation_error"], machine=False)
    if not acceptance.openable:
        return result("L0", ["not_openable"], machine=False)

    # --- L0 image-only, L1 untagged --------------------------------------
    # NOTE: the research's literal L0 gate ("no text OR (no struct AND no
    # marked)") makes L1 unreachable for every untagged-but-text PDF. Corrected
    # here: L0 = no text layer (image-only); L1 = text present but no structure.
    if not probe.has_text:
        return result("L0", ["no_text_layer"])
    if not probe.has_struct_tree:
        return result("L1", ["untagged"])

    # --- Tagged: decide L2 vs L3 (PDF/UA-1 machine gate) ------------------
    l3_blockers: list[str] = []
    if not (acceptance.verapdf_result.checked and acceptance.verapdf_result.passed):
        l3_blockers.append("verapdf_failed")
    if profile.require_lang and not probe.has_lang:
        l3_blockers.append("missing_lang")
    if profile.require_display_doc_title and not probe.display_doc_title:
        l3_blockers.append("missing_display_doc_title")
    if profile.require_uaid and not probe.has_uaid:
        l3_blockers.append("missing_uaid")

    if l3_blockers:
        return result("L2", l3_blockers)

    # --- L3 reached; decide L4 (WCAG AA engine-complete) ------------------
    l4_blockers: list[str] = []
    if _determine_conformance(acceptance) != Conformance.CONFORMANT:
        l4_blockers.append("not_conformant")
    if acceptance.quality_result is None:
        l4_blockers.append("quality_layer_not_run")
    elif not acceptance.quality_result.overall_pass:
        l4_blockers.append("quality_failed")

    if l4_blockers:
        return result("L3", l4_blockers)
    return result("L4", [])


def _build_sub_scores(
    acceptance: PDFAcceptanceResult | None, probe: StructureProbe
) -> dict[str, Any]:
    if acceptance is None:
        return {
            "verapdf_violations": 0,
            "blocking_checker_failures": 0,
            "sr_errors": 0,
            "alt_coverage": None,
            "contrast_violations": None,
            "page_count": probe.page_count,
        }
    verapdf_violations = (
        len(acceptance.verapdf_result.violations)
        if acceptance.verapdf_result.checked
        else 0
    )
    return {
        "verapdf_violations": verapdf_violations,
        "blocking_checker_failures": len(acceptance.checker_failures),
        "sr_errors": len(acceptance.screen_reader_errors),
        "alt_coverage": _alt_coverage(acceptance),
        # Contrast is not tracked on PDFAcceptanceResult; None = not evaluated.
        "contrast_violations": None,
        "page_count": probe.page_count,
    }


def _alt_coverage(acceptance: PDFAcceptanceResult) -> float | None:
    """Percent of Figure nodes carrying non-empty alt text, or None if none."""
    nodes = getattr(acceptance.tag_tree_result.tag_tree, "nodes", []) or []
    figures = [n for n in nodes if getattr(n, "tag", "") == "Figure"]
    if not figures:
        return None
    with_alt = sum(1 for n in figures if (getattr(n, "alt_text", "") or "").strip())
    return round(100.0 * with_alt / len(figures), 1)


def summarize_levels(
    records: list[dict[str, Any]],
    *,
    vision_enabled: bool,
    generated_at: str,
    profile_name: str = LACCD_DISTRICT_UA1.name,
) -> dict[str, Any]:
    """Aggregate per-document level records into a burndown summary.

    Pure: ``generated_at`` is passed in (the corpus tool stamps it) rather than
    read from the clock here, so the function stays deterministic/testable.
    """
    by_root: dict[str, dict[str, int]] = {}
    totals: dict[str, int] = {}
    needs_human_total = 0
    for record in records:
        root = record.get("root") or "(unknown)"
        level = record.get("level") or "unclassified"
        by_root.setdefault(root, {})
        by_root[root][level] = by_root[root].get(level, 0) + 1
        totals[level] = totals.get(level, 0) + 1
        needs_human_total += len(record.get("needs_human") or [])
    return {
        "profile": profile_name,
        "vision_enabled": vision_enabled,
        "generated_at": generated_at,
        "by_root": by_root,
        "totals": totals,
        "needs_human_total": needs_human_total,
    }


def _build_needs_human(acceptance: PDFAcceptanceResult | None) -> list[str]:
    """Checker conditions flagged 'Manual Check Needed' that the LLM can't handle."""
    if acceptance is None:
        return []
    return [
        r.rule_id
        for r in acceptance.checker_report.results
        if r.status == "Manual Check Needed" and r.rule_id not in _LLM_HANDLED_CHECKS
    ]
