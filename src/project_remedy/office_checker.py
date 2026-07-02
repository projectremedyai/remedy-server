"""office-verify deterministic rule engine (PRD_ooxml_a11y_validator.md §4).

Each rule is a pure function ``(DocxContext) -> OfficeCheckResult`` registered
in ``DOCX_RULES`` under its canonical rule id. Determinism contract (NFR1):
no network, no LLM, no clock — same input bytes, same report.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from project_remedy.models import FileType
from project_remedy.office_acceptance import (
    OfficeCheckReport,
    OfficeCheckResult,
    _docx_outline_level,
    _docx_paragraph_has_heading_structure,
    _infer_file_type,
)
from project_remedy.office_rules import NS, RULE_CATALOG, RULE_SPECS_BY_ID, qn_w


@dataclass
class DocxContext:
    """Everything a docx rule may read, loaded exactly once per document."""

    path: Path
    document: Any                 # python-docx Document
    body_root: ET.Element         # parsed word/document.xml

    @classmethod
    def load(cls, path: Path) -> "DocxContext":
        from docx import Document

        with zipfile.ZipFile(path) as zf:
            body_root = ET.fromstring(zf.read("word/document.xml"))
        return cls(path=Path(path), document=Document(str(path)), body_root=body_root)


DOCX_RULES: dict[str, Callable[[DocxContext], OfficeCheckResult]] = {}


def docx_rule(rule_id: str):
    def wrap(fn: Callable[[DocxContext], OfficeCheckResult]):
        DOCX_RULES[rule_id] = fn
        return fn
    return wrap


def _make_result(rule_id: str, *, flagged: bool, details: list[str]) -> OfficeCheckResult:
    spec = RULE_SPECS_BY_ID[rule_id]
    return OfficeCheckResult(
        rule_id=spec.emitted_id,
        description=spec.description,
        status=spec.flag_status if flagged else "Passed",
        details=details if flagged else [],
        fixable=spec.fixable,
        checkpoint=spec.checkpoint,
        wcag_ref=spec.wcag_ref,
    )


# --- Checkpoint 1: document metadata ---------------------------------------

@docx_rule("OOXML-DOCX-1.1")
def rule_docx_title(ctx: DocxContext) -> OfficeCheckResult:
    title = (ctx.document.core_properties.title or "").strip()
    return _make_result("OOXML-DOCX-1.1", flagged=not title,
                        details=["docProps/core.xml dc:title is empty"])


@docx_rule("OOXML-DOCX-1.2")
def rule_docx_language(ctx: DocxContext) -> OfficeCheckResult:
    language = (getattr(ctx.document.core_properties, "language", "") or "").strip()
    return _make_result("OOXML-DOCX-1.2", flagged=not language,
                        details=["docProps/core.xml dc:language is empty"])


# --- Checkpoint 2: heading structure ----------------------------------------

_HEADING_STYLE_RE = re.compile(r"^(?:accessibility )?heading\s*(\d+)?", re.IGNORECASE)


def _heading_level(paragraph: Any) -> int | None:
    """1-based heading level, or None if the paragraph is not a heading.

    Title (and Accessibility Title) count as level 1; "Heading N" is level N;
    a bare w:outlineLvl value v maps to level v+1.
    """
    style_name = (getattr(getattr(paragraph, "style", None), "name", "") or "").strip()
    lowered = style_name.lower()
    if lowered.startswith(("title", "accessibility title")):
        return 1
    match = _HEADING_STYLE_RE.match(lowered)
    if match:
        return int(match.group(1) or 1)
    outline = _docx_outline_level(paragraph)
    if outline is not None:
        return outline + 1
    return None


@docx_rule("OOXML-DOCX-2.1")
def rule_docx_headings_present(ctx: DocxContext) -> OfficeCheckResult:
    has_heading = any(_docx_paragraph_has_heading_structure(p) for p in ctx.document.paragraphs)
    return _make_result("OOXML-DOCX-2.1", flagged=not has_heading,
                        details=["no paragraph carries a Heading/Title style or w:outlineLvl"])


@docx_rule("OOXML-DOCX-2.2")
def rule_docx_heading_skips(ctx: DocxContext) -> OfficeCheckResult:
    skips: list[str] = []
    previous = 0  # virtual document root: the first heading must be level 1
    for index, paragraph in enumerate(ctx.document.paragraphs):
        level = _heading_level(paragraph)
        if level is None:
            continue
        if level > previous + 1:
            snippet = paragraph.text.strip()[:48]
            skips.append(f"paragraph {index}: heading level jumps {previous} -> {level} ('{snippet}')")
        previous = level
    return _make_result("OOXML-DOCX-2.2", flagged=bool(skips), details=skips)


@docx_rule("OOXML-DOCX-2.3")
def rule_docx_no_orphan_intro(ctx: DocxContext) -> OfficeCheckResult:
    first = next((p for p in ctx.document.paragraphs if p.text.strip()), None)
    if first is None:
        return _make_result("OOXML-DOCX-2.3", flagged=False, details=[])
    flagged = not _docx_paragraph_has_heading_structure(first)
    return _make_result(
        "OOXML-DOCX-2.3", flagged=flagged,
        details=[f"document opens on body paragraph: '{first.text.strip()[:64]}'"],
    )
