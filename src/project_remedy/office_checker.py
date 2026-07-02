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
