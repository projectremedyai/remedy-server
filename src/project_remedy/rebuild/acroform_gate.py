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
