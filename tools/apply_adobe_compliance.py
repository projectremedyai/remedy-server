#!/usr/bin/env python3
"""Apply the deterministic Adobe-compliance post-pass to a PDF or directory.

Thin CLI wrapper around ``project_remedy.adobe_compliance``. This is a local
operations helper (not part of the HTTP API): it clears the residual Acrobat /
PDF-UA checker failures — orphan alt text ("Associated with content"), alt text
that "Hides annotation", and table "Summary" shape — that the LLM + veraPDF
pipeline leaves behind, without re-running any model.

Examples
--------
    # fix every PDF in the remediated corpus, in place
    python tools/apply_adobe_compliance.py ~/code/lamc_district_forms/lamc_remediated/remediated_pdfs

    # see what *would* change, without touching any file
    python tools/apply_adobe_compliance.py ~/code/lamc_district_forms/lamc_remediated/remediated_pdfs --dry-run

    # one file
    python tools/apply_adobe_compliance.py path/to/doc.pdf

NOTE: this does NOT fix "Character encoding" failures — those are a font
ToUnicode problem requiring the font-rebuild tier, not a structural post-pass.
"""

from __future__ import annotations

from project_remedy.adobe_compliance import run_cli

if __name__ == "__main__":
    raise SystemExit(run_cli())
