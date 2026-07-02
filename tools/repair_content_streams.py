#!/usr/bin/env python3
"""Repair unbalanced BT/ET in PDF page content streams (rendering fix).

Thin CLI wrapper around ``project_remedy.content_stream_repair``. The remediation
engine's tag injection leaves text objects unbalanced (dropped ``ET``/``BT``),
which renders in Preview/Quartz but fails in Acrobat/Ghostscript. This re-inserts
the missing ``BT``/``ET`` without touching text, fonts, marked content, or the
struct tree (PDF/UA-1 compliance preserved). Idempotent.

Examples
--------
    # fix every PDF in the remediated corpus, in place
    python tools/repair_content_streams.py ~/code/lamc_district_forms/lamc_remediated/remediated_pdfs

    # see how many files / ops would change, without writing
    python tools/repair_content_streams.py ~/code/lamc_district_forms/lamc_remediated/remediated_pdfs --dry-run
"""

from __future__ import annotations

from project_remedy.content_stream_repair import run_cli

if __name__ == "__main__":
    raise SystemExit(run_cli())
