#!/usr/bin/env python3
"""TEAM A escalation tool: font-replacer tier on the 7.21.x-residue files.

Standalone escalation runner. For each target file whose veraPDF failures are
ALL in the 7.21.x family (per verapdf_sweep_results.json), it:

  1. copies the delivered original into a scratch results dir (never mutates
     the source),
  2. runs the built-but-unwired font-replacer tier behind the engine's own
     eligibility gates:
       - Phase 1 encoding repair (targets 7.21.6-3) on every file (safe
         metadata edit; no-op elsewhere),
       - Simple-font replacement for page-level non-CID Type1/TrueType fonts
         (targets 7.21.4.1-1),
       - Type0 / Mode-B replacement for CID fonts, applying the PUA per-font
         filter exactly as mode_b_production does,
  3. re-runs veraPDF and applies the DELTA GATE: the output is KEPT only if the
     set of failed 7.21.x clauses strictly shrank AND no NEW failed clause (any
     family) appeared. Otherwise the output is deleted and a skip reason is
     recorded.

Nothing in the existing engine is edited; all engine behavior is imported.

Usage:
    uv run python tools/escalate_font_residue.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from pathlib import Path

import pikepdf

# --- Engine imports (never edited; reused as-is) ---------------------------
from project_remedy.faithful_rebuild.simple_font_replacer import (
    check_simple_multifont_eligibility,
    repair_encoding_on_pdf,
)
from project_remedy.faithful_rebuild.simple_font_orchestrator import (
    SimpleMultiFontReplacer,
)
from project_remedy.faithful_rebuild.font_analysis import (
    check_multifont_eligibility_with_recovery,
)
from project_remedy.faithful_rebuild.multifont_replacer import MultiFontReplacer
from project_remedy.faithful_rebuild.pua_handler import should_skip_font_for_pua

# --- Fixed paths -----------------------------------------------------------
SCRATCH = Path(
    "/private/tmp/claude-503/-Users-laccd-Desktop-lamc-district-forms/"
    "57f9a73e-c2d0-47a7-80dd-bca4424a3609/scratchpad"
)
SWEEP_JSON = SCRATCH / "verapdf_sweep_results.json"
OUT_DIR = SCRATCH / "font_fixed"
REPORT_JSON = SCRATCH / "font_fixed_results.json"
SRC_DIR = Path("/Users/laccd/code/lamc_district_forms/lamc_remediated/remediated_pdfs")

VERAPDF_BIN = "/opt/homebrew/bin/verapdf"


# --- veraPDF: failed-clause extraction -------------------------------------
def verapdf_failed_clauses(pdf_path: Path) -> list[str]:
    """Return the sorted list of unique failed rule clauses for a PDF.

    A file PASSES iff the returned list is empty. Clauses are reported at the
    veraPDF ``clause`` granularity (e.g. ``7.21.4.1``), matching the sweep JSON.
    """
    proc = subprocess.run(
        [VERAPDF_BIN, "-f", "ua1", "--format", "xml", str(pdf_path)],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    xml_text = proc.stdout.strip()
    if not xml_text:
        raise RuntimeError(
            f"veraPDF returned no XML for {pdf_path.name}: "
            f"{proc.stderr.strip()[:400]}"
        )
    # The XML is produced by our own local veraPDF subprocess (trusted), but we
    # still parse with entity expansion disabled to defuse XXE / billion-laughs.
    parser = ET.XMLParser()
    try:  # CPython/expat: refuse DTDs and external/internal entity definitions
        parser.parser.DefaultHandler = lambda data: None
        parser.parser.EntityDeclHandler = lambda *a, **k: (_ for _ in ()).throw(
            ValueError("entity declarations are not allowed")
        )
    except AttributeError:  # pragma: no cover - non-expat backend
        pass
    root = ET.fromstring(xml_text, parser=parser)
    clauses: set[str] = set()
    # rule elements carry clause= and status="failed" (namespace-agnostic scan)
    for el in root.iter():
        tag = el.tag.rsplit("}", 1)[-1]
        if tag != "rule":
            continue
        if el.attrib.get("status") == "failed":
            clause = el.attrib.get("clause")
            if clause:
                clauses.add(clause)
    return sorted(clauses)


# --- font-replacer tier ----------------------------------------------------
def run_font_tier(in_path: Path, out_path: Path) -> tuple[bool, list[str]]:
    """Run the font-replacer tier on a COPY. Returns (mutated, notes).

    Applies, behind the engine's own eligibility gates:
      * Phase 1 encoding repair (7.21.6-3),
      * simple-font replacement (7.21.4.1-1, non-CID page fonts),
      * Type0/Mode-B replacement (CID fonts) with the PUA per-font filter.
    Saves to ``out_path`` only if something actually changed.
    """
    notes: list[str] = []
    with pikepdf.open(in_path) as pdf:
        # Phase 1: encoding repair (7.21.6-3). Safe metadata edit; no-op if
        # nothing qualifies.
        enc = repair_encoding_on_pdf(pdf)
        if enc.fonts_repaired:
            notes.append(f"encoding_repair: {enc.fonts_repaired} font(s) repaired")

        # Phase 2: simple-font replacement (7.21.4.1-1) for non-CID page fonts.
        simple_replaced = 0
        try:
            multi_simple = check_simple_multifont_eligibility(pdf)
            if multi_simple.qualifies_document:
                reports = SimpleMultiFontReplacer().replace_all(pdf, multi_simple)
                simple_replaced = sum(1 for r in reports if r.status == "replaced")
                if simple_replaced:
                    notes.append(f"simple_font: {simple_replaced} replaced")
                else:
                    skips = {r.reason for r in reports if r.status != "replaced"}
                    if skips:
                        notes.append("simple_font skips: " + "; ".join(sorted(s for s in skips if s)))
            else:
                if multi_simple.disqualifying_reasons:
                    notes.append(
                        "simple_font not-qualified: "
                        + "; ".join(multi_simple.disqualifying_reasons)
                    )
                else:
                    # surface per-font reasons for visibility
                    per = {
                        r
                        for e in multi_simple.font_eligibilities
                        for r in (e.disqualifying_reasons or [])
                    }
                    if per:
                        notes.append("simple_font eligibility: " + "; ".join(sorted(per)))
        except Exception as exc:  # noqa: BLE001 - record, continue to Mode B
            notes.append(f"simple_font ERROR: {exc}")

        # Phase 3: Type0 / Mode-B replacement (CID fonts). Apply the PUA
        # per-font filter exactly as mode_b_production does before replacing.
        mode_b_replaced = 0
        try:
            multi_cid = check_multifont_eligibility_with_recovery(pdf)
            # PUA pre-filter: disqualify fonts whose CID->Unicode map is PUA-
            # dominated or whose name looks like custom glyph naming.
            pua_reasons: list[str] = []
            for e in multi_cid.font_eligibilities:
                if not e.qualifies or e.font_object is None:
                    continue
                font_dict = (
                    e.font_object.get_object()
                    if hasattr(e.font_object, "get_object")
                    else e.font_object
                )
                skip, reason = should_skip_font_for_pua(e.cid_unicode_map, font_dict)
                if skip:
                    e.qualifies = False
                    e.disqualifying_reasons = [
                        *e.disqualifying_reasons,
                        f"PUA/custom-glyph — {reason}",
                    ]
                    pua_reasons.append(reason)
            if pua_reasons:
                notes.append("mode_b PUA-skipped: " + "; ".join(sorted(set(pua_reasons))))

            if multi_cid.qualifies_document:
                reports = MultiFontReplacer().replace_all(pdf, multi_cid)
                mode_b_replaced = sum(1 for r in reports if r.status == "replaced")
                if mode_b_replaced:
                    notes.append(f"mode_b: {mode_b_replaced} replaced")
                else:
                    skips = {r.reason for r in reports if r.status != "replaced"}
                    if skips:
                        notes.append("mode_b skips: " + "; ".join(sorted(s for s in skips if s)))
            else:
                if multi_cid.disqualifying_reasons:
                    notes.append(
                        "mode_b not-qualified: "
                        + "; ".join(multi_cid.disqualifying_reasons)
                    )
                else:
                    per = {
                        r
                        for e in multi_cid.font_eligibilities
                        for r in (e.disqualifying_reasons or [])
                    }
                    if per:
                        notes.append("mode_b eligibility: " + "; ".join(sorted(per)))
        except Exception as exc:  # noqa: BLE001 - record, continue
            notes.append(f"mode_b ERROR: {exc}")

        mutated = bool(enc.fonts_repaired) or simple_replaced > 0 or mode_b_replaced > 0
        if mutated:
            # Save to a sibling temp file (pikepdf refuses to overwrite the file
            # it currently has open), then atomically move it into place. This
            # supports out_path == in_path.
            tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
            pdf.save(tmp_path)
    if mutated:
        tmp_path.replace(out_path)
    return mutated, notes


# --- main ------------------------------------------------------------------
def compute_targets() -> list[dict]:
    data = json.loads(SWEEP_JSON.read_text())
    return [
        x
        for x in data
        if (not x.get("pass"))
        and x.get("clauses")
        and all(str(c).startswith("7.21") for c in x["clauses"])
    ]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = compute_targets()

    results: list[dict] = []
    counts = {"kept": 0, "discarded": 0, "errored": 0, "no_change": 0}
    errored_files: list[str] = []

    for t in targets:
        fname = t["file"]
        src = SRC_DIR / fname
        dst = OUT_DIR / fname
        entry = {
            "file": fname,
            "before": None,
            "after": None,
            "cleared": False,
            "reason": "",
        }

        if not src.exists():
            entry["reason"] = f"source not found: {src}"
            counts["errored"] += 1
            errored_files.append(fname)
            results.append(entry)
            continue

        try:
            # (1) copy to scratch
            shutil.copy2(src, dst)

            # before-state: measure on the copy for an apples-to-apples delta
            before = verapdf_failed_clauses(dst)
            entry["before"] = before

            # (2)+(3) run the font-replacer tier on the copy
            mutated, notes = run_font_tier(dst, dst)

            if not mutated:
                # nothing changed -> discard copy, record no-op
                after = before
                entry["after"] = after
                entry["reason"] = "no mutation (" + ("; ".join(notes) or "no eligible fonts") + ")"
                dst.unlink(missing_ok=True)
                counts["no_change"] += 1
                counts["discarded"] += 1
                results.append(entry)
                continue

            # (4) re-validate and apply DELTA GATE
            after = verapdf_failed_clauses(dst)
            entry["after"] = after

            before_721 = {c for c in before if c.startswith("7.21")}
            after_721 = {c for c in after if c.startswith("7.21")}
            new_clauses = set(after) - set(before)

            font_reduced = after_721 < before_721  # strict subset => dropped some
            no_regression = len(new_clauses) == 0

            note_str = "; ".join(notes)
            if font_reduced and no_regression:
                entry["cleared"] = True
                entry["reason"] = (
                    "KEPT: 7.21.x reduced "
                    f"{sorted(before_721)} -> {sorted(after_721)}"
                    + (f" [{note_str}]" if note_str else "")
                )
                counts["kept"] += 1
            else:
                # discard: no improvement or a new failed clause appeared
                if new_clauses:
                    entry["reason"] = f"introduced {sorted(new_clauses)}"
                elif not font_reduced:
                    entry["reason"] = (
                        "no 7.21.x reduction "
                        f"({sorted(before_721)} unchanged)"
                    )
                if note_str:
                    entry["reason"] += f" [{note_str}]"
                dst.unlink(missing_ok=True)
                counts["discarded"] += 1
            results.append(entry)

        except Exception as exc:  # noqa: BLE001
            entry["reason"] = f"ERROR: {exc}\n{traceback.format_exc()}"
            counts["errored"] += 1
            errored_files.append(fname)
            dst.unlink(missing_ok=True)
            results.append(entry)

    REPORT_JSON.write_text(json.dumps(results, indent=2))

    summary = {
        "targets": len(targets),
        "kept": counts["kept"],
        "discarded": counts["discarded"],
        "no_change_discards": counts["no_change"],
        "errored": counts["errored"],
        "errored_files": errored_files,
        "report": str(REPORT_JSON),
        "kept_files": [r["file"] for r in results if r["cleared"]],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
