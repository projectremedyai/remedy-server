"""Standalone OCG-config (PDF/UA-1 clause 7.10-1) fixer + delta gate.

Team B / IMPLEMENT.

For each target file (failed clauses include 7.10 per the veraPDF sweep JSON):
  1. Copy the delivered original to a scratch results dir (originals never touched).
  2. Apply the minimal /OCProperties fix by reusing the engine helper
     ``fix_optional_content_config_names`` (sets /Name on the /D default config
     dict and on every /Configs entry that lacks a non-empty /Name). Nothing
     else is touched -- /OCGs, /Order, /OFF, /RBGroups, /AS are left as-is
     because clause 7.10-1 only requires the Name key.
  3. Run veraPDF (ua1) before and after; parse the set of failed rule clauses.
  4. DELTA GATE: keep the fixed copy only if 7.10 cleared AND no new failed
     clause was introduced vs the file's own before-state. Otherwise delete the
     copy and record the reason.

"cleared" means the file now PASSES PDF/UA-1 entirely (zero failed clauses).
If 7.10 is fixed but residual font clauses (7.21.x) remain, the file is a real
partial win: kept, but cleared=False with a descriptive reason.

This module ONLY creates new files (this tool + scratch outputs). It does not
edit any existing engine source; it imports and reuses engine functions.

Run from the repo root via the project venv:
    uv run python tools/fix_ocg_config.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pikepdf

from project_remedy.pdf_fixer import fix_optional_content_config_names

# --- fixed paths -----------------------------------------------------------
SCRATCH = Path(
    "/private/tmp/claude-503/-Users-laccd-Desktop-lamc-district-forms/"
    "57f9a73e-c2d0-47a7-80dd-bca4424a3609/scratchpad"
)
SWEEP_JSON = SCRATCH / "verapdf_sweep_results.json"
SRC_DIR = Path("/Users/laccd/code/lamc_district_forms/lamc_remediated/remediated_pdfs")
OUT_DIR = SCRATCH / "ocg_fixed"
RESULTS_JSON = SCRATCH / "ocg_fixed_results.json"

TARGET_CLAUSE = "7.10"


def failed_clauses(pdf_path: Path) -> list[str]:
    """Return the sorted list of failed veraPDF rule clauses for a PDF.

    A file PASSES iff this list is empty. Parses the raw veraPDF XML for
    <rule clause=.. testNumber=.. status="failed">.
    """
    cmd = ["verapdf", "-f", "ua1", "--format", "xml", str(pdf_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    xml_text = proc.stdout.strip()
    if not xml_text:
        raise RuntimeError(
            f"veraPDF returned no XML for {pdf_path.name}: "
            f"{proc.stderr.strip()[:400]}"
        )
    # XML comes from the local trusted `verapdf` binary, not untrusted input.
    # Harden the stdlib parser anyway: disable DTD/entity processing so any
    # XXE / entity-expansion constructs are rejected rather than expanded.
    # (defusedxml is not a project dependency and must not be added here.)
    parser = ET.XMLParser()
    try:
        parser.parser.DefaultHandler = lambda data: None
        parser.entity = {}  # refuse to expand any entity definitions
    except Exception:  # noqa: BLE001 - best-effort hardening only
        pass
    root = ET.fromstring(xml_text, parser=parser)
    clauses: set[str] = set()
    for rule in root.iter("rule"):
        if rule.get("status") == "failed":
            clause = rule.get("clause")
            if clause:
                clauses.add(clause)
    return sorted(clauses)


def compute_targets() -> list[str]:
    """Files whose failed clauses include 7.10, per the sweep JSON."""
    data = json.loads(SWEEP_JSON.read_text())
    targets = [
        entry["file"]
        for entry in data
        if TARGET_CLAUSE in (entry.get("clauses") or [])
    ]
    return targets


def apply_ocg_fix(copy_path: Path) -> list[str]:
    """Apply the minimal /OCProperties /Name fix in place. Returns changes."""
    pdf = pikepdf.open(copy_path, allow_overwriting_input=True)
    try:
        changes = fix_optional_content_config_names(pdf)
        pdf.save(copy_path)
    finally:
        pdf.close()
    return changes


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = compute_targets()
    print(f"Computed {len(targets)} target files (failed clauses include {TARGET_CLAUSE}).")

    results: list[dict] = []

    for name in targets:
        src = SRC_DIR / name
        copy_path = OUT_DIR / name
        record: dict = {
            "file": name,
            "before": [],
            "after": [],
            "cleared": False,
            "reason": "",
        }

        if not src.exists():
            record["reason"] = f"source missing: {src}"
            results.append(record)
            print(f"[MISSING] {name}")
            continue

        try:
            shutil.copy2(src, copy_path)

            before = failed_clauses(copy_path)
            record["before"] = before

            changes = apply_ocg_fix(copy_path)
            record["changes"] = changes

            after = failed_clauses(copy_path)
            record["after"] = after

            before_set = set(before)
            after_set = set(after)
            new_fails = sorted(after_set - before_set)
            cleared_710 = TARGET_CLAUSE in before_set and TARGET_CLAUSE not in after_set

            # --- DELTA GATE ---------------------------------------------
            # Reject if 7.10 not reduced, or any new failed clause introduced.
            if not cleared_710:
                copy_path.unlink(missing_ok=True)
                record["reason"] = (
                    f"discarded: {TARGET_CLAUSE} not cleared "
                    f"(before={before}, after={after}, changes={changes})"
                )
                print(f"[DISCARD] {name}: {TARGET_CLAUSE} not cleared")
            elif new_fails:
                copy_path.unlink(missing_ok=True)
                record["reason"] = (
                    f"discarded: introduced new failed clause(s) {new_fails}"
                )
                print(f"[DISCARD] {name}: new fails {new_fails}")
            else:
                # Gate passed: 7.10 cleared, no regressions -> keep the copy.
                if not after_set:
                    record["cleared"] = True
                    record["reason"] = "7.10 fixed; file now passes UA-1 entirely"
                    print(f"[PASS]    {name}: full UA-1 pass")
                else:
                    record["cleared"] = False
                    residual = ", ".join(after)
                    record["reason"] = (
                        f"7.10 fixed; residual font/other clauses remain: {residual}"
                    )
                    print(f"[PARTIAL] {name}: 7.10 fixed, residual {after}")
        except Exception as exc:  # noqa: BLE001 - record and continue
            copy_path.unlink(missing_ok=True)
            record["reason"] = f"error: {type(exc).__name__}: {exc}"
            print(f"[ERROR]   {name}: {exc}")

        results.append(record)

    RESULTS_JSON.write_text(json.dumps(results, indent=2))

    n_full = sum(1 for r in results if r["cleared"])
    n_partial = sum(
        1
        for r in results
        if not r["cleared"] and "7.10 fixed" in r["reason"]
    )
    n_discard = len(results) - n_full - n_partial
    print(
        f"\nSummary: {len(results)} targets | "
        f"{n_full} full UA-1 pass | {n_partial} partial (7.10 fixed, font residual) | "
        f"{n_discard} discarded/error"
    )
    print(f"Results written to {RESULTS_JSON}")


if __name__ == "__main__":
    main()
