"""Click ``office`` subgroup — office-verify check and level classification.

Commands::

    remedy-office check <file> [--json]
    remedy-office classify-level <file>

FR8: legacy binary formats (.doc/.ppt/.xls, OLE2 magic) fail closed with a
clear conversion-required error — never silently mis-parsed as ZIP.
(The ``report`` subcommand ships with office_compliance_report in Phase 4.)
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from project_remedy.models import FileType
from project_remedy.office_acceptance import (
    _infer_file_type,
    evaluate_office_acceptance,
    summarize_office_acceptance,
)
from project_remedy.office_levels import classify_level, probe_office_structure

console = Console()

_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_LEGACY_SUFFIXES = {".doc", ".ppt", ".xls"}
_CONVERT_MSG = "unsupported legacy format — requires OOXML conversion first (.docx/.pptx/.xlsx)"


def _guard_ooxml(path: Path) -> FileType:
    """FR8 fail-closed guard: reject legacy/OLE2/non-ZIP input before parsing."""
    if path.suffix.lower() in _LEGACY_SUFFIXES:
        raise click.ClickException(f"{_CONVERT_MSG} (got '{path.suffix}')")
    head = path.open("rb").read(8)
    if head.startswith(_OLE2_MAGIC):
        raise click.ClickException(f"{_CONVERT_MSG} (OLE2 container detected)")
    if not head.startswith(b"PK"):
        raise click.ClickException("not an OOXML package (missing ZIP signature)")
    try:
        return _infer_file_type(path)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc


@click.group("office")
def office_group() -> None:
    """office-verify: deterministic OOXML accessibility validation."""


@office_group.command()
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json", "as_json", is_flag=True, help="Output as JSON.")
def check(file: Path, as_json: bool) -> None:
    """Run the full deterministic rule catalog against FILE."""
    file_type = _guard_ooxml(file)
    result = evaluate_office_acceptance(file, file_type=file_type)
    summary = summarize_office_acceptance(result)
    if as_json:
        payload = dict(summary)
        payload["file_type"] = file_type.value
        payload["checks"] = [asdict(r) for r in result.checker_report.results]
        click.echo(json.dumps(payload, indent=2))
    else:
        table = Table(title=f"office-verify: {file.name}")
        table.add_column("Rule")
        table.add_column("Status")
        table.add_column("Details")
        for r in result.checker_report.results:
            table.add_row(r.rule_id, r.status, "; ".join(r.details))
        console.print(table)
        console.print(f"[bold]{'PASS' if result.passed else 'FAIL'}[/bold] — {result.summary()}")
    sys.exit(0 if result.passed else 1)


@office_group.command("classify-level")
@click.argument("file", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def classify_level_cmd(file: Path) -> None:
    """Classify FILE onto the L0-L4 remediation ladder (never L5)."""
    file_type = _guard_ooxml(file)
    if file_type != FileType.DOCX:
        raise click.ClickException("classify-level supports .docx only in Phase 1 (pptx/xlsx: Phase 2/3)")
    acceptance = evaluate_office_acceptance(file, file_type=file_type)
    probe = probe_office_structure(file, file_type)
    level = classify_level(acceptance, probe)
    click.echo(json.dumps(asdict(level), indent=2, default=str))
