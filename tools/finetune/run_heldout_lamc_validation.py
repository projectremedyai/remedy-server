#!/usr/bin/env python3
"""Run held-out LAMC PDF remediation validation through the configured router.

This is intentionally a thin operational harness around the production CLI:

1. ``remedy-pdf fix`` on each source PDF, usually with ``--thorough`` so the
   vision router is exercised.
2. ``remedy-pdf check --json`` on the output.
3. ``remedy-pdf report --json --original`` for the compliance readout.
4. Normalized text extraction comparison as a content-fidelity guard.

The script writes one JSONL record per input and a compact summary JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any


ROUTER_ENV_KEYS = (
    "OLLAMA_VISION_MODEL",
    "VISION_BASE_URL",
    "OLLAMA_BASE_URL",
    "OLLAMA_VISION_TASK_MODELS",
    "OLLAMA_VISION_TASK_BASE_URLS",
    "OLLAMA_VISION_ROUTER_ALLOW_FALLBACK",
    "OLLAMA_ESCALATION_MAX_INFLIGHT",
    "OLLAMA_VISION_MAX_INFLIGHT",
    "OLLAMA_VISION_GATE_TIMEOUT_SECONDS",
    "OLLAMA_VISION_MAX_TOKENS",
)


def parse_first_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    """Parse the first JSON object from CLI output, returning trailing footer.

    Several Remedy CLI commands print a token-usage footer after ``--json``.
    ``json.load`` rejects that as "extra data"; this helper keeps the structured
    part and returns the footer for auditability.
    """

    start = text.find("{")
    if start < 0:
        return None, text.strip()
    try:
        data, end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None, text.strip()
    if not isinstance(data, dict):
        return None, text[start + end :].strip()
    return data, text[start + end :].strip()


def slugify(path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._-")
    return stem[:120] or "document"


def collect_sources(args: argparse.Namespace) -> list[Path]:
    sources: list[Path] = []
    for value in args.sources:
        sources.append(Path(value).expanduser())
    if args.source_list:
        for line in args.source_list.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                sources.append(Path(stripped).expanduser())
    if args.source_dir:
        sources.extend(sorted(args.source_dir.expanduser().glob("*.pdf")))

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in sources:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if args.limit:
        unique = unique[: args.limit]
    return unique


def command_base(args: argparse.Namespace) -> list[str]:
    if args.remedy_pdf_bin:
        return [str(args.remedy_pdf_bin)]
    found = shutil.which("remedy-pdf")
    if found:
        return [found]
    return ["remedy-pdf"]


def run_command(
    cmd: list[str],
    *,
    timeout: int,
    log_path: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
        status = {
            "returncode": proc.returncode,
            "timed_out": False,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
        }
    except subprocess.TimeoutExpired as exc:
        output = (exc.stdout or "") + (exc.stderr or "")
        if isinstance(output, bytes):
            output = output.decode(errors="replace")
        status = {
            "returncode": None,
            "timed_out": True,
            "elapsed_seconds": round(time.perf_counter() - start, 3),
        }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(output, encoding="utf-8")
    status["log"] = str(log_path)
    return status


def normalized_text(path: Path) -> dict[str, Any]:
    try:
        import fitz  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local optional dep
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        doc = fitz.open(path)
        text = "\n".join(page.get_text("text") for page in doc)
        normalized = " ".join(text.split())
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "normalized_chars": len(normalized),
        "preview": normalized[:300],
        "_text": normalized,
    }


def _text_tokens(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.casefold(), flags=re.UNICODE)


def _counter_sample(counter: Counter, *, limit: int = 20) -> list[str]:
    return list(counter.elements())[:limit]


def _compact_chars(text: str, *, alnum_only: bool = False) -> str:
    if alnum_only:
        return re.sub(r"[^\w]+", "", text).casefold()
    return re.sub(r"\s+", "", text).casefold()


def text_fidelity(source: Path, output: Path) -> dict[str, Any]:
    src = normalized_text(source)
    out = normalized_text(output)
    result: dict[str, Any] = {
        "available": bool(src.get("available") and out.get("available")),
        "source_normalized_chars": src.get("normalized_chars"),
        "output_normalized_chars": out.get("normalized_chars"),
        "normalized_text_equal": None,
    }
    if src.get("error"):
        result["source_error"] = src["error"]
    if out.get("error"):
        result["output_error"] = out["error"]
    if result["available"]:
        source_text = str(src.get("_text") or "")
        output_text = str(out.get("_text") or "")
        source_tokens = _text_tokens(source_text)
        output_tokens = _text_tokens(output_text)
        source_counter = Counter(source_tokens)
        output_counter = Counter(output_tokens)
        whitespace_compact_source = _compact_chars(source_text)
        whitespace_compact_output = _compact_chars(output_text)
        alnum_source = _compact_chars(source_text, alnum_only=True)
        alnum_output = _compact_chars(output_text, alnum_only=True)
        result.update(
            normalized_text_equal=source_text == output_text,
            whitespace_insensitive_equal=whitespace_compact_source == whitespace_compact_output,
            whitespace_insensitive_char_multiset_equal=(
                Counter(whitespace_compact_source) == Counter(whitespace_compact_output)
            ),
            alnum_char_multiset_equal=Counter(alnum_source) == Counter(alnum_output),
            sequence_similarity=round(
                SequenceMatcher(None, source_text, output_text, autojunk=False).ratio(),
                4,
            ),
            source_token_count=len(source_tokens),
            output_token_count=len(output_tokens),
            token_count_delta=len(output_tokens) - len(source_tokens),
            token_multiset_equal=source_counter == output_counter,
            missing_token_sample=_counter_sample(source_counter - output_counter),
            added_token_sample=_counter_sample(output_counter - source_counter),
        )
    return result


def visual_fidelity(
    source: Path,
    output: Path,
    *,
    dpi: int = 72,
    mean_threshold: float = 0.5,
    max_threshold: int = 4,
) -> dict[str, Any]:
    """Compare rendered source/output pages as a guard against visual drift."""

    try:
        import fitz  # type: ignore
        from PIL import Image, ImageChops, ImageStat
    except Exception as exc:  # pragma: no cover - depends on local optional deps
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    def page_image(page: Any) -> Any:
        pix = page.get_pixmap(
            matrix=fitz.Matrix(dpi / 72, dpi / 72),
            alpha=False,
            colorspace=fitz.csRGB,
        )
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    try:
        source_doc = fitz.open(source)
        output_doc = fitz.open(output)
        page_count_equal = len(source_doc) == len(output_doc)
        page_metrics: list[dict[str, Any]] = []
        for page_index, (source_page, output_page) in enumerate(zip(source_doc, output_doc), start=1):
            source_image = page_image(source_page)
            output_image = page_image(output_page)
            sizes_equal = source_image.size == output_image.size
            metric: dict[str, Any] = {
                "page": page_index,
                "source_size": source_image.size,
                "output_size": output_image.size,
                "sizes_equal": sizes_equal,
            }
            if sizes_equal:
                diff = ImageChops.difference(source_image, output_image)
                stat = ImageStat.Stat(diff)
                mean_delta = sum(stat.mean) / len(stat.mean)
                max_delta = max(channel[1] for channel in stat.extrema)
                metric.update(
                    mean_abs_pixel_delta=round(mean_delta, 4),
                    max_abs_pixel_delta=int(max_delta),
                )
            page_metrics.append(metric)
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    comparable = [item for item in page_metrics if item.get("sizes_equal")]
    mean_values = [float(item.get("mean_abs_pixel_delta", 255.0)) for item in comparable]
    max_values = [int(item.get("max_abs_pixel_delta", 255)) for item in comparable]
    visual_match = (
        page_count_equal
        and len(comparable) == len(page_metrics)
        and bool(page_metrics)
        and (max(mean_values) if mean_values else 255.0) <= mean_threshold
        and (max(max_values) if max_values else 255) <= max_threshold
    )
    return {
        "available": True,
        "dpi": dpi,
        "page_count_equal": page_count_equal,
        "source_pages": len(source_doc),
        "output_pages": len(output_doc),
        "mean_abs_pixel_delta_avg": round(sum(mean_values) / len(mean_values), 4)
        if mean_values else None,
        "mean_abs_pixel_delta_max_page": round(max(mean_values), 4) if mean_values else None,
        "max_abs_pixel_delta": max(max_values) if max_values else None,
        "visual_match": visual_match,
        "thresholds": {
            "mean_abs_pixel_delta_max_page": mean_threshold,
            "max_abs_pixel_delta": max_threshold,
        },
        "pages": page_metrics,
    }


def check_passed(check_json: dict[str, Any] | None) -> bool:
    if not check_json:
        return False
    summary = check_json.get("summary") or {}
    return (
        int(summary.get("failed", -1)) == 0
        and int(summary.get("manual", -1)) == 0
        and int(summary.get("fixable", -1)) == 0
    )


def report_passed(report_json: dict[str, Any] | None) -> bool:
    if not report_json:
        return False
    summary = report_json.get("summary") or {}
    return (
        int(summary.get("failed_checks", -1)) == 0
        and int(summary.get("sr_errors", -1)) == 0
        and bool(summary.get("verapdf_checked"))
        and bool(summary.get("verapdf_passed"))
        and int(summary.get("verapdf_violations", -1)) == 0
        and int(summary.get("wcag_fail", -1)) == 0
    )


def validate_one(source: Path, args: argparse.Namespace) -> dict[str, Any]:
    base = command_base(args)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    name = slugify(source)
    output_pdf = out_dir / f"{name}.router_fixed.pdf"
    fix_log = out_dir / f"{name}.fix.log"
    check_json_path = out_dir / f"{name}.check.json"
    report_json_path = out_dir / f"{name}.report.json"

    env = os.environ.copy()
    fix_cmd = [*base, "fix"]
    if args.thorough:
        fix_cmd.append("--thorough")
    if args.env_file:
        fix_cmd.extend(["--env", str(args.env_file)])
    if args.config_file:
        fix_cmd.extend(["--config", str(args.config_file)])
    fix_cmd.extend([str(source), "-o", str(output_pdf)])

    record: dict[str, Any] = {
        "source": str(source),
        "output": str(output_pdf),
        "router_env": {key: env.get(key) for key in ROUTER_ENV_KEYS if env.get(key) is not None},
    }

    if args.skip_existing and output_pdf.exists():
        record["fix"] = {"skipped_existing": True, "returncode": 0, "timed_out": False}
    else:
        record["fix"] = run_command(
            fix_cmd,
            timeout=args.fix_timeout,
            log_path=fix_log,
            env=env,
        )
    record["output_exists"] = output_pdf.exists()

    if output_pdf.exists():
        check_cmd = [*base, "check"]
        if args.env_file:
            check_cmd.extend(["--env", str(args.env_file)])
        if args.config_file:
            check_cmd.extend(["--config", str(args.config_file)])
        check_cmd.extend(["--json", str(output_pdf)])
        check_run = run_command(
            check_cmd,
            timeout=args.check_timeout,
            log_path=check_json_path,
            env=env,
        )
        check_text = check_json_path.read_text(encoding="utf-8")
        check_data, check_footer = parse_first_json_object(check_text)
        record["check"] = {
            **check_run,
            "json_path": str(check_json_path),
            "parsed": check_data is not None,
            "footer": check_footer,
            "summary": (check_data or {}).get("summary"),
            "passed": check_passed(check_data),
        }

        report_cmd = [
            *base,
            "report",
            str(output_pdf),
            "--original",
            str(source),
            "--json",
        ]
        report_run = run_command(
            report_cmd,
            timeout=args.report_timeout,
            log_path=report_json_path,
            env=env,
        )
        report_text = report_json_path.read_text(encoding="utf-8")
        report_data, report_footer = parse_first_json_object(report_text)
        record["report"] = {
            **report_run,
            "json_path": str(report_json_path),
            "parsed": report_data is not None,
            "footer": report_footer,
            "summary": (report_data or {}).get("summary"),
            "visual_diff": (report_data or {}).get("visual_diff"),
            "passed": report_passed(report_data),
        }
        record["text_fidelity"] = text_fidelity(source, output_pdf)
        record["visual_fidelity"] = visual_fidelity(
            source,
            output_pdf,
            dpi=args.visual_dpi,
            mean_threshold=args.visual_mean_threshold,
            max_threshold=args.visual_max_threshold,
        )
    else:
        record["check"] = {"passed": False, "parsed": False}
        record["report"] = {"passed": False, "parsed": False}
        record["text_fidelity"] = {"available": False, "normalized_text_equal": False}
        record["visual_fidelity"] = {"available": False, "visual_match": False}

    text_diag = record.get("text_fidelity") or {}
    visual_diag = record.get("visual_fidelity") or {}
    text_ok = text_diag.get("normalized_text_equal") is True
    content_ok = bool(
        text_ok
        or (
            text_diag.get("alnum_char_multiset_equal") is True
            and visual_diag.get("visual_match") is True
        )
    )
    record["content_fidelity_passed"] = content_ok
    fix_ok = (
        record.get("fix", {}).get("returncode") == 0
        and not record.get("fix", {}).get("timed_out")
    )
    record["passed"] = bool(
        fix_ok
        and record["output_exists"]
        and record.get("check", {}).get("passed")
        and record.get("report", {}).get("passed")
        and content_ok
    )
    return record


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [record for record in records if record.get("passed")]
    return {
        "count": len(records),
        "passed": len(passed),
        "failed": len(records) - len(passed),
        "pass_rate": round(len(passed) / len(records), 4) if records else None,
        "verapdf_passed": sum(
            bool(((record.get("report") or {}).get("summary") or {}).get("verapdf_passed"))
            for record in records
        ),
        "check_zero_failures": sum(bool((record.get("check") or {}).get("passed")) for record in records),
        "report_zero_failures": sum(bool((record.get("report") or {}).get("passed")) for record in records),
        "text_fidelity_passed": sum(
            (record.get("text_fidelity") or {}).get("normalized_text_equal") is True
            for record in records
        ),
        "text_token_multiset_passed": sum(
            (record.get("text_fidelity") or {}).get("token_multiset_equal") is True
            for record in records
        ),
        "text_alnum_char_multiset_passed": sum(
            (record.get("text_fidelity") or {}).get("alnum_char_multiset_equal") is True
            for record in records
        ),
        "visual_fidelity_passed": sum(
            (record.get("visual_fidelity") or {}).get("visual_match") is True
            for record in records
        ),
        "content_fidelity_passed": sum(
            record.get("content_fidelity_passed") is True
            for record in records
        ),
        "text_min_sequence_similarity": min(
            (
                float((record.get("text_fidelity") or {}).get("sequence_similarity"))
                for record in records
                if (record.get("text_fidelity") or {}).get("sequence_similarity") is not None
            ),
            default=None,
        ),
        "failures": [
            {
                "source": record.get("source"),
                "output_exists": record.get("output_exists"),
                "fix": record.get("fix"),
                "check_summary": (record.get("check") or {}).get("summary"),
                "report_summary": (record.get("report") or {}).get("summary"),
                "text_fidelity": record.get("text_fidelity"),
                "visual_fidelity": record.get("visual_fidelity"),
                "content_fidelity_passed": record.get("content_fidelity_passed"),
            }
            for record in records
            if not record.get("passed")
        ],
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sources", nargs="*", help="PDF files to validate")
    ap.add_argument("--source-list", type=Path, help="text file with one PDF path per line")
    ap.add_argument("--source-dir", type=Path, help="directory of PDFs to validate")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--records-out", type=Path, default=None)
    ap.add_argument("--summary-out", type=Path, default=None)
    ap.add_argument("--remedy-pdf-bin", type=Path, default=None)
    ap.add_argument("--env-file", type=Path, default=None)
    ap.add_argument("--config-file", type=Path, default=None)
    ap.add_argument("--fix-timeout", type=int, default=1800)
    ap.add_argument("--check-timeout", type=int, default=600)
    ap.add_argument("--report-timeout", type=int, default=600)
    ap.add_argument("--visual-dpi", type=int, default=72)
    ap.add_argument("--visual-mean-threshold", type=float, default=0.5)
    ap.add_argument("--visual-max-threshold", type=int, default=4)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-thorough", dest="thorough", action="store_false")
    ap.set_defaults(thorough=True)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records_path = args.records_out or args.out_dir / "heldout_lamc_validation.jsonl"
    summary_path = args.summary_out or args.out_dir / "heldout_lamc_validation.summary.json"
    sources = collect_sources(args)
    if not sources:
        raise SystemExit("No source PDFs selected")

    records = [validate_one(source, args) for source in sources]
    summary = summarize(records)
    summary.update({
        "sources": [str(source) for source in sources],
        "records_path": str(records_path),
        "out_dir": str(args.out_dir),
    })

    write_jsonl(records_path, records)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
