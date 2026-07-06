#!/usr/bin/env python3
"""Task-aware production metrics for Remedy vision eval JSONL results.

This is a companion to ``tools/run_vision_eval.py``. The production harness is
a document-level gold-vs-bad severity check, while the multitask adapter needs a
task-aware readout that understands the current JSON schemas and the known
limits of the real eval corpus.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from eval_task_metrics import normalized_status, parse_jsonish
except ModuleNotFoundError:  # pragma: no cover - import path differs under pytest loaders
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from eval_task_metrics import normalized_status, parse_jsonish


HEADING_TAGS = {f"H{i}" for i in range(1, 7)}
BODY_TAGS = {"P", "SPAN", "LI", "LBODY", "DOCUMENT", "TH", "TD", "TR", "TABLE", "ARTIFACT"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def prediction_text(row: dict[str, Any]) -> str:
    for key in ("response", "prediction", "generated", "output", "text"):
        if key in row:
            return str(row[key])
    return json.dumps(row, ensure_ascii=False)


def tag_name(value: Any) -> str:
    text = str(value or "").strip().lstrip("/").upper()
    return "SPAN" if text == "Span".upper() else text


def task_status(task: str, parsed: Any) -> str | None:
    return normalized_status(parsed, task)


def schema_valid(task: str, parsed: Any) -> bool:
    if not isinstance(parsed, dict):
        return False
    if task == "reading_order":
        return (
            isinstance(parsed.get("page_layout"), str)
            and isinstance(parsed.get("issues"), list)
            and isinstance(parsed.get("summary"), str)
        )
    if task == "contrast":
        return isinstance(parsed.get("issues"), list)
    if task == "heading_hierarchy":
        return (
            str(parsed.get("status", "")).strip().lower() in {"pass", "fail"}
            and isinstance(parsed.get("findings"), list)
        )
    if task == "alt_text_quality":
        return isinstance(parsed.get("figures", parsed.get("issues")), list)
    return parsed is not None


def severity_value(value: Any, default: float = 1.0) -> float:
    severity = str(value or "").strip().lower()
    if severity in {"error", "fail", "failed", "critical", "high"}:
        return 2.0
    if severity == "info":
        return 0.0
    if severity in {"warning", "warn", "minor", "low", "medium", ""}:
        return 1.0 if severity else default
    return default


def severity_score(task: str, parsed: Any) -> float | None:
    if not isinstance(parsed, dict):
        return None
    if task in {"reading_order", "contrast"}:
        issues = parsed.get("issues")
        if not isinstance(issues, list):
            return None
        return sum(severity_value(issue.get("severity")) for issue in issues if isinstance(issue, dict))
    if task == "heading_hierarchy":
        findings = parsed.get("findings")
        status = str(parsed.get("status", "")).strip().lower()
        if not isinstance(findings, list):
            return 1.0 if status == "fail" else 0.0 if status == "pass" else None
        score = sum(
            severity_value(finding.get("severity"))
            for finding in findings
            if isinstance(finding, dict)
        )
        return 1.0 if status == "fail" and score == 0 else score
    if task == "alt_text_quality":
        figures = parsed.get("figures", parsed.get("issues"))
        if not isinstance(figures, list):
            return None
        score = 0.0
        for figure in figures:
            if not isinstance(figure, dict):
                continue
            if str(figure.get("status", "pass")).strip().lower() in {"fail", "failed", "error"}:
                score += severity_value(figure.get("severity"), default=2.0)
        return score
    return None


def expected_status_from_variant(row: dict[str, Any]) -> str:
    return "fail" if str(row.get("variant", "")).startswith("bad") else "pass"


def parse_logical_order(text: str) -> dict[int, dict[str, str]]:
    entries: dict[int, dict[str, str]] = {}
    pattern = re.compile(r"^\s*(\d+)\.\s+/(?:StructElem\s+)?([A-Za-z0-9_-]+)(?:.*?text:\s*\"([^\"]*)\")?")
    for line in str(text or "").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        entries[int(match.group(1))] = {
            "tag": tag_name(match.group(2)),
            "text": match.group(3) or "",
            "line": line,
        }
    return entries


def heading_finding_classification(
    source: dict[str, Any],
    finding: dict[str, Any],
) -> dict[str, Any]:
    prompt_inputs = source.get("prompt_inputs") or {}
    logical_order = str(prompt_inputs.get("logical_order") or prompt_inputs.get("structure_order") or "")
    entries = parse_logical_order(logical_order)
    try:
        element_index = int(finding.get("element_index"))
    except Exception:
        return {
            "classification": "needs_manual_audit",
            "reason": "finding has no numeric element_index",
            "structure_tag": None,
        }

    entry = entries.get(element_index)
    structure_tag = entry["tag"] if entry else None
    model_current = tag_name(finding.get("current_tag"))
    correct_tag = tag_name(finding.get("correct_tag"))

    if not entry:
        return {
            "classification": "likely_model_false_positive",
            "reason": "element_index is outside the provided logical order",
            "structure_tag": None,
            "model_current_tag": model_current,
            "correct_tag": correct_tag,
        }
    if model_current and model_current != structure_tag:
        return {
            "classification": "likely_model_false_positive",
            "reason": "model current_tag disagrees with the provided logical order",
            "structure_tag": structure_tag,
            "model_current_tag": model_current,
            "correct_tag": correct_tag,
            "structure_line": entry["line"],
        }
    if correct_tag == structure_tag:
        return {
            "classification": "likely_model_false_positive",
            "reason": "suggested correct_tag already matches the logical-order tag",
            "structure_tag": structure_tag,
            "model_current_tag": model_current,
            "correct_tag": correct_tag,
            "structure_line": entry["line"],
        }
    if correct_tag in HEADING_TAGS and structure_tag not in HEADING_TAGS:
        return {
            "classification": "likely_true_residual_structure_issue",
            "reason": "model says visible heading text is carried by a non-heading structure tag",
            "structure_tag": structure_tag,
            "model_current_tag": model_current,
            "correct_tag": correct_tag,
            "structure_line": entry["line"],
        }
    if structure_tag in HEADING_TAGS and correct_tag in BODY_TAGS:
        return {
            "classification": "likely_true_residual_structure_issue",
            "reason": "model says body text is over-promoted as a heading",
            "structure_tag": structure_tag,
            "model_current_tag": model_current,
            "correct_tag": correct_tag,
            "structure_line": entry["line"],
        }
    if structure_tag in HEADING_TAGS and correct_tag in HEADING_TAGS and structure_tag != correct_tag:
        return {
            "classification": "likely_true_residual_structure_issue",
            "reason": "model says the visible hierarchy needs a different H-level",
            "structure_tag": structure_tag,
            "model_current_tag": model_current,
            "correct_tag": correct_tag,
            "structure_line": entry["line"],
        }
    return {
        "classification": "needs_manual_audit",
        "reason": "structure/tag relationship is not enough to classify confidently",
        "structure_tag": structure_tag,
        "model_current_tag": model_current,
        "correct_tag": correct_tag,
        "structure_line": entry["line"],
    }


def pair_discrimination(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pairs: dict[tuple[str, Any], dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row.get("severity_score") is None:
            continue
        key = (str(row.get("doc_id")), row.get("page_index"))
        pairs[key][str(row.get("variant"))] = float(row["severity_score"])

    wins = ties = losses = 0
    for pair in pairs.values():
        bad_scores = [score for variant, score in pair.items() if variant.startswith("bad")]
        if "gold" not in pair or not bad_scores:
            continue
        bad = max(bad_scores)
        if bad > pair["gold"]:
            wins += 1
        elif bad == pair["gold"]:
            ties += 1
        else:
            losses += 1
    total = wins + ties + losses
    return {
        "pairs": total,
        "bad_flagged_more": wins,
        "ties": ties,
        "gold_flagged_more": losses,
        "win_rate": round(wins / total, 4) if total else None,
    }


def image_delta_stats(eval_rows: list[dict[str, Any]], eval_dir: Path) -> dict[str, Any]:
    try:
        from PIL import Image, ImageChops, ImageStat
    except Exception as exc:  # pragma: no cover - depends on optional local Pillow
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}

    by_pair: dict[tuple[str, Any], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in eval_rows:
        if row.get("task") != "contrast":
            continue
        variant = str(row.get("variant", ""))
        if variant == "gold" or variant.startswith("bad"):
            by_pair[(str(row.get("doc_id")), row.get("page_index"))][variant] = row

    deltas: list[float] = []
    missing = 0
    for pair in by_pair.values():
        if "gold" not in pair:
            continue
        bad_rows = [row for variant, row in pair.items() if variant.startswith("bad")]
        if not bad_rows:
            continue
        gold_path = eval_dir / str(pair["gold"].get("image", ""))
        bad_path = eval_dir / str(bad_rows[0].get("image", ""))
        if not gold_path.exists() or not bad_path.exists():
            missing += 1
            continue
        try:
            with Image.open(gold_path) as gold_img, Image.open(bad_path) as bad_img:
                gold_rgb = gold_img.convert("RGB")
                bad_rgb = bad_img.convert("RGB")
                if gold_rgb.size != bad_rgb.size:
                    bad_rgb = bad_rgb.resize(gold_rgb.size)
                diff = ImageChops.difference(gold_rgb, bad_rgb)
                mean_delta = sum(ImageStat.Stat(diff).mean) / 3.0
                deltas.append(float(mean_delta))
        except Exception:
            missing += 1

    return {
        "available": True,
        "pairs_checked": len(deltas),
        "pairs_missing_or_failed": missing,
        "identical_pairs": sum(1 for delta in deltas if delta == 0),
        "median_mean_abs_pixel_delta": round(statistics.median(deltas), 4) if deltas else None,
        "max_mean_abs_pixel_delta": round(max(deltas), 4) if deltas else None,
    }


def load_contrast_gate_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return ((data.get("by_task") or {}).get("contrast") or None)


def join_rows(eval_rows: list[dict[str, Any]], result_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_by_id = {str(row.get("example_id")): row for row in eval_rows}
    joined = []
    for result in result_rows:
        example_id = str(result.get("example_id"))
        source = source_by_id.get(example_id, {})
        parsed = parse_jsonish(prediction_text(result))
        task = str(result.get("task") or source.get("task") or "")
        joined.append(
            {
                "example_id": example_id,
                "doc_id": result.get("doc_id", source.get("doc_id")),
                "page_index": result.get("page_index", source.get("page_index")),
                "task": task,
                "variant": str(result.get("variant", source.get("variant", ""))),
                "source": source,
                "result": result,
                "parsed": parsed,
                "json_parsed": parsed is not None,
                "schema_valid": schema_valid(task, parsed),
                "pred_status": task_status(task, parsed),
                "expected_status": expected_status_from_variant(source or result),
                "severity_score": severity_score(task, parsed),
            }
        )
    return joined


def task_summary(task: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    variants = Counter(str(row.get("variant")) for row in rows)
    expected_matches = [
        row["pred_status"] == row["expected_status"]
        for row in rows
        if row.get("pred_status") in {"pass", "fail"}
    ]
    gold_rows = [row for row in rows if str(row.get("variant")) == "gold"]
    bad_rows = [row for row in rows if str(row.get("variant", "")).startswith("bad")]
    status_by_variant: dict[str, dict[str, int]] = {}
    for variant in sorted(variants):
        status_by_variant[variant] = dict(
            Counter(
                str(row.get("pred_status"))
                for row in rows
                if str(row.get("variant")) == variant
            )
        )

    out: dict[str, Any] = {
        "count": len(rows),
        "variants": dict(sorted(variants.items())),
        "json_parsed_rate": round(sum(row["json_parsed"] for row in rows) / len(rows), 4),
        "schema_valid_rate": round(sum(row["schema_valid"] for row in rows) / len(rows), 4),
        "status_by_variant": status_by_variant,
        "variant_expected_status_accuracy": (
            round(sum(expected_matches) / len(expected_matches), 4) if expected_matches else None
        ),
        "pass_false_positive_rate": (
            round(sum(row.get("pred_status") == "fail" for row in gold_rows) / len(gold_rows), 4)
            if gold_rows
            else None
        ),
        "bad_detection_rate": (
            round(sum(row.get("pred_status") == "fail" for row in bad_rows) / len(bad_rows), 4)
            if bad_rows
            else None
        ),
        "mean_severity_by_variant": {
            variant: round(statistics.mean(scores), 4)
            for variant in sorted(variants)
            if (
                scores := [
                    float(row["severity_score"])
                    for row in rows
                    if str(row.get("variant")) == variant and row.get("severity_score") is not None
                ]
            )
        },
        "gold_vs_bad_discrimination": pair_discrimination(rows),
    }

    if task == "reading_order":
        out.update(
            schema="page_layout + issues + summary",
            empty_issues_means_pass=True,
            corrected_order_required=False,
            corrected_order_accuracy=None,
            miss_counts={
                "bad_not_flagged": sum(row.get("pred_status") == "pass" for row in bad_rows),
                "gold_flagged": sum(row.get("pred_status") == "fail" for row in gold_rows),
                "schema_invalid": sum(not row.get("schema_valid") for row in rows),
            },
        )
    if task == "heading_hierarchy":
        classifications = Counter()
        flagged_gold = 0
        for row in gold_rows:
            parsed = row.get("parsed")
            findings = parsed.get("findings") if isinstance(parsed, dict) else None
            if not isinstance(findings, list) or not findings:
                continue
            flagged_gold += 1
            for finding in findings:
                if isinstance(finding, dict):
                    classifications[
                        heading_finding_classification(row["source"], finding)["classification"]
                    ] += 1
        out.update(
            gold_flagged_pages=flagged_gold,
            gold_flag_classification_counts=dict(sorted(classifications.items())),
        )
    if task == "contrast":
        relevant = Counter(
            str((row.get("source") or {}).get("relevant_dimension"))
            for row in rows
        )
        out.update(
            gate_applicability="not_applicable_current_production_corpus",
            gate_reason=(
                "production contrast records have no annotated contrast dimension; "
                "gold/bad variants are not verified foreground/background contrast pairs"
            ),
            relevant_dimension_counts=dict(sorted(relevant.items())),
            required_gate_source="tools/finetune/data_contrast/val.jsonl or verified contrast-specific real set",
        )
    return out


def sample_base(row: dict[str, Any]) -> dict[str, Any]:
    source = row.get("source") or {}
    result = row.get("result") or {}
    return {
        "example_id": row.get("example_id"),
        "doc_id": row.get("doc_id"),
        "task": row.get("task"),
        "variant": row.get("variant"),
        "page_index": row.get("page_index"),
        "image": source.get("image"),
        "artifact_path": source.get("artifact_path"),
        "pred_status": row.get("pred_status"),
        "expected_status": row.get("expected_status"),
        "severity_score": row.get("severity_score"),
        "response": result.get("response", result.get("prediction")),
    }


def representative_samples(joined: list[dict[str, Any]], limit_per_type: int = 8) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()

    def add(sample_type: str, row: dict[str, Any], extra: dict[str, Any] | None = None) -> None:
        if counts[sample_type] >= limit_per_type:
            return
        item = sample_base(row)
        item["sample_type"] = sample_type
        if extra:
            item.update(extra)
        samples.append(item)
        counts[sample_type] += 1

    for row in joined:
        if row.get("task") == "reading_order":
            if str(row.get("variant", "")).startswith("bad") and row.get("pred_status") == "pass":
                add(
                    "reading_order_bad_not_flagged",
                    row,
                    {"structure_order": (row.get("source") or {}).get("prompt_inputs", {}).get("structure_order")},
                )
            elif row.get("variant") == "gold" and row.get("pred_status") == "fail":
                add(
                    "reading_order_gold_flagged",
                    row,
                    {"structure_order": (row.get("source") or {}).get("prompt_inputs", {}).get("structure_order")},
                )
            elif not row.get("schema_valid"):
                add("reading_order_schema_invalid", row)

        if row.get("task") == "heading_hierarchy" and row.get("variant") == "gold":
            parsed = row.get("parsed")
            findings = parsed.get("findings") if isinstance(parsed, dict) else None
            if not isinstance(findings, list) or not findings:
                continue
            classified_findings = []
            for finding in findings:
                if isinstance(finding, dict):
                    classified_findings.append(
                        {
                            "finding": finding,
                            "classification": heading_finding_classification(row["source"], finding),
                        }
                    )
            add(
                "heading_gold_flag",
                row,
                {
                    "logical_order": (row.get("source") or {}).get("prompt_inputs", {}).get("logical_order"),
                    "classified_findings": classified_findings,
                },
            )
    return samples


def build_summary(
    eval_rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    *,
    eval_dir: Path,
    result_path: Path | None = None,
    contrast_gate_metrics: dict[str, Any] | None = None,
    compute_image_delta: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    joined = join_rows(eval_rows, result_rows)
    source_ids = {str(row.get("example_id")) for row in eval_rows}
    result_ids = {str(row.get("example_id")) for row in result_rows}
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_task[str(row.get("task"))].append(row)

    task_summaries = {task: task_summary(task, rows) for task, rows in sorted(by_task.items())}
    if "contrast" in task_summaries:
        task_summaries["contrast"]["verified_contrast_gate_metrics"] = contrast_gate_metrics

    corpus_checks: dict[str, Any] = {
        "contrast": {
            "current_production_corpus_is_valid_gate": False,
            "reason": (
                "TASK_DIMENSION for contrast is None and the source rows do not contain "
                "verified color-ratio labels."
            ),
        }
    }
    if compute_image_delta:
        corpus_checks["contrast"]["image_delta_stats"] = image_delta_stats(eval_rows, eval_dir)

    recommendation = {
        "decision": "gather_better_eval_data_first",
        "stable_alias": "qwen3vl-32b-remedy remains the stable alt-text v2 alias",
        "candidate_alias": "qwen3vl-32b-remedy-multitask-v1",
        "summary": (
            "Do not promote the weighted multitask adapter to qwen3vl-32b-remedy from this "
            "production corpus alone. Keep the stable alt alias and router fallback while "
            "contrast uses the verified contrast gate and heading/reading-order production "
            "examples get task-specific ground truth or manual audit."
        ),
    }

    summary = {
        "result_path": str(result_path) if result_path else "",
        "source_rows": len(eval_rows),
        "result_rows": len(result_rows),
        "matched_result_rows": len(source_ids & result_ids),
        "missing_source_for_results": sorted(result_ids - source_ids)[:20],
        "source_rows_without_results": len(source_ids - result_ids),
        "model_alias_semantics": {
            "stable_current_alias": "qwen3vl-32b-remedy",
            "stable_current_alias_points_to": "alt-text v2 adapter",
            "weighted_multitask_candidate": "qwen3vl-32b-remedy-multitask-v1",
            "do_not_reuse_stable_alias_for_candidate": True,
        },
        "tasks": task_summaries,
        "corpus_checks": corpus_checks,
        "recommendation": recommendation,
    }
    return summary, representative_samples(joined)


def default_contrast_metrics_path(results_path: Path) -> Path:
    return results_path.parent / "multitask_contrast_weighted.full.metrics.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval", type=Path, required=True, help="production eval source JSONL")
    ap.add_argument("--results", type=Path, required=True, help="run_vision_eval result JSONL")
    ap.add_argument("--out", type=Path, required=True, help="task-aware summary JSON")
    ap.add_argument("--samples-out", type=Path, default=None, help="representative samples JSONL")
    ap.add_argument("--contrast-metrics", type=Path, default=None,
                    help="optional verified contrast metrics JSON from eval_task_metrics")
    ap.add_argument("--skip-image-delta", action="store_true",
                    help="skip rendered gold/bad image-delta check")
    args = ap.parse_args()

    eval_path = args.eval.resolve()
    results_path = args.results.resolve()
    contrast_metrics_path = (
        args.contrast_metrics.resolve()
        if args.contrast_metrics
        else default_contrast_metrics_path(results_path)
    )
    summary, samples = build_summary(
        load_jsonl(eval_path),
        load_jsonl(results_path),
        eval_dir=eval_path.parent,
        result_path=results_path,
        contrast_gate_metrics=load_contrast_gate_metrics(contrast_metrics_path),
        compute_image_delta=not args.skip_image_delta,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    samples_out = args.samples_out or args.out.with_suffix(".samples.jsonl")
    write_jsonl(samples_out, samples)

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote samples: {samples_out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
