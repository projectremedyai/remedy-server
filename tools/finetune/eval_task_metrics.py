#!/usr/bin/env python3
"""Task-aware metrics for Remedy vision fine-tune JSONL outputs.

The trainer val files contain conversations with a gold assistant JSON target.
Prediction JSONL may come from eval_adapter_hf.py, tools/run_vision_eval.py, or a
small custom generation loop. This scorer accepts either same-order predictions
or rows keyed by example_id/id/doc_id+page+task+variant.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def parse_jsonish(text: str | dict | list | None) -> Any:
    if isinstance(text, (dict, list)):
        return text
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        return None


def target_text(rec: dict) -> str:
    return str(rec["messages"][-1]["content"][0]["text"])


def task_name(rec: dict) -> str:
    meta = rec.get("meta") or {}
    return str(meta.get("task") or rec.get("task") or "")


def record_key(rec: dict, index: int) -> str:
    meta = rec.get("meta") or {}
    if meta.get("example_id"):
        return str(meta["example_id"])
    parts = [
        str(meta.get("doc_id") or rec.get("doc_id") or ""),
        str(meta.get("page") or rec.get("page") or rec.get("page_index") or ""),
        str(meta.get("task") or rec.get("task") or ""),
        str(meta.get("variant") or rec.get("variant") or ""),
    ]
    if any(parts):
        return "|".join(parts)
    return str(index)


def prediction_text(row: dict) -> str:
    for key in ("response", "prediction", "generated", "output", "text"):
        if key in row:
            return str(row[key])
    if "messages" in row:
        return target_text(row)
    return json.dumps(row, ensure_ascii=False)


def normalized_status(parsed: Any, task: str) -> str | None:
    if not isinstance(parsed, dict):
        return None
    status = str(parsed.get("status", "")).strip().lower()
    if status in {"pass", "fail"}:
        return status
    if task == "alt_text_quality":
        figures = parsed.get("figures", parsed.get("issues"))
        if isinstance(figures, list):
            return "fail" if any(
                isinstance(item, dict)
                and str(item.get("status", "pass")).strip().lower() in {"fail", "failed", "error"}
                for item in figures
            ) else "pass"
    issues = parsed.get("issues")
    if isinstance(issues, list):
        return "fail" if issues else "pass"
    findings = parsed.get("findings")
    if isinstance(findings, list):
        return "fail" if findings else "pass"
    return None


def heading_pairs(parsed: Any) -> set[tuple[int, str]]:
    if not isinstance(parsed, dict):
        return set()
    items: list[Any] = []
    for key in ("findings", "heading_corrections", "corrections", "heading_issues", "issues"):
        value = parsed.get(key)
        if isinstance(value, list):
            items.extend(value)
    pairs: set[tuple[int, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            element_index = int(item.get("element_index") or item.get("index") or item.get("element"))
        except Exception:
            continue
        tag = str(item.get("correct_tag") or item.get("target_tag") or item.get("expected_tag") or "")
        tag = tag.strip().lstrip("/").upper()
        if re.fullmatch(r"H[1-6]|P|SPAN", tag):
            pairs.add((element_index, "Span" if tag == "SPAN" else tag))
    return pairs


def heading_level_counts(pairs: set[tuple[int, str]]) -> Counter:
    return Counter(tag for _idx, tag in pairs if re.fullmatch(r"H[1-6]", tag))


def reading_order(parsed: Any) -> tuple[int, ...] | None:
    if not isinstance(parsed, dict):
        return None
    value = parsed.get("corrected_order", parsed.get("reading_order"))
    if value in (None, False):
        return None
    if not isinstance(value, list):
        return None
    out: list[int] = []
    for item in value:
        try:
            out.append(int(item))
        except Exception:
            return None
    return tuple(out)


def contrast_ratios(parsed: Any) -> list[float]:
    if not isinstance(parsed, dict):
        return []
    issues = parsed.get("issues", parsed.get("contrast_issues"))
    if not isinstance(issues, list):
        return []
    ratios = []
    for item in issues:
        if not isinstance(item, dict):
            continue
        try:
            ratios.append(float(item.get("ratio")))
        except Exception:
            pass
    return ratios


def score_one(task: str, gold: Any, pred: Any) -> dict:
    gold_status = normalized_status(gold, task)
    pred_status = normalized_status(pred, task)
    valid_json = pred is not None
    status_match = gold_status is not None and pred_status == gold_status
    out = {
        "task": task,
        "valid_json": valid_json,
        "gold_status": gold_status,
        "pred_status": pred_status,
        "status_match": status_match,
    }

    if task == "heading_hierarchy":
        gold_pairs = heading_pairs(gold)
        pred_pairs = heading_pairs(pred)
        out.update(
            exact_corrections=gold_pairs == pred_pairs,
            correction_recall=(len(gold_pairs & pred_pairs) / len(gold_pairs) if gold_pairs else 1.0),
            correction_precision=(len(gold_pairs & pred_pairs) / len(pred_pairs) if pred_pairs else 1.0),
            gold_levels=dict(heading_level_counts(gold_pairs)),
            pred_levels=dict(heading_level_counts(pred_pairs)),
            pass_false_positive=(gold_status == "pass" and pred_status == "fail"),
        )
    elif task == "reading_order":
        gold_order = reading_order(gold)
        pred_order = reading_order(pred)
        out.update(
            corrected_order_match=(gold_order == pred_order if gold_order is not None else None),
            pass_false_positive=(gold_status == "pass" and pred_status == "fail"),
        )
    elif task == "contrast":
        gold_ratios = contrast_ratios(gold)
        pred_ratios = contrast_ratios(pred)
        near_threshold = any(4.2 <= ratio <= 4.8 for ratio in gold_ratios + pred_ratios)
        out.update(
            gold_ratios=gold_ratios,
            pred_ratios=pred_ratios,
            near_threshold=near_threshold,
            pass_false_positive=(gold_status == "pass" and pred_status == "fail"),
        )
    elif task == "table_structure":
        out["pass_false_positive"] = gold_status == "pass" and pred_status == "fail"
    return out


def summarize(scores: list[dict]) -> dict:
    by_task: dict[str, list[dict]] = defaultdict(list)
    for score in scores:
        by_task[score["task"]].append(score)

    summary = {"total": len(scores), "by_task": {}}
    for task, rows in by_task.items():
        confusion = Counter((r.get("gold_status"), r.get("pred_status")) for r in rows)
        task_summary = {
            "count": len(rows),
            "valid_json_rate": round(sum(r["valid_json"] for r in rows) / len(rows), 4),
            "status_accuracy": round(sum(r["status_match"] for r in rows) / len(rows), 4),
            "confusion": {f"{g}->{p}": n for (g, p), n in sorted(confusion.items())},
            "pass_false_positive_rate": round(
                sum(bool(r.get("pass_false_positive")) for r in rows) / len(rows), 4
            ),
        }
        if task == "heading_hierarchy":
            fail_rows = [r for r in rows if r.get("gold_status") == "fail"]
            task_summary.update(
                exact_correction_accuracy=round(
                    sum(bool(r.get("exact_corrections")) for r in fail_rows) / len(fail_rows), 4
                ) if fail_rows else None,
                correction_recall=round(
                    sum(float(r.get("correction_recall", 0.0)) for r in fail_rows) / len(fail_rows), 4
                ) if fail_rows else None,
                correction_precision=round(
                    sum(float(r.get("correction_precision", 0.0)) for r in fail_rows) / len(fail_rows), 4
                ) if fail_rows else None,
            )
        if task == "reading_order":
            order_rows = [r for r in rows if r.get("corrected_order_match") is not None]
            task_summary["corrected_order_accuracy"] = round(
                sum(bool(r.get("corrected_order_match")) for r in order_rows) / len(order_rows), 4
            ) if order_rows else None
        if task == "contrast":
            near = [r for r in rows if r.get("near_threshold")]
            task_summary["near_threshold_status_accuracy"] = round(
                sum(bool(r["status_match"]) for r in near) / len(near), 4
            ) if near else None
        summary["by_task"][task] = task_summary
    return summary


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_predictions(path: Path | None, gold_rows: list[dict], gold_as_predictions: bool) -> dict[str, str]:
    if gold_as_predictions:
        return {str(i): target_text(rec) for i, rec in enumerate(gold_rows)}
    if path is None:
        raise SystemExit("pass --predictions or --gold-as-predictions")
    rows = load_jsonl(path)
    keyed: dict[str, str] = {}
    for i, row in enumerate(rows):
        key = str(row.get("example_id") or row.get("id") or "")
        if not key:
            key = record_key(row, i)
        keyed[key] = prediction_text(row)
        keyed[str(i)] = prediction_text(row)
    if not keyed and rows:
        keyed = {str(i): prediction_text(row) for i, row in enumerate(rows)}
    return keyed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=Path, required=True, help="conversation val JSONL")
    ap.add_argument("--predictions", type=Path, default=None,
                    help="JSONL with response/prediction/generated fields")
    ap.add_argument("--gold-as-predictions", action="store_true",
                    help="Smoke-test the scorer by treating gold targets as predictions")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    gold_rows = load_jsonl(args.val)
    preds = load_predictions(args.predictions, gold_rows, args.gold_as_predictions)
    scores = []
    for i, rec in enumerate(gold_rows):
        key = record_key(rec, i)
        pred_text = preds.get(key, preds.get(str(i), ""))
        task = task_name(rec)
        scores.append(score_one(task, parse_jsonish(target_text(rec)), parse_jsonish(pred_text)))
    summary = summarize(scores)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
