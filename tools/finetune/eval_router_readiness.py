#!/usr/bin/env python3
"""Summarize whether the Remedy task-router profile is ready for live smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TASK_ALIASES = {
    "alt_text_quality": "qwen3vl-32b-remedy",
    "table_structure": "qwen3vl-32b-remedy-table-v1",
    "contrast": "qwen3vl-32b-remedy-contrast-v1",
    "reading_order": "qwen3vl-32b-remedy-reading-order-v1",
    "heading_hierarchy": "qwen3vl-32b-remedy-heading-v1",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def task_metrics(data: dict[str, Any], task: str) -> dict[str, Any]:
    return dict(((data.get("by_task") or {}).get(task) or {}))


def gate(
    name: str,
    observed: Any,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    equals: Any = None,
    source: str = "",
) -> dict[str, Any]:
    passed = observed is not None
    if passed and min_value is not None:
        passed = float(observed) >= min_value
    if passed and max_value is not None:
        passed = float(observed) <= max_value
    if passed and equals is not None:
        passed = observed == equals
    expected: dict[str, Any] = {}
    if min_value is not None:
        expected["min"] = min_value
    if max_value is not None:
        expected["max"] = max_value
    if equals is not None:
        expected["equals"] = equals
    return {
        "name": name,
        "passed": bool(passed),
        "observed": observed,
        "expected": expected,
        "source": source,
    }


def adapter_check(alias: str, path: Path) -> dict[str, Any]:
    return {
        "alias": alias,
        "path": str(path),
        "has_config": (path / "adapter_config.json").exists(),
        "has_weights": (path / "adapter_model.safetensors").exists(),
        "passed": (path / "adapter_config.json").exists()
        and (path / "adapter_model.safetensors").exists(),
    }


def all_passed(items: list[dict[str, Any]]) -> bool:
    return all(bool(item.get("passed")) for item in items)


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    eval_dir = args.eval_dir
    main_repo = args.main_repo
    adapter_paths = {
        "alt_text_quality": args.alt_adapter or main_repo / "artifacts/lamc-qwen3vl-32b-lora-v2",
        "table_structure": args.table_adapter or main_repo / "artifacts/lamc-qwen3vl-32b-table-lora",
        "contrast": args.contrast_adapter or Path("outputs_runpod/lamc-qwen3vl-32b-contrast-lora"),
        "reading_order": args.reading_order_adapter
        or Path("outputs_runpod/lamc-qwen3vl-32b-reading-order-lora"),
        "heading_hierarchy": args.heading_adapter or Path("outputs_runpod/lamc-qwen3vl-32b-heading-lora"),
    }
    adapters = [
        adapter_check(TASK_ALIASES[task], path)
        for task, path in adapter_paths.items()
    ]

    alt_summary = load_json(args.alt_summary)
    alt_disc = (alt_summary.get("gold_vs_bad_discrimination") or {}).get("alt_text_quality") or {}
    alt_gates = [
        gate(
            "alt valid_json_rate >= 0.90",
            alt_summary.get("valid_json_rate"),
            min_value=0.90,
            source=str(args.alt_summary),
        ),
        gate(
            "alt win_rate >= 0.90",
            alt_disc.get("win_rate"),
            min_value=0.90,
            source=str(args.alt_summary),
        ),
        gate(
            "alt gold_flagged_more == 0",
            alt_disc.get("gold_flagged_more"),
            equals=0,
            source=str(args.alt_summary),
        ),
    ]

    table_metrics = task_metrics(load_json(args.table_metrics), "table_structure")
    table_gates = [
        gate("table status_accuracy >= 1.00", table_metrics.get("status_accuracy"), min_value=1.0, source=str(args.table_metrics)),
        gate("table valid_json_rate >= 0.90", table_metrics.get("valid_json_rate"), min_value=0.90, source=str(args.table_metrics)),
        gate("table pass_false_positive_rate <= 0.10", table_metrics.get("pass_false_positive_rate"), max_value=0.10, source=str(args.table_metrics)),
    ]

    contrast_metrics = task_metrics(load_json(eval_dir / "contrast.tuned.metrics.json"), "contrast")
    contrast_gates = [
        gate("contrast status_accuracy >= 0.90", contrast_metrics.get("status_accuracy"), min_value=0.90, source=str(eval_dir / "contrast.tuned.metrics.json")),
        gate("contrast near_threshold_status_accuracy >= 0.85", contrast_metrics.get("near_threshold_status_accuracy"), min_value=0.85, source=str(eval_dir / "contrast.tuned.metrics.json")),
        gate("contrast valid_json_rate >= 0.90", contrast_metrics.get("valid_json_rate"), min_value=0.90, source=str(eval_dir / "contrast.tuned.metrics.json")),
        gate("contrast pass_false_positive_rate <= 0.10", contrast_metrics.get("pass_false_positive_rate"), max_value=0.10, source=str(eval_dir / "contrast.tuned.metrics.json")),
    ]

    reading_metrics = task_metrics(load_json(eval_dir / "reading_order.tuned.metrics.json"), "reading_order")
    reading_gates = [
        gate("reading_order status_accuracy >= 0.80", reading_metrics.get("status_accuracy"), min_value=0.80, source=str(eval_dir / "reading_order.tuned.metrics.json")),
        gate("reading_order valid_json_rate >= 0.90", reading_metrics.get("valid_json_rate"), min_value=0.90, source=str(eval_dir / "reading_order.tuned.metrics.json")),
        gate("reading_order pass_false_positive_rate <= 0.10", reading_metrics.get("pass_false_positive_rate"), max_value=0.10, source=str(eval_dir / "reading_order.tuned.metrics.json")),
    ]

    heading_metrics = task_metrics(load_json(eval_dir / "heading.tuned.metrics.json"), "heading_hierarchy")
    heading_gates = [
        gate("heading status_accuracy >= 0.95", heading_metrics.get("status_accuracy"), min_value=0.95, source=str(eval_dir / "heading.tuned.metrics.json")),
        gate("heading exact_correction_accuracy >= 0.85", heading_metrics.get("exact_correction_accuracy"), min_value=0.85, source=str(eval_dir / "heading.tuned.metrics.json")),
        gate("heading valid_json_rate >= 0.90", heading_metrics.get("valid_json_rate"), min_value=0.90, source=str(eval_dir / "heading.tuned.metrics.json")),
        gate("heading pass_false_positive_rate <= 0.10", heading_metrics.get("pass_false_positive_rate"), max_value=0.10, source=str(eval_dir / "heading.tuned.metrics.json")),
    ]

    prod = load_json(args.production_task_metrics)
    contrast_corpus = ((prod.get("corpus_checks") or {}).get("contrast") or {})
    production_gates = [
        gate(
            "production contrast corpus marked non-gating",
            contrast_corpus.get("current_production_corpus_is_valid_gate"),
            equals=False,
            source=str(args.production_task_metrics),
        ),
        gate(
            "stable alias remains alt-text v2",
            ((prod.get("model_alias_semantics") or {}).get("stable_current_alias")),
            equals="qwen3vl-32b-remedy",
            source=str(args.production_task_metrics),
        ),
        gate(
            "multitask candidate not promoted",
            (prod.get("recommendation") or {}).get("decision"),
            equals="gather_better_eval_data_first",
            source=str(args.production_task_metrics),
        ),
    ]

    live_router_gates: list[dict[str, Any]] = []
    live_router_summary = getattr(args, "live_router_summary", None)
    if live_router_summary:
        live = load_json(live_router_summary)
        valid_by_task = live.get("valid_json_rate_by_task") or {}
        live_router_gates.extend([
            gate(
                "live router errors == 0",
                live.get("errors"),
                equals=0,
                source=str(live_router_summary),
            ),
            gate(
                "live router valid_json_rate >= 0.90",
                live.get("valid_json_rate"),
                min_value=0.90,
                source=str(live_router_summary),
            ),
        ])
        for task in sorted(valid_by_task):
            live_router_gates.append(
                gate(
                    f"live router {task} valid_json_rate >= 0.90",
                    valid_by_task.get(task),
                    min_value=0.90,
                    source=str(live_router_summary),
                )
            )

    task_gates = {
        "alt_text_quality": alt_gates,
        "table_structure": table_gates,
        "contrast": contrast_gates,
        "reading_order": reading_gates,
        "heading_hierarchy": heading_gates,
    }
    metric_gates = [gate_item for gates in task_gates.values() for gate_item in gates]
    metric_gates.extend(production_gates)
    metric_gates.extend(live_router_gates)
    ready_for_live_smoke = all_passed(adapters) and all_passed(metric_gates)
    live_smoke_passed = bool(live_router_gates) and all_passed(live_router_gates)
    if ready_for_live_smoke and live_smoke_passed:
        decision = "ready_for_heldout_lamc_pipeline_validation"
        final_ship_blocker = (
            "Held-out real LAMC remediation pipeline validation has not been run "
            "against the live router profile."
        )
    else:
        decision = (
            "ready_for_live_router_smoke"
            if ready_for_live_smoke
            else "not_ready_for_live_router_smoke"
        )
        final_ship_blocker = (
            "Live multi-LoRA serving smoke and held-out LAMC pipeline validation have not been run."
            if ready_for_live_smoke
            else "One or more adapter, metric, production, or live router gates failed."
        )
    return {
        "decision": decision,
        "final_ship_ready": False,
        "final_ship_blocker": final_ship_blocker,
        "stable_alias": "qwen3vl-32b-remedy",
        "stable_alias_points_to": "alt-text v2 adapter",
        "task_aliases": TASK_ALIASES,
        "adapters": adapters,
        "task_gates": task_gates,
        "production_gates": production_gates,
        "live_router_gates": live_router_gates,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-dir", type=Path, default=Path("eval_runs/runpod_h200_2026-07-06"))
    ap.add_argument("--main-repo", type=Path, default=Path("../remedy-server"))
    ap.add_argument("--alt-summary", type=Path, required=True)
    ap.add_argument("--table-metrics", type=Path, default=Path("output/eval_runs/table.tuned.capped.metrics.json"))
    ap.add_argument(
        "--production-task-metrics",
        type=Path,
        default=Path("eval_runs/runpod_h200_2026-07-06/remedy_multitask_v1.production.task_metrics.json"),
    )
    ap.add_argument(
        "--live-router-summary",
        type=Path,
        default=None,
        help="optional run_vision_eval summary from the live multi-LoRA router",
    )
    ap.add_argument("--alt-adapter", type=Path, default=None)
    ap.add_argument("--table-adapter", type=Path, default=None)
    ap.add_argument("--contrast-adapter", type=Path, default=None)
    ap.add_argument("--reading-order-adapter", type=Path, default=None)
    ap.add_argument("--heading-adapter", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_summary(args)
    text = json.dumps(summary, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if summary["decision"] in {
        "ready_for_live_router_smoke",
        "ready_for_heldout_lamc_pipeline_validation",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
