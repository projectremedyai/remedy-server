#!/usr/bin/env python3
"""Run a vision model over the exported eval JSONL and score the harness gates.

Step 1 of the vision-model eval harness. Consumes the output of
tools/export_corpus_jsonl.py and measures, per model:

  - valid-JSON rate (overall and per task)
  - latency p50 / p95 per call
  - gold-vs-bad discrimination per task: for each (doc, page, task) pair the
    model's issue-severity score on the known-bad render must EXCEED its score
    on the gold render

The provider is built with the production factory (load_config ->
create_provider_from_config), so model/endpoint swaps use the same env vars as
the pipeline: OLLAMA_VISION_MODEL, VISION_BASE_URL / OLLAMA_BASE_URL,
OLLAMA_API_KEY, OLLAMA_VISION_FALLBACK_*. --model / --base-url flags override
the env for convenience.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _severity_score(task: str, parsed) -> float | None:
    """Issue-severity score for one parsed response (higher = more issues)."""
    if not isinstance(parsed, dict):
        return None

    def _sev(value, default: float = 1.0) -> float:
        s = str(value or "").strip().lower()
        if s in {"error", "fail", "failed", "critical", "high"}:
            return 2.0
        if s == "info":
            return 0.0
        if s in {"warning", "warn", "minor", "low", "medium", ""}:
            return 1.0 if s else default
        return default

    if task in {"reading_order", "contrast"}:
        issues = parsed.get("issues")
        if not isinstance(issues, list):
            return None
        return sum(_sev(i.get("severity")) for i in issues if isinstance(i, dict))

    if task == "heading_hierarchy":
        findings = parsed.get("findings")
        status = str(parsed.get("status", "")).strip().lower()
        if not isinstance(findings, list):
            if status in {"pass", "fail"}:
                return 1.0 if status == "fail" else 0.0
            return None
        score = sum(_sev(f.get("severity")) for f in findings if isinstance(f, dict))
        if status == "fail" and score == 0:
            score = 1.0
        return score

    if task == "alt_text_quality":
        figures = parsed.get("figures", parsed.get("issues"))
        if not isinstance(figures, list):
            return None
        score = 0.0
        for fig in figures:
            if not isinstance(fig, dict):
                continue
            if str(fig.get("status", "pass")).strip().lower() in {"fail", "failed", "error"}:
                score += _sev(fig.get("severity"), default=2.0)
        return score

    return None


async def _run_records(records, provider, parse_json, args, out_fh) -> list[dict]:
    sem = asyncio.Semaphore(args.concurrency)
    response_format = {"type": "json_object"} if args.response_format else None
    results: list[dict] = []
    done = 0

    async def one(rec: dict) -> dict:
        nonlocal done
        image = Path(args.eval).parent / rec["image"]
        result = {
            "example_id": rec["example_id"],
            "doc_id": rec["doc_id"],
            "task": rec["task"],
            "variant": rec["variant"],
            "page_index": rec["page_index"],
            "model": args.model_label,
            "response_format": bool(response_format),
        }
        async with sem:
            t0 = time.perf_counter()
            try:
                response = await provider.analyze_image(
                    image, rec["prompt"], response_format=response_format
                )
                result["latency_s"] = round(time.perf_counter() - t0, 3)
                parsed = parse_json(response)
                score = _severity_score(rec["task"], parsed)
                result.update(
                    ok=True,
                    json_parsed=parsed is not None,
                    json_valid=score is not None,
                    severity_score=score,
                    response=response,
                )
            except Exception as e:  # noqa: BLE001
                result.update(
                    ok=False,
                    json_valid=False,
                    severity_score=None,
                    latency_s=round(time.perf_counter() - t0, 3),
                    error=f"{type(e).__name__}: {e}",
                )
        out_fh.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_fh.flush()
        done += 1
        if done % 10 == 0 or done == len(records):
            print(f"  {done}/{len(records)} calls done", file=sys.stderr)
        return result

    return list(await asyncio.gather(*(one(r) for r in records)))


def _summarize(results: list[dict], model_label: str) -> dict:
    by_task: dict[str, list[dict]] = {}
    for r in results:
        by_task.setdefault(r["task"], []).append(r)

    def _valid_rate(rows):
        return round(sum(1 for r in rows if r.get("json_valid")) / len(rows), 4) if rows else None

    latencies = sorted(r["latency_s"] for r in results if r.get("ok") and "latency_s" in r)

    def _pct(p):
        if not latencies:
            return None
        return round(latencies[min(len(latencies) - 1, int(p * len(latencies)))], 2)

    discrimination = {}
    for task, rows in by_task.items():
        scores: dict[tuple, dict[str, float]] = {}
        for r in rows:
            if r.get("severity_score") is None:
                continue
            key = (r["doc_id"], r["page_index"])
            scores.setdefault(key, {})[r["variant"]] = r["severity_score"]
        wins = ties = losses = 0
        for pair in scores.values():
            bad_scores = [v for k, v in pair.items() if k.startswith("bad")]
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
        discrimination[task] = {
            "pairs": total,
            "bad_flagged_more": wins,
            "ties": ties,
            "gold_flagged_more": losses,
            "win_rate": round(wins / total, 4) if total else None,
        }

    return {
        "model": model_label,
        "calls": len(results),
        "errors": sum(1 for r in results if not r.get("ok")),
        "json_parsed_rate": (
            round(sum(1 for r in results if r.get("json_parsed")) / len(results), 4)
            if results
            else None
        ),
        "valid_json_rate": _valid_rate(results),
        "valid_json_rate_by_task": {t: _valid_rate(rows) for t, rows in by_task.items()},
        "latency_p50_s": _pct(0.50),
        "latency_p95_s": _pct(0.95),
        "mean_severity_by_task_variant": {
            t: {
                v: round(statistics.mean(s), 3)
                for v in sorted({r["variant"] for r in rows})
                if (
                    s := [
                        r["severity_score"]
                        for r in rows
                        if r["variant"] == v and r.get("severity_score") is not None
                    ]
                )
            }
            for t, rows in by_task.items()
        },
        "gold_vs_bad_discrimination": discrimination,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval", type=Path, required=True, help="eval JSONL from the exporter")
    ap.add_argument("--out", type=Path, required=True, help="per-call results JSONL")
    ap.add_argument("--model", default="", help="override OLLAMA_VISION_MODEL")
    ap.add_argument("--base-url", default="", help="override VISION_BASE_URL")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N records")
    ap.add_argument("--tasks", default="", help="comma-separated task filter")
    ap.add_argument("--docs", default="", help="comma-separated doc_id filter")
    ap.add_argument("--response-format", action="store_true",
                    help='send response_format={"type": "json_object"}')
    ap.add_argument("--resume", action="store_true",
                    help="skip example_ids already present in --out")
    ap.add_argument("--dry-run", action="store_true",
                    help="print planned calls, no model traffic")
    args = ap.parse_args()

    if args.model:
        os.environ["OLLAMA_VISION_MODEL"] = args.model
    if args.base_url:
        os.environ["VISION_BASE_URL"] = args.base_url

    from project_remedy.config import load_config
    from project_remedy.pdf_vision import _parse_json_response, create_provider_from_config

    config = load_config(env_path=REPO_ROOT / ".env", yaml_path=REPO_ROOT / "config.yaml")
    args.model_label = config.api.vision_model
    provider = create_provider_from_config(config)
    if provider is None:
        print("No vision provider available (check OLLAMA_API_KEY / base URLs)",
              file=sys.stderr)
        return 1

    eval_path = args.eval if args.eval.is_absolute() else REPO_ROOT / args.eval
    args.eval = eval_path
    records = [json.loads(line) for line in eval_path.read_text().splitlines() if line.strip()]

    if args.tasks:
        wanted = {t.strip() for t in args.tasks.split(",") if t.strip()}
        records = [r for r in records if r["task"] in wanted]
    if args.docs:
        wanted = {d.strip() for d in args.docs.split(",") if d.strip()}
        records = [r for r in records if r["doc_id"] in wanted]

    out_path = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prior: list[dict] = []
    if args.resume and out_path.exists():
        prior = [json.loads(line) for line in out_path.read_text().splitlines() if line.strip()]
        seen = {r["example_id"] for r in prior}
        records = [r for r in records if r["example_id"] not in seen]
        print(f"resume: {len(seen)} already done, {len(records)} remaining", file=sys.stderr)

    if args.limit:
        records = records[: args.limit]

    print(f"model={args.model_label}  records={len(records)}  "
          f"concurrency={args.concurrency}  response_format={args.response_format}",
          file=sys.stderr)

    if args.dry_run:
        for r in records[:20]:
            print(f"  {r['example_id']}  image={r['image']}  prompt={len(r['prompt'])} chars")
        print(f"(dry run - {len(records)} calls planned, none made)")
        return 0

    with out_path.open("a", encoding="utf-8") as out_fh:
        results = asyncio.run(_run_records(records, provider, _parse_json_response, args, out_fh))

    summary = _summarize(prior + results, args.model_label)
    summary_path = out_path.with_suffix(out_path.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nper-call results: {out_path}\nsummary: {summary_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
