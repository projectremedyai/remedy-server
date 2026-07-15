"""Tests for grouped five-task campaign dataset rebuilding."""

from __future__ import annotations

import json
from pathlib import Path

from tools.finetune.remedy_nemo_rl.dataset import DEFAULT_HOLDOUT_PATTERNS, build_dataset
from tools.finetune.finalize_dataset import to_conversation


TASKS = (
    "alt_text_quality",
    "table_structure",
    "contrast",
    "reading_order",
    "heading_hierarchy",
)


def _target(task: str, failed: bool) -> dict:
    if task == "alt_text_quality":
        return {
            "figures": [
                {
                    "figure_index": 1,
                    "status": "fail" if failed else "pass",
                    "issue_type": "inaccurate" if failed else "",
                }
            ]
        }
    if task == "table_structure":
        return {
            "status": "fail" if failed else "pass",
            "findings": [{"issue_id": "headers", "fixer": "fix_table_headers"}] if failed else [],
        }
    if task == "contrast":
        return {"issues": [{"issue_id": "body", "ratio": 3.9}] if failed else []}
    if task == "reading_order":
        return {
            "issues": [
                {"issue_id": "order", "suggestion": "Restore reading order: 1, 2, 3"}
            ]
            if failed
            else []
        }
    return {
        "status": "fail" if failed else "pass",
        "findings": [{"element_index": 3, "correct_tag": "H2"}] if failed else [],
    }


def _row(task: str, doc_id: str, failed: bool, image_name: str, variant: int) -> dict:
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_name},
                    {"type": "text", "text": f"Inspect {task}."},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": json.dumps(_target(task, failed))}],
            },
        ],
        "meta": {
            "doc_id": doc_id,
            "page": variant + 1,
            "task": task,
            "variant": f"v{variant}",
            "source_family": "synthetic" if "synthetic" in doc_id else "real",
        },
    }


def _write_sources(tmp_path: Path) -> list[Path]:
    image = tmp_path / "page.png"
    image.write_bytes(b"not-a-real-png-but-resolvable")
    paths = []
    for task in TASKS:
        path = tmp_path / f"{task}.jsonl"
        rows = []
        for doc_index in range(20):
            doc_id = f"{'synthetic' if doc_index % 3 == 0 else 'real'}_{task}_{doc_index:02d}"
            rows.append(_row(task, doc_id, False, image.name, 0))
            rows.append(_row(task, doc_id, True, image.name, 1))
        rows.append(_row(task, "LAMC_2025-26_Catalog", True, image.name, 0))
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_grouped_splits_holdouts_balance_and_gym_targets(tmp_path: Path) -> None:
    sources = _write_sources(tmp_path)
    out = tmp_path / "campaign"
    holdout_pdf = tmp_path / "1d3a9d09c6f7_LAMC-Catalog-2025-2026.pdf"
    holdout_pdf.write_bytes(b"held-out-catalog")

    manifest = build_dataset(sources, out, seed=20260714, holdout_files=[holdout_pdf])

    split_docs = {}
    for split in ("train", "validation", "test"):
        rows = _load(out / "sft" / f"{split}.jsonl")
        split_docs[split] = {row["meta"]["doc_id"] for row in rows}
        assert all("catalog" not in doc_id.lower() for doc_id in split_docs[split])
        assert all(not Path(row["messages"][0]["content"][0]["image"]).is_absolute() for row in rows)
        assert all(
            ((out / "sft") / row["messages"][0]["content"][0]["image"]).resolve().is_file()
            for row in rows
        )

        gym_rows = _load(out / "gym" / f"{split}.jsonl")
        assert len(gym_rows) == len(rows)
        assert all(row["responses_create_params"]["input"][0]["content"][0]["type"] == "input_image" for row in gym_rows)
        assert all(row["responses_create_params"]["input"][0]["content"][0]["image_url"].startswith("data:image/png;base64,") for row in gym_rows)
        assert all("verifier_target" in row for row in gym_rows)
        assert all("verifier_target" not in row["responses_create_params"] for row in gym_rows)

    assert split_docs["train"].isdisjoint(split_docs["validation"])
    assert split_docs["train"].isdisjoint(split_docs["test"])
    assert split_docs["validation"].isdisjoint(split_docs["test"])
    assert manifest["split_percentages"] == {"train": 70, "validation": 15, "test": 15}
    assert manifest["excluded_rows"] == len(TASKS)
    assert tuple(manifest["holdout_patterns"]) == DEFAULT_HOLDOUT_PATTERNS
    assert manifest["holdout_files"][str(holdout_pdf.resolve())] == {
        "sha256": "9f50c9795fb244b145da6533451ee27b37ce64ef80e8e7a031aa71a2d702f531",
        "size_bytes": 16,
    }
    for task in TASKS:
        counts = manifest["counts"]["train"][task]
        fail_fraction = counts["fail"] / counts["total"]
        assert 0.45 <= fail_fraction <= 0.55
        assert set(counts["source_types"]) >= {"real", "synthetic"}
        for split in ("train", "validation", "test"):
            task_sft = _load(out / "sft" / task / f"{split}.jsonl")
            task_gym = _load(out / "gym" / task / f"{split}.jsonl")
            assert task_sft
            assert len(task_sft) == len(task_gym)
            assert {row["meta"]["task"] for row in task_sft} == {task}
            assert all(
                ((out / "sft" / task) / row["messages"][0]["content"][0]["image"]).resolve().is_file()
                for row in task_sft
            )


def test_dataset_hashes_are_reproducible(tmp_path: Path) -> None:
    sources = _write_sources(tmp_path)

    first = build_dataset(sources, tmp_path / "first", seed=20260714)
    second = build_dataset(sources, tmp_path / "second", seed=20260714)

    assert first["dataset_hashes"] == second["dataset_hashes"]
    assert first["source_hashes"] == second["source_hashes"]


def test_default_holdouts_include_exact_blocked_catalog_identifiers() -> None:
    assert {"ad142f824d25", "0312e204645b", "1d3a9d09c6f7"} <= set(DEFAULT_HOLDOUT_PATTERNS)


def test_delivered_builder_provenance_survives_finalization() -> None:
    row = to_conversation(
        {
            "image": "/tmp/page.png",
            "prompt": "inspect",
            "target": '{"status":"pass","findings":[]}',
            "doc_id": "doc",
            "page": 1,
            "task": "table_structure",
            "provenance": "delivered-derived-pass",
        }
    )

    assert row["meta"]["source_type"] == "delivered"
    assert row["meta"]["provenance"] == "delivered-derived-pass"
