from __future__ import annotations

import importlib.util
import asyncio
import json
import sys
import types
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]


def load_tool(name: str):
    path = ROOT / "tools" / "finetune" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_repo_tool(name: str):
    path = ROOT / "tools" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_heading_metric_scores_exact_corrections_and_pass_false_positive():
    metrics = load_tool("eval_task_metrics")
    gold = {
        "status": "fail",
        "findings": [
            {"element_index": 3, "correct_tag": "H2"},
            {"element_index": 5, "correct_tag": "H3"},
        ],
    }
    pred = {
        "status": "fail",
        "findings": [
            {"element_index": 3, "correct_tag": "H2"},
            {"element_index": 5, "correct_tag": "H3"},
        ],
    }

    score = metrics.score_one("heading_hierarchy", gold, pred)

    assert score["status_match"] is True
    assert score["exact_corrections"] is True
    assert score["correction_recall"] == 1.0
    assert score["correction_precision"] == 1.0

    false_positive = metrics.score_one(
        "heading_hierarchy",
        {"status": "pass", "findings": []},
        {"status": "fail", "findings": [{"element_index": 1, "correct_tag": "H1"}]},
    )
    assert false_positive["pass_false_positive"] is True


def test_table_metric_confusion_and_status_accuracy():
    metrics = load_tool("eval_task_metrics")
    scores = [
        metrics.score_one("table_structure", {"status": "fail"}, {"status": "fail"}),
        metrics.score_one("table_structure", {"status": "pass"}, {"status": "fail"}),
        metrics.score_one("table_structure", {"status": "pass"}, None),
    ]

    summary = metrics.summarize(scores)

    table = summary["by_task"]["table_structure"]
    assert table["status_accuracy"] == 0.3333
    assert table["confusion"] == {"fail->fail": 1, "pass->None": 1, "pass->fail": 1}
    assert table["pass_false_positive_rate"] == 0.3333


def test_reading_order_corruption_emits_balanced_fail_and_pass():
    builder = load_tool("build_delivered_dataset")
    order = "\n".join(
        f"{i:3d}. /P  (text: \"Block {i}\")"
        for i in range(1, 9)
    )
    parsed = builder._parse_structure_order(order)

    examples = builder._reading_order_corruption_examples(order, parsed, emit_pass=True)

    assert len(examples) == 2
    fail_prompt, fail_target, fail_prov = examples[0]
    pass_prompt, pass_target, pass_prov = examples[1]
    assert fail_target["issues"]
    assert pass_target["issues"] == []
    assert fail_prov.endswith("-corrupt")
    assert pass_prov.endswith("-pass")
    assert fail_prompt != pass_prompt


def test_contrast_builder_labels_ratios_around_threshold(tmp_path):
    contrast = load_tool("build_contrast_corpus")
    fail_fg = contrast.gray_for_ratio_on_white(4.35)
    pass_fg = contrast.gray_for_ratio_on_white(4.7)

    fail_ratio = contrast.contrast_ratio(fail_fg, (255, 255, 255))
    pass_ratio = contrast.contrast_ratio(pass_fg, (255, 255, 255))

    assert fail_ratio < 4.5
    assert pass_ratio >= 4.5
    assert contrast.target_for_ratio(fail_ratio, fail_fg, (255, 255, 255))["issues"]
    assert contrast.target_for_ratio(pass_ratio, pass_fg, (255, 255, 255))["issues"] == []

    records = contrast.build_records(tmp_path, count=8, seed=7)
    assert {rec["variant"] for rec in records} == {"fail", "pass"}
    assert all((tmp_path / rec["image"]).exists() for rec in records)


def test_multitask_union_preserves_resolvable_relative_image_paths(tmp_path):
    union = load_tool("build_multitask_dataset")
    source = tmp_path / "data_table"
    renders = source / "renders"
    renders.mkdir(parents=True)
    Image.new("RGB", (20, 20), (255, 255, 255)).save(renders / "sample.png")
    row = {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": "renders/sample.png"},
                {"type": "text", "text": "prompt"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "{\"status\":\"pass\"}"}]},
        ],
        "meta": {"doc_id": "d", "page": 1, "task": "table_structure"},
    }
    for split in ("train", "val"):
        (source / f"{split}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

    out = tmp_path / "data_multitask"
    manifest = union.build(out, [source], seed=1)
    train_row = json.loads((out / "train.jsonl").read_text().splitlines()[0])
    image_rel = train_row["messages"][0]["content"][0]["image"]

    assert manifest["tasks_train"] == {"table_structure": 1}
    assert not Path(image_rel).is_absolute()
    assert (out / image_rel).resolve().exists()


def test_multitask_union_weights_train_rows_only(tmp_path):
    union = load_tool("build_multitask_dataset")
    source = tmp_path / "data_mixed"
    renders = source / "renders"
    renders.mkdir(parents=True)
    Image.new("RGB", (20, 20), (255, 255, 255)).save(renders / "sample.png")

    def row(task: str, variant: str) -> dict:
        return {
            "messages": [
                {"role": "user", "content": [
                    {"type": "image", "image": "renders/sample.png"},
                    {"type": "text", "text": "prompt"},
                ]},
                {"role": "assistant", "content": [{"type": "text", "text": "{\"status\":\"pass\"}"}]},
            ],
            "meta": {"doc_id": f"{task}-{variant}", "page": 1, "task": task, "variant": variant},
        }

    train_rows = [
        row("contrast", "fail"),
        row("contrast", "pass"),
        row("table_structure", "pass"),
    ]
    val_rows = [
        row("contrast", "fail"),
        row("table_structure", "pass"),
    ]
    (source / "train.jsonl").write_text(
        "\n".join(json.dumps(item) for item in train_rows) + "\n",
        encoding="utf-8",
    )
    (source / "val.jsonl").write_text(
        "\n".join(json.dumps(item) for item in val_rows) + "\n",
        encoding="utf-8",
    )

    manifest = union.build(
        tmp_path / "weighted",
        [source],
        seed=1,
        task_weights={"contrast": 3},
    )

    assert manifest["tasks_train_unweighted"] == {
        "contrast": 2,
        "table_structure": 1,
    }
    assert manifest["tasks_train"] == {
        "contrast": 6,
        "table_structure": 1,
    }
    assert manifest["tasks_val"] == {
        "contrast": 1,
        "table_structure": 1,
    }


def test_generate_predictions_rows_align_with_eval_metrics(tmp_path):
    gen = load_tool("generate_predictions_hf")
    metrics = load_tool("eval_task_metrics")
    renders = tmp_path / "renders"
    renders.mkdir()
    Image.new("RGB", (20, 20), (255, 255, 255)).save(renders / "sample.png")
    row = {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": "renders/sample.png"},
                {"type": "text", "text": "prompt"},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": "{\"status\":\"pass\"}"}]},
        ],
        "meta": {
            "doc_id": "doc-a",
            "page": 2,
            "task": "table_structure",
            "variant": "pass",
        },
    }
    val = tmp_path / "val.jsonl"
    val.write_text(json.dumps(row) + "\n", encoding="utf-8")

    loaded = gen.load_records(val)
    pred_row = gen.prediction_row(loaded[0], 0, "{\"status\":\"pass\"}")

    assert Path(loaded[0]["messages"][0]["content"][0]["image"]).is_absolute()
    assert pred_row["example_id"] == metrics.record_key(row, 0)
    assert pred_row["task"] == "table_structure"
    assert pred_row["prediction"] == "{\"status\":\"pass\"}"
    assert pred_row["meta"]["doc_id"] == "doc-a"


def test_eval_metrics_prefers_same_order_predictions_for_duplicate_keys(tmp_path):
    metrics = load_tool("eval_task_metrics")
    fail_row = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "prompt"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "{\"status\":\"fail\"}"}]},
        ],
        "meta": {"doc_id": "doc-a", "page": 1, "task": "table_structure"},
    }
    pass_row = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "prompt"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "{\"status\":\"pass\"}"}]},
        ],
        "meta": {"doc_id": "doc-a", "page": 1, "task": "table_structure"},
    }
    val = tmp_path / "val.jsonl"
    val.write_text(json.dumps(fail_row) + "\n" + json.dumps(pass_row) + "\n", encoding="utf-8")
    preds = tmp_path / "preds.jsonl"
    duplicate_key = metrics.record_key(fail_row, 0)
    preds.write_text(
        json.dumps({"example_id": duplicate_key, "prediction": "{\"status\":\"fail\"}"}) + "\n"
        + json.dumps({"example_id": duplicate_key, "prediction": "{\"status\":\"pass\"}"}) + "\n",
        encoding="utf-8",
    )

    gold_rows = metrics.load_jsonl(val)
    keyed = metrics.load_predictions(preds, gold_rows, False)
    scores = []
    for i, rec in enumerate(gold_rows):
        key = metrics.record_key(rec, i)
        pred_text = keyed.get(str(i), keyed.get(key, ""))
        scores.append(metrics.score_one(
            metrics.task_name(rec),
            metrics.parse_jsonish(metrics.target_text(rec)),
            metrics.parse_jsonish(pred_text),
        ))

    summary = metrics.summarize(scores)
    assert summary["by_task"]["table_structure"]["status_accuracy"] == 1.0


def test_production_task_metrics_marks_contrast_corpus_as_non_gate(tmp_path):
    prod = load_tool("eval_production_task_metrics")
    renders = tmp_path / "renders"
    renders.mkdir()
    Image.new("RGB", (20, 20), (255, 255, 255)).save(renders / "gold.png")
    Image.new("RGB", (20, 20), (255, 255, 255)).save(renders / "bad.png")
    eval_rows = [
        {
            "example_id": "d_gold_p1_contrast",
            "doc_id": "d",
            "page_index": 1,
            "variant": "gold",
            "task": "contrast",
            "image": "renders/gold.png",
            "relevant_dimension": None,
        },
        {
            "example_id": "d_bad_p1_contrast",
            "doc_id": "d",
            "page_index": 1,
            "variant": "bad",
            "task": "contrast",
            "image": "renders/bad.png",
            "relevant_dimension": None,
        },
    ]
    results = [
        {**row, "response": "{\"issues\": []}"}
        for row in eval_rows
    ]

    summary, _samples = prod.build_summary(
        eval_rows,
        results,
        eval_dir=tmp_path,
        contrast_gate_metrics={"status_accuracy": 1.0},
    )

    contrast = summary["tasks"]["contrast"]
    assert contrast["gate_applicability"] == "not_applicable_current_production_corpus"
    assert contrast["verified_contrast_gate_metrics"] == {"status_accuracy": 1.0}
    assert summary["corpus_checks"]["contrast"]["current_production_corpus_is_valid_gate"] is False


def test_production_task_metrics_scores_reading_order_current_schema():
    prod = load_tool("eval_production_task_metrics")
    eval_rows = [
        {
            "example_id": "d_gold_p1_reading_order",
            "doc_id": "d",
            "page_index": 1,
            "variant": "gold",
            "task": "reading_order",
            "prompt_inputs": {"structure_order": "  1. /P"},
        },
        {
            "example_id": "d_bad_p1_reading_order",
            "doc_id": "d",
            "page_index": 1,
            "variant": "bad",
            "task": "reading_order",
            "prompt_inputs": {"structure_order": "  1. /P"},
        },
    ]
    results = [
        {**eval_rows[0], "response": "{\"page_layout\":\"single_column\",\"issues\":[],\"summary\":\"ok\"}"},
        {**eval_rows[1], "response": "{\"page_layout\":\"single_column\",\"issues\":[],\"summary\":\"ok\"}"},
    ]

    summary, samples = prod.build_summary(
        eval_rows,
        results,
        eval_dir=Path("."),
        compute_image_delta=False,
    )

    reading = summary["tasks"]["reading_order"]
    assert reading["schema"] == "page_layout + issues + summary"
    assert reading["empty_issues_means_pass"] is True
    assert reading["corrected_order_required"] is False
    assert reading["corrected_order_accuracy"] is None
    assert reading["miss_counts"]["bad_not_flagged"] == 1
    assert any(sample["sample_type"] == "reading_order_bad_not_flagged" for sample in samples)


def test_production_task_metrics_classifies_heading_gold_flags_from_logical_order():
    prod = load_tool("eval_production_task_metrics")
    source = {
        "example_id": "d_gold_p1_heading_hierarchy",
        "doc_id": "d",
        "page_index": 1,
        "variant": "gold",
        "task": "heading_hierarchy",
        "prompt_inputs": {"logical_order": "  1.   /Document\n  2.       /LI"},
    }
    result = {
        **source,
        "response": json.dumps({
            "status": "fail",
            "findings": [
                {
                    "severity": "warning",
                    "element_index": 2,
                    "current_tag": "LI",
                    "visible_text": "Program Review",
                    "correct_tag": "H1",
                }
            ],
        }),
    }

    summary, samples = prod.build_summary(
        [source],
        [result],
        eval_dir=Path("."),
        compute_image_delta=False,
    )

    heading = summary["tasks"]["heading_hierarchy"]
    assert heading["gold_flagged_pages"] == 1
    assert heading["gold_flag_classification_counts"] == {
        "likely_true_residual_structure_issue": 1
    }
    assert samples[0]["classified_findings"][0]["classification"]["classification"] == (
        "likely_true_residual_structure_issue"
    )


def test_train_lora_init_adapter_loads_existing_adapter_as_trainable(monkeypatch, tmp_path):
    trainer = load_tool("train_lora_vision_hf")
    calls = {}

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, path, is_trainable=False):
            calls["model"] = model
            calls["path"] = path
            calls["is_trainable"] = is_trainable
            return "attached-adapter"

    monkeypatch.setitem(
        sys.modules,
        "peft",
        types.SimpleNamespace(PeftModel=FakePeftModel),
    )
    base_model = object()
    adapter_dir = tmp_path / "adapter"

    attached = trainer._attach_or_create_lora(
        base_model,
        init_adapter=adapter_dir,
        rank=16,
        alpha=32,
        tune_vision=False,
    )

    assert attached == "attached-adapter"
    assert calls == {
        "model": base_model,
        "path": str(adapter_dir),
        "is_trainable": True,
    }


def test_remedy_router_vllm_command_and_env_profile(tmp_path):
    serve = load_tool("serve_remedy_router_vllm")
    adapter_dirs = [
        "artifacts/lamc-qwen3vl-32b-lora-v2",
        "artifacts/lamc-qwen3vl-32b-table-lora",
        "outputs_runpod/lamc-qwen3vl-32b-contrast-lora",
        "outputs_runpod/lamc-qwen3vl-32b-reading-order-lora",
        "outputs_runpod/lamc-qwen3vl-32b-heading-lora",
    ]
    for rel in adapter_dirs:
        path = tmp_path / rel
        path.mkdir(parents=True)
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")

    args = serve.parse_args([
        "--adapter-root", str(tmp_path),
        "--python", "/workspace/ft/bin/python",
        "--dry-run",
    ])
    modules = serve.adapter_modules(args)
    command = serve.build_command(args)

    assert serve.missing_adapters(modules) == []
    assert "--enable-lora" in command
    assert command[command.index("--lora-modules") + 1:] == [
        f"qwen3vl-32b-remedy={tmp_path / 'artifacts/lamc-qwen3vl-32b-lora-v2'}",
        f"qwen3vl-32b-remedy-alt-v2={tmp_path / 'artifacts/lamc-qwen3vl-32b-lora-v2'}",
        f"qwen3vl-32b-remedy-table-v1={tmp_path / 'artifacts/lamc-qwen3vl-32b-table-lora'}",
        f"qwen3vl-32b-remedy-contrast-v1={tmp_path / 'outputs_runpod/lamc-qwen3vl-32b-contrast-lora'}",
        f"qwen3vl-32b-remedy-reading-order-v1={tmp_path / 'outputs_runpod/lamc-qwen3vl-32b-reading-order-lora'}",
        f"qwen3vl-32b-remedy-heading-v1={tmp_path / 'outputs_runpod/lamc-qwen3vl-32b-heading-lora'}",
    ]
    env = serve.router_env("http://router.test/v1")
    assert "OLLAMA_VISION_MODEL=qwen3vl-32b-remedy" in env
    assert "contrast:qwen3vl-32b-remedy-contrast-v1" in env
    assert "OLLAMA_VISION_ROUTER_ALLOW_FALLBACK=0" in env


def test_run_vision_eval_passes_task_to_provider(tmp_path):
    runner = load_repo_tool("run_vision_eval")
    eval_path = tmp_path / "eval.jsonl"
    (tmp_path / "page.png").write_bytes(b"fake")
    records = [{
        "example_id": "doc_bad_p1_contrast",
        "doc_id": "doc",
        "task": "contrast",
        "variant": "bad",
        "page_index": 1,
        "image": "page.png",
        "prompt": "Find contrast issues",
    }]

    class RecordingProvider:
        def __init__(self) -> None:
            self.calls = []

        async def analyze_image(
            self,
            image_path,
            prompt,
            *,
            response_format=None,
            task=None,
        ):
            self.calls.append({
                "image_path": image_path,
                "prompt": prompt,
                "response_format": response_format,
                "task": task,
            })
            return '{"issues":[{"severity":"error"}]}'

    provider = RecordingProvider()
    args = types.SimpleNamespace(
        concurrency=1,
        response_format=True,
        eval=eval_path,
        model_label="router",
    )
    out_path = tmp_path / "results.jsonl"

    with out_path.open("w", encoding="utf-8") as out_fh:
        results = asyncio.run(
            runner._run_records(records, provider, json.loads, args, out_fh)
        )

    assert provider.calls == [{
        "image_path": tmp_path / "page.png",
        "prompt": "Find contrast issues",
        "response_format": {"type": "json_object"},
        "task": "contrast",
    }]
    assert results[0]["json_valid"] is True
    assert results[0]["severity_score"] == 2.0


def test_router_readiness_combines_adapter_and_metric_gates(tmp_path):
    readiness = load_tool("eval_router_readiness")
    main_repo = tmp_path / "main"
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    table_metrics = tmp_path / "table.metrics.json"
    prod_metrics = tmp_path / "production.task_metrics.json"
    alt_summary = tmp_path / "alt.summary.json"

    adapter_paths = [
        main_repo / "artifacts/lamc-qwen3vl-32b-lora-v2",
        main_repo / "artifacts/lamc-qwen3vl-32b-table-lora",
        tmp_path / "outputs_runpod/lamc-qwen3vl-32b-contrast-lora",
        tmp_path / "outputs_runpod/lamc-qwen3vl-32b-reading-order-lora",
        tmp_path / "outputs_runpod/lamc-qwen3vl-32b-heading-lora",
    ]
    for path in adapter_paths:
        path.mkdir(parents=True)
        (path / "adapter_config.json").write_text("{}", encoding="utf-8")
        (path / "adapter_model.safetensors").write_text("weights", encoding="utf-8")

    alt_summary.write_text(json.dumps({
        "valid_json_rate": 0.902,
        "gold_vs_bad_discrimination": {
            "alt_text_quality": {
                "win_rate": 0.9333,
                "gold_flagged_more": 0,
            }
        },
    }), encoding="utf-8")
    table_metrics.write_text(json.dumps({
        "by_task": {
            "table_structure": {
                "status_accuracy": 1.0,
                "valid_json_rate": 1.0,
                "pass_false_positive_rate": 0.0,
            }
        }
    }), encoding="utf-8")
    (eval_dir / "contrast.tuned.metrics.json").write_text(json.dumps({
        "by_task": {
            "contrast": {
                "status_accuracy": 1.0,
                "near_threshold_status_accuracy": 1.0,
                "valid_json_rate": 1.0,
                "pass_false_positive_rate": 0.0,
            }
        }
    }), encoding="utf-8")
    (eval_dir / "reading_order.tuned.metrics.json").write_text(json.dumps({
        "by_task": {
            "reading_order": {
                "status_accuracy": 1.0,
                "valid_json_rate": 1.0,
                "pass_false_positive_rate": 0.0,
            }
        }
    }), encoding="utf-8")
    (eval_dir / "heading.tuned.metrics.json").write_text(json.dumps({
        "by_task": {
            "heading_hierarchy": {
                "status_accuracy": 1.0,
                "exact_correction_accuracy": 0.8857,
                "valid_json_rate": 1.0,
                "pass_false_positive_rate": 0.0,
            }
        }
    }), encoding="utf-8")
    prod_metrics.write_text(json.dumps({
        "corpus_checks": {
            "contrast": {"current_production_corpus_is_valid_gate": False}
        },
        "model_alias_semantics": {"stable_current_alias": "qwen3vl-32b-remedy"},
        "recommendation": {"decision": "gather_better_eval_data_first"},
    }), encoding="utf-8")
    args = types.SimpleNamespace(
        eval_dir=eval_dir,
        main_repo=main_repo,
        alt_summary=alt_summary,
        table_metrics=table_metrics,
        production_task_metrics=prod_metrics,
        alt_adapter=None,
        table_adapter=None,
        contrast_adapter=tmp_path / "outputs_runpod/lamc-qwen3vl-32b-contrast-lora",
        reading_order_adapter=tmp_path / "outputs_runpod/lamc-qwen3vl-32b-reading-order-lora",
        heading_adapter=tmp_path / "outputs_runpod/lamc-qwen3vl-32b-heading-lora",
    )

    summary = readiness.build_summary(args)

    assert summary["decision"] == "ready_for_live_router_smoke"
    assert summary["final_ship_ready"] is False
    assert all(item["passed"] for item in summary["adapters"])
    assert all(
        item["passed"]
        for gates in summary["task_gates"].values()
        for item in gates
    )

    live_summary = tmp_path / "router_full.production.jsonl.summary.json"
    live_summary.write_text(json.dumps({
        "errors": 0,
        "valid_json_rate": 0.9887,
        "valid_json_rate_by_task": {
            "alt_text_quality": 1.0,
            "contrast": 1.0,
            "heading_hierarchy": 0.9615,
            "reading_order": 1.0,
        },
    }), encoding="utf-8")
    args.live_router_summary = live_summary

    summary = readiness.build_summary(args)

    assert summary["decision"] == "ready_for_heldout_lamc_pipeline_validation"
    assert summary["final_ship_ready"] is False
    assert all(item["passed"] for item in summary["live_router_gates"])

    heldout_summary = tmp_path / "heldout_lamc_validation.summary.json"
    heldout_summary.write_text(json.dumps({
        "count": 2,
        "passed": 1,
        "failed": 1,
        "pass_rate": 0.5,
        "verapdf_passed": 2,
        "check_zero_failures": 1,
        "report_zero_failures": 2,
        "text_fidelity_passed": 2,
        "visual_fidelity_passed": 2,
        "content_fidelity_passed": 1,
    }), encoding="utf-8")
    args.heldout_validation_summary = heldout_summary

    summary = readiness.build_summary(args)

    assert summary["decision"] == "heldout_lamc_pipeline_validation_failed"
    assert summary["final_ship_ready"] is False
    assert any(not item["passed"] for item in summary["heldout_validation_gates"])

    heldout_summary.write_text(json.dumps({
        "count": 2,
        "passed": 2,
        "failed": 0,
        "pass_rate": 1.0,
        "verapdf_passed": 2,
        "check_zero_failures": 2,
        "report_zero_failures": 2,
        "text_fidelity_passed": 2,
        "visual_fidelity_passed": 2,
        "content_fidelity_passed": 2,
    }), encoding="utf-8")

    summary = readiness.build_summary(args)

    assert summary["decision"] == "router_final_gates_passed"
    assert summary["final_ship_ready"] is True
    assert summary["final_ship_blocker"] == ""
    assert all(item["passed"] for item in summary["heldout_validation_gates"])


def test_heldout_lamc_validation_parses_json_with_token_footer():
    heldout = load_tool("run_heldout_lamc_validation")
    payload, footer = heldout.parse_first_json_object(
        '{"summary":{"failed":0}}\nTokens: 10 in + 2 out = 12 billed'
    )

    assert payload == {"summary": {"failed": 0}}
    assert footer == "Tokens: 10 in + 2 out = 12 billed"


def test_heldout_lamc_validation_summary_counts_final_gates():
    heldout = load_tool("run_heldout_lamc_validation")
    records = [
        {
            "source": "pass.pdf",
            "passed": True,
            "output_exists": True,
            "check": {"passed": True, "summary": {"failed": 0}},
            "report": {
                "passed": True,
                "summary": {"verapdf_passed": True, "failed_checks": 0},
            },
            "text_fidelity": {"normalized_text_equal": True},
            "visual_fidelity": {"visual_match": True},
            "content_fidelity_passed": True,
        },
        {
            "source": "fail.pdf",
            "passed": False,
            "output_exists": True,
            "fix": {"returncode": 0},
            "check": {"passed": False, "summary": {"failed": 1}},
            "report": {
                "passed": True,
                "summary": {"verapdf_passed": True, "failed_checks": 0},
            },
            "text_fidelity": {"normalized_text_equal": True},
            "visual_fidelity": {"visual_match": True},
            "content_fidelity_passed": True,
        },
    ]

    summary = heldout.summarize(records)

    assert summary["count"] == 2
    assert summary["passed"] == 1
    assert summary["failed"] == 1
    assert summary["pass_rate"] == 0.5
    assert summary["verapdf_passed"] == 2
    assert summary["check_zero_failures"] == 1
    assert summary["text_fidelity_passed"] == 2
    assert summary["visual_fidelity_passed"] == 2
    assert summary["content_fidelity_passed"] == 2
    assert summary["failures"][0]["source"] == "fail.pdf"


def test_heldout_lamc_text_fidelity_reports_token_preservation(monkeypatch):
    heldout = load_tool("run_heldout_lamc_validation")

    def fake_normalized_text(path):
        text = {
            "source.pdf": "Alpha beta gamma.",
            "output.pdf": "Gamma alpha beta.",
        }[path.name]
        return {
            "available": True,
            "normalized_chars": len(text),
            "_text": text,
        }

    monkeypatch.setattr(heldout, "normalized_text", fake_normalized_text)

    result = heldout.text_fidelity(Path("source.pdf"), Path("output.pdf"))

    assert result["normalized_text_equal"] is False
    assert result["token_multiset_equal"] is True
    assert result["alnum_char_multiset_equal"] is True
    assert result["missing_token_sample"] == []
    assert result["added_token_sample"] == []
    assert result["sequence_similarity"] < 1.0
