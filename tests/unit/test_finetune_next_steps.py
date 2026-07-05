from __future__ import annotations

import importlib.util
import json
import sys
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
