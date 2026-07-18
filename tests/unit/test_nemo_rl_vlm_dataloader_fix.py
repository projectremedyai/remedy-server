"""Tests for the NeMo RL VLM SFT dataloader fix.

Root cause (proven locally against pinned NeMo RL c339070 + datasets==4.4.1):
``load_dataset("json", ...)`` unifies the Arrow struct schema across every
multimodal ``content`` list, injecting phantom ``image: None`` keys into text
parts. Qwen-style chat templates test key MEMBERSHIP (``'image' in content``),
so corrupted text parts render as image placeholders — 2 ``<|image_pad|>``
occurrences against 1 real image — and the HF processor's unbounded expansion
loop dies with ``IndexError: index 1 is out of bounds for dimension 0 with
size 1`` on the first dataloader batch.

The fix has three parts, each pinned by a test here:
1. a read-time None-strip patch applied to NeMo RL's ``sft_processor``
   (read time is the ONLY correct placement: ``Dataset.map`` re-encodes
   through Arrow and re-injects the Nones, and ``use_preserving_dataset=True``
   is rejected by ``concatenate_datasets`` at run_sft.py:92);
2. an idempotent ``git apply`` hook in brev_setup.sh;
3. a paid-run gate: a dataloader preflight that pushes real rows through the
   real processing path before any training money is spent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from tools.finetune.remedy_nemo_rl.campaign import (
    build_preflight_command,
    build_sft_command,
)
from tools.finetune.remedy_nemo_rl.dataloader_preflight import row_failures

ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "tools/finetune/patches/nemo_rl_strip_none_multimodal_content.patch"
SETUP = ROOT / "tools/finetune/brev_setup.sh"

VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"


def test_strip_none_patch_targets_sft_processor_read_time() -> None:
    text = PATCH.read_text(encoding="utf-8")
    assert "nemo_rl/data/processors.py" in text
    assert "if v is not None" in text
    # The strip must run before message formatting, i.e. inside sft_processor,
    # not inside format_data (Dataset.map would re-encode through Arrow and
    # re-inject the Nones).
    assert "def sft_processor" in text or "get_formatted_message_log" in text


def test_brev_setup_applies_the_patch_idempotently() -> None:
    text = SETUP.read_text(encoding="utf-8")
    assert "nemo_rl_strip_none_multimodal_content.patch" in text
    # Re-running setup on a box that already has the patch must not fail:
    # skip when the patch is already present (reverse-apply check succeeds).
    assert "apply --reverse --check" in text
    # The apply must precede the import smoke so a failed apply fails setup.
    assert text.index("_content.patch") < text.index("nemo_rl_and_gym_import_ok")


def test_brev_setup_exposes_patched_clone_over_baked_image_copy() -> None:
    """The official image bakes its own nemo_rl at /opt/nemo-rl, which shadows
    the pinned+patched clone at import time (proven on the 2026-07-16 paid
    smoke: preflight crashed identically because /opt/nemo-rl has no patch).
    Setup must symlink the clone's nemo_rl package into the payload dir, which
    is first on the container PYTHONPATH. The RL repo ROOT must never go on
    PYTHONPATH: its tools/ is a regular package and would shadow this repo's
    tools/ namespace package."""
    setup_text = SETUP.read_text(encoding="utf-8")
    assert "ln -sfn /home/ubuntu/RL/nemo_rl /home/ubuntu/workspace/remedy-server/nemo_rl" in setup_text
    run_text = (ROOT / "tools/finetune/brev_vm_container_run.sh").read_text(encoding="utf-8")
    assert "PYTHONPATH=/home/ubuntu/workspace/remedy-server" in run_text
    assert "PYTHONPATH=/home/ubuntu/RL" not in run_text


def test_row_failures_flags_placeholder_desync_and_dropped_text() -> None:
    corrupted = VISION_BLOCK * 2  # phantom-key rendering: text part became an image
    failures = row_failures(
        corrupted, image_count=1, must_contain="verifying image alt"
    )
    assert any("vision" in failure for failure in failures)
    assert any("text" in failure for failure in failures)


def test_row_failures_passes_a_healthy_row() -> None:
    healthy = VISION_BLOCK + "You are verifying image alt text quality for one page."
    assert row_failures(healthy, image_count=1, must_contain="verifying image alt") == []


def test_row_failures_requires_text_only_when_snippet_given() -> None:
    text_only = "plain text row with no figures"
    assert row_failures(text_only, image_count=0, must_contain=None) == []


def test_lora_target_modules_are_language_scoped_wildcards() -> None:
    """Bare module names like "q_proj" silently match NOTHING in NeMo
    Automodel's ModuleMatcher: it compares the FULL dotted path with an
    anchored re.match, so only wildcard patterns can hit nested modules.
    The 2026-07-16 smoke trained for 28 steps with ZERO LoRA modules applied
    (val_loss frozen at 1.6127, adapter file empty) because of this.

    Patterns must also be scoped to the language model: Qwen2.5-VL's VISION
    tower reuses gate/up/down_proj names, and unscoped '*.gate_proj' would
    train it — violating the campaign's 0-visual-trainables constraint."""
    import yaml

    for config_name in ("sft_qwen25_vl_3b_h200.yaml", "sft_qwen35_9b_h200.yaml"):
        config = yaml.safe_load(
            (ROOT / "tools/finetune/nemo_rl_configs" / config_name).read_text(
                encoding="utf-8"
            )
        )
        targets = config["policy"]["dtensor_cfg"]["lora_cfg"]["target_modules"]
        assert targets, f"{config_name}: empty target_modules"
        for pattern in targets:
            assert pattern.startswith("*.language_model."), (
                f"{config_name}: target module {pattern!r} must be a "
                "language-scoped wildcard ('*.language_model.*.<proj>') — "
                "bare names match nothing and unscoped wildcards hit the "
                "vision tower"
            )


def test_truncation_patch_drops_media_with_the_token_stub() -> None:
    """sft_processor's over-length truncation chops token_ids to a ~4-token
    stub but leaves pixel_values attached, so ANY row longer than
    max_total_sequence_length crashes the whole run in the model forward:
    'Image features and image tokens do not match, tokens: 0, features: N'
    (hit live on table_structure, 2026-07-16). The patch must strip media
    keys alongside the token stub so overlong rows are masked, not fatal."""
    patch = ROOT / "tools/finetune/patches/nemo_rl_truncation_drops_media.patch"
    text = patch.read_text(encoding="utf-8")
    assert "nemo_rl/data/processors.py" in text
    # Modality-agnostic: media rides on messages as PackedTensor values, so
    # the patch drops every PackedTensor-valued key rather than hardcoding
    # pixel_values/image_grid_thw/audio/video key names.
    assert "isinstance(v, PackedTensor)" in text
    assert "del message[_media_key]" in text


def test_brev_setup_applies_every_patch_in_the_patches_dir() -> None:
    """Patches accumulate; setup must apply all of them idempotently rather
    than hardcoding one filename (the second patch silently not shipping is
    exactly the class of failure that burned the first smoke)."""
    text = SETUP.read_text(encoding="utf-8")
    assert "patches/" in text and "*.patch" in text


def test_preflight_probes_the_longest_row_per_split() -> None:
    """The truncation crash escaped the preflight because the first N rows
    are short. The preflight must also probe each split's LONGEST row (by
    raw line length) so length-dependent failures surface in seconds."""
    from tools.finetune.remedy_nemo_rl.dataloader_preflight import select_probe_rows

    lines = ["{'a': 1}", "x" * 500, "{'b': 2}", "x" * 90, "{'c': 3}"]
    picked = select_probe_rows(lines, rows=2)
    assert picked[:2] == [0, 1]  # the first N...
    assert 1 in picked  # ...and the longest (index 1) is guaranteed present
    picked_tail_longest = select_probe_rows(["short", "aa", "b" * 999], rows=2)
    assert picked_tail_longest == [0, 1, 2]  # longest appended when not in head


def test_collate_patch_tolerates_media_less_rows() -> None:
    """Batches mixing media rows with media-less rows (e.g. rows truncated by
    sft_processor) crashed collation: batched_message_log_to_flat_message
    passes seq.get(key)=None entries into PackedTensor.flattened_concat
    ('NoneType' has no attribute 'dim_to_pack' — hit live on all three window-A
    tasks, 2026-07-16). It also dispatched on values[0] only, silently
    bypassing packing when the FIRST row lacks media. The patch must filter
    to actual PackedTensors and dispatch on ANY row having one."""
    patch = ROOT / "tools/finetune/patches/nemo_rl_collate_skips_missing_media.patch"
    text = patch.read_text(encoding="utf-8")
    assert "nemo_rl/data/llm_message_utils.py" in text
    assert "isinstance(v, PackedTensor)" in text


def test_preflight_command_gates_the_same_task_and_config() -> None:
    command, environment = build_preflight_command(
        task="contrast",
        model_role="control",
        dataset_root=Path("/ephemeral/nemo-rl/datasets"),
    )
    assert command[:3] == [
        "python",
        "-m",
        "tools.finetune.remedy_nemo_rl.dataloader_preflight",
    ]
    assert "--task-root" in command
    assert "/ephemeral/nemo-rl/datasets/sft/contrast" in command
    config = command[command.index("--config") + 1]
    assert config.endswith("sft_qwen25_vl_3b_h200.yaml")
    assert environment["PYTHONPATH"] == "/home/ubuntu/workspace/remedy-server"


def _fake_manifest(tmp_path: Path) -> Path:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "counts": {"train": {"contrast": {"total": 114}}},
                "length_filter": {"max_tokens": 8128},
            }
        ),
        encoding="utf-8",
    )
    return manifest


def _fake_dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "datasets"
    task = root / "sft" / "contrast"
    task.mkdir(parents=True)
    (task / "train.jsonl").write_text("{}\n", encoding="utf-8")
    (task / "validation.jsonl").write_text("{}\n", encoding="utf-8")
    return root


def test_run_sft_runs_preflight_before_training(tmp_path, monkeypatch) -> None:
    from tools.finetune.remedy_nemo_rl import campaign

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ANN001
        calls.append(list(command))
        return type("Result", (), {"returncode": 0})()

    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    monkeypatch.setenv("REMEDY_LOG_ROOT", str(tmp_path / "logs"))

    args = argparse.Namespace(
        manifest=_fake_manifest(tmp_path),
        dataset_root=_fake_dataset_root(tmp_path),
        task="contrast",
        model_role="control",
    )
    # Route logs somewhere writable for the test.
    monkeypatch.setattr(
        campaign,
        "build_sft_command",
        lambda **kwargs: (
            ["python", "/home/ubuntu/RL/examples/run_vlm_sft.py"],
            {"REMEDY_LOG_DIR": str(tmp_path / "logs")},
        ),
    )

    assert campaign._run_sft(args) == 0
    assert len(calls) == 2
    assert calls[0][:3] == [
        "python",
        "-m",
        "tools.finetune.remedy_nemo_rl.dataloader_preflight",
    ]
    assert calls[1][1].endswith("run_vlm_sft.py")


def test_run_sft_aborts_when_preflight_fails(tmp_path, monkeypatch) -> None:
    from tools.finetune.remedy_nemo_rl import campaign

    calls: list[list[str]] = []

    def fake_run(command, **kwargs):  # noqa: ANN001
        calls.append(list(command))
        return type("Result", (), {"returncode": 3})()

    monkeypatch.setattr(campaign.subprocess, "run", fake_run)
    monkeypatch.setattr(
        campaign,
        "build_sft_command",
        lambda **kwargs: (
            ["python", "/home/ubuntu/RL/examples/run_vlm_sft.py"],
            {"REMEDY_LOG_DIR": str(tmp_path / "logs")},
        ),
    )

    args = argparse.Namespace(
        manifest=_fake_manifest(tmp_path),
        dataset_root=_fake_dataset_root(tmp_path),
        task="contrast",
        model_role="control",
    )
    assert campaign._run_sft(args) == 3
    # Training must never start after a failed preflight.
    assert len(calls) == 1
    assert calls[0][:3] == [
        "python",
        "-m",
        "tools.finetune.remedy_nemo_rl.dataloader_preflight",
    ]


def test_run_sft_rejects_manifest_without_exact_length_filter(tmp_path) -> None:
    from tools.finetune.remedy_nemo_rl import campaign

    manifest = tmp_path / "unfiltered.json"
    manifest.write_text(
        json.dumps({"counts": {"train": {"contrast": {"total": 114}}}}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        manifest=manifest,
        dataset_root=_fake_dataset_root(tmp_path),
        task="contrast",
        model_role="control",
    )

    with pytest.raises(SystemExit, match="exact length filter"):
        campaign._run_sft(args)


def test_length_filter_partitions_recounts_and_refreshes_manifest(tmp_path) -> None:
    """Overlong rows must be dropped at BUILD time: runtime truncation of
    multimodal rows violates NeMo's uniform-batch invariants layer after
    layer (forward feature mismatch -> collation NoneType -> BatchedDataDict
    size assertion, all hit live 2026-07-16). Every shipped row must fit."""
    from tools.finetune.filter_overlong_sft_rows import (
        partition_rows,
        recount,
        update_manifest_after_filter,
    )

    rows = [
        {
            "verifier_target": {"issues": []},
            "meta": {"task": "contrast", "source_type": "real"},
        },
        {
            "verifier_target": {"issues": [{"kind": "low"}]},
            "meta": {"task": "contrast", "source_type": "synthetic"},
        },
        {
            "verifier_target": {"issues": []},
            "meta": {"task": "contrast", "source_type": "real"},
        },
    ]
    kept, dropped = partition_rows(rows, [100, 9000, 200], max_tokens=8128)
    assert len(kept) == 2
    assert dropped == [(1, 9000)]
    counts = recount(kept, "contrast")
    assert counts["total"] == 2
    assert counts["pass"] == 2
    assert counts["fail"] == 0
    assert counts["source_types"] == {"real": 2}

    root = tmp_path / "dataset"
    jsonl = root / "sft" / "contrast" / "train.jsonl"
    jsonl.parent.mkdir(parents=True)
    payload = "\n".join(json.dumps(row, sort_keys=True) for row in kept) + "\n"
    jsonl.write_text(payload, encoding="utf-8")
    manifest = {
        "counts": {"train": {"contrast": {"total": 3}}},
        "dataset_hashes": {"sft/contrast/train.jsonl": "stale"},
    }

    update_manifest_after_filter(manifest, root, jsonl, kept, "contrast", "train")

    expected_hash = hashlib.sha256(payload.encode()).hexdigest()
    assert manifest["counts"]["train"]["contrast"] == counts
    assert manifest["dataset_hashes"]["sft/contrast/train.jsonl"] == expected_hash
