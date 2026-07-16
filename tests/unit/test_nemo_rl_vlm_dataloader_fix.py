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
        json.dumps({"counts": {"train": {"contrast": {"total": 114}}}}),
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
