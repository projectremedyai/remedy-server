"""Tests for bounded NeMo RL campaign command construction."""

from __future__ import annotations

from pathlib import Path

from tools.finetune.remedy_nemo_rl.campaign import build_sft_command, warmup_steps


def test_warmup_is_five_percent_of_two_epoch_optimizer_steps() -> None:
    assert warmup_steps(114, global_batch_size=8, epochs=2) == 2
    assert warmup_steps(1202, global_batch_size=8, epochs=2) == 16


def test_sft_command_is_single_gpu_and_uses_task_specific_paths() -> None:
    command, environment = build_sft_command(
        task="contrast",
        model_role="target",
        dataset_root=Path("/ephemeral/nemo-rl/datasets"),
        train_count=114,
    )

    assert command[:5] == [
        "uv",
        "run",
        "--project",
        "/home/ubuntu/RL",
        "python",
    ]
    assert "cluster.gpus_per_node=1" in command
    assert "policy.scheduler.0.kwargs.total_iters=2" in command
    assert environment["REMEDY_SFT_TRAIN"].endswith("/sft/contrast/train.jsonl")
    assert environment["REMEDY_CHECKPOINT_DIR"].endswith("/sft/target/contrast")
    assert environment["PYTHONPATH"] == "/home/ubuntu/workspace/remedy-server"
