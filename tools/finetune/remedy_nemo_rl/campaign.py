"""Build and run bounded single-GPU NeMo RL campaign stages on Brev."""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


TASKS_BY_PAID_PRIORITY = (
    "contrast",
    "table_structure",
    "alt_text_quality",
    "reading_order",
    "heading_hierarchy",
)
MAX_FILTERED_SEQUENCE_TOKENS = 8128
MODEL_CONFIGS = {
    "target": "/home/ubuntu/RL/examples/configs/remedy/sft_qwen35_9b_h200.yaml",
    "control": "/home/ubuntu/RL/examples/configs/remedy/sft_qwen25_vl_3b_h200.yaml",
}


def warmup_steps(train_count: int, *, global_batch_size: int = 8, epochs: int = 2) -> int:
    """Return the nearest whole step for a five-percent optimizer warmup."""

    optimizer_steps = math.ceil(train_count / global_batch_size) * epochs
    return max(1, math.ceil(optimizer_steps * 0.05))


def build_sft_command(
    *,
    task: str,
    model_role: str,
    dataset_root: Path,
    train_count: int,
    overrides: tuple[str, ...] = (),
) -> tuple[list[str], dict[str, str]]:
    """Construct one task-specific, one-GPU SFT command and environment."""

    if task not in TASKS_BY_PAID_PRIORITY:
        raise ValueError(f"unsupported task: {task}")
    if model_role not in MODEL_CONFIGS:
        raise ValueError(f"unsupported model role: {model_role}")
    task_root = dataset_root / "sft" / task
    warmup = warmup_steps(train_count)
    command = [
        "python",
        "/home/ubuntu/RL/examples/run_vlm_sft.py",
        "--config",
        MODEL_CONFIGS[model_role],
        "cluster.gpus_per_node=1",
        "cluster.num_nodes=1",
        f"policy.scheduler.0.kwargs.total_iters={warmup}",
        f"policy.scheduler.2.milestones=[{warmup}]",
        *overrides,
    ]
    environment = {
        "PYTHONPATH": "/home/ubuntu/workspace/remedy-server",
        "HF_HOME": "/ephemeral/nemo-rl/cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/ephemeral/nemo-rl/cache/huggingface/hub",
        "TORCH_HOME": "/ephemeral/nemo-rl/cache/torch",
        "RAY_TMPDIR": "/ephemeral/nemo-rl/ray",
        "TMPDIR": "/ephemeral/nemo-rl/tmp",
        "REMEDY_SFT_TRAIN": str(task_root / "train.jsonl"),
        "REMEDY_SFT_VALIDATION": str(task_root / "validation.jsonl"),
        "REMEDY_CHECKPOINT_DIR": f"/ephemeral/nemo-rl/checkpoints/sft/{model_role}/{task}",
        "REMEDY_LOG_DIR": f"/ephemeral/nemo-rl/logs/sft/{model_role}/{task}",
    }
    return command, environment


def build_preflight_command(
    *,
    task: str,
    model_role: str,
    dataset_root: Path,
) -> tuple[list[str], dict[str, str]]:
    """Construct the dataloader preflight that must pass before SFT starts.

    The 2026-07-15 paid smoke died on the first dataloader batch after the
    VM, image pull, and setup were already paid for. This gate replays the
    real data path on real rows in seconds, so a data-processing defect
    costs a log line instead of a training window.
    """

    if task not in TASKS_BY_PAID_PRIORITY:
        raise ValueError(f"unsupported task: {task}")
    if model_role not in MODEL_CONFIGS:
        raise ValueError(f"unsupported model role: {model_role}")
    command = [
        "python",
        "-m",
        "tools.finetune.remedy_nemo_rl.dataloader_preflight",
        "--task-root",
        str(dataset_root / "sft" / task),
        "--config",
        MODEL_CONFIGS[model_role],
        "--rows",
        "4",
    ]
    environment = {
        "PYTHONPATH": "/home/ubuntu/workspace/remedy-server",
        "HF_HOME": "/ephemeral/nemo-rl/cache/huggingface",
        "HUGGINGFACE_HUB_CACHE": "/ephemeral/nemo-rl/cache/huggingface/hub",
    }
    return command, environment


def campaign_plan(manifest: dict[str, Any], dataset_root: Path) -> list[dict[str, Any]]:
    """Return deterministic target SFT stages ordered to maximize evidence per paid minute."""

    counts = manifest["counts"]["train"]
    plan = []
    for task in TASKS_BY_PAID_PRIORITY:
        command, environment = build_sft_command(
            task=task,
            model_role="target",
            dataset_root=dataset_root,
            train_count=int(counts[task]["total"]),
        )
        plan.append(
            {
                "task": task,
                "train_examples": counts[task]["total"],
                "warmup_steps": warmup_steps(int(counts[task]["total"])),
                "command": command,
                "environment": environment,
            }
        )
    return plan


def _run_sft(args: argparse.Namespace) -> int:
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    length_filter = manifest.get("length_filter") or {}
    filtered_max = length_filter.get("max_tokens")
    if not isinstance(filtered_max, int) or filtered_max > MAX_FILTERED_SEQUENCE_TOKENS:
        raise SystemExit(
            "dataset manifest does not prove the exact length filter; run "
            "filter_overlong_sft_rows.py --max-tokens 8128 --apply before SFT"
        )
    count = int(manifest["counts"]["train"][args.task]["total"])
    command, environment = build_sft_command(
        task=args.task,
        model_role=args.model_role,
        dataset_root=args.dataset_root,
        train_count=count,
        overrides=tuple(getattr(args, "override", None) or ()),
    )
    task_root = args.dataset_root / "sft" / args.task
    for split in ("train", "validation"):
        if not (task_root / f"{split}.jsonl").is_file():
            raise SystemExit(f"missing {split} data: {task_root / f'{split}.jsonl'}")
    merged_env = os.environ.copy()
    merged_env.update(environment)
    log = Path(environment["REMEDY_LOG_DIR"]) / "command.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "environment=" + json.dumps(environment, sort_keys=True) + "\ncommand=" + shlex.join(command) + "\n",
        encoding="utf-8",
    )
    preflight_command, preflight_env = build_preflight_command(
        task=args.task,
        model_role=args.model_role,
        dataset_root=args.dataset_root,
    )
    preflight_merged_env = os.environ.copy()
    preflight_merged_env.update(preflight_env)
    with log.open("a", encoding="utf-8") as stream:
        stream.write("preflight=" + shlex.join(preflight_command) + "\n")
        stream.flush()
        preflight = subprocess.run(
            preflight_command,
            cwd=task_root,
            env=preflight_merged_env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
        if preflight.returncode != 0:
            stream.write(
                f"preflight failed with exit code {preflight.returncode}; "
                "training NOT started\n"
            )
            return preflight.returncode
        return subprocess.run(command, cwd=task_root, env=merged_env, stdout=stream, stderr=subprocess.STDOUT).returncode


def main() -> int:
    """Print the campaign plan or execute one explicitly selected SFT stage."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--manifest", type=Path, required=True)
    plan.add_argument("--dataset-root", type=Path, default=Path("/ephemeral/nemo-rl/datasets"))

    sft = subparsers.add_parser("sft")
    sft.add_argument("--manifest", type=Path, required=True)
    sft.add_argument("--dataset-root", type=Path, default=Path("/ephemeral/nemo-rl/datasets"))
    sft.add_argument("--task", choices=TASKS_BY_PAID_PRIORITY, required=True)
    sft.add_argument("--model-role", choices=tuple(MODEL_CONFIGS), default="target")
    sft.add_argument(
        "--override",
        action="append",
        help="extra config override appended to the training command (repeatable)",
    )

    args = parser.parse_args()
    if args.command == "plan":
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        print(json.dumps(campaign_plan(manifest, args.dataset_root), indent=2, sort_keys=True))
        return 0
    return _run_sft(args)


if __name__ == "__main__":
    raise SystemExit(main())
