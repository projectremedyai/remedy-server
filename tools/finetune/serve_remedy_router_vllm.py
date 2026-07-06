#!/usr/bin/env python3
"""Build and run the vLLM command for the Remedy task-router adapter profile."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


BASE_MODEL = "Qwen/Qwen3-VL-32B-Instruct"
STABLE_ALIAS = "qwen3vl-32b-remedy"
TASK_MODEL_MAP = {
    "contrast": "qwen3vl-32b-remedy-contrast-v1",
    "reading_order": "qwen3vl-32b-remedy-reading-order-v1",
    "heading_hierarchy": "qwen3vl-32b-remedy-heading-v1",
    "table_structure": "qwen3vl-32b-remedy-table-v1",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--adapter-root", type=Path, default=Path("."))
    ap.add_argument("--alt-adapter", type=Path, default=None)
    ap.add_argument("--table-adapter", type=Path, default=None)
    ap.add_argument("--contrast-adapter", type=Path, default=None)
    ap.add_argument("--reading-order-adapter", type=Path, default=None)
    ap.add_argument("--heading-adapter", type=Path, default=None)
    ap.add_argument("--multitask-adapter", type=Path, default=None)
    ap.add_argument("--include-multitask", action="store_true")
    ap.add_argument("--max-loras", type=int, default=8)
    ap.add_argument("--max-lora-rank", type=int, default=16)
    ap.add_argument("--max-model-len", type=int, default=0)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.0)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--skip-missing", action="store_true")
    ap.add_argument("--print-env", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable that has vLLM installed",
    )
    return ap.parse_args(argv)


def _resolve(root: Path, override: Path | None, default: str) -> Path:
    path = override if override is not None else root / default
    return path.expanduser()


def adapter_modules(args: argparse.Namespace) -> list[tuple[str, Path]]:
    root = args.adapter_root.expanduser()
    alt = _resolve(root, args.alt_adapter, "artifacts/lamc-qwen3vl-32b-lora-v2")
    modules = [
        (STABLE_ALIAS, alt),
        ("qwen3vl-32b-remedy-alt-v2", alt),
        (
            TASK_MODEL_MAP["table_structure"],
            _resolve(root, args.table_adapter, "artifacts/lamc-qwen3vl-32b-table-lora"),
        ),
        (
            TASK_MODEL_MAP["contrast"],
            _resolve(root, args.contrast_adapter, "outputs_runpod/lamc-qwen3vl-32b-contrast-lora"),
        ),
        (
            TASK_MODEL_MAP["reading_order"],
            _resolve(root, args.reading_order_adapter, "outputs_runpod/lamc-qwen3vl-32b-reading-order-lora"),
        ),
        (
            TASK_MODEL_MAP["heading_hierarchy"],
            _resolve(root, args.heading_adapter, "outputs_runpod/lamc-qwen3vl-32b-heading-lora"),
        ),
    ]
    if args.include_multitask:
        modules.append((
            "qwen3vl-32b-remedy-multitask-v1",
            _resolve(
                root,
                args.multitask_adapter,
                "outputs_runpod/lamc-qwen3vl-32b-multitask-contrast-weighted-lora",
            ),
        ))
    return modules


def missing_adapters(modules: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    missing: list[tuple[str, Path]] = []
    for alias, path in modules:
        if not (path / "adapter_config.json").exists():
            missing.append((alias, path))
    return missing


def router_env(base_url: str = "http://<served-host>:8000/v1") -> str:
    return "\n".join([
        "OLLAMA_API_KEY=dummy",
        f"VISION_BASE_URL={base_url}",
        f"OLLAMA_VISION_MODEL={STABLE_ALIAS}",
        "OLLAMA_VISION_TASK_MODELS="
        + ",".join(f"{task}:{model}" for task, model in TASK_MODEL_MAP.items()),
        "OLLAMA_VISION_TASK_BASE_URLS=",
        "OLLAMA_VISION_ROUTER_ALLOW_FALLBACK=0",
        "OLLAMA_ESCALATION_MAX_INFLIGHT=8",
        "OLLAMA_VISION_MAX_INFLIGHT=8",
        "OLLAMA_VISION_GATE_TIMEOUT_SECONDS=600",
        "OLLAMA_VISION_MAX_TOKENS=768",
    ])


def build_command(args: argparse.Namespace) -> list[str]:
    modules = adapter_modules(args)
    command = [
        args.python,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        args.base,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--enable-lora",
        "--max-loras",
        str(args.max_loras),
        "--max-lora-rank",
        str(args.max_lora_rank),
        "--lora-modules",
        *[f"{alias}={path}" for alias, path in modules],
    ]
    if args.max_model_len:
        command.extend(["--max-model-len", str(args.max_model_len)])
    if args.gpu_memory_utilization:
        command.extend(["--gpu-memory-utilization", str(args.gpu_memory_utilization)])
    if args.trust_remote_code:
        command.append("--trust-remote-code")
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    modules = adapter_modules(args)
    missing = missing_adapters(modules)
    if missing and not args.skip_missing:
        for alias, path in missing:
            print(f"missing adapter for {alias}: {path}", file=sys.stderr)
        return 2

    command = build_command(args)
    print(shlex.join(command))
    if args.print_env:
        print("\n# Router runtime env")
        print(router_env(f"http://<served-host>:{args.port}/v1"))
    if args.dry_run:
        return 0
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
