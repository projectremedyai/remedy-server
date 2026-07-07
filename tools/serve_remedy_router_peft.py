#!/usr/bin/env python3
"""Serve Remedy Qwen3-VL PEFT adapters behind a tiny OpenAI-compatible API."""

from __future__ import annotations

import argparse
import base64
import io
import json
import traceback
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


BASE_MODEL = "Qwen/Qwen3-VL-32B-Instruct"
STABLE_ALIAS = "qwen3vl-32b-remedy"
TASK_MODEL_MAP = {
    "contrast": "qwen3vl-32b-remedy-contrast-v1",
    "reading_order": "qwen3vl-32b-remedy-reading-order-v1",
    "heading_hierarchy": "qwen3vl-32b-remedy-heading-v1",
    "table_structure": "qwen3vl-32b-remedy-table-v1",
}


def _adapter_path(root: Path, override: Path | None, default: str) -> Path:
    return (override if override is not None else root / default).expanduser()


def router_env(base_url: str) -> str:
    return "\n".join([
        "OLLAMA_API_KEY=dummy",
        f"VISION_BASE_URL={base_url.rstrip('/')}",
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


class RouterState:
    def __init__(
        self,
        *,
        base_model: str,
        adapters: dict[str, Path],
        aliases: dict[str, str],
        max_pixels: int,
        attn_implementation: str,
    ) -> None:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.lock = threading.Lock()
        self.model_names = tuple(aliases)
        self.aliases = aliases
        self.processor = AutoProcessor.from_pretrained(
            base_model,
            max_pixels=max_pixels,
            use_fast=True,
        )
        base = AutoModelForImageTextToText.from_pretrained(
            base_model,
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation=attn_implementation,
        )
        first_alias, first_adapter = next(iter(adapters.items()))
        print(f"[serve-peft] attaching adapter={first_alias} path={first_adapter}", flush=True)
        self.model = PeftModel.from_pretrained(
            base,
            str(first_adapter),
            adapter_name=first_alias,
        )
        for adapter_name, path in list(adapters.items())[1:]:
            print(f"[serve-peft] loading adapter={adapter_name} path={path}", flush=True)
            self.model.load_adapter(str(path), adapter_name=adapter_name)
        self.model.config.use_cache = True
        self.model.eval()

    def generate(self, payload: dict[str, Any]) -> tuple[str, str]:
        import torch

        requested_model = str(payload.get("model") or STABLE_ALIAS)
        adapter_name = self.aliases.get(requested_model)
        if adapter_name is None:
            known = ", ".join(self.model_names)
            raise ValueError(f"unknown model {requested_model!r}; known models: {known}")

        messages, images = _convert_messages(payload.get("messages") or [])
        if not messages:
            raise ValueError("messages must include at least one message")
        max_tokens = int(payload.get("max_tokens") or 768)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if images:
            inputs = self.processor(text=[text], images=images, return_tensors="pt").to(
                self.model.device
            )
        else:
            inputs = self.processor(text=[text], return_tensors="pt").to(self.model.device)
        with self.lock, torch.inference_mode():
            self.model.set_adapter(adapter_name)
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        gen_ids = out[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()
        return requested_model, text


def _decode_image_url(url: str):
    from PIL import Image

    if not url.startswith("data:") or ";base64," not in url:
        raise ValueError("only data:...;base64 image URLs are supported")
    _, encoded = url.split(";base64,", 1)
    raw = base64.b64decode(encoded)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _convert_messages(messages: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Any]]:
    converted: list[dict[str, Any]] = []
    images: list[Any] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content")
        if isinstance(content, str):
            converted.append({"role": role, "content": [{"type": "text", "text": content}]})
            continue
        parts: list[dict[str, Any]] = []
        for part in content or []:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                parts.append({"type": "text", "text": str(part.get("text") or "")})
            elif part.get("type") == "image_url":
                image_url = part.get("image_url") or {}
                images.append(_decode_image_url(str(image_url.get("url") or "")))
                parts.append({"type": "image", "image": "inline"})
        converted.append({"role": role, "content": parts})
    return converted, images


class OpenAIHandler(BaseHTTPRequestHandler):
    server_version = "RemedyPEFTRouter/0.1"

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        state: RouterState = self.server.router_state  # type: ignore[attr-defined]
        if self.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if self.path in {"/v1/models", "/models"}:
            self._send_json(
                HTTPStatus.OK,
                {
                    "object": "list",
                    "data": [
                        {
                            "id": name,
                            "object": "model",
                            "created": 0,
                            "owned_by": "project-remedy",
                        }
                        for name in state.model_names
                    ],
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        state: RouterState = self.server.router_state  # type: ignore[attr-defined]
        if self.path not in {"/v1/chat/completions", "/chat/completions"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            started = time.time()
            model_name, text = state.generate(payload)
            elapsed = time.time() - started
            now = int(time.time())
            self._send_json(
                HTTPStatus.OK,
                {
                    "id": f"chatcmpl-{now}",
                    "object": "chat.completion",
                    "created": now,
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
            print(
                f"[serve-peft] model={model_name} status=200 "
                f"elapsed={elapsed:.2f}s chars={len(text)}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[serve-peft] request failed: {type(exc).__name__}: {exc}",
                flush=True,
            )
            traceback.print_exc()
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": {"message": f"{type(exc).__name__}: {exc}"}},
            )

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[serve-peft] {self.address_string()} {fmt % args}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--adapter-root", type=Path, default=Path("."))
    ap.add_argument("--alt-adapter", type=Path, default=None)
    ap.add_argument("--table-adapter", type=Path, default=None)
    ap.add_argument("--contrast-adapter", type=Path, default=None)
    ap.add_argument("--reading-order-adapter", type=Path, default=None)
    ap.add_argument("--heading-adapter", type=Path, default=None)
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    ap.add_argument("--attn-implementation", default="sdpa")
    ap.add_argument("--print-env", action="store_true")
    return ap.parse_args(argv)


def build_adapter_maps(args: argparse.Namespace) -> tuple[dict[str, Path], dict[str, str]]:
    root = args.adapter_root.expanduser()
    adapters = {
        "alt_v2": _adapter_path(root, args.alt_adapter, "artifacts/lamc-qwen3vl-32b-lora-v2"),
        "table_v1": _adapter_path(
            root,
            args.table_adapter,
            "artifacts/lamc-qwen3vl-32b-table-lora",
        ),
        "contrast_v1": _adapter_path(
            root,
            args.contrast_adapter,
            "outputs/lamc-qwen3vl-32b-contrast-lora",
        ),
        "reading_order_v1": _adapter_path(
            root,
            args.reading_order_adapter,
            "outputs/lamc-qwen3vl-32b-reading-order-lora",
        ),
        "heading_v1": _adapter_path(
            root,
            args.heading_adapter,
            "outputs/lamc-qwen3vl-32b-heading-lora",
        ),
    }
    aliases = {
        STABLE_ALIAS: "alt_v2",
        "qwen3vl-32b-remedy-alt-v2": "alt_v2",
        TASK_MODEL_MAP["table_structure"]: "table_v1",
        TASK_MODEL_MAP["contrast"]: "contrast_v1",
        TASK_MODEL_MAP["reading_order"]: "reading_order_v1",
        TASK_MODEL_MAP["heading_hierarchy"]: "heading_v1",
    }
    return adapters, aliases


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    adapters, aliases = build_adapter_maps(args)
    missing = [
        f"{name}: {path}"
        for name, path in adapters.items()
        if not (path / "adapter_config.json").exists()
    ]
    if missing:
        raise SystemExit("missing adapter(s):\n" + "\n".join(missing))
    if args.print_env:
        print(router_env(f"http://<served-host>:{args.port}/v1"))

    state = RouterState(
        base_model=args.model,
        adapters=adapters,
        aliases=aliases,
        max_pixels=args.max_pixels,
        attn_implementation=args.attn_implementation,
    )
    server = ThreadingHTTPServer((args.host, args.port), OpenAIHandler)
    server.router_state = state  # type: ignore[attr-defined]
    print(
        f"[serve-peft] listening on http://{args.host}:{args.port} "
        f"with {len(state.model_names)} aliases",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
