#!/usr/bin/env python3
"""Serve a PEFT vision adapter behind a tiny OpenAI-compatible HTTP API.

This is a fallback validation server for RunPod pods that have Transformers/PEFT
installed but do not have vLLM or another OpenAI-compatible serving stack.
It intentionally supports only the endpoints and request shape used by
``tools/run_vision_eval.py``:

* GET /health
* GET /v1/models
* POST /v1/chat/completions
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class ModelState:
    def __init__(
        self,
        *,
        model_name: str,
        base_model: str,
        adapter: Path | None,
        max_pixels: int,
        attn_implementation: str,
    ) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_name = model_name
        self.lock = threading.Lock()
        self.processor = AutoProcessor.from_pretrained(
            base_model,
            max_pixels=max_pixels,
            use_fast=True,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            base_model,
            dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation=attn_implementation,
        )
        if adapter is not None:
            from peft import PeftModel

            print(f"[serve-hf] attaching adapter={adapter}", flush=True)
            self.model = PeftModel.from_pretrained(self.model, str(adapter))
        self.model.config.use_cache = True
        self.model.eval()

    def generate(self, payload: dict[str, Any]) -> str:
        import torch

        messages, images = _convert_messages(payload.get("messages") or [])
        if not messages:
            raise ValueError("messages must include at least one user message")
        max_tokens = int(payload.get("max_tokens") or 768)
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(
            self.model.device
        )
        with self.lock, torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
            )
        gen_ids = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()


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
    server_version = "RemedyHFOpenAI/0.1"

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:  # noqa: N802
        state: ModelState = self.server.model_state  # type: ignore[attr-defined]
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
                            "id": state.model_name,
                            "object": "model",
                            "created": 0,
                            "owned_by": "project-remedy",
                        }
                    ],
                },
            )
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        state: ModelState = self.server.model_state  # type: ignore[attr-defined]
        if self.path not in {"/v1/chat/completions", "/chat/completions"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})
            return
        try:
            length = int(self.headers.get("Content-Length") or "0")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            text = state.generate(payload)
            now = int(time.time())
            self._send_json(
                HTTPStatus.OK,
                {
                    "id": f"chatcmpl-{now}",
                    "object": "chat.completion",
                    "created": now,
                    "model": state.model_name,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": text},
                            "finish_reason": "stop",
                        }
                    ],
                },
            )
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": {"message": f"{type(exc).__name__}: {exc}"}},
            )

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[serve-hf] {self.address_string()} {fmt % args}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help="base HF model id or local model dir")
    ap.add_argument("--adapter", type=Path, default=None, help="optional PEFT adapter dir")
    ap.add_argument("--served-model-name", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
    ap.add_argument("--attn-implementation", default="sdpa")
    args = ap.parse_args()

    state = ModelState(
        model_name=args.served_model_name,
        base_model=args.model,
        adapter=args.adapter,
        max_pixels=args.max_pixels,
        attn_implementation=args.attn_implementation,
    )
    server = ThreadingHTTPServer((args.host, args.port), OpenAIHandler)
    server.model_state = state  # type: ignore[attr-defined]
    print(
        f"[serve-hf] listening on http://{args.host}:{args.port} "
        f"as {args.served_model_name}",
        flush=True,
    )
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
