"""Probe an OpenAI-compatible VLM chat-completions endpoint with one image."""

from __future__ import annotations

import argparse
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def image_data_uri(image_path: Path) -> str:
    """Return an image file as a base64 data URI."""

    suffix = image_path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"


def build_payload(model: str, image_path: Path, *, max_tokens: int = 128) -> dict[str, Any]:
    """Build the minimal one-image chat-completions request."""

    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_uri(image_path)}},
                    {
                        "type": "text",
                        "text": (
                            'Inspect the page. Return ONLY valid JSON with this exact shape: '
                            '{"status":"pass","findings":[]}.'
                        ),
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }


def post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    """POST JSON and return a decoded JSON response."""

    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(base_url: str, *, timeout_seconds: float) -> None:
    """Wait for a vLLM/OpenAI-compatible server to answer `/v1/models`."""

    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    url = base_url.rstrip("/") + "/models"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status < 500:
                    return
        except (OSError, urllib.error.URLError) as error:
            last_error = str(error)
        time.sleep(2)
    raise TimeoutError(f"server did not become ready before timeout: {last_error}")


def extract_message_content(response: dict[str, Any]) -> str:
    """Extract the assistant message content from a chat-completions response."""

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("response choice has no message")
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("response message content is not a string")
    return content.strip()


def run_probe(
    *,
    base_url: str,
    model: str,
    image_path: Path,
    report_path: Path,
    ready_timeout_seconds: float,
    request_timeout_seconds: float,
) -> dict[str, Any]:
    """Run the one-image serving gate and write a machine-readable report."""

    report: dict[str, Any] = {
        "base_url": base_url,
        "model": model,
        "image": str(image_path),
        "server_ready": False,
        "one_image_chat_completions": False,
        "zero_shot_json_valid": False,
        "technical_pass": False,
    }
    try:
        wait_for_server(base_url, timeout_seconds=ready_timeout_seconds)
        report["server_ready"] = True
        response = post_json(
            base_url.rstrip("/") + "/chat/completions",
            build_payload(model, image_path),
            timeout=request_timeout_seconds,
        )
        content = extract_message_content(response)
        report["one_image_chat_completions"] = True
        report["raw_content"] = content
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError("assistant content is not a JSON object")
        report["parsed_content"] = parsed
        report["zero_shot_json_valid"] = True
        report["technical_pass"] = True
    except Exception as error:  # The report is the probe's durable artifact.
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> int:
    """Run the serving probe from the command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--ready-timeout-seconds", type=float, default=600)
    parser.add_argument("--request-timeout-seconds", type=float, default=120)
    args = parser.parse_args()
    if not args.image.is_file():
        raise SystemExit(f"image does not exist: {args.image}")
    report = run_probe(
        base_url=args.base_url,
        model=args.model,
        image_path=args.image,
        report_path=args.report,
        ready_timeout_seconds=args.ready_timeout_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
