"""Adapter between NeMo Gym Responses API objects and Remedy rewards."""

from __future__ import annotations

from typing import Any

from .reward import VerificationResult, verify_response


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def extract_output_text(response: Any) -> str:
    """Extract assistant output text from a NeMo Gym response object or dict."""

    direct = _field(response, "output_text")
    if isinstance(direct, str):
        return direct
    pieces = []
    for output_item in _field(response, "output", []) or []:
        if _field(output_item, "type") != "message":
            continue
        for content_item in _field(output_item, "content", []) or []:
            if _field(content_item, "type") == "output_text":
                pieces.append(str(_field(content_item, "text", "")))
    return "".join(pieces)


def verify_gym_response(
    *,
    task: str,
    response: Any,
    verifier_target: dict[str, Any],
) -> VerificationResult:
    """Score a NeMo Gym response with the shared offline verifier."""

    return verify_response(task, extract_output_text(response), verifier_target)
