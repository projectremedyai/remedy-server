"""OLLAMA_VISION_TEMPERATURE: env-configurable sampling temperature.

The vision request temperature was hardcoded to 0.2 at every call site, so
every acceptance re-rolls the sampler — the headings-nesting verifier flags
*different* headings run-to-run even on an unchanged page. t=0 makes the
verify path deterministic: a fixed file stays fixed on re-eval. Default stays
0.2 (current behavior) when the env var is unset.
"""
from __future__ import annotations

import httpx

import project_remedy.pdf_vision as PV
from project_remedy.pdf_vision import OllamaVisionProvider


class _FakeResp:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("err", request=None, response=self)  # type: ignore[arg-type]

    def json(self) -> dict:
        return self._payload


def _install(monkeypatch, handler):
    calls: list[tuple[str, dict]] = []

    class _FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def post(self, endpoint, json=None):
            calls.append((endpoint, json))
            return handler(endpoint, json, calls)

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)
    return calls


def _chat_payload(content: str) -> dict:
    return {"message": {"content": content}, "prompt_eval_count": 1, "eval_count": 1}


def _provider(**kw) -> OllamaVisionProvider:
    kw.setdefault("base_url", "https://ollama.com/v1")
    kw.setdefault("model", "minimax-m3:cloud")
    kw.setdefault("retry_backoff_seconds", 0.0)
    return OllamaVisionProvider(**kw)


def test_vision_temperature_default_is_current_behavior(monkeypatch):
    monkeypatch.delenv("OLLAMA_VISION_TEMPERATURE", raising=False)
    assert PV._vision_temperature() == 0.2


def test_vision_temperature_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_VISION_TEMPERATURE", "0")
    assert PV._vision_temperature() == 0.0
    monkeypatch.setenv("OLLAMA_VISION_TEMPERATURE", "0.7")
    assert PV._vision_temperature() == 0.7


def test_vision_temperature_invalid_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("OLLAMA_VISION_TEMPERATURE", "hot")
    assert PV._vision_temperature() == 0.2
    monkeypatch.setenv("OLLAMA_VISION_TEMPERATURE", "-1")
    assert PV._vision_temperature() == 0.2


async def test_chat_payload_carries_env_temperature(monkeypatch):
    """The native /api/chat request body must honor the env temperature."""
    monkeypatch.setenv("OLLAMA_VISION_TEMPERATURE", "0")

    def handler(endpoint, body, calls):
        return _FakeResp(_chat_payload('{"status": "pass"}'))

    calls = _install(monkeypatch, handler)
    out = await _provider().analyze_image(None, "prompt")
    assert out == '{"status": "pass"}'
    body = calls[0][1]
    assert body["options"]["temperature"] == 0.0, \
        "env t=0 must reach the request payload (deterministic verify)"


async def test_chat_payload_default_temperature_unchanged(monkeypatch):
    monkeypatch.delenv("OLLAMA_VISION_TEMPERATURE", raising=False)

    def handler(endpoint, body, calls):
        return _FakeResp(_chat_payload("ok"))

    calls = _install(monkeypatch, handler)
    await _provider().analyze_image(None, "prompt")
    assert calls[0][1]["options"]["temperature"] == 0.2, "default stays 0.2"
