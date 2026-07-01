"""Unit tests for OllamaVisionProvider empty-response handling (fix #2).

Covers the reliability fix: an empty completion on the chat transport is
recovered via the more-reliable ``/api/generate`` transport, and — when the
fallback is disabled — an empty response is classified *transient* so a
multi-node setup rotates to the next node instead of failing fast.

All HTTP is mocked (no network, no model). pytest-asyncio ``asyncio_mode=auto``
is configured in pyproject, so ``async def test_*`` needs no decorator.
"""
from __future__ import annotations

import httpx
import pytest

from project_remedy.pdf_vision import EmptyVisionResponse, OllamaVisionProvider


# --------------------------------------------------------------------------- #
# httpx mock harness
# --------------------------------------------------------------------------- #
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
    """Patch httpx.AsyncClient so POSTs are served by *handler*.

    handler(endpoint, json, calls) -> _FakeResp. `calls` is the running list of
    (endpoint, json) tuples, returned to the test for assertions."""
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
    # cloud native /api/chat response shape
    return {"message": {"content": content}, "prompt_eval_count": 1, "eval_count": 1}


def _compat_payload(content: str) -> dict:
    # /v1/chat/completions response shape
    return {"choices": [{"message": {"content": content}}], "usage": {}}


def _generate_payload(content: str) -> dict:
    # /api/generate response shape
    return {"response": content, "prompt_eval_count": 1, "eval_count": 1}


def _provider(**kw) -> OllamaVisionProvider:
    # retry_backoff_seconds=0 keeps tests instant; cloud base_url -> /api/chat.
    kw.setdefault("base_url", "https://ollama.com/v1")
    kw.setdefault("model", "minimax-m3:cloud")
    kw.setdefault("retry_backoff_seconds", 0.0)
    return OllamaVisionProvider(**kw)


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
async def test_empty_chat_recovers_via_generate(monkeypatch):
    """Chat returns empty on a 200 -> /api/generate fallback returns text."""
    def handler(endpoint, body, calls):
        if endpoint == "/api/chat":
            return _FakeResp(_chat_payload(""))
        if endpoint == "/api/generate":
            return _FakeResp(_generate_payload('{"order": [1, 2, 3]}'))
        raise AssertionError(f"unexpected endpoint {endpoint}")

    calls = _install(monkeypatch, handler)
    out = await _provider().analyze_image(None, "prompt")
    assert out == '{"order": [1, 2, 3]}'
    endpoints = [c[0] for c in calls]
    assert endpoints == ["/api/chat", "/api/generate"]
    # generate body carries the native shape (prompt + no data-URL image list)
    gen_body = calls[1][1]
    assert gen_body["prompt"] == "prompt"
    assert gen_body["stream"] is False
    assert "images" not in gen_body  # image_path was None


async def test_nonempty_chat_never_calls_generate(monkeypatch):
    """A good chat response short-circuits — no fallback POST."""
    def handler(endpoint, body, calls):
        assert endpoint == "/api/chat", f"generate should not be called; got {endpoint}"
        return _FakeResp(_chat_payload("real answer"))

    calls = _install(monkeypatch, handler)
    out = await _provider().analyze_image(None, "prompt")
    assert out == "real answer"
    assert [c[0] for c in calls] == ["/api/chat"]


async def test_generate_retries_until_nonempty(monkeypatch):
    """First /api/generate is also empty; second returns text (within budget)."""
    monkeypatch.setenv("OLLAMA_VISION_EMPTY_RETRIES", "2")

    def handler(endpoint, body, calls):
        if endpoint == "/api/chat":
            return _FakeResp(_chat_payload(""))
        gen_calls = [c for c in calls if c[0] == "/api/generate"]
        # first generate call empty, subsequent non-empty
        return _FakeResp(_generate_payload("" if len(gen_calls) == 1 else "recovered"))

    calls = _install(monkeypatch, handler)
    out = await _provider().analyze_image(None, "prompt")
    assert out == "recovered"
    gen = [c[0] for c in calls if c[0] == "/api/generate"]
    assert len(gen) == 2


async def test_fallback_disabled_raises_empty(monkeypatch):
    """With the fallback disabled, an all-empty single node fails (no generate)."""
    monkeypatch.setenv("OLLAMA_VISION_GENERATE_FALLBACK", "0")

    def handler(endpoint, body, calls):
        assert endpoint == "/api/chat"
        return _FakeResp(_chat_payload(""))

    calls = _install(monkeypatch, handler)
    with pytest.raises(RuntimeError) as ei:
        await _provider().analyze_image(None, "prompt")
    # wrapped as "All vision attempts ... empty vision response"
    assert "empty vision response" in str(ei.value)
    assert all(c[0] == "/api/chat" for c in calls)  # never hit /api/generate


async def test_empty_is_transient_rotates_nodes(monkeypatch):
    """Fallback disabled + multi-node: empty on node1 rotates to node2 (transient)."""
    monkeypatch.setenv("OLLAMA_VISION_GENERATE_FALLBACK", "0")
    # two LOCAL /v1 nodes -> OpenAI-compat /chat/completions transport
    prov = OllamaVisionProvider(
        base_urls=["http://n1:11434/v1", "http://n2:11434/v1"],
        model="minimax-m3",
        retry_backoff_seconds=0.0,
    )

    def handler(endpoint, body, calls):
        assert endpoint == "/chat/completions"
        # first node empty, second node good — keyed by call count
        compat_calls = [c for c in calls if c[0] == "/chat/completions"]
        return _FakeResp(_compat_payload("" if len(compat_calls) == 1 else "node2 answer"))

    calls = _install(monkeypatch, handler)
    out = await prov.analyze_image(None, "prompt")
    assert out == "node2 answer"
    assert len([c for c in calls if c[0] == "/chat/completions"]) == 2


async def test_empty_vision_response_is_runtimeerror():
    """EmptyVisionResponse stays a RuntimeError subclass (callers catch broadly)."""
    assert issubclass(EmptyVisionResponse, RuntimeError)
