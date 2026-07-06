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
from types import SimpleNamespace

from project_remedy.config import load_config
from project_remedy.pdf_vision import (
    EmptyVisionResponse,
    OllamaVisionProvider,
    TaskRoutedVisionProvider,
    create_provider_from_config,
    _parse_task_provider_map,
)


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


class _RecordingVisionProvider:
    def __init__(self, name: str, *, fail: bool = False) -> None:
        self.model = name
        self.base_url = "http://example.test/v1"
        self.fail = fail
        self.calls: list[dict] = []

    async def analyze_image(
        self,
        image_path,
        prompt,
        *,
        max_tokens=4096,
        response_format=None,
        task=None,
    ):
        self.calls.append({
            "image_path": image_path,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "task": task,
        })
        if self.fail:
            raise RuntimeError(f"{self.model} failed")
        return f"{self.model}:{task or 'default'}"


def test_task_provider_map_accepts_model_and_url_values():
    assert _parse_task_provider_map(
        "contrast:qwen3vl-32b-contrast, reading-order:http://host:8000/v1",
        env_name="TEST_ROUTES",
    ) == {
        "contrast": "qwen3vl-32b-contrast",
        "reading_order": "http://host:8000/v1",
    }


def test_task_provider_map_rejects_malformed_entries():
    with pytest.raises(ValueError, match="task:value"):
        _parse_task_provider_map("contrast", env_name="TEST_ROUTES")
    with pytest.raises(ValueError, match="non-empty"):
        _parse_task_provider_map("contrast:", env_name="TEST_ROUTES")


async def test_task_routed_provider_sends_contrast_to_override():
    primary = _RecordingVisionProvider("primary")
    contrast = _RecordingVisionProvider("contrast")
    provider = TaskRoutedVisionProvider(primary, {"contrast": contrast})

    assert await provider.analyze_image(None, "prompt", task="contrast") == "contrast:contrast"
    assert await provider.analyze_image(None, "prompt", task="heading_hierarchy") == (
        "primary:heading_hierarchy"
    )

    assert [call["task"] for call in contrast.calls] == ["contrast"]
    assert [call["task"] for call in primary.calls] == ["heading_hierarchy"]


async def test_task_routed_provider_does_not_fallback_by_default():
    primary = _RecordingVisionProvider("primary")
    contrast = _RecordingVisionProvider("contrast", fail=True)
    provider = TaskRoutedVisionProvider(primary, {"contrast": contrast})

    with pytest.raises(RuntimeError, match="contrast failed"):
        await provider.analyze_image(None, "prompt", task="contrast")
    assert primary.calls == []


async def test_task_routed_provider_fallback_is_opt_in():
    primary = _RecordingVisionProvider("primary")
    contrast = _RecordingVisionProvider("contrast", fail=True)
    provider = TaskRoutedVisionProvider(
        primary,
        {"contrast": contrast},
        allow_fallback=True,
    )

    assert await provider.analyze_image(None, "prompt", task="contrast") == "primary:contrast"
    assert [call["task"] for call in primary.calls] == ["contrast"]


def test_provider_config_builds_full_env_task_router(monkeypatch):
    monkeypatch.setenv(
        "OLLAMA_VISION_TASK_MODELS",
        ",".join([
            "contrast:qwen3vl-32b-remedy-contrast-v1",
            "reading_order:qwen3vl-32b-remedy-reading-order-v1",
            "heading-hierarchy:qwen3vl-32b-remedy-heading-v1",
            "table_structure:qwen3vl-32b-remedy-table-v1",
        ]),
    )
    monkeypatch.setenv(
        "OLLAMA_VISION_TASK_BASE_URLS",
        "contrast:http://contrast.test/v1,table_structure:http://table.test/v1",
    )
    monkeypatch.delenv("OLLAMA_VISION_ROUTER_ALLOW_FALLBACK", raising=False)
    config = SimpleNamespace(
        api=SimpleNamespace(
            api_key="dummy",
            base_url="http://primary.test/v1",
            vision_base_url="http://primary.test/v1",
            vision_cluster_nodes=[],
            vision_model="qwen3vl-32b-remedy",
            ollama_stream=False,
            ollama_reasoning_effort="low",
        )
    )

    provider = create_provider_from_config(config)

    assert isinstance(provider, TaskRoutedVisionProvider)
    assert provider.allow_fallback is False
    assert provider.primary.model == "qwen3vl-32b-remedy"
    assert provider.task_providers["contrast"].model == "qwen3vl-32b-remedy-contrast-v1"
    assert provider.task_providers["contrast"].base_url == "http://contrast.test/v1"
    assert provider.task_providers["reading_order"].model == (
        "qwen3vl-32b-remedy-reading-order-v1"
    )
    assert provider.task_providers["reading_order"].base_url == "http://primary.test/v1"
    assert provider.task_providers["heading_hierarchy"].model == (
        "qwen3vl-32b-remedy-heading-v1"
    )
    assert provider.task_providers["table_structure"].model == "qwen3vl-32b-remedy-table-v1"
    assert provider.task_providers["table_structure"].base_url == "http://table.test/v1"


def test_provider_config_rejects_task_base_url_without_model(monkeypatch):
    monkeypatch.setenv("OLLAMA_VISION_TASK_MODELS", "contrast:qwen3vl-32b-remedy-contrast-v1")
    monkeypatch.setenv("OLLAMA_VISION_TASK_BASE_URLS", "heading_hierarchy:http://heading.test/v1")
    config = SimpleNamespace(
        api=SimpleNamespace(
            api_key="dummy",
            base_url="http://primary.test/v1",
            vision_base_url="",
            vision_cluster_nodes=[],
            vision_model="qwen3vl-32b-remedy",
            ollama_stream=False,
            ollama_reasoning_effort="low",
        )
    )

    with pytest.raises(ValueError, match="without models: heading_hierarchy"):
        create_provider_from_config(config)


def test_env_file_can_configure_task_router(tmp_path, monkeypatch):
    for name in (
        "OLLAMA_API_KEY",
        "OLLAMA_BASE_URL",
        "VISION_BASE_URL",
        "OLLAMA_VISION_MODEL",
        "OLLAMA_VISION_TASK_MODELS",
        "OLLAMA_VISION_TASK_BASE_URLS",
        "OLLAMA_VISION_ROUTER_ALLOW_FALLBACK",
    ):
        monkeypatch.delenv(name, raising=False)
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join([
            "OLLAMA_API_KEY=dummy",
            "VISION_BASE_URL=http://primary.test/v1",
            "OLLAMA_VISION_MODEL=qwen3vl-32b-remedy",
            (
                "OLLAMA_VISION_TASK_MODELS="
                "contrast:qwen3vl-32b-remedy-contrast-v1,"
                "reading_order:qwen3vl-32b-remedy-reading-order-v1,"
                "heading_hierarchy:qwen3vl-32b-remedy-heading-v1,"
                "table_structure:qwen3vl-32b-remedy-table-v1"
            ),
            (
                "OLLAMA_VISION_TASK_BASE_URLS="
                "contrast:http://contrast.test/v1,"
                "table_structure:http://table.test/v1"
            ),
            "OLLAMA_VISION_ROUTER_ALLOW_FALLBACK=0",
        ]),
        encoding="utf-8",
    )

    config = load_config(env_path=env_path, yaml_path=tmp_path / "missing.yaml")
    provider = create_provider_from_config(config)

    assert isinstance(provider, TaskRoutedVisionProvider)
    assert provider.allow_fallback is False
    assert provider.primary.model == "qwen3vl-32b-remedy"
    assert provider.primary.base_url == "http://primary.test/v1"
    assert provider.task_providers["contrast"].model == "qwen3vl-32b-remedy-contrast-v1"
    assert provider.task_providers["contrast"].base_url == "http://contrast.test/v1"
    assert provider.task_providers["reading_order"].base_url == "http://primary.test/v1"
