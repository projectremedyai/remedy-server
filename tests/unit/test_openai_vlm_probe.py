"""Tests for the OpenAI-compatible VLM serving probe."""

from __future__ import annotations

import json

from tools.finetune.remedy_nemo_rl.openai_vlm_probe import (
    build_payload,
    extract_message_content,
    image_data_uri,
)


def test_image_data_uri_png(tmp_path):
    image = tmp_path / "page.png"
    image.write_bytes(b"fake-png")

    assert image_data_uri(image) == "data:image/png;base64,ZmFrZS1wbmc="


def test_build_payload_uses_openai_compatible_image_url(tmp_path):
    image = tmp_path / "page.jpg"
    image.write_bytes(b"jpeg")

    payload = build_payload("Qwen/Qwen2.5-VL-3B-Instruct", image)

    assert payload["model"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert payload["response_format"] == {"type": "json_object"}
    content = payload["messages"][0]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert json.loads('{"status":"pass","findings":[]}') == {"status": "pass", "findings": []}


def test_extract_message_content():
    response = {"choices": [{"message": {"content": '{"status":"pass","findings":[]}'}}]}

    assert extract_message_content(response) == '{"status":"pass","findings":[]}'
