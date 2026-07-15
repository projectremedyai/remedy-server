"""Tests for adapting NeMo Gym response objects to the shared verifier."""

from __future__ import annotations

from types import SimpleNamespace

from tools.finetune.remedy_nemo_rl.gym_adapter import extract_output_text, verify_gym_response


def test_extracts_output_text_from_responses_api_object() -> None:
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", text='{"status":"pass","findings":[]}')
                ],
            )
        ]
    )

    assert extract_output_text(response) == '{"status":"pass","findings":[]}'


def test_extracts_output_text_from_serialized_response() -> None:
    response = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": '{"issues":[]}'}],
            }
        ]
    }

    assert extract_output_text(response) == '{"issues":[]}'


def test_gym_adapter_uses_shared_asymmetric_reward() -> None:
    result = verify_gym_response(
        task="heading_hierarchy",
        response={"output_text": '{"status":"fail","findings":[{"element_index":3,"correct_tag":"H2"}]}'},
        verifier_target={
            "status": "fail",
            "findings": [{"element_index": 3, "correct_tag": "H2"}],
        },
    )

    assert result.reward == 1.0
    assert result.passed is True
