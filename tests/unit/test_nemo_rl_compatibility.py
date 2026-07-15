"""Tests for model compatibility reporting and target/control selection."""

from __future__ import annotations

import pytest

from tools.finetune.remedy_nemo_rl.compatibility import (
    CompatibilityFailure,
    choose_model_family,
    validate_trainable_parameter_names,
)


def test_trainable_lora_parameters_must_be_nonzero_and_language_only() -> None:
    report = validate_trainable_parameter_names(
        {
            "model.language_model.layers.0.q_proj.lora_A.default.weight": 256,
            "model.language_model.layers.0.q_proj.lora_B.default.weight": 512,
        }
    )

    assert report["trainable_parameters"] == 768
    assert report["visual_trainable_parameters"] == 0


@pytest.mark.parametrize(
    "names",
    [
        {},
        {"model.visual.blocks.0.attn.qkv.lora_A.default.weight": 256},
        {"model.vision_tower.encoder.layers.0.lora_B.default.weight": 256},
    ],
)
def test_invalid_adapter_trainable_shapes_fail(names: dict[str, int]) -> None:
    with pytest.raises(CompatibilityFailure):
        validate_trainable_parameter_names(names)


def test_target_is_selected_when_technical_gates_pass_and_scores_are_close() -> None:
    selection = choose_model_family(
        target_technical_pass=True,
        target_scores={"heading": 0.90, "contrast": 0.82, "table": 0.95},
        control_scores={"heading": 0.95, "contrast": 0.85, "table": 0.96},
    )

    assert selection.model_role == "target"
    assert selection.trailing_tasks == ()


def test_control_is_selected_after_a_target_technical_failure() -> None:
    selection = choose_model_family(
        target_technical_pass=False,
        target_scores={},
        control_scores={},
    )

    assert selection.model_role == "control"
    assert selection.reason == "target_failed_technical_gate"


def test_control_is_selected_when_target_trails_by_over_ten_points_on_two_tasks() -> None:
    selection = choose_model_family(
        target_technical_pass=True,
        target_scores={"heading": 0.70, "contrast": 0.69, "table": 0.95},
        control_scores={"heading": 0.82, "contrast": 0.81, "table": 0.96},
    )

    assert selection.model_role == "control"
    assert selection.trailing_tasks == ("contrast", "heading")
    assert selection.reason == "target_trails_control_on_two_tasks"
