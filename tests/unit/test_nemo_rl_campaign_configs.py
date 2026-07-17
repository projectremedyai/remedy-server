"""Static contract tests for the pinned Brev and NeMo RL campaign recipes."""

from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "tools" / "finetune" / "nemo_rl_configs"


def _yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def test_campaign_manifest_pins_versions_models_storage_and_credit_limits() -> None:
    campaign = _yaml("campaign.yaml")

    assert campaign["container"] == "nvcr.io/nvidia/nemo-rl:v0.6.0"
    assert campaign["nemo_rl"]["revision"] == "r0.6.0"
    assert campaign["models"]["target"] == "Qwen/Qwen3.5-9B"
    assert campaign["models"]["control"] == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert campaign["budget"] == {
        "hard_limit_usd": 50.0,
        "no_new_work_usd": 40.0,
        "reserve_usd": 10.0,
        "max_instance_hours": 3.0,
        "max_gpu_count": 1,
    }
    assert campaign["brev"]["primary"]["gpu_count"] == 1
    assert campaign["brev"]["multi_gpu_escalation"] is False
    assert campaign["storage"]["source_root"] == "/home/ubuntu/workspace"
    assert campaign["storage"]["artifact_root"].startswith("/ephemeral/nemo-rl")


def test_sft_recipes_have_identical_five_adapter_hyperparameters() -> None:
    target = _yaml("sft_qwen35_9b_h200.yaml")
    control = _yaml("sft_qwen25_vl_3b_h200.yaml")

    for recipe, model in (
        (target, "Qwen/Qwen3.5-9B"),
        (control, "Qwen/Qwen2.5-VL-3B-Instruct"),
    ):
        assert recipe["policy"]["model_name"] == model
        assert recipe["sft"]["max_num_epochs"] == 2
        assert recipe["policy"]["train_micro_batch_size"] == 1
        assert recipe["policy"]["train_global_batch_size"] == 8
        assert recipe["policy"]["max_total_sequence_length"] == 8192
        assert recipe["policy"]["precision"] == "bfloat16"
        assert recipe["policy"]["optimizer"]["kwargs"]["lr"] == 2e-5
        lora = recipe["policy"]["dtensor_cfg"]["lora_cfg"]
        assert lora["enabled"] is True
        assert lora["dim"] == 16
        assert lora["alpha"] == 32
        assert lora["dropout"] == 0.0
        # Language-scoped wildcards: bare names silently match nothing in
        # NeMo Automodel's ModuleMatcher (anchored re.match on the full dotted
        # path), and unscoped wildcards would train the vision tower.
        assert set(lora["target_modules"]) == {
            "*.language_model.*.q_proj",
            "*.language_model.*.k_proj",
            "*.language_model.*.v_proj",
            "*.language_model.*.o_proj",
            "*.language_model.*.gate_proj",
            "*.language_model.*.up_proj",
            "*.language_model.*.down_proj",
        }
        assert lora["exclude_modules"] == []
        assert lora["match_all_linear"] is False
        assert recipe["policy"]["dtensor_cfg"]["activation_checkpointing"] is True
        assert recipe["data"]["default"]["dataset_name"] == "openai_format"
        assert recipe["data"]["train"]["dataset_name"] == "openai_format"
        assert recipe["data"]["validation"]["dataset_name"] == "openai_format"
        assert recipe["cluster"] == {"gpus_per_node": 1, "num_nodes": 1}


def test_grpo_recipe_uses_nemo_gym_and_approved_limits() -> None:
    recipe = _yaml("grpo_vlm_h200.yaml")

    assert recipe["grpo"]["num_generations_per_prompt"] == 4
    assert recipe["grpo"]["num_prompts_per_step"] == 8
    assert recipe["grpo"]["max_num_steps"] == 20
    assert recipe["grpo"]["val_period"] == 5
    assert recipe["policy"]["generation"]["max_new_tokens"] == 512
    assert recipe["policy"]["generation"]["temperature"] == 0.7
    assert recipe["policy"]["generation"]["top_p"] == 0.95
    assert recipe["policy"]["optimizer"]["kwargs"]["lr"] == 1e-6
    assert recipe["policy"]["generation"]["vllm_cfg"]["async_engine"] is True
    assert recipe["policy"]["generation"]["vllm_cfg"]["expose_http_server"] is True
    assert recipe["data"]["default"]["dataset_name"] == "NemoGymDataset"
    assert recipe["data"]["train"]["dataset_name"] == "NemoGymDataset"
    assert recipe["data"]["validation"]["dataset_name"] == "NemoGymDataset"
    assert recipe["env"]["should_use_nemo_gym"] is True
    # VLMEnvironment is constructed even in gym mode and hard-fails without
    # these keys (proven live, 2026-07-17).
    # validate() computes max_val_samples // val_batch_size; null TypeErrors
    # at validation-at-start (proven live, attempt 3).
    assert isinstance(recipe["grpo"]["max_val_samples"], int)
    assert recipe["env"]["nemo_gym"]["num_workers"] == 1
    assert recipe["env"]["nemo_gym"]["reward_functions"]
    assert recipe["cluster"] == {"gpus_per_node": 1, "num_nodes": 1}
    # Lessons from the SFT campaign, applied to GRPO before its first run:
    # bare target_modules silently match NOTHING in Automodel's ModuleMatcher,
    # and 4096 ctx truncates alt_text/table rows (max measured 6.5k-17k tokens).
    lora = recipe["policy"]["dtensor_cfg"]["lora_cfg"]
    for pattern in lora["target_modules"]:
        assert pattern.startswith("*.language_model."), pattern
    assert recipe["policy"]["max_total_sequence_length"] == 8192
    assert recipe["policy"]["generation"]["vllm_cfg"]["max_model_len"] == 8192


def test_gym_resource_server_dispatches_one_step_five_task_environment() -> None:
    config = yaml.safe_load(
        (ROOT / "tools" / "finetune" / "nemo_gym" / "resources_servers" / "remedy_pdf" / "configs" / "remedy_pdf.yaml").read_text(
            encoding="utf-8"
        )
    )

    server = config["remedy_pdf"]["resources_servers"]["remedy_pdf"]
    agent = config["remedy_pdf_simple_agent"]["responses_api_agents"]["simple_agent"]
    assert server["entrypoint"] == "app.py"
    assert agent["max_steps"] == 1
    assert agent["resources_server"]["name"] == "remedy_pdf"
