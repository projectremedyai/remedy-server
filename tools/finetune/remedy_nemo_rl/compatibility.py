"""Qwen VLM compatibility spike and target/control selection policy."""

from __future__ import annotations

import argparse
import base64
import gc
import hashlib
import json
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


VISUAL_NAME_MARKERS = (
    "visual",
    "vision_tower",
    "vision_model",
    "multi_modal_projector",
)


class CompatibilityFailure(RuntimeError):
    """Raised when a model violates an adapter compatibility gate."""


@dataclass(frozen=True)
class ModelSelection:
    """Result of the approved target-versus-control selection rule."""

    model_role: str
    reason: str
    trailing_tasks: tuple[str, ...]


def validate_trainable_parameter_names(parameter_sizes: Mapping[str, int]) -> dict[str, int]:
    """Require nonzero trainable parameters with no visual-tower adapters."""

    total = sum(int(size) for size in parameter_sizes.values())
    visual = sum(
        int(size)
        for name, size in parameter_sizes.items()
        if any(marker in name.lower() for marker in VISUAL_NAME_MARKERS)
    )
    if total <= 0:
        raise CompatibilityFailure("adapter has no trainable parameters")
    if visual:
        raise CompatibilityFailure(f"adapter includes {visual} visual-tower trainable parameters")
    return {"trainable_parameters": total, "visual_trainable_parameters": visual}


def choose_model_family(
    *,
    target_technical_pass: bool,
    target_scores: Mapping[str, float],
    control_scores: Mapping[str, float],
) -> ModelSelection:
    """Apply the technical and ten-point two-task fallback rule."""

    if not target_technical_pass:
        return ModelSelection("control", "target_failed_technical_gate", ())
    trailing = tuple(
        sorted(
            task
            for task in set(target_scores) & set(control_scores)
            if float(control_scores[task]) - float(target_scores[task]) > 0.10
        )
    )
    if len(trailing) >= 2:
        return ModelSelection("control", "target_trails_control_on_two_tasks", trailing)
    return ModelSelection("target", "target_passed_selection_gate", trailing)


def _adapter_digest(state: Mapping[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
        digest.update(raw)
    return digest.hexdigest()


def _to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}


def run_training_compatibility(model_name: str, image_path: Path) -> dict[str, Any]:
    """Run image forward/backward and language-only LoRA save/reload gates."""

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to("cuda")

    language_linear_modules = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name != "lm_head"
        and not any(marker in name.lower() for marker in VISUAL_NAME_MARKERS)
    ]
    if not language_linear_modules:
        raise CompatibilityFailure("no language-backbone linear modules were found")
    config_kwargs: dict[str, Any] = {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": language_linear_modules,
        "task_type": "CAUSAL_LM",
    }
    peft_model = get_peft_model(model, LoraConfig(**config_kwargs))
    trainable = {
        name: parameter.numel()
        for name, parameter in peft_model.named_parameters()
        if parameter.requires_grad
    }
    trainable_report = validate_trainable_parameter_names(trainable)

    image = Image.open(image_path).convert("RGB")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Return only JSON describing whether this page has a clear accessibility issue."},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": '{"status":"pass","findings":[]}'}]},
    ]
    batch = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_tensors="pt",
    )
    batch = _to_device(batch, peft_model.device)
    batch["labels"] = batch["input_ids"].clone()
    peft_model.train()
    output = peft_model(**batch)
    output.loss.backward()
    gradient_parameters = sum(
        parameter.numel()
        for parameter in peft_model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )
    if gradient_parameters <= 0:
        raise CompatibilityFailure("image forward/backward produced no LoRA gradients")

    before = _adapter_digest(get_peft_model_state_dict(peft_model))
    with tempfile.TemporaryDirectory(prefix="remedy-lora-") as temp_dir:
        peft_model.save_pretrained(temp_dir)
        base_model = peft_model.unload()
        reloaded = PeftModel.from_pretrained(base_model, temp_dir, is_trainable=False)
        after = _adapter_digest(get_peft_model_state_dict(reloaded))
        saved_files = sorted(path.name for path in Path(temp_dir).iterdir())
    if before != after:
        raise CompatibilityFailure("PEFT adapter state changed across save/reload")

    del reloaded, base_model, peft_model, model, batch, output
    gc.collect()
    torch.cuda.empty_cache()
    return {
        **trainable_report,
        "gradient_parameters": gradient_parameters,
        "forward_backward": True,
        "adapter_save_reload_identical": True,
        "adapter_sha256": before,
        "saved_files": saved_files,
    }


def _data_uri(image_path: Path) -> str:
    suffix = image_path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    return f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"


def run_vllm_compatibility(model_name: str, image_path: Path) -> dict[str, Any]:
    """Serve one image through vLLM and require a strict zero-shot JSON object."""

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        dtype="bfloat16",
        max_model_len=4096,
        gpu_memory_utilization=0.72,
        limit_mm_per_prompt={"image": 1},
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _data_uri(image_path)}},
                {
                    "type": "text",
                    "text": 'Inspect the page. Return ONLY valid JSON with this exact shape: {"status":"pass","findings":[]}.',
                },
            ],
        }
    ]
    outputs = llm.chat(
        messages,
        sampling_params=SamplingParams(temperature=0.0, max_tokens=128),
        use_tqdm=False,
    )
    text = outputs[0].outputs[0].text.strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise CompatibilityFailure("zero-shot response is not a JSON object")
    return {"one_image_vllm": True, "zero_shot_json_valid": True, "response": parsed}


def run_spike(model_name: str, image_path: Path) -> dict[str, Any]:
    """Run all technical gates and preserve failures in a machine-readable report."""

    report: dict[str, Any] = {"model": model_name, "image": str(image_path), "technical_pass": False}
    try:
        report["training"] = run_training_compatibility(model_name, image_path)
        report["inference"] = run_vllm_compatibility(model_name, image_path)
        report["technical_pass"] = True
    except Exception as error:  # The report is the compatibility spike's primary artifact.
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
    return report


def main() -> int:
    """Run the GPU compatibility spike and write its complete report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if not args.image.is_file():
        raise SystemExit(f"image does not exist: {args.image}")
    report = run_spike(args.model, args.image)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["technical_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
