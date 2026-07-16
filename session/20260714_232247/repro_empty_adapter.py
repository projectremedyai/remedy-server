#!/usr/bin/env python3
"""CPU repro of the empty adapter_model.safetensors from the 2026-07-16 smoke.

Chain under test (exactly what the box ran):
  nemo_rl save -> nemo_automodel Checkpointer.save_model
    -> ModelState(model, is_peft=True).state_dict()
       = get_model_state_dict(FSDP2 model, StateDictOptions(full_state_dict=True,
         cpu_offload=True, ignore_frozen_params=True))  then  filter "lora_" in key
    -> rank0 save_file(state_dict, "adapter_model.safetensors")

The retrieved file was 16 bytes == save_file({}) — the state dict was EMPTY.

Probes:
  A. plain module (automodel LoRA applied, base frozen)  -> expect lora keys
  B. FSDP2 fully_shard single rank (gloo)                -> the suspected {}
  C. isolate the ignore_frozen_params flag on the wrapped model
"""

import sys
import types
from pathlib import Path

SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRATCH.parent / "automodel-src"))

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.fsdp import fully_shard

from nemo_automodel.components._peft.lora import (
    PeftConfig,
    apply_lora_to_linear_modules,
)
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState

print(f"torch {torch.__version__}")


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(32, 32)
        self.up_proj = nn.Linear(32, 64)
        self.down_proj = nn.Linear(64, 32)

    def forward(self, x):
        return self.down_proj(torch.relu(self.up_proj(self.q_proj(x))))


class VisionBlock(nn.Module):
    """Mirrors Qwen2.5-VL: the vision MLP ALSO uses gate/up/down_proj names."""

    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(16, 48)
        self.gate_proj = nn.Linear(16, 32)
        self.up_proj = nn.Linear(16, 32)
        self.down_proj = nn.Linear(32, 16)


class Visual(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([VisionBlock() for _ in range(2)])


class LanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Layer() for _ in range(2)])


class Inner(nn.Module):
    """model.visual + model.language_model, like Qwen2_5_VLModel."""

    def __init__(self):
        super().__init__()
        self.visual = Visual()
        self.language_model = LanguageModel()


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = Inner()
        self.lm_head = nn.Linear(32, 100)
        self.config = types.SimpleNamespace(
            tie_word_embeddings=False,
            architectures=["TinyForConditionalGeneration"],
        )

    def forward(self, x):
        for layer in self.model.language_model.layers:
            x = layer(x)
        return self.lm_head(x)


def report(tag, sd):
    lora = [k for k in sd if "lora_" in k]
    print(f"[{tag}] total_keys={len(sd)} lora_keys={len(lora)}")
    for k in lora[:3]:
        print(f"    {k}")


model = Tiny()
import os

# Bare names (the shipped YAML) match NOTHING: ModuleMatcher compares the FULL
# dotted path via re.match, so only wildcard patterns like '*.q_proj' can hit
# nested modules. REPRO_TARGET_STYLE=bare reproduces the defect; default runs
# the fix.
if os.environ.get("REPRO_TARGET_STYLE") == "bare":
    targets = ["q_proj", "up_proj", "down_proj"]
elif os.environ.get("REPRO_TARGET_STYLE") == "unscoped":
    targets = ["*.q_proj", "*.up_proj", "*.down_proj"]  # WOULD hit vision MLP
else:
    targets = [
        "*.language_model.*.q_proj",
        "*.language_model.*.up_proj",
        "*.language_model.*.down_proj",
    ]

peft_config = PeftConfig.from_dict(
    {
        "target_modules": targets,
        "dim": 8,
        "alpha": 32,
        "use_triton": False,
        "lora_dtype": "torch.float32",
    }
)
patched = apply_lora_to_linear_modules(model, peft_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
visual_trainable = sum(
    p.numel() for p in model.model.visual.parameters() if p.requires_grad
)
print(
    f"patched_modules={patched} trainable_params={trainable} "
    f"visual_trainable={visual_trainable}"
)

# ---- probe A: plain module -------------------------------------------------
report("A plain ModelState", ModelState(model, is_peft=True).state_dict())

# ---- probe B: FSDP2 single-rank (matches the smoke: 1 GPU, dtensor v2) -----
dist.init_process_group(
    backend="gloo", init_method="tcp://127.0.0.1:29517", rank=0, world_size=1
)
from torch.distributed.device_mesh import init_device_mesh

mesh = init_device_mesh("cpu", (1,))
for layer in model.model.language_model.layers:
    fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)
report("B fsdp2 ModelState", ModelState(model, is_peft=True).state_dict())

# ---- probe C: isolate the ignore_frozen_params flag ------------------------
sd_all = get_model_state_dict(
    model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
)
sd_ifp = get_model_state_dict(
    model,
    options=StateDictOptions(
        full_state_dict=True, cpu_offload=True, ignore_frozen_params=True
    ),
)
report("C without ignore_frozen", sd_all)
report("C with ignore_frozen", sd_ifp)

dist.destroy_process_group()
