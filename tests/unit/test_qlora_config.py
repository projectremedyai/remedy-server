"""Guards the QLoRA branch of the HF vision LoRA trainer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


_TRAINER = (
    Path(__file__).resolve().parents[2]
    / "tools"
    / "finetune"
    / "train_lora_vision_hf.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("train_lora_vision_hf", _TRAINER)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def test_bf16_path_uses_no_quantization():
    mod = _load_module()
    assert mod._bnb_config(False) is None


def test_qlora_path_is_4bit_nf4_bf16_compute():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("bitsandbytes")
    mod = _load_module()

    cfg = mod._bnb_config(True)

    assert cfg is not None
    assert cfg.load_in_4bit is True
    assert cfg.bnb_4bit_quant_type == "nf4"
    assert cfg.bnb_4bit_compute_dtype == torch.bfloat16
    assert cfg.bnb_4bit_use_double_quant is True
