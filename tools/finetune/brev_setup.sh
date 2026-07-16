#!/usr/bin/env bash
set -euo pipefail

export HF_HOME=/ephemeral/nemo-rl/cache/huggingface
export HUGGINGFACE_HUB_CACHE=/ephemeral/nemo-rl/cache/huggingface/hub
export TORCH_HOME=/ephemeral/nemo-rl/cache/torch
export RAY_TMPDIR=/ephemeral/nemo-rl/ray
export TMPDIR=/ephemeral/nemo-rl/tmp
if [[ -x /opt/nemo_rl_venv/bin/python ]]; then
  python_bin=/opt/nemo_rl_venv/bin/python
else
  python_bin=$(command -v python3 || command -v python)
fi

if [[ ! -d /home/ubuntu/RL/.git ]]; then
  git clone --branch r0.6.0 --depth 1 https://github.com/NVIDIA-NeMo/RL.git /home/ubuntu/RL
fi

if [[ ! -d /home/ubuntu/Gym/.git ]]; then
  git clone https://github.com/NVIDIA-NeMo/Gym.git /home/ubuntu/Gym
fi

git config --global --add safe.directory /home/ubuntu/RL
git config --global --add safe.directory /home/ubuntu/Gym
git -C /home/ubuntu/RL checkout --detach c339070fa3bfa83a5ac58ff80d73518911e14b81

# Fix the VLM SFT dataloader crash: datasets==4.4.1 None-pads heterogeneous
# multimodal content lists at load time, which makes Qwen chat templates
# render text parts as extra <|image_pad|> placeholders and crash the HF
# processor (IndexError in image_grid_thw indexing) on the first batch.
# The patch strips the None padding at read time inside sft_processor.
strip_none_patch=/home/ubuntu/workspace/remedy-server/tools/finetune/patches/nemo_rl_strip_none_multimodal_content.patch
if git -C /home/ubuntu/RL apply --reverse --check "$strip_none_patch" 2>/dev/null; then
  echo "strip-none multimodal patch already applied"
else
  git -C /home/ubuntu/RL apply "$strip_none_patch"
  echo "strip-none multimodal patch applied"
fi

# The official image bakes its own nemo_rl copy at /opt/nemo-rl, which shadows
# the pinned+patched clone at import time (proven on the 2026-07-16 smoke:
# the first preflight crashed because /opt/nemo-rl carries no patch). Expose
# the clone's nemo_rl package through the payload dir, which is first on the
# container PYTHONPATH. Do NOT put the RL repo root on PYTHONPATH instead:
# its tools/ is a regular package and shadows this repo's tools/ namespace.
ln -sfn /home/ubuntu/RL/nemo_rl /home/ubuntu/workspace/remedy-server/nemo_rl
# nemo_rl/__init__.py injects Megatron-LM from <parent-of-package>/3rdparty/…,
# which now resolves relative to the payload dir. The shallow clone has no
# submodules; borrow the image's baked copy (path is only valid in-container,
# which is fine — symlinks resolve lazily).
ln -sfn /opt/nemo-rl/3rdparty /home/ubuntu/workspace/remedy-server/3rdparty
"$python_bin" - <<'PYEOF'
import inspect
import nemo_rl.data.processors as processors

source = inspect.getsource(processors.sft_processor)
assert "if v is not None" in source, (
    "imported nemo_rl.data.processors is missing the strip-none patch: "
    + processors.__file__
)
print("patched_nemo_rl_import_ok: " + processors.__file__)
PYEOF

git -C /home/ubuntu/Gym fetch --depth 1 origin 25d471edfc6db9d783b31140a4e10e6194455f71
git -C /home/ubuntu/Gym checkout --detach 25d471edfc6db9d783b31140a4e10e6194455f71
"$python_bin" -m pip install --no-deps -e /home/ubuntu/Gym

mkdir -p /home/ubuntu/Gym/resources_servers/remedy_pdf
cp -R \
  /home/ubuntu/workspace/remedy-server/tools/finetune/nemo_gym/resources_servers/remedy_pdf/. \
  /home/ubuntu/Gym/resources_servers/remedy_pdf/

mkdir -p /home/ubuntu/RL/examples/configs/remedy
cp \
  /home/ubuntu/workspace/remedy-server/tools/finetune/nemo_rl_configs/*.yaml \
  /home/ubuntu/RL/examples/configs/remedy/

test "$(git -C /home/ubuntu/RL rev-parse HEAD)" = c339070fa3bfa83a5ac58ff80d73518911e14b81
test "$(git -C /home/ubuntu/Gym rev-parse HEAD)" = 25d471edfc6db9d783b31140a4e10e6194455f71
PYTHONPATH=/home/ubuntu/workspace/remedy-server "$python_bin" -c 'import nemo_rl, nemo_gym; print("nemo_rl_and_gym_import_ok")'
