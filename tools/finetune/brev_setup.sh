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

git -C /home/ubuntu/RL checkout --detach c339070fa3bfa83a5ac58ff80d73518911e14b81
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
