#!/usr/bin/env bash
set -euo pipefail

mkdir -p \
  /home/ubuntu/workspace \
  /ephemeral/nemo-rl/checkpoints \
  /ephemeral/nemo-rl/datasets \
  /ephemeral/nemo-rl/logs \
  /ephemeral/nemo-rl/cache/huggingface \
  /ephemeral/nemo-rl/cache/torch \
  /ephemeral/nemo-rl/ray \
  /ephemeral/nemo-rl/tmp

if id ubuntu >/dev/null 2>&1; then
  chown -R ubuntu:ubuntu /home/ubuntu/workspace /ephemeral/nemo-rl
fi
chmod 700 /ephemeral/nemo-rl
nvidia-smi
