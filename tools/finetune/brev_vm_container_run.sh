#!/usr/bin/env bash
set -euo pipefail

image=nvcr.io/nvidia/nemo-rl:v0.6.0
workspace=/home/ubuntu/workspace
runtime_home=/home/ubuntu/nemo-runtime

mkdir -p "$workspace" "$runtime_home" /ephemeral/nemo-rl

exec docker run --rm \
  --gpus all \
  --ipc=host \
  --network=host \
  --shm-size=16g \
  --user "$(id -u):$(id -g)" \
  --env HOME=/home/ubuntu \
  --env PYTHONPATH=/home/ubuntu/workspace/remedy-server \
  --env HF_HOME=/ephemeral/nemo-rl/cache/huggingface \
  --env HUGGINGFACE_HUB_CACHE=/ephemeral/nemo-rl/cache/huggingface/hub \
  --env TORCH_HOME=/ephemeral/nemo-rl/cache/torch \
  --env RAY_TMPDIR=/ephemeral/nemo-rl/ray \
  --env TMPDIR=/ephemeral/nemo-rl/tmp \
  --volume "$runtime_home:/home/ubuntu" \
  --volume "$workspace:/home/ubuntu/workspace" \
  --volume /ephemeral/nemo-rl:/ephemeral/nemo-rl \
  --workdir /home/ubuntu/workspace/remedy-server \
  "$image" "$@"
