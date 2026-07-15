#!/usr/bin/env bash
set -euo pipefail

image=nvcr.io/nvidia/nemo-rl:v0.6.0
workspace=/home/ubuntu/workspace
runtime_home=/home/ubuntu/nemo-runtime
container_user=${REMEDY_BREV_CONTAINER_USER:-root}
docker_cmd=(docker)

mkdir -p "$workspace" "$runtime_home" /ephemeral/nemo-rl

if ! docker info >/dev/null 2>&1; then
  docker_cmd=(sudo docker)
fi

exec "${docker_cmd[@]}" run --rm \
  --gpus all \
  --ipc=host \
  --network=host \
  --shm-size=16g \
  --user "$container_user" \
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
