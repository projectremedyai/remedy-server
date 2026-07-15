#!/usr/bin/env bash
set -euo pipefail

instance=${1:?usage: retrieve_brev_artifacts.sh INSTANCE DESTINATION}
destination=${2:?usage: retrieve_brev_artifacts.sh INSTANCE DESTINATION}

mkdir -p "$destination"
brev exec "$instance" \
  "cd /ephemeral/nemo-rl && find checkpoints logs -type f -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS"
brev copy "$instance:/ephemeral/nemo-rl/checkpoints" "$destination/checkpoints"
brev copy "$instance:/ephemeral/nemo-rl/logs" "$destination/logs"
brev copy "$instance:/ephemeral/nemo-rl/SHA256SUMS" "$destination/SHA256SUMS"

cd "$destination"
shasum -a 256 -c SHA256SUMS
