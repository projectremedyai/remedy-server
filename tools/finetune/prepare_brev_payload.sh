#!/usr/bin/env bash
set -euo pipefail

repo_root=$(git rev-parse --show-toplevel)
dataset_root=${REMEDY_DATASET_ROOT:-"$repo_root/tools/finetune/generated/nemo_campaign_dataset"}
payload_root=${1:-$(mktemp -d /tmp/remedy-nemo-brev-payload.XXXXXX)}
payload_repo="$payload_root/remedy-server"

test -f "$dataset_root/manifest.json"
test -d "$dataset_root/sft"
test -d "$dataset_root/media"

test ! -e "$payload_repo" || { echo "payload destination already exists: $payload_repo" >&2; exit 1; }
mkdir -p "$payload_repo/tools/finetune/generated/nemo_campaign_dataset"
git -C "$repo_root" archive HEAD | tar -xf - -C "$payload_repo"

rsync -a --exclude '._*' "$dataset_root/sft/" "$payload_repo/tools/finetune/generated/nemo_campaign_dataset/sft/"
rsync -a --exclude '._*' "$dataset_root/media/" "$payload_repo/tools/finetune/generated/nemo_campaign_dataset/media/"
cp "$dataset_root/manifest.json" "$payload_repo/tools/finetune/generated/nemo_campaign_dataset/manifest.json"

du -sh "$payload_root"
find "$payload_repo/tools/finetune/generated/nemo_campaign_dataset/media" -type f | wc -l
