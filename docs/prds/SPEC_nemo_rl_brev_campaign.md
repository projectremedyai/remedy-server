# Five-Adapter NeMo RL Campaign on NVIDIA Brev

**Status:** Five control adapters trained and evaluated; v3 alt/heading retrain approved

**Prepared:** 2026-07-14

**Branch:** `codex/autoresearch/remedy-vlm-20260714/datafix-v3-sft`

## Objective

Train and evaluate separate language-backbone LoRAs for `alt_text_quality`,
`table_structure`, `contrast`, `reading_order`, and `heading_hierarchy`. The
primary is `Qwen/Qwen3.5-9B`; `Qwen/Qwen2.5-VL-3B-Instruct` is the technical and
quality control. The existing Qwen3-VL-32B router remains the rollback until all
promotion and serving gates pass.

The original allocation was a $50 hard ceiling with a $10 reserve. Provider billing
reconciliation on 2026-07-17 put the `remedy-*` campaign at $50.82. The user then
approved the v3 experiment and raised the hard ceiling to $60. The v3 window uses:

- hard credit ceiling: $60;
- provider-reconciled starting spend: $50.82;
- conservative accounting rate: $3.00/hour for an advertised $1.98/hour A100;
- $0.60 reserve below the no-new-work line;
- at most one GPU and 2.85 hours for the v3 instance;
- no 4-GPU escalation without new user approval.

## Implemented Components

- deterministic document-level 70/15/15 splitting and 50/50 train balancing;
- content-addressed portable images and per-task SFT and Gym JSONLs;
- immutable holdout manifest for the 2023-24, 2024-25, and 2025-26 catalogs;
- one NeMo Gym resource server dispatching all five task schemas;
- asymmetric rewards shared by Gym and offline evaluation;
- Qwen3.5 compatibility spike with forward/backward, language-only LoRA,
  save/reload, one-image vLLM, and strict JSON checks;
- pinned NeMo RL v0.6.0 SFT and GRPO recipes;
- a cost-authorized Brev launcher and detached auto-stop watchdog;
- task promotion metrics and a five-stage SFT command planner.

The production router is intentionally unchanged. New aliases are added only
after all five candidates pass the frozen tests, 20-request cross-talk test,
throughput-per-dollar gate, and end-to-end PDF checks.

## Rebuild and Verify Locally

The generated corpus is ignored by Git. Its authoritative manifest is
`tools/finetune/generated/nemo_campaign_dataset/manifest.json` and records every
source and output SHA-256. Rebuild from task-specific source builder outputs;
never train from an old multitask union.

```bash
uv run pytest -q tests/unit
uv run python -m tools.finetune.remedy_nemo_rl.campaign plan \
  --manifest tools/finetune/generated/nemo_campaign_dataset/manifest.json \
  --dataset-root /ephemeral/nemo-rl/datasets
```

The final dataset currently contains these balanced training counts:

| Task | Train | Pass | Fail |
|---|---:|---:|---:|
| contrast | 114 | 57 | 57 |
| table structure | 228 | 114 | 114 |
| alt text quality | 252 | 126 | 126 |
| reading order | 282 | 141 | 141 |
| heading hierarchy | 1,202 | 601 | 601 |

The heading source includes 84 new LAMC true-fail pages, 109 false-flag pass
pages, and 100 delivered pass pages. This replaces the earlier heading cohort
with too few real LAMC correction examples.

## Guarded Brev Run

The launcher is dry-run unless `--execute` is present. The hard limit and reserve are
persisted into campaign state so the detached watchdog enforces the same policy that
authorized the launch. For the approved v3 A100 window:

```bash
uv run python -m tools.finetune.remedy_nemo_rl.brev_control launch \
  --state session/20260714_232247/brev_state.json \
  --instance remedy-qwen25-v3-sft-20260717 \
  --instance-type a100-80gb.1x \
  --hourly-rate 3.00 \
  --hours 2.85 \
  --hard-limit-usd 60 \
  --reserve-override-usd 0.60 \
  --startup-script tools/finetune/brev_startup.sh
```

After checking the printed cost decision, repeat with `--execute`. The detached
watchdog stops the instance at its deadline or at the hard limit. Check it with:

```bash
uv run python -m tools.finetune.remedy_nemo_rl.brev_control status \
  --state session/20260714_232247/brev_state.json
brev ls
```

Copy the committed source plus only the SFT JSONLs, media, and manifest to
`/home/ubuntu/workspace/remedy-server`. Do not copy the 2.9 GB base64 Gym corpus
unless a GRPO stage is actually authorized. On the instance:

```bash
cd /home/ubuntu/workspace/remedy-server
bash tools/finetune/brev_setup.sh
PYTHONPATH=/home/ubuntu/workspace/remedy-server \
python -m tools.finetune.remedy_nemo_rl.compatibility \
  --model Qwen/Qwen3.5-9B \
  --image /ephemeral/nemo-rl/datasets/media/COMPATIBILITY_IMAGE.png \
  --report /ephemeral/nemo-rl/logs/compatibility/qwen35.json
```

Run the same spike for the 3B control. Qwen3.5 is rejected if a technical gate
fails, or if it trails the control by more than ten points on at least two
tasks. SFT is intentionally ordered by corpus size to maximize completed
evidence inside the paid window:

1. contrast;
2. table structure;
3. alt text quality;
4. reading order;
5. heading hierarchy.

Example stage:

```bash
PYTHONPATH=/home/ubuntu/workspace/remedy-server \
python -m tools.finetune.remedy_nemo_rl.campaign sft \
  --manifest /ephemeral/nemo-rl/datasets/manifest.json \
  --dataset-root /ephemeral/nemo-rl/datasets \
  --model-role target \
  --task contrast
```

GRPO remains ordered heading, contrast, table, reading order, and alt text. It
starts only after five-task baseline and SFT evidence fits the remaining time
and budget. Otherwise the campaign stops with SFT checkpoints and records GRPO
as a measured feasibility blocker.

## Teardown and Artifact Integrity

Before stopping, create SHA-256 manifests for checkpoints, reports, logs, and
the experiment ledger under `/ephemeral/nemo-rl`. Copy them locally and verify
the hashes before deleting anything. Then use the guarded stop command:

```bash
uv run python -m tools.finetune.remedy_nemo_rl.brev_control stop \
  --state session/20260714_232247/brev_state.json \
  --instance remedy-nemo-rl-20260714
```

Stopping avoids compute charges but does not guarantee that capacity will be
available again. Keep only small reproducibility records in
`/home/ubuntu/workspace`; checkpoints, datasets, caches, logs, Ray state, and
temporary files belong under `/ephemeral/nemo-rl`.

## Provisioning Fallback

On 2026-07-15, three Brev custom-container launches using the pinned NeMo RL
image failed to expose a shell: two Crusoe A100 builds and one GCP A100 build
remained `UNHEALTHY/BUILDING` through the providers' advertised seven-minute
boot window. No payload or training command reached any instance.

Do not repeat container-mode provisioning after this evidence. Once every
failed instance is fully deleted, the bounded fallback is Brev VM mode with an
explicit 100 GB disk and the same official image launched inside the VM by
`tools/finetune/brev_vm_container_run.sh`. Brev VM mode supplies Docker and the
NVIDIA Container Toolkit, while the wrapper keeps source under
`/home/ubuntu/workspace` and all heavy state under `/ephemeral/nemo-rl`.

Example setup after the VM exposes a shell:

```bash
brev copy --host /tmp/remedy-nemo-brev-payload-3f3f23d/remedy-server/ \
  INSTANCE:/home/ubuntu/workspace/remedy-server/
brev exec INSTANCE --host \
  "cd /home/ubuntu/workspace/remedy-server && tools/finetune/brev_vm_container_run.sh bash tools/finetune/brev_setup.sh"
```

All failed instances were deleted before this handoff, and final `brev ls`
reported no instances. The VM fallback was intentionally not launched after
the repeated failures under the user's few-hours constraint. Never overlap
paid instances for this campaign.
