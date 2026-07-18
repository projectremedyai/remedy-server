# Files

## Inspected
- `HANDOFF_20260713.md` - current catalog blocker and heading-v3 requirement.
- `session/20260714_232247/{handoff.md,session_state.md,timeline.md,datafix_v3_handoff.md}` - recovered current adapter campaign state and v3 continuation.
- `session/20260714_232247/brev_state.json` and NVIDIA Billing dashboard - reconciled the understated local ledger against provider instance costs.
- `tools/finetune/generated/nemo_campaign_dataset_v3/manifest.json` - v3 data hashes and counts are intact; restored the omitted immutable holdout metadata after independent hash/size verification.
- `tools/finetune/remedy_nemo_rl/{campaign.py,evaluation.py,preflight.py,brev_control.py}` - confirmed launch, promotion, dataset, and budget gates.
- `docs/prds/PRD_vlm_multi_lora_serving.md` - five-adapter router, lower-hardware candidates, metrics, and serving gates.
- `/Users/laccd/code/lamc_district_forms/remedy-server-multitask-next/tools/finetune/` - reusable historical builders, evaluators, and trainers.
- `artifacts/lamc-qwen3vl-32b-heading-v2-lora/adapter_config.json` - current language-backbone LoRA shape.

## Changed
- `session/20260714_232247/{session_state.md,timeline.md,handoff.md,files.md,datafix_v3_handoff.md}` - takeover checkpoint, provider billing reconciliation, and current no-launch blocker.
- `tools/finetune/remedy_nemo_rl/brev_control.py` - persist the user-approved hard ceiling/reserve so launch and watchdog enforce the same policy.
- `tests/unit/test_nemo_rl_budget.py` and `tests/unit/test_brev_campaign_control.py` - TDD coverage for the $60 ceiling and state round trip.
- `tests/unit/test_build_delivered_dataset_alt_labels.py` - objective placeholder and human-rewording label coverage.
- `docs/prds/SPEC_nemo_rl_brev_campaign.md` - current branch, provider starting spend, $60 ceiling, and guarded v3 launch command.
- `tools/finetune/remedy_nemo_rl/` - rewards, dataset rebuild, preflight, compatibility, evaluation, campaign planning, and Brev cost control.
- `tools/finetune/nemo_gym/` - one deterministic five-task resource server.
- `tools/finetune/nemo_rl_configs/` - pinned campaign, SFT, and GRPO recipes.
- `tools/finetune/brev_*.sh` and artifact scripts - Brev storage, setup, payload, and checksum workflow.
- `tools/finetune/brev_vm_container_run.sh` - unexecuted VM fallback that runs the pinned NeMo RL container with heavy state under `/ephemeral`.
- `tools/finetune/build_*.py` and evaluators - selectively recovered source builders and evaluation utilities.
- `tests/unit/test_nemo_*.py` and `test_brev_campaign_control.py` - deterministic local contracts.
- `docs/prds/SPEC_nemo_rl_brev_campaign.md` and `docs/index.md` - campaign runbook and documentation index.

## Generated
- `session/20260714_232247/` - durable campaign state and handoff notes.
- `session/20260714_232247/datafix_v3_preflight.json` - authoritative green v3 dataset acceptance report.
- `/tmp/remedy-v3-payload-145e1fa.tar.gz` - v3 Brev payload; SHA-256 `d1757d1dbc7c34dd2c5e332b8f2f67f426fa29f417c7f41663f4cdeac4499e27`.
- `tools/finetune/generated/nemo_campaign_dataset/` - ignored, content-addressed SFT/Gym corpus and manifest.
- `session/20260714_232247/dataset_preflight.json` - successful full-corpus acceptance report.
- `session/20260714_232247/brev_state.json` - reconciled three-attempt cost ledger with no active instance and an estimated $2.3856 total.
