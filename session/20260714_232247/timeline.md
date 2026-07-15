# Timeline

## 2026-07-14 23:22:47 PDT
- User approved implementation of the five-adapter NeMo RL campaign on NVIDIA Brev with a hard $100 ceiling.
- Verified clean `main` at `24f94a0` and preserved the older `codex/multitask-next` worktree.
- Installed and read the four NVIDIA NeMo RL skills and all required auto-research references.
- Created `/Users/laccd/code/lamc_district_forms/remedy-server-nemo-rl-brev` on `codex/nemo-rl-brev-five-adapter`.
- No GPU instance was launched; current campaign spend is $0.

## 2026-07-14 23:26:08 PDT
- Verified Brev CLI v0.6.330, authenticated organization `johnny-01be29-vebe`, no instances, and $0 campaign spend.
- Recorded stoppable inventory: 1x H200 141GB at $5.40/hour, 1x H100 80GB at $4.62/hour, and 4x A100 80GB at $7.92/hour.
- User reduced the available credit ceiling to $50 and requested only a few paid hours.
- Enforced a three-hour first-instance limit, $10 reserve, no new work at $40, and a hard stop at $50. Multi-GPU escalation is disabled for this allocation.

## 2026-07-15 00:21:00 PDT
- Selectively recovered reusable task builders and evaluators without merging the older worktree.
- Rebuilt fresh delivered, contrast, and expanded LAMC heading sources, then created document-grouped 70/15/15 splits.
- Balanced every task's training split to 50% pass and 50% fail; heading now includes 84 LAMC true-fail examples in the fresh cohort.
- Added exact catalog holdout identifiers and a manifest containing the three full catalog SHA-256 hashes.
- Implemented the shared deterministic verifier, single-step NeMo Gym resource server, Qwen compatibility spike, promotion evaluator, pinned SFT/GRPO recipes, single-GPU campaign planner, and Brev cost watchdog.
- Full dataset preflight passed with 640 train documents, 139 validation documents, 137 test documents, and zero leakage or hash mismatches.
- Paid Brev spend remains $0.

## 2026-07-15 00:36:11 PDT
- Committed the local campaign harness as `3f3f23d` and created `codex/autoresearch/remedy-vlm-20260714/baseline` at that commit.
- Prepared a roughly 535 MB transfer payload containing the committed source plus SFT JSONL, images, and manifest; no Gym corpus was included.
- Attempted two Crusoe A100 80GB custom-container launches and one GCP A100 80GB custom-container launch. All remained `UNHEALTHY/BUILDING` past the advertised provisioning window and never exposed a shell.
- Deleted all three instances without transferring the payload or running a model command. Final `brev ls` reports no instances.
- Reconciled an estimated elapsed-time cost of $2.3856: $0.0929, $0.2749, and $2.0178. Provider billing is authoritative.
- Added guarded deletion reconciliation, a 100 GB disk floor, and an unexecuted VM-mode Docker fallback. No fourth paid attempt was started.
- Fresh verification passed with 347 unit tests passed and one skipped, plus successful shell syntax, Python compilation, and YAML parsing checks.
- Committed the provisioning hardening, reconciled spend ledger, and final handoff on `codex/nemo-rl-brev-five-adapter`.
