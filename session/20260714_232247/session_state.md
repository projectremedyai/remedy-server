# Session State

- Session: 20260714_232247
- Repo: /Users/laccd/code/lamc_district_forms/remedy-server-nemo-rl-brev
- Branch: codex/nemo-rl-brev-five-adapter
- Started: 2026-07-14 23:22:47 PDT
- Updated: 2026-07-15 00:21:00 PDT

## Goal
Implement the approved five-adapter NeMo RL campaign on NVIDIA Brev, with Qwen3.5-9B as the target, Qwen2.5-VL-3B as the control, deterministic NeMo Gym rewards, and a hard $50 total Brev credit ceiling.

## Current Subtask
Build and verify the local campaign harness before launching paid GPU compute.

## Loaded Skills
- `nemo-rl-auto-research` - baseline-first experiments, one branch per hypothesis, durable TSV ledger, and explicit stop conditions.
- `nemo-rl-session-memory` - durable checkpoints and handoff state for this long-running campaign.
- `nemo-rl-docs` - Google-style public docstrings and documentation index updates for new docs.
- `nemo-rl-brev-etiquette` - keep source small and route checkpoints, caches, logs, and Ray state to `/ephemeral` on Brev.

## Current Status
- Current `main` was clean at `24f94a0` when the worktree was created.
- The older `codex/multitask-next` worktree remains separate and dirty only with its prior generated evaluation state.
- Four NVIDIA skills were installed globally and verified with `npx skills list -g`.
- Brev CLI v0.6.330 is authenticated to `johnny-01be29-vebe`; `brev ls` reports no instances.
- Live stoppable prices observed on 2026-07-14 are $5.40/hour for 1x H200 141GB, $4.62/hour for 1x H100 80GB, and $7.92/hour for 4x A100 80GB.
- No paid compute has been launched and recorded spend is $0.
- The user has $50 in credits. Reserve $10, refuse new work at $40, stop at $50, and cap the first paid instance at three hours of wall time.
- Rebuilt 2,078 balanced training rows across the five task adapters; the largest task is heading hierarchy with 1,202 rows.
- Recorded the three full catalog holdouts by exact identifier, path, size, and SHA-256. No held-out identifier appears in train, validation, or test.
- Full dataset preflight passes: no document leakage, missing images, schema failures, balance failures, holdout leaks, or hash mismatches.
- The generated SFT plus media payload is about 527 MB. The 2.9 GB base64 Gym corpus will not be transferred unless GRPO is authorized.

## Plan
- [x] Recover only the reusable five-task builders, evaluators, and trainer scaffolding.
- [x] Implement grouped dataset rebuilding, normalized verifier targets, deterministic rewards, and tests.
- [x] Add pinned NeMo RL SFT/GRPO recipes, Brev setup, storage, budget, and artifact-transfer tooling.
- [ ] Commit the integration baseline, create the required auto-research baseline branch, and run local verification.
- [x] Authenticate Brev and inspect live prices.
- [ ] Launch at most one single-GPU instance only when projected spend remains below the stop threshold and an automatic three-hour shutdown is armed.

## Assumptions
- Five adapters means five separate language-backbone LoRAs and no consolidated multitask adapter.
- Qwen3.5-9B is primary unless the compatibility and control-model gates select Qwen2.5-VL-3B.
- Zero false positives on frozen real pass pages is a hard promotion constraint.
- Existing Qwen3-VL-32B routing remains the production rollback until every promotion gate passes.
- Under the revised credit constraint, compatibility, frozen baselines, and SFT evidence take priority; unfinished GRPO is reported as budget-limited rather than exceeding the ceiling.

## Blockers
- The original mandatory five-task baseline plus SFT scope may not fit a three-hour paid window; the campaign must retain partial evidence and stop cleanly instead of overrunning credits.
