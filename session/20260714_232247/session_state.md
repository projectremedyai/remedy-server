# Session State

- Session: 20260714_232247
- Repo: /Users/laccd/code/lamc_district_forms/remedy-server-nemo-rl-brev
- Branch: codex/nemo-rl-brev-five-adapter
- Started: 2026-07-14 23:22:47 PDT
- Updated: 2026-07-15 00:42:00 PDT

## Goal
Implement the approved five-adapter NeMo RL campaign on NVIDIA Brev, with Qwen3.5-9B as the target, Qwen2.5-VL-3B as the control, deterministic NeMo Gym rewards, and a hard $50 total Brev credit ceiling.

## Current Subtask
Close out the measured Brev provisioning blocker without starting another paid instance.

## Loaded Skills
- `nemo-rl-auto-research` - baseline-first experiments, one branch per hypothesis, durable TSV ledger, and explicit stop conditions.
- `nemo-rl-session-memory` - durable checkpoints and handoff state for this long-running campaign.
- `nemo-rl-docs` - Google-style public docstrings and documentation index updates for new docs.
- `nemo-rl-brev-etiquette` - keep source small and route checkpoints, caches, logs, and Ray state to `/ephemeral` on Brev.

## Current Status
- Current `main` was clean at `24f94a0` when the worktree was created.
- The older `codex/multitask-next` worktree remains separate and dirty only with its prior generated evaluation state.
- Four NVIDIA skills were installed globally and verified with `npx skills list -g`.
- Brev CLI v0.6.330 is authenticated to `johnny-01be29-vebe`; final `brev ls` reports no instances.
- The launch-time inventory no longer contained a single H200/H100. The bounded attempts used Crusoe 1x A100 80GB at $1.98/hour and GCP 1x A100 80GB at $6.0504/hour including the selected disk estimate.
- Three paid custom-container provisions were deleted after remaining `UNHEALTHY/BUILDING`; no payload, model, inference, SFT, or GRPO command reached an instance.
- The local elapsed-time ledger estimates $2.3856 total spend. Provider billing remains authoritative.
- The user has $50 in credits. Reserve $10, refuse new work at $40, stop at $50, and cap the first paid instance at three hours of wall time.
- Rebuilt 2,078 balanced training rows across the five task adapters; the largest task is heading hierarchy with 1,202 rows.
- Recorded the three full catalog holdouts by exact identifier, path, size, and SHA-256. No held-out identifier appears in train, validation, or test.
- Full dataset preflight passes: no document leakage, missing images, schema failures, balance failures, holdout leaks, or hash mismatches.
- The generated SFT plus media payload is about 527 MB. It was prepared locally but never transferred. The 2.9 GB base64 Gym corpus was not transferred.
- Integration commit `3f3f23d` and baseline branch `codex/autoresearch/remedy-vlm-20260714/baseline` preserve the verified local harness.
- Fresh final verification passed: 347 unit tests passed, one skipped; shell syntax, Python compilation, and all campaign YAML files also passed.

## Plan
- [x] Recover only the reusable five-task builders, evaluators, and trainer scaffolding.
- [x] Implement grouped dataset rebuilding, normalized verifier targets, deterministic rewards, and tests.
- [x] Add pinned NeMo RL SFT/GRPO recipes, Brev setup, storage, budget, and artifact-transfer tooling.
- [x] Commit the integration baseline and create the required auto-research baseline branch.
- [x] Authenticate Brev and inspect live prices.
- [x] Attempt bounded single-GPU provisioning with an automatic watchdog; delete every failed build and reconcile the local cost ledger.
- [x] Run a fresh full local verification.
- [x] Commit the provisioning hardening and final handoff.

## Assumptions
- Five adapters means five separate language-backbone LoRAs and no consolidated multitask adapter.
- Qwen3.5-9B is primary unless the compatibility and control-model gates select Qwen2.5-VL-3B.
- Zero false positives on frozen real pass pages is a hard promotion constraint.
- Existing Qwen3-VL-32B routing remains the production rollback until every promotion gate passes.
- Under the revised credit constraint, compatibility, frozen baselines, and SFT evidence take priority; unfinished GRPO is reported as budget-limited rather than exceeding the ceiling.

## Blockers
- Brev custom-container provisioning failed before shell access across three attempts and two providers. This prevents compatibility, baseline, SFT, and GRPO work from starting.
- A VM-mode Docker fallback is prepared but intentionally not launched after the repeated failures under the user's few-hours constraint.
