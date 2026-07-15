# Session State

- Session: 20260714_232247
- Repo: /Users/laccd/code/lamc_district_forms/remedy-server-nemo-rl-brev
- Branch: codex/nemo-rl-brev-five-adapter
- Started: 2026-07-14 23:22:47 PDT
- Updated: 2026-07-15 10:18:00 PDT

## Goal
Implement the approved five-adapter NeMo RL campaign on NVIDIA Brev, with Qwen3.5-9B as the target, Qwen2.5-VL-3B as the control, deterministic NeMo Gym rewards, and a hard $50 total Brev credit ceiling.

## Current Subtask
Close out the first model compatibility spike under the $50 Brev credit constraint.

## Loaded Skills
- `nemo-rl-auto-research` - baseline-first experiments, one branch per hypothesis, durable TSV ledger, and explicit stop conditions.
- `nemo-rl-session-memory` - durable checkpoints and handoff state for this long-running campaign.
- `nemo-rl-docs` - Google-style public docstrings and documentation index updates for new docs.
- `nemo-rl-brev-etiquette` - keep source small and route checkpoints, caches, logs, and Ray state to `/ephemeral` on Brev.

## Current Status
- Current `main` was clean at `24f94a0` when the worktree was created.
- The older `codex/multitask-next` worktree remains separate and dirty only with its prior generated evaluation state.
- Four NVIDIA skills were installed globally and verified with `npx skills list -g`.
- Brev CLI v0.6.330 is authenticated to `johnny-01be29-vebe`.
- Three earlier paid custom-container provisions were deleted after remaining `UNHEALTHY/BUILDING`; no payload, model, inference, SFT, or GRPO command reached those instances.
- A targeted tiny custom-container preflight with `nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04` confirmed that the host and GPU can come up, but the custom container did not start, no Docker container/image was present on the host, and the startup script did not run. This is the evidence for switching away from Brev custom-container mode.
- A stoppable H100 VM, `remedy-nemo-rl-vm-20260715`, launched successfully with `gpu-h100-sxm.1gpu-16vcpu-200gb` at $4.62/hour.
- The VM exposed an H100 80GB GPU, allowed host Docker with sudo, and successfully ran the official `nvcr.io/nvidia/nemo-rl:v0.6.0` container.
- The official NeMo image digest was `sha256:336aa41391a99e01d018d17d327107fd6d1023ad4b2812c8d8c913dee95fd3f2`; the pulled image was about 36.4 GB.
- The pinned setup completed inside the official image with NeMo RL at `c339070fa3bfa83a5ac58ff80d73518911e14b81`, NeMo Gym at `25d471edfc6db9d783b31140a4e10e6194455f71`, and `nemo_rl_and_gym_import_ok`.
- The H100 VM was stopped through the guarded budget controller. `brev_state.json` records the run as stopped at 2026-07-15T09:54:37Z with $3.0055 VM cost and $5.3911 tracked campaign spend. After a short consistency lag, `brev ls` reports `remedy-nemo-rl-vm-20260715` as `STOPPED`.
- A short tiny-container preflight cost is not included in the numeric ledger because exact elapsed billing data was not captured; expected cost is below about $0.15 at the A100 rate.
- The user has $50 in credits. Reserve $10, refuse new work at $40, stop at $50, and cap the first paid instance at three hours of wall time.
- Rebuilt 2,078 balanced training rows across the five task adapters; the largest task is heading hierarchy with 1,202 rows.
- Recorded the three full catalog holdouts by exact identifier, path, size, and SHA-256. No held-out identifier appears in train, validation, or test.
- Full dataset preflight passes: no document leakage, missing images, schema failures, balance failures, holdout leaks, or hash mismatches.
- The generated SFT plus media payload is about 527 MB. It was prepared locally but never transferred. The 2.9 GB base64 Gym corpus was not transferred.
- Integration commit `3f3f23d` and baseline branch `codex/autoresearch/remedy-vlm-20260714/baseline` preserve the verified local harness.
- Provisioning follow-up commits on `codex/nemo-rl-brev-five-adapter` are `df57ea4`, `3295ba6`, `98f07b9`, `db4da1d`, and `f4f2f1d`.
- Split compatibility runner commit `97251e4` added `--mode training|inference|both` so the NeMo training image and vLLM serving runtime can be tested independently.
- A second H100 VM, `remedy-nemo-rl-compat-20260715`, ran from 2026-07-15T16:34:46Z to 2026-07-15T17:15:15Z at $4.62/hour. `brev_state.json` records $3.1170 for this window and $8.5081 total tracked campaign spend.
- The second payload archive SHA-256 was `95143884f86935852edbed433c30b4e99790772fcfcfa636bd6df122b551cb6f`.
- Qwen3.5-9B failed the training-side compatibility gate on single H100 80GB with CUDA OOM during image forward/backward.
- Qwen2.5-VL-3B-Instruct passed the training-side compatibility gate: image forward/backward, PEFT save/reload identity, 29,933,568 trainable LoRA parameters, and 0 visual-tower trainable parameters.
- A separate `vllm/vllm-openai:v0.25.1` runtime was attempted for Qwen2.5-VL-3B serving. The vLLM image pulled successfully, but `docker run -d` for the OpenAI server stuck before visible container creation with only about 16 GB root disk free. Treat this as a serving-runtime feasibility blocker, not a model failure.
- Remote reports could not be copied after the stop because SSH reset during shutdown. The captured JSON output is summarized in `session/20260714_232247/compatibility_results_20260715.md`.
- Fresh final verification passed: 347 unit tests passed, one skipped; shell syntax, Python compilation, and all campaign YAML files also passed.

## Plan
- [x] Recover only the reusable five-task builders, evaluators, and trainer scaffolding.
- [x] Implement grouped dataset rebuilding, normalized verifier targets, deterministic rewards, and tests.
- [x] Add pinned NeMo RL SFT/GRPO recipes, Brev setup, storage, budget, and artifact-transfer tooling.
- [x] Commit the integration baseline and create the required auto-research baseline branch.
- [x] Authenticate Brev and inspect live prices.
- [x] Attempt bounded single-GPU custom-container provisioning with an automatic watchdog; delete every failed build and reconcile the local cost ledger.
- [x] Prove Brev VM mode with the official NeMo RL container after a tiny custom-container preflight failure.
- [x] Run target/control training-side compatibility under the split runtime plan.
- [ ] Re-run the serving-side vLLM gate on a fresh/larger serving-only runtime.
- [x] Run a fresh full local verification.
- [x] Commit the provisioning hardening and final handoff.

## Assumptions
- Five adapters means five separate language-backbone LoRAs and no consolidated multitask adapter.
- Qwen3.5-9B is primary unless the compatibility and control-model gates select Qwen2.5-VL-3B. Current measured evidence selects Qwen2.5-VL-3B for the next low-cost training-side work unless Qwen3.5 is revisited with a different memory strategy.
- Zero false positives on frozen real pass pages is a hard promotion constraint.
- Existing Qwen3-VL-32B routing remains the production rollback until every promotion gate passes.
- Under the revised credit constraint, compatibility, frozen baselines, and SFT evidence take priority; unfinished GRPO is reported as budget-limited rather than exceeding the ceiling.

## Blockers
- Brev custom-container mode is rejected for this campaign. It failed across earlier full attempts, and the tiny NVIDIA container preflight showed the host/GPU became available while the requested custom container never started.
- The official NeMo RL training container does not include PEFT or vLLM by default. PEFT can be installed, but current vLLM installation attempts replace the NeMo-pinned Torch/Transformers stack and conflict with `nemo-rl==0.6.0`.
- Do not install vLLM into the NeMo RL training image for the next spike. Treat training and serving as separate runtimes or build explicit derived images.
- The second H100 VM stop showed Brev status lag: guarded stop recorded success, direct stop then reported the backend state was already `stopped`, while `brev ls` displayed `STOPPING`. Re-check `brev ls` before any paid restart and delete stopped instances if storage cost becomes a concern.
