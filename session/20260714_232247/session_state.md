# Session State

- Session: 20260714_232247
- Repo: /Users/laccd/code/lamc_district_forms/remedy-server-nemo-rl-brev
- Branch: codex/autoresearch/remedy-vlm-20260714/datafix-v3-sft
- Started: 2026-07-14 23:22:47 PDT
- Updated: 2026-07-17 22:34:03 PDT

## Goal
Finish the linked LAMC remediation and adapter-training workstreams: preserve the
398/464 machine-clean PDF deliverable, and validate only adapters that can safely
improve it. Keep Qwen2.5-VL-3B as the measured training path, the existing production
routes as rollback, and the campaign's hard $50 Brev ceiling unless the user explicitly
changes it.

## Current Subtask
Take over the v3 data-fix continuation for `alt_text_quality` and
`heading_hierarchy`. The missing immutable catalog metadata was restored and the v3
dataset now passes the authoritative local preflight. No Brev instance exists. Provider
billing shows the campaign at $50.82. On 2026-07-17 the user approved the exact v3
experiment plan and raised the hard ceiling to $60. Next: create the dedicated branch,
commit the hypothesis, update/test the guard for $60, then launch one guarded A100.

## Loaded Skills
- `nemo-rl-auto-research` - baseline-first experiments, one branch per hypothesis, durable TSV ledger, and explicit stop conditions.
- `nemo-rl-session-memory` - durable checkpoints and handoff state for this long-running campaign.
- `nemo-rl-docs` - Google-style public docstrings and documentation index updates for new docs.
- `nemo-rl-brev-etiquette` - keep source small and route checkpoints, caches, logs, and Ray state to `/ephemeral` on Brev.

## Current Status
- Dedicated experiment branch created:
  `codex/autoresearch/remedy-vlm-20260714/datafix-v3-sft`; unrelated MOVE3 remains
  untracked and unstaged.
- Budget controller now persists approved hard-limit/reserve values in campaign state,
  and the detached watchdog reads the same hard limit. Provider reconciliation added a
  $12.4293 adjustment, taking recorded starting spend to $50.82. Approved state:
  hard limit $60, conservative rate $3/hour, 2.85-hour window, $0.60 reserve.
- Guard verification is green: full unit suite 378 passed / 1 skipped; shell syntax and
  Python compilation passed; launch dry-run authorized projected spend $59.37 with
  $0.63 remaining below the hard ceiling. Live Brev inventory offers stoppable
  `a100-80gb.1x` at an advertised $1.98/hour; conservative accounting stays $3/hour.
- User authorization received verbatim on 2026-07-17: "I Approve the v3 experiment
  plan and raise the campaign hard ceiling to $60." This authorizes the dedicated v3
  branch and the previously documented one-A100, at-most-three-hour retrain/eval plan.
- Stop rules now are: provider-reconciled starting spend $50.82; hard ceiling $60;
  one GPU; at most three hours; retrieve and SHA-verify artifacts before stop; no GRPO,
  MOVE3, Qwen3.5, or unrelated adapter work.
- Recovery on 2026-07-17 confirmed the authoritative whole-project handoffs:
  `remedy-server/HANDOFF_20260713.md` for the 398/464 PDF deliverable and this session
  directory plus `datafix_v3_handoff.md` for adapter work. Deleted/archived handoffs are
  superseded and must not drive new work.
- `brev ls` reports no instances in org `johnny-01be29-vebe`; nothing is currently billing.
- The NVIDIA dashboard for 2026-07-01 through 2026-07-17 reports $74.24 org total,
  $72.75 compute, $1.49 storage, and $75.77 current balance. Excluding the two parallel
  workstream instances (`brevp1sftr2` $16.79 and `brevp1sft20260716r1` $6.63) leaves
  $50.82 attributable to the `remedy-*` campaign instances. The local $38.3907 ledger
  was understated.
- The inherited worktree is intentionally dirty: modified `brev_state.json`,
  `handoff.md`, and `build_delivered_dataset.py`; untracked
  `MOVE3_task_input_redesign.md` and `datafix_v3_handoff.md`. No branch switch, stash,
  reset, or overwrite was performed during recovery.
- v3 dataset verification is fully green after restoring the three catalog holdout
  entries from the prior accepted manifest and independently rechecking each file hash
  and size. Evidence: `datafix_v3_preflight.json` (`passed=true`), 1,456 media files,
  exact SFT/Gym example-ID alignment for both tasks, zero subjective alt labels, and six
  placeholder-classification spot checks. Full unit suite: 368 passed, 1 skipped.
- Recovery check on 2026-07-16 confirmed the worktree was clean before this documentation refresh, `brev ls` reported no instances in org `johnny-01be29-vebe`, and `brev_state.json` reported no active instance, $0.00 active accrued cost, and $9.9713 conservative local tracked spend.
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
- The user provided the authoritative NVIDIA Billing dashboard state before the serving-only rerun: total cost $7.28 and current balance $45.94. The local elapsed-time ledger remains conservative and may differ from provider settlement.
- Created branch `codex/autoresearch/remedy-vlm-20260714/qwen25-vllm-serving` from `573fd0a`.
- Added an OpenAI-compatible one-image vLLM serving probe in commit `8c95465`, then tightened it to support vLLM versions that reject `response_format=json_object` for VLM requests and to require raw JSON without Markdown code fences.
- Launched fresh serving-only VM `remedy-qwen25-vllm-serving-20260715` on `a100-80gb.1x` at $1.98/hour with 128 GB disk.
- `vllm/vllm-openai:v0.25.1` pulled successfully but was rejected on this A100 host because the image required a newer NVIDIA driver/CUDA runtime than the host's CUDA 12.7 driver path exposed.
- `vllm/vllm-openai:v0.8.5` pulled successfully and served `Qwen/Qwen2.5-VL-3B-Instruct` on the single A100.
- The 4096-token serving attempt rejected the image request because the decoder prompt length was 4863 tokens. Restarting vLLM with `--max-model-len 8192` and `--gpu-memory-utilization 0.80` fixed the serving capacity issue.
- Qwen2.5-VL-3B passed the one-image OpenAI-compatible `/v1/chat/completions` gate on vLLM 0.8.5 with strict zero-shot JSON after the raw-JSON prompt: `server_ready=true`, `one_image_chat_completions=true`, `zero_shot_json_valid=true`, and `technical_pass=true`.
- Local serving proof artifacts are under `session/20260714_232247/remote_artifacts/qwen25_vllm_serving/`, including probe reports, vLLM logs, pull logs, and remote SHA-256 manifest.
- The serving VM was stopped through the budget controller at 2026-07-15T18:36:42Z. `brev_state.json` records $0.5285 for this serving-only window and $9.0366 conservative local tracked spend.
- Delete was requested after artifact transfer with `brev delete` by name, by ID, and through stdin. It still showed `STOPPED`, not `RUNNING`, on 2026-07-15; the 2026-07-16 recovery check showed no instances, so the delete/state convergence eventually completed.
- Added guarded restart support in commit `4dcb5bb`, but `brev start remedy-qwen25-vllm-serving-20260715` stayed in a loop reporting `instance is stopped`; the command was interrupted before the budget controller recorded an active window.
- Launched fresh A100 VM `remedy-qwen25-sft-smoke-20260715` for a 1.5-hour Qwen2.5 SFT smoke at $1.98/hour. The payload SHA-256 was `29685583c1d51c5b439705541dcff36cfa59caf6d0df3bbb8efdf53a3c2f3f47`.
- The SFT smoke proved the fresh A100 VM, official NeMo RL image, pinned RL/Gym setup, Qwen2.5 model load, and explicit language-module LoRA recipe up to the first dataloader batch.
- The SFT smoke did not produce a checkpoint. It stopped on a NeMo RL VLM SFT processor compatibility blocker: `IndexError: index 1 is out of bounds for dimension 0 with size 1` in Qwen2.5-VL `image_grid_thw`.
- Variants tried before stopping: text-first content order, absolute image paths, native Qwen chat template, disabling BOS/EOS, and disabling validation. Training still fails on the first dataloader batch.
- The SFT smoke VM was stopped through the budget controller at 2026-07-15T19:13:26Z. `brev_state.json` records $0.9347 for this SFT smoke window and $9.9713 conservative local tracked spend.
- `brev delete remedy-qwen25-sft-smoke-20260715` initially returned successfully while `brev ls` still showed the SFT VM as `STOPPED`; the 2026-07-16 recovery check confirmed that Brev later converged and now reports no instances.
- Remote reports could not be copied after the stop because SSH reset during shutdown. The captured JSON output is summarized in `session/20260714_232247/compatibility_results_20260715.md`.
- Focused final verification after the Qwen2.5 serving/SFT smoke changes passed: `uv run pytest -q tests/unit/test_nemo_rl_campaign.py tests/unit/test_nemo_rl_campaign_configs.py tests/unit/test_brev_campaign_control.py tests/unit/test_openai_vlm_probe.py` reported 15 passed. Earlier full local verification passed with 347 unit tests passed and one skipped; shell syntax, Python compilation, and all campaign YAML files also passed.
- 2026-07-16: the `image_grid_thw` blocker was root-caused, reproduced, and fixed locally at $0. Root cause: NeMo RL's pinned `datasets==4.4.1` None-pads heterogeneous multimodal `content` lists at `load_dataset("json", ...)` time (text parts gain a phantom `"image": None` key); Qwen2.5-VL's chat template tests key MEMBERSHIP (`'image' in content`), so corrupted text parts render as extra `<|image_pad|>` placeholders (their real text silently dropped), desyncing the placeholder count from the single loaded image and crashing the HF processor's unbounded expansion loop on the first batch. This explains why all six variants tried on the paid smoke failed identically — they were all downstream of the loader.
- The exact paid-smoke `IndexError` was reproduced on this Mac (CPU, no GPU) by running the REAL pinned NeMo code (`OpenAIFormatDataset` -> `sft_processor` -> `Qwen2_5_VLProcessor`) on the first real row of `sft/train.jsonl`: `session/20260714_232247/repro_image_grid_thw.py`.
- Fix shipped: `tools/finetune/patches/nemo_rl_strip_none_multimodal_content.patch` strips None-valued keys from content parts at READ time inside `sft_processor` (the only correct placement — `Dataset.map` re-encodes through Arrow and re-injects the Nones). `brev_setup.sh` applies it idempotently after the pinned checkout. RED->GREEN verified: the same real row flips from IndexError to a clean 2-turn log with exactly one vision block and the prompt text intact.
- `use_preserving_dataset: true` (the obvious YAML-only alternative) was tested and RULED OUT: `run_sft.py:92` `concatenate_datasets` type-rejects `PreservingDataset` ("Expected a list of Dataset objects...").
- New paid-run gate: `tools/finetune/remedy_nemo_rl/dataloader_preflight.py` replays real rows through the real processing path inside the container in seconds; `campaign._run_sft` now runs it before training and aborts on failure, so a data-processing defect costs a log line instead of a paid training window.
- Full local verification after the fix: 359 unit tests passed, 1 skipped; `bash -n brev_setup.sh` and `py_compile` clean.
- NOTE: `.gitignore` line 150 (`tools/finetune/*`) ignores non-`.py` files there — the `.patch` file must be committed with `git add -f` or it silently never ships in a payload.

## Plan
- [x] Recover the two authoritative workstreams, current branch, inherited changes,
  experiment ledger, promotion gates, and stop rules.
- [x] Verify live Brev state and reconcile the local ledger against provider billing.
- [x] Restore the three verified catalog holdout entries to the v3 manifest and rerun
  the full local preflight at $0.
- [x] Obtain explicit user confirmation of the experiment plan before creating the v3
  experiment branch, as required by `nemo-rl-auto-research`.
- [x] Obtain an explicit new hard ceiling: user raised it from $50 to $60.
- [x] Create the dedicated v3 branch without staging MOVE3.
- [x] Reconcile `brev_state.json` to provider spend and make the controller/watchdog
  persist and enforce the $60 ceiling; test before launch.
- [ ] Commit the v3 data hypothesis and budget guard on the dedicated branch.
- [ ] Package v3, use one guarded A100 window, retrain only alt text then heading,
  retrieve and SHA-verify artifacts, evaluate frozen test splits, and stop.
- [x] Recover only the reusable five-task builders, evaluators, and trainer scaffolding.
- [x] Implement grouped dataset rebuilding, normalized verifier targets, deterministic rewards, and tests.
- [x] Add pinned NeMo RL SFT/GRPO recipes, Brev setup, storage, budget, and artifact-transfer tooling.
- [x] Commit the integration baseline and create the required auto-research baseline branch.
- [x] Authenticate Brev and inspect live prices.
- [x] Attempt bounded single-GPU custom-container provisioning with an automatic watchdog; delete every failed build and reconcile the local cost ledger.
- [x] Prove Brev VM mode with the official NeMo RL container after a tiny custom-container preflight failure.
- [x] Run target/control training-side compatibility under the split runtime plan.
- [x] Re-run the serving-side vLLM gate on a fresh/larger serving-only runtime.
- [x] Run a low-cost Qwen2.5 SFT smoke and identify the next blocker.
- [x] Run a fresh full local verification.
- [x] Commit the provisioning hardening and final handoff.

## Assumptions
- Five adapters means five separate language-backbone LoRAs and no consolidated multitask adapter.
- Qwen3.5-9B is primary only if it is intentionally revisited with a different memory strategy. Current measured evidence selects Qwen2.5-VL-3B for the next low-cost baseline/SFT work.
- Zero false positives on frozen real pass pages is a hard promotion constraint.
- Existing Qwen3-VL-32B routing remains the production rollback until every promotion gate passes.
- Under the revised credit constraint, compatibility, frozen baselines, and SFT evidence take priority; unfinished GRPO is reported as budget-limited rather than exceeding the ceiling.

## Blockers
- RESOLVED 2026-07-17: provider-derived campaign spend was $50.82 above the old $50
  ceiling; the user explicitly raised the hard ceiling to $60 and approved the v3 plan.
- RESOLVED 2026-07-17: launch and detached watchdog now share the hard limit persisted
  in campaign state. Tests and the exact dry-run are green.
- RESOLVED 2026-07-17: the v3 manifest omitted the three immutable catalog holdout
  entries. Their paths, hashes, and sizes were independently verified and restored;
  authoritative preflight now passes.
- Brev custom-container mode is rejected for this campaign. It failed across earlier full attempts, and the tiny NVIDIA container preflight showed the host/GPU became available while the requested custom container never started.
- The official NeMo RL training container does not include PEFT or vLLM by default. PEFT can be installed, but current vLLM installation attempts replace the NeMo-pinned Torch/Transformers stack and conflict with `nemo-rl==0.6.0`.
- Do not install vLLM into the NeMo RL training image for the next spike. Treat training and serving as separate runtimes or build explicit derived images.
- The second H100 VM stop showed Brev status lag: guarded stop recorded success, direct stop then reported the backend state was already `stopped`, while `brev ls` displayed `STOPPING`. Re-check `brev ls` before any paid restart and delete stopped instances if storage cost becomes a concern.
- `vllm/vllm-openai:v0.25.1` is not compatible with the current Crusoe A100 80GB driver stack observed in this Brev VM. Use `vllm/vllm-openai:v0.8.5` for the Qwen2.5 serving-only gate unless a newer driver image/host is intentionally selected.
- RESOLVED 2026-07-16: the Qwen2.5-VL `sft_processor` / `image_grid_thw` failure is reproduced and fixed locally (strip-None patch + dataloader preflight gate). A paid SFT smoke rerun is unblocked, subject to the usual budget guards and fresh user authorization.
- Qwen3.5-9B remains parked: activation checkpointing was already on and a single-image forward/backward filled 79.16 GiB of the 80 GiB H100. Realistic single-GPU rescues (8-bit/paged optimizer, CPU-offload optimizer state, reduced `max_pixels` vision resolution, shorter max sequence length) each trade fidelity or wall-time and none is proven; multi-GPU escalation requires fresh user approval. Do not spend on Qwen3.5 without deliberately choosing one of those strategies first.
