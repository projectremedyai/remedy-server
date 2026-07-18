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

## 2026-07-15 02:11:00 PDT
- Re-read the user plan from `/Users/laccd/Downloads/Five-Adapter NeMo RL Campaign on NVIDIA Brev.md`; the campaign remains the five-adapter NeMo RL on Brev plan, with the user's later $50/few-hours constraint overriding the original $100 ceiling.
- Tested Brev custom-container mode first with tiny known-good `nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04` on an A100 80GB instance.
- Inspected the Brev CLI surface before launching the large NeMo image: `brev create` supports `--container-image`, `--startup-script`, and `--mode`, but exposes no custom container entrypoint/command override; `brev exec --host` can bypass the container and run on the VM host.
- The tiny preflight host came up with an A100 visible through `nvidia-smi`, but the custom container never existed, Docker had no images or containers, and the startup marker files were absent. The instance was deleted. This rejected Brev custom-container mode without pulling the 18.73+ GB NeMo image.

## 2026-07-15 02:15:35 PDT
- Switched to Brev VM mode after the tiny custom-container failure.
- Launched `remedy-nemo-rl-vm-20260715` as a stoppable `gpu-h100-sxm.1gpu-16vcpu-200gb` VM at $4.62/hour with a three-hour watchdog.
- Verified the host exposed an H100 80GB GPU. Nebius did not provide a mounted `/ephemeral`, so the setup scripts now create `/ephemeral` on the root disk when needed.
- Transferred a 477 MB payload archive with SHA-256 `cb30edf480cb845b10da139d33495c57b45379c71fb952141b1e8862695cf9ae`, verified the checksum remotely, and extracted it under `/home/ubuntu/workspace`.
- Pulled and ran official `nvcr.io/nvidia/nemo-rl:v0.6.0`; image digest resolved to `sha256:336aa41391a99e01d018d17d327107fd6d1023ad4b2812c8d8c913dee95fd3f2`.
- Fixed VM/container setup issues in commits `df57ea4`, `3295ba6`, `98f07b9`, `db4da1d`, and `f4f2f1d`.
- Completed pinned setup inside the official image with NeMo RL `c339070fa3bfa83a5ac58ff80d73518911e14b81`, NeMo Gym `25d471edfc6db9d783b31140a4e10e6194455f71`, and final `nemo_rl_and_gym_import_ok`.

## 2026-07-15 02:54:37 PDT
- Stopped the H100 VM through the budget controller. `brev_state.json` records the VM run as stopped with $3.0055 cost and $5.3911 tracked campaign spend.
- A follow-up `brev stop remedy-nemo-rl-vm-20260715` returned an internal Brev status-transition error saying the environment was already `stopped`, while `brev ls` briefly continued to display `STOPPING`. The listing later converged to `STOPPED`. Re-check `brev ls` before any restart or delete the instance if storage charges/capacity risk outweigh retaining it.
- Measured the next blocker: vLLM should not be installed into the official NeMo RL training container because current vLLM installation attempts replace NeMo-pinned Torch and Transformers versions. Next compatibility work should split NeMo RL training from vLLM serving.

## 2026-07-15 09:34:46 PDT
- Launched fresh H100 VM `remedy-nemo-rl-compat-20260715` for a 1.5-hour compatibility window. Projected run cost was $6.93 and projected total was $12.3211, below the $40 no-new-work threshold.
- Added and committed `97251e4`, which lets `tools.finetune.remedy_nemo_rl.compatibility` run `--mode training`, `--mode inference`, or `--mode both`.
- Packaged and uploaded a 477 MB payload archive with SHA-256 `95143884f86935852edbed433c30b4e99790772fcfcfa636bd6df122b551cb6f`.
- Official NeMo RL container setup succeeded again with `nemo_rl_and_gym_import_ok`.
- Qwen3.5-9B failed the NeMo training-side compatibility gate with CUDA OOM during image forward/backward on a single H100 80GB.
- Qwen2.5-VL-3B-Instruct passed the NeMo training-side gate with image forward/backward, PEFT save/reload identity, 29,933,568 trainable LoRA parameters, and 0 visual-tower trainables.
- Pulled separate `vllm/vllm-openai:v0.25.1` image. Serving server start for Qwen2.5-VL-3B blocked because `docker run -d` stuck before visible container creation when root disk had about 16 GB free.
- Stopped the VM to protect budget. `brev_state.json` records $3.1170 for this window and $8.5081 total tracked campaign spend. Brev status lagged at `STOPPING` after the backend reported the instance was already `stopped`.

## 2026-07-15 11:16:33 PDT
- User provided the authoritative NVIDIA Billing dashboard state: total cost $7.28 and current balance $45.94. This overrides the local elapsed-time ledger for actual provider billing.
- User directed the campaign to proceed with Qwen2.5-VL-3B for the low-cost training path, redo vLLM serving on a fresh serving-only VM or larger disk, and avoid more Qwen3.5 spend unless intentionally testing a different memory strategy.

## 2026-07-15 11:20:41 PDT
- Created branch `codex/autoresearch/remedy-vlm-20260714/qwen25-vllm-serving` from `573fd0a`.
- Added reusable OpenAI-compatible one-image serving probe in commit `8c95465`.
- Launched serving-only Brev VM `remedy-qwen25-vllm-serving-20260715` on `a100-80gb.1x` at $1.98/hour with 128 GB disk and a 1.25-hour watchdog.
- Uploaded the 119 KB probe bundle with SHA-256 `ab14067de81a38f037fc85a3a9bf12b520d38075a20a4ed7ab8ee1c695729879`.
- Verified the host GPU as NVIDIA A100 80GB PCIe with driver 565.57.01 and CUDA 12.7.
- Pulled `vllm/vllm-openai:v0.25.1` successfully, but rejected it because the container required a newer NVIDIA driver/CUDA path than the host exposed.
- Pulled `vllm/vllm-openai:v0.8.5` successfully. The 4096-token server came up but rejected the one-image request because the decoder prompt length was 4863 tokens.
- Restarted vLLM 0.8.5 with `--max-model-len 8192` and `--gpu-memory-utilization 0.80`; the first non-guided probe hit Markdown code fences and therefore failed strict JSON validity.
- Tightened the probe prompt to forbid Markdown/code fences, reran against the warm 8192 server, and passed the one-image OpenAI-compatible gate with strict JSON.
- Copied serving reports, vLLM logs, pull logs, and remote SHA-256 manifest to `session/20260714_232247/remote_artifacts/qwen25_vllm_serving/`.
- Stopped `remedy-qwen25-vllm-serving-20260715` through the budget controller at 2026-07-15T18:36:42Z. The local ledger records $0.5285 for this serving-only window and $9.0366 cumulative conservative spend.
- Requested delete after artifact transfer with `brev delete` by name, by ID, and through stdin. The final 2026-07-15 `brev ls` still showed the VM as `STOPPED`, not `RUNNING`; this later converged, and the 2026-07-16 recovery check showed no instances.

## 2026-07-15 11:45:07 PDT
- Added guarded `start-existing` support in commit `4dcb5bb`.
- Attempted to restart stopped serving VM `remedy-qwen25-vllm-serving-20260715`; `brev start` looped on `instance is stopped`. The command was interrupted before the budget controller recorded an active paid window.
- Launched fresh A100 VM `remedy-qwen25-sft-smoke-20260715` for a 1.5-hour Qwen2.5 SFT smoke at $1.98/hour.
- Uploaded 477 MB payload `remedy-nemo-sft-payload-4dcb5bb.tar.gz` with SHA-256 `29685583c1d51c5b439705541dcff36cfa59caf6d0df3bbb8efdf53a3c2f3f47`.
- Official `nvcr.io/nvidia/nemo-rl:v0.6.0` setup completed on A100 with `nemo_rl_and_gym_import_ok`.
- Initial SFT wrapper failed because `uv run --project /home/ubuntu/RL` cannot resolve NeMo's `nemo-gym` workspace source in this cloned layout. The campaign launcher was patched to call `python /home/ubuntu/RL/examples/run_vlm_sft.py` directly.
- NeMo Automodel rejected `match_all_linear=true` with non-empty `exclude_modules`; the recipes were patched to explicit Qwen language-layer LoRA targets.
- The SFT smoke then loaded Qwen2.5 on A100 and reached the first dataloader batch, but every tested variant failed with Qwen2.5-VL `image_grid_thw` index error:
  - original relative paths / image-first content
  - text-first content
  - absolute image paths
  - native Qwen chat template
  - native Qwen chat template with BOS/EOS disabled
  - validation disabled
- Stopped `remedy-qwen25-sft-smoke-20260715` through the budget controller at 2026-07-15T19:13:26Z. The local ledger records $0.9347 for this SFT smoke and $9.9713 cumulative conservative spend.
- Requested deletion for `remedy-qwen25-sft-smoke-20260715`; the CLI returned successfully, but final 2026-07-15 `brev ls` still showed it as `STOPPED`. This later converged, and the 2026-07-16 recovery check showed no instances.

## 2026-07-16 09:31:43 PDT
- User asked to update `handoff.md`, `session_state.md`, and `timeline.md`.
- Re-read the session-memory skill and verified current state before editing.
- `brev ls` reported no instances in org `johnny-01be29-vebe`; the stopped SFT VM from 2026-07-15 has disappeared, so Brev deletion eventually converged.
- `brev_state.json` reported `active_instance=null`, `active_accrued_cost_usd=0.0`, and conservative local tracked spend of $9.9713.
- Git branch remained `codex/autoresearch/remedy-vlm-20260714/qwen25-vllm-serving`; the worktree was clean before this markdown refresh.
- Updated the handoff/session/timeline notes to reflect the no-instance Brev state and the current next action: fix the NeMo RL VLM SFT `image_grid_thw` processor blocker locally before any further paid training run.

## 2026-07-16 10:29:31 PDT
- User asked for a solution to the campaign blockers; work was done locally at $0 with no Brev instance launched.
- Delegated a source-excavation agent over pinned NeMo RL `c339070`; its finding was then independently verified line-by-line and live-reproduced before being relied on.
- Root-caused the `image_grid_thw` SFT dataloader crash: pinned `datasets==4.4.1` None-pads heterogeneous multimodal `content` lists at `load_dataset("json", ...)` time; Qwen2.5-VL's chat template tests key membership (`'image' in content`), so text parts with phantom `image: None` keys render as extra `<|image_pad|>` placeholders (their text silently dropped), desyncing placeholders (2) from loaded images (1) and crashing the HF processor's unbounded expansion loop on the first batch.
- Reproduced the exact paid-smoke IndexError on the Mac (CPU-only) by running the real pinned NeMo path (`OpenAIFormatDataset` -> `sft_processor` -> `Qwen2_5_VLProcessor`, transformers 5.3.0-equivalent loop) against the first real row of `sft/train.jsonl` — `session/20260714_232247/repro_image_grid_thw.py`.
- Empirically RULED OUT `use_preserving_dataset: true` as a YAML-only fix: `run_sft.py:92` `concatenate_datasets` raises `ValueError: Expected a list of Dataset objects ... element at position 0 is a PreservingDataset`.
- Shipped the fix as a read-time None-strip inside `sft_processor` (`tools/finetune/patches/nemo_rl_strip_none_multimodal_content.patch`), applied idempotently by `brev_setup.sh` after the pinned checkout; `git apply --check` verified against the pinned tree; RED->GREEN proven on the same real row (clean 2-turn log, one vision block, prompt text intact).
- Added a paid-run gate: `tools/finetune/remedy_nemo_rl/dataloader_preflight.py` replays real rows through the real processing path in-container in seconds, and `campaign._run_sft` now aborts before training if it fails.
- TDD: `tests/unit/test_nemo_rl_vlm_dataloader_fix.py` written first (8 tests: patch placement, idempotent setup hook, preflight row checks, preflight-before-training ordering, abort-on-preflight-failure). Full suite after: 359 passed, 1 skipped; shell syntax and py_compile clean.
- Recorded experiments 21 (root cause + repro) and 22 (fix + gate) in `experiments.tsv`; campaign spend unchanged at $9.9713 conservative local / $7.28 last authoritative provider view.
- Qwen3.5-9B stays parked with an explicit memory-strategy decision note in the handoff; GRPO-stage loaders flagged for the same datasets None-padding trap.

## 2026-07-16 12:05:00 PDT
- User authorized the branch push and the paid SFT smoke rerun ("yes to both").
- Pushed `codex/autoresearch/remedy-vlm-20260714/qwen25-vllm-serving` to origin (commit `98ad504`).
- Launched `remedy-qwen25-sft-smoke2-20260716` (a100-80gb.1x, $1.98/hr, 1.5h watchdog) after `brev ls` showed no instances. Payload `98ad504` uploaded and SHA-verified; setup applied the strip-None patch to the pinned clone.
- The dataloader preflight gate CAUGHT a second latent defect on attempt 1: the official image imports its own baked NeMo RL at `/opt/nemo-rl`, shadowing the pinned+patched clone. Training never started; the defect cost a log line instead of a training window — the gate's exact design purpose.
- Fixed import resolution live: symlinked the clone's `nemo_rl` into the payload dir (PYTHONPATH-first) and `/opt/nemo-rl/3rdparty` for Megatron; rejected putting the RL repo root on PYTHONPATH (NeMo's `tools/` regular package shadows our `tools/` namespace package). All fixes mirrored into `brev_setup.sh` with a `patched_nemo_rl_import_ok` assert, plus a pinning unit test.
- Preflight then PASSED both splits and the SFT smoke ran end to end: 28 steps / 2 epochs at ~18.7s/step, loss ~1.42–1.50, validation at start and end, checkpoint `step_28` with `adapter_model.safetensors`, `SFT_EXIT_CODE=0`. FIRST CAMPAIGN CHECKPOINT.
- Artifacts SHA-verified and copied BEFORE stop to `session/20260714_232247/remote_artifacts/qwen25_sft_smoke2/` (26 MB). Guarded stop recorded $1.0673 (window ~17:39–18:57Z); cumulative conservative local spend $11.0386. Delete requested; `brev ls` convergence being polled.
- Suite after all changes: 360 passed, 1 skipped.

## 2026-07-16 12:20:00 PDT
- Post-retrieval audit caught that `step_28/policy/weights/model/adapter_model.safetensors` is a 16-byte EMPTY stub: NeMo's consolidated LoRA save wrote no weights (trainer warned `save_consolidated=True but v4_compatible=False`). Training itself was real (consumed_samples=224, end val_loss 1.6127).
- Corrected the smoke record from "first checkpoint" to "pipeline proven, adapter export defective". New small blocker: diagnose NeMo's consolidated save path at the pin at $0 before any re-run; a validating re-run needs only a few steps. Plain `peft save_pretrained` is proven working on this stack (2026-07-15 compat spike) as the fallback.
- VM stopped ($1.0673 recorded, $11.0386 cumulative); delete requested, `brev ls` convergence being polled.

## 2026-07-16 12:50:00 PDT
- User confirmed the campaign stays on Brev (NVIDIA credits) and asked for the export diagnosis.
- REAL root cause of the empty adapter found at $0: bare `target_modules: [q_proj, ...]` match NOTHING in NeMo Automodel's ModuleMatcher (anchored re.match on the full dotted path; apply_lora_to_linear_modules silently accepts 0 matches). The smoke trained 28 steps with ZERO adapters: no LinearLoRA in the printed model tree, val_loss bit-identical 1.6127 at start/mid/end, 12.5 MB metadata-only optimizer state (no param ever got a gradient; activation checkpointing is why backward didn't raise). The v4_compatible warning was a red herring (config.json format only).
- The save path itself is HEALTHY: CPU repro through pinned Automodel 92635e74 (ModelState + FSDP2 + ignore_frozen_params + "lora_" filter) returns 12/12 adapter keys once modules are actually patched.
- Vision-tower trap caught before it shipped: Qwen2.5-VL's vision MLP reuses gate/up/down_proj names, so unscoped '*.gate_proj' would train the vision tower (repro: 1536 visual params leak). Fix = language-scoped '*.language_model.*.<proj>' patterns in BOTH SFT YAMLs (TDD: new scoping test + updated parity test; suite 361 passed).
- 3-way repro preserved at session/20260714_232247/repro_empty_adapter.py (bare=0 modules / unscoped=vision leak / scoped=language-only).
- Smoke claims corrected: dataloader fix, preflight gate, import-shadowing fix, checkpoint mechanics, and budget tooling are genuinely proven; the training itself was a frozen-model no-op. A few-step paid re-run with the fixed YAML is REQUIRED before any full adapter training.

## 2026-07-16 18:40:00 PDT
- User authorized the remaining four adapters ("if it validates, kick off the other four").
- ALL FIVE CONTROL ADAPTERS (Qwen2.5-VL-3B, rank-16 language-scoped LoRA) now trained, exported, and SHA-verified locally: contrast (val 1.6127->0.0593), table_structure (->0.0310), alt_text_quality (->0.1278), reading_order (->0.0193), heading_hierarchy (->0.0105). Every adapter exactly 119,809,056 bytes.
- Window A surfaced two more latent defects: (1) sft_processor truncation stubs tokens but keeps media -> forward crash 'Image features and image tokens do not match'; (2) after patching that, media-less rows crashed collation (NoneType dim_to_pack) and then BatchedDataDict's uniformity assert ([8,7]). Three fixes deep = architectural smell: the REAL fix is build-time - filter_overlong_sft_rows.py drops rows that cannot fit the 8192 context (48/~2600 dropped, worst 41,321 tokens; exact lengths, real processor, multiprocess). Patches 2+3 remain as defense in depth; preflight now probes longest rows and mixed-batch collation.
- seq len 4096->8192 in both recipes; campaign sft gained repeatable --override (heading ran val_period=50/save_period=50, saving ~1h of validation overhead).
- Windows: A $4.1767 (two failed attempts + two adapters), B $1.7683 (reading_order clean first try), C $5.8744 (heading, 237 steps, checkpoint_must_save_by closed the run cleanly; adapter retrieved 3 minutes before the watchdog).
- Campaign spend: $23.7687 conservative local of the $50 cap. All boxes stopped+delete requested. Commits pushed through dded267 (+ final docs commit).
- NEXT MILESTONE: evaluation gates - score all five adapters against the frozen baselines (eval harness + promotion evaluator already in tools/finetune), then vLLM serving probes with the trained adapters, then GRPO stage decision (Gym corpus was never transferred). The smoke-level val losses prove LEARNING, not production quality: zero-false-positive promotion constraint still ungated.

## 2026-07-16 20:45:00 PDT
- Ran the eval phase per the NVIDIA autoresearch workflow: 778 greedy generations (base vs adapter, five test splits, training-faithful max_pixels) on one A100 window ($2.687), scored locally with the campaign's evaluation.py promotion gates.
- Preflight before renting: sft<->gym test example_ids match 389/389; adapter format verified as genuine HF PEFT (504 keys, base_model.model.* naming, fp32) loadable by PeftModel.
- VERDICT: **table_structure PROMOTED** (perfect 1.000 across every gate). Four adapters beat base massively but fail gates, each with a distinct error signature: contrast collapsed to always-fail (near-threshold 100% is vacuous); reading_order collapsed to always-pass (missed 29/30 gold fails); alt_text_quality genuinely close (0.883 vs 0.90, 6 real-pass FPs, 1 invalid JSON); heading_hierarchy perfect on synthetic (118/118) but misses 26/35 REAL fails - the synthetic/real domain gap again.
- Next hypotheses recorded in handoff: GRPO with the FP-penalizing deterministic reward for alt_text + heading (closest to gates, verifier already penalizes exactly what fails); more REAL fail examples for heading; contrast/reading_order need task-input redesign (numeric contrast ratios / structural hints in prompt), not more epochs.
- Ledger $26.4557 of $50. Eval box stopped+delete requested.

## 2026-07-17 21:43:31 PDT
- User asked this agent to take over the linked PDF-remediation and adapter-training
  workstreams using `nemo-rl-docs`, `nemo-rl-auto-research`,
  `nemo-rl-brev-etiquette`, and `nemo-rl-session-memory`.
- Recovered `HANDOFF_20260713.md`, this session's handoff/state/timeline, the v3
  data-fix handoff, git state, experiment ledger, recipes, evaluator, and budget guards.
- Verified `brev ls`: no instances in org `johnny-01be29-vebe`; no active billing.
- Read the NVIDIA Billing dashboard for 2026-07-01 through 2026-07-17: $74.24 org
  total ($72.75 compute + $1.49 storage), $75.77 current balance. Removing the two
  parallel-workstream instances (`brevp1sftr2` $16.79 and `brevp1sft20260716r1`
  $6.63) leaves $50.82 for `remedy-*` campaign instances. This exceeds the explicit
  $50 hard stop; no paid launch is authorized.
- Ran the v3 authoritative dataset preflight. All content, split, image, schema,
  balance, leakage, and recorded dataset-hash checks pass, but the manifest omitted all
  three immutable catalog holdout hashes, so overall `passed=false`. Focused local
  campaign tests passed: 30 passed.
- Decision: repair and revalidate the manifest locally at $0, but do not create a new
  experiment branch or launch a GPU until the user confirms the plan; a paid launch also
  requires the user to set a new hard ceiling.
- Restored the three immutable catalog holdout entries to the ignored v3 manifest only
  after independently rechecking all source hashes and sizes against the prior accepted
  manifest. No dataset row or dataset hash changed.
- Re-ran authoritative preflight: `passed=true`, with zero document leakage, missing
  images, schema errors, holdout leaks, balance errors, dataset-hash mismatches, or
  missing holdout hashes. Persisted evidence as `datafix_v3_preflight.json`.
- Verified 1,456 media files; exact SFT/Gym example-ID alignment for alt text and heading
  across train/validation/test; alt issue types limited to
  `missing_or_placeholder`/`decorative`; zero subjective rows; six placeholder spot
  checks passed. Full unit suite: 368 passed, 1 skipped.
- Stop condition remains met: provider-derived campaign spend is $50.82/$50. No branch
  creation, payload packaging, or paid launch was performed.

## 2026-07-17 22:34:03 PDT
- User explicitly approved the v3 experiment plan and raised the campaign hard ceiling
  from $50 to $60.
- Restated scoped plan: dedicated v3 hypothesis branch; provider-reconciled starting
  spend $50.82; one A100; at most three hours; alt text then heading only; frozen v3
  test evaluation; retrieve and SHA-verify artifacts before stop. GRPO, MOVE3, Qwen3.5,
  and the other three adapters remain out of scope.
- Rechecked `brev ls`: no instances in org `johnny-01be29-vebe`.
- Pre-branch finding: the controller launch path can override reserve but the detached
  watchdog still hardcodes the historical $50 policy. Decision: persist the approved
  hard limit in campaign state and test launch/watch behavior before any paid command.
- Created branch `codex/autoresearch/remedy-vlm-20260714/datafix-v3-sft`; MOVE3 was
  preserved untracked and unstaged.
- Added state-persisted `hard_limit_usd`/`reserve_usd` plus explicit
  `--hard-limit-usd`; launch/start previews now expose the active policy and the detached
  watchdog reads the persisted hard limit. Added Google-style public function docs and
  updated the indexed campaign spec; no new docs page was added, so `docs/index.md`
  required no change.
- Reconciled campaign state from $38.3907 local to $50.82 provider-derived spend with a
  $12.4293 explicit adjustment. Stored approved hard limit $60 and reserve $0.60.
- Live inventory check: stoppable Crusoe `a100-80gb.1x`, 80 GB VRAM, 128 GB disk,
  advertised $1.98/hour. Guard uses conservative $3.00/hour for 2.85 hours.
- TDD: new budget/state tests failed first on missing persistent-policy support, then
  passed after implementation. Full unit suite: 378 passed, 1 skipped. Shell syntax,
  Python compilation, and exact launch dry-run passed. Projection: $8.55 run,
  $59.37 cumulative, $0.63 remaining under the hard limit.

## 2026-07-17 22:41:36 PDT
- Committed the v3 objective-label hypothesis as `19f31a7` and the persistent $60
  watchdog policy as `145e1fa` on the dedicated datafix-v3 branch. MOVE3 remains
  untracked and unstaged.
- Packaged v3 SFT plus 1,456 media files into
  `/tmp/remedy-v3-payload-145e1fa.tar.gz` (SHA-256
  `d1757d1dbc7c34dd2c5e332b8f2f67f426fa29f417c7f41663f4cdeac4499e27`). Gym
  base64 payloads were intentionally excluded from the paid SFT/eval window.
- The old two-epoch heading run took 175 minutes by itself, so the authorized window
  will use one complete epoch per fixed task, no validation at start, end validation,
  and end checkpoint. Expected work: about 21 minutes alt + 88 minutes heading + about
  22 minutes for 208 adapter-only frozen-test generations, leaving setup, transfer,
  retrieval, and stop margin inside 171 minutes. This is a real one-epoch experiment,
  not a smoke; a missed gate is reported as budget-limited evidence and is not extended.

## 2026-07-17 23:08:07 PDT
- Guarded A100 `remedy-qwen25-v3-sft-20260717` created; payload copied and independently
  SHA-verified on the host. Setup passed pinned NeMo/Gym commits, three patches, and the
  patched import assertion. Dataset copy has 1,456 real media files.
- Found and fixed a live watchdog-detach defect before relying on it: inherited PTY
  stdin became a bad descriptor when the launcher exited. Added `stdin=DEVNULL` with a
  RED-to-GREEN regression test; full unit suite 379 passed / 1 skipped. Corrected
  watchdog PID 67914 is detached under init and enforcing the original deadline.
- Alt attempt 1 stopped before a successful optimizer step on `Batch sizes [8,7]`.
  Exact processor audit found the v3 data had not been passed through the mandatory
  8,128-token build-time filter: dropped 4/238 alt train, 49/1,202 heading train, and
  4/188 heading validation rows; frozen test splits unchanged. Alt stays exactly
  balanced at 117 pass / 117 fail; heading is 576 fail / 577 pass.
- Alt attempt 2 passed dataloader preflight on the filtered corpus, loaded genuine
  language-scoped LoRA modules, and began the one-epoch 29-step run.
