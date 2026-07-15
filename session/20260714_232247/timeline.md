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
- Requested delete after artifact transfer with `brev delete` by name, by ID, and through stdin. The final `brev ls` still showed the VM as `STOPPED`, not `RUNNING`; compute is stopped, but the Brev UI should be checked for lingering storage charges.

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
- Requested deletion for `remedy-qwen25-sft-smoke-20260715`; the CLI returned successfully, but final `brev ls` still showed it as `STOPPED`. The earlier serving VM no longer appeared in `brev ls`.
