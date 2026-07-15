# Handoff

## Resume From Here
Continue in `/Users/laccd/code/lamc_district_forms/remedy-server-nemo-rl-brev` on `codex/autoresearch/remedy-vlm-20260714/qwen25-vllm-serving`. The five-task corpus, shared verifier, Gym server, pinned recipes, compatibility spike, evaluation gates, and Brev budget watchdog are implemented. Dataset preflight passes. Integration commit `3f3f23d` has the baseline branch `codex/autoresearch/remedy-vlm-20260714/baseline`.

Three earlier paid Brev custom-container provisions failed before shell access across Crusoe and GCP. All were deleted, and no payload, inference, training, checkpoint, or evaluation result reached those instances.

A tiny known-good NVIDIA CUDA custom-container preflight was then run first, as requested. It confirmed the Brev host and A100 GPU came up, but the requested custom container did not start, no Docker container/image existed on the host, and the startup script did not run. That made custom-container mode a measured reject instead of a guess.

After that first targeted container failure, VM mode was used. `remedy-nemo-rl-vm-20260715` launched on a stoppable H100 80GB VM, accepted the payload archive, pulled `nvcr.io/nvidia/nemo-rl:v0.6.0`, and successfully completed the pinned NeMo RL/Gym setup inside the official image. The setup ended with `nemo_rl_and_gym_import_ok` using NeMo RL `c339070fa3bfa83a5ac58ff80d73518911e14b81` and Gym `25d471edfc6db9d783b31140a4e10e6194455f71`.

Provisioning follow-up commits on the current branch are `df57ea4`, `3295ba6`, `98f07b9`, `db4da1d`, and `f4f2f1d`. Fresh local verification previously finished with 347 unit tests passed and one skipped. Shell syntax, Python compilation, and campaign YAML parsing also passed.

`brev_state.json` records the H100 VM as stopped at 2026-07-15T09:54:37Z with $3.0055 VM cost and $5.3911 tracked campaign spend. After a short consistency lag, `brev ls` reports `remedy-nemo-rl-vm-20260715` as `STOPPED`. A short tiny-container preflight cost is not included in the numeric ledger because exact elapsed billing data was not captured; expected cost is below about $0.15 at the A100 rate.

The current measured blocker is not Brev VM provisioning anymore. The official NeMo RL training image does not include PEFT or vLLM; PEFT can be added in-run, but current vLLM installation attempts replace the NeMo-pinned Torch/Transformers stack and conflict with `nemo-rl==0.6.0`. Do not install vLLM into the NeMo RL training image.

The first split compatibility spike is complete. Commit `97251e4` added `--mode training|inference|both`. On `remedy-nemo-rl-compat-20260715`, Qwen3.5-9B failed the NeMo training-side gate with CUDA OOM on single H100 80GB during image forward/backward. Qwen2.5-VL-3B-Instruct passed the same gate with image forward/backward, 29,933,568 language-backbone LoRA trainables, 0 visual-tower trainables, and PEFT save/reload identity.

The serving-only follow-up is also complete. Commit `8c95465` added the reusable OpenAI-compatible one-image vLLM probe, and the current branch tightens it for vLLM versions that reject guided JSON on VLM requests. Fresh VM `remedy-qwen25-vllm-serving-20260715` used `a100-80gb.1x`, 128 GB disk, and a 1.25-hour watchdog. `vllm/vllm-openai:v0.25.1` was rejected on the Crusoe A100 host because the image required a newer NVIDIA driver path than the host exposed. `vllm/vllm-openai:v0.8.5` served `Qwen/Qwen2.5-VL-3B-Instruct` successfully after restarting with `--max-model-len 8192`; the 4096 attempt failed because the image prompt was 4863 tokens. Final probe report `qwen25_vllm_openai_probe_8192_raw_json_prompt.json` passed with `server_ready=true`, `one_image_chat_completions=true`, `zero_shot_json_valid=true`, and `technical_pass=true`.

Serving artifacts are copied locally under `session/20260714_232247/remote_artifacts/qwen25_vllm_serving/`. The serving VM was stopped through the guarded controller after artifact transfer. Delete was requested with `brev delete` by name, by ID, and through the documented stdin form, but the last `brev ls` still showed `STOPPED`, not `RUNNING`; treat compute as stopped and verify/delete manually in the Brev UI if storage charges appear.

Tracked campaign spend in the local elapsed-time ledger is now $9.9713. The user's NVIDIA Billing screenshot is authoritative provider state and showed $7.28 total cost with $45.94 balance before the short serving-only and SFT-smoke reruns settled. The serving VM cost $0.5285 locally, and the SFT smoke cost $0.9347 locally. Earlier Brev stops/deletes had status lag, so always re-check `brev ls` before any paid restart.

The first low-cost Qwen2.5 SFT smoke is complete and stopped. `remedy-qwen25-sft-smoke-20260715` launched on `a100-80gb.1x` at $1.98/hour with a 1.5-hour watchdog. It cost $0.9347 in the local ledger and raised conservative tracked spend to $9.9713. It did not produce a checkpoint. It proved the A100 VM, official NeMo image, pinned RL/Gym setup, Qwen2.5 model load, and explicit language-module LoRA recipe, then failed at the first NeMo VLM SFT dataloader batch with `IndexError: index 1 is out of bounds for dimension 0 with size 1` in Qwen2.5-VL `image_grid_thw`. See `session/20260714_232247/qwen25_sft_smoke_20260715.md`.

Final Brev state from the CLI: no active compute in the local budget controller; `brev ls` shows only `remedy-qwen25-sft-smoke-20260715` as `STOPPED`. `brev delete remedy-qwen25-sft-smoke-20260715` returned successfully but did not remove it from the list immediately. The earlier stopped serving VM has disappeared from `brev ls`.

## Next Actions
- Do not repeat Brev custom-container mode; it failed across earlier full attempts and the tiny known-good NVIDIA preflight.
- Before any new paid command, run `brev ls`. If the stopped serving VM still appears, delete it in the Brev UI or confirm storage billing before recreating a fresh VM.
- Before any new paid command, run `brev ls` and confirm no instance is `RUNNING`.
- For training-side work, use Qwen2.5-VL-3B-Instruct as the measured fallback unless Qwen3.5 is deliberately re-tested with a different memory strategy.
- For serving-side work, use a separate serving runtime. The measured working path is `vllm/vllm-openai:v0.8.5` with `--max-model-len 8192` for Qwen2.5-VL-3B on a single A100 80GB. Do not co-locate the NeMo image, vLLM image, payload, and both model caches on a 100 GB root disk again.
- Keep the same $50 hard stop, $40 no-new-work threshold, one-GPU limit, and automatic wall-time watchdog.
- Next work should be local/offline if possible: reproduce and fix the NeMo RL VLM `sft_processor` / `image_grid_thw` failure with a minimal Qwen2.5 row before another paid SFT run. Do not spend more on Qwen3.5 unless intentionally testing a different memory strategy.

## Watch Outs
- Never train from or overwrite the three held-out catalogs.
- Do not modify `main` or the older dirty multitask worktree.
- No new paid job may start at $40 recorded/projected spend, all work stops at $50, and the first paid instance must stop within three hours.
- Do not use the 4x A100 escalation under the current allocation without fresh user approval.
- Keep Brev-generated artifacts under `/ephemeral` and never print secrets from `/home/ubuntu/RL/.env`.
- The $8.5081 spend is a local elapsed-time estimate, not a provider invoice.
- The SFT smoke logs were not copied before shutdown; SSH closed immediately after stop. The important failure traces are summarized in `qwen25_sft_smoke_20260715.md` from the captured terminal output.
