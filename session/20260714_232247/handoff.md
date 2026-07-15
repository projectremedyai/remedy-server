# Handoff

## Resume From Here
Continue in `/Users/laccd/code/lamc_district_forms/remedy-server-nemo-rl-brev` on `codex/nemo-rl-brev-five-adapter`. The five-task corpus, shared verifier, Gym server, pinned recipes, compatibility spike, evaluation gates, and Brev budget watchdog are implemented. Dataset preflight passes. Integration commit `3f3f23d` has the baseline branch `codex/autoresearch/remedy-vlm-20260714/baseline`.

Three earlier paid Brev custom-container provisions failed before shell access across Crusoe and GCP. All were deleted, and no payload, inference, training, checkpoint, or evaluation result reached those instances.

A tiny known-good NVIDIA CUDA custom-container preflight was then run first, as requested. It confirmed the Brev host and A100 GPU came up, but the requested custom container did not start, no Docker container/image existed on the host, and the startup script did not run. That made custom-container mode a measured reject instead of a guess.

After that first targeted container failure, VM mode was used. `remedy-nemo-rl-vm-20260715` launched on a stoppable H100 80GB VM, accepted the payload archive, pulled `nvcr.io/nvidia/nemo-rl:v0.6.0`, and successfully completed the pinned NeMo RL/Gym setup inside the official image. The setup ended with `nemo_rl_and_gym_import_ok` using NeMo RL `c339070fa3bfa83a5ac58ff80d73518911e14b81` and Gym `25d471edfc6db9d783b31140a4e10e6194455f71`.

Provisioning follow-up commits on the current branch are `df57ea4`, `3295ba6`, `98f07b9`, `db4da1d`, and `f4f2f1d`. Fresh local verification previously finished with 347 unit tests passed and one skipped. Shell syntax, Python compilation, and campaign YAML parsing also passed.

`brev_state.json` records the H100 VM as stopped at 2026-07-15T09:54:37Z with $3.0055 VM cost and $5.3911 tracked campaign spend. After a short consistency lag, `brev ls` reports `remedy-nemo-rl-vm-20260715` as `STOPPED`. A short tiny-container preflight cost is not included in the numeric ledger because exact elapsed billing data was not captured; expected cost is below about $0.15 at the A100 rate.

The current measured blocker is not Brev VM provisioning anymore. The blocker is runtime separation and serving capacity. The official NeMo RL training image does not include PEFT or vLLM; PEFT can be added in-run, but current vLLM installation attempts replace the NeMo-pinned Torch/Transformers stack and conflict with `nemo-rl==0.6.0`. Do not install vLLM into the NeMo RL training image.

The first split compatibility spike is complete. Commit `97251e4` added `--mode training|inference|both`. On `remedy-nemo-rl-compat-20260715`, Qwen3.5-9B failed the NeMo training-side gate with CUDA OOM on single H100 80GB during image forward/backward. Qwen2.5-VL-3B-Instruct passed the same gate with image forward/backward, 29,933,568 language-backbone LoRA trainables, 0 visual-tower trainables, and PEFT save/reload identity. A separate `vllm/vllm-openai:v0.25.1` serving runtime pulled successfully, but the server `docker run -d` stuck before visible container creation when root disk had about 16 GB free. See `session/20260714_232247/compatibility_results_20260715.md`.

Tracked campaign spend is now $8.5081. The latest Brev stop had status lag: guarded stop recorded success and a direct stop reported the backend was already `stopped`, while `brev ls` briefly showed `STOPPING`.

## Next Actions
- Do not repeat Brev custom-container mode; it failed across earlier full attempts and the tiny known-good NVIDIA preflight.
- Before any new paid command, run `brev ls`. If retaining the stopped VM is not worth possible storage/capacity cost, delete it before recreating a fresh VM.
- Before any new paid command, run `brev ls` and confirm no instance is `RUNNING`.
- For training-side work, use Qwen2.5-VL-3B-Instruct as the measured fallback unless Qwen3.5 is deliberately re-tested with a different memory strategy.
- For serving-side work, create a fresh serving-only VM or a larger disk VM and run only `vllm/vllm-openai:v0.25.1` plus the shared Qwen2.5-VL-3B cache. Do not co-locate the NeMo image, vLLM image, payload, and both model caches on a 100 GB root disk again.
- Keep the same $50 hard stop, $40 no-new-work threshold, one-GPU limit, and automatic wall-time watchdog.
- Next paid work should be either the vLLM serving-only gate or a very small Qwen2.5-VL-3B frozen baseline. Do not start SFT until the user accepts that Qwen3.5 failed the current single-H100 gate.

## Watch Outs
- Never train from or overwrite the three held-out catalogs.
- Do not modify `main` or the older dirty multitask worktree.
- No new paid job may start at $40 recorded/projected spend, all work stops at $50, and the first paid instance must stop within three hours.
- Do not use the 4x A100 escalation under the current allocation without fresh user approval.
- Keep Brev-generated artifacts under `/ephemeral` and never print secrets from `/home/ubuntu/RL/.env`.
- The $8.5081 spend is a local elapsed-time estimate, not a provider invoice.
