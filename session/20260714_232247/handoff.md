# Handoff

## Resume From Here
Continue in `/Users/laccd/code/lamc_district_forms/remedy-server-nemo-rl-brev` on `codex/nemo-rl-brev-five-adapter`. The five-task corpus, shared verifier, Gym server, pinned recipes, compatibility spike, evaluation gates, and Brev budget watchdog are implemented. Dataset preflight passes. Integration commit `3f3f23d` has the baseline branch `codex/autoresearch/remedy-vlm-20260714/baseline`.

Three earlier paid Brev custom-container provisions failed before shell access across Crusoe and GCP. All were deleted, and no payload, inference, training, checkpoint, or evaluation result reached those instances.

A tiny known-good NVIDIA CUDA custom-container preflight was then run first, as requested. It confirmed the Brev host and A100 GPU came up, but the requested custom container did not start, no Docker container/image existed on the host, and the startup script did not run. That made custom-container mode a measured reject instead of a guess.

After that first targeted container failure, VM mode was used. `remedy-nemo-rl-vm-20260715` launched on a stoppable H100 80GB VM, accepted the payload archive, pulled `nvcr.io/nvidia/nemo-rl:v0.6.0`, and successfully completed the pinned NeMo RL/Gym setup inside the official image. The setup ended with `nemo_rl_and_gym_import_ok` using NeMo RL `c339070fa3bfa83a5ac58ff80d73518911e14b81` and Gym `25d471edfc6db9d783b31140a4e10e6194455f71`.

Provisioning follow-up commits on the current branch are `df57ea4`, `3295ba6`, `98f07b9`, `db4da1d`, and `f4f2f1d`. Fresh local verification previously finished with 347 unit tests passed and one skipped. Shell syntax, Python compilation, and campaign YAML parsing also passed.

`brev_state.json` records the H100 VM as stopped at 2026-07-15T09:54:37Z with $3.0055 VM cost and $5.3911 tracked campaign spend. After a short consistency lag, `brev ls` reports `remedy-nemo-rl-vm-20260715` as `STOPPED`. A short tiny-container preflight cost is not included in the numeric ledger because exact elapsed billing data was not captured; expected cost is below about $0.15 at the A100 rate.

The current measured blocker is not Brev VM provisioning anymore. The blocker is runtime separation: the official NeMo RL training image does not include PEFT or vLLM, PEFT can be added, but current vLLM installation attempts replace the NeMo-pinned Torch/Transformers stack and conflict with `nemo-rl==0.6.0`. Do not install vLLM into the NeMo RL training image for the next spike.

## Next Actions
- Do not repeat Brev custom-container mode; it failed across earlier full attempts and the tiny known-good NVIDIA preflight.
- Before any new paid command, run `brev ls`. If retaining the stopped VM is not worth possible storage/capacity cost, delete it before recreating a fresh VM.
- For the next compatibility spike, restart or recreate one stoppable Brev VM with a strict wall-time cap, copy the prepared payload, and run the pinned image with `tools/finetune/brev_vm_container_run.sh`.
- Split compatibility into two tracks: NeMo RL training/PEFT/forward-backward in the official training image, and vLLM one-image `/v1/chat/completions` serving in a separate serving runtime or explicit derived image.
- Keep the same $50 hard stop, $40 no-new-work threshold, one-GPU limit, and automatic wall-time watchdog.
- In a usable VM, run Qwen3.5 and 3B compatibility first. Preserve reports and stop if the target fails technically.
- Use remaining paid time for frozen baselines and task SFT in size order; stop rather than overrunning the window. Treat GRPO as optional under this credit allocation.

## Watch Outs
- Never train from or overwrite the three held-out catalogs.
- Do not modify `main` or the older dirty multitask worktree.
- No new paid job may start at $40 recorded/projected spend, all work stops at $50, and the first paid instance must stop within three hours.
- Do not use the 4x A100 escalation under the current allocation without fresh user approval.
- Keep Brev-generated artifacts under `/ephemeral` and never print secrets from `/home/ubuntu/RL/.env`.
- The $2.3856 spend is a local elapsed-time estimate, not a provider invoice.
