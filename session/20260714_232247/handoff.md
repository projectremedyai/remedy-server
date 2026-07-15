# Handoff

## Resume From Here
Continue in `/Users/laccd/code/lamc_district_forms/remedy-server-nemo-rl-brev` on `codex/nemo-rl-brev-five-adapter`. The five-task corpus, shared verifier, Gym server, pinned recipes, compatibility spike, evaluation gates, and Brev budget watchdog are implemented. Dataset preflight passes. Integration commit `3f3f23d` has the baseline branch `codex/autoresearch/remedy-vlm-20260714/baseline`.

Three paid Brev custom-container provisions failed before shell access across Crusoe and GCP. All were deleted, final `brev ls` reports no instances, and the local elapsed-time ledger estimates $2.3856 spend. No payload, inference, training, checkpoint, or evaluation result reached Brev.

Fresh local verification finished with 347 unit tests passed and one skipped. Shell syntax, Python compilation, and campaign YAML parsing also passed.

## Next Actions
- Do not repeat Brev custom-container mode; it failed three times across two providers.
- If the user authorizes another paid attempt, create one stoppable Brev VM with a 100 GB disk, copy the prepared payload, and run the pinned image with `tools/finetune/brev_vm_container_run.sh`.
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
