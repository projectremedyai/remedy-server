# Handoff

## Resume From Here
Continue in `/Users/laccd/code/lamc_district_forms/remedy-server-nemo-rl-brev` on `codex/nemo-rl-brev-five-adapter`. The five-task corpus, shared verifier, Gym server, pinned recipes, compatibility spike, evaluation gates, and Brev budget watchdog are implemented. Dataset preflight passes. Skills are installed, Brev is authenticated, and no compute has been launched.

## Next Actions
- Run the complete unit suite and commit the integration baseline.
- Create `codex/autoresearch/remedy-vlm-20260714/baseline` at that commit.
- Build the roughly 527 MB Brev payload, run the H200 launch dry-run, then execute only with the three-hour watchdog armed.
- Run Qwen3.5 and 3B compatibility first. Preserve reports and stop if the target fails technically.
- Use remaining paid time for frozen baselines and task SFT in size order; stop rather than overrunning the window.

## Watch Outs
- Never train from or overwrite the three held-out catalogs.
- Do not modify `main` or the older dirty multitask worktree.
- No new paid job may start at $40 recorded/projected spend, all work stops at $50, and the first paid instance must stop within three hours.
- Do not use the 4x A100 escalation under the current allocation without fresh user approval.
- Keep Brev-generated artifacts under `/ephemeral` and never print secrets from `/home/ubuntu/RL/.env`.
