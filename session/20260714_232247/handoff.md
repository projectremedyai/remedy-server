# Handoff

## 2026-07-16 evening — SMOKE: pipeline SUCCEEDED end-to-end; adapter EXPORT defective

`remedy-qwen25-sft-smoke2-20260716` (A100, $1.07) ran `campaign sft --task
contrast --model-role control` end to end on payload `98ad504`: preflight
passed both splits, 28 steps / 2 epochs, val at start+end, exit 0. The
dataloader blocker is DEAD in production conditions. Artifacts (SHA-verified,
copied BEFORE stop): `session/20260714_232247/remote_artifacts/qwen25_sft_smoke2/`.
Details: `qwen25_sft_smoke2_20260716.md`. Spend: $11.0386 conservative local.

**BUT the adapter was EMPTY — root cause found and fixed the same evening ($0):**
the smoke trained **ZERO LoRA adapters**. Bare `target_modules: [q_proj, ...]`
match NOTHING in NeMo Automodel's ModuleMatcher (anchored `re.match` on the
FULL dotted path; `apply_lora_to_linear_modules` silently accepts 0 matches).
Proof: no `LinearLoRA` in the printed model tree, val_loss bit-identical
(1.6127) at start/mid/end, 12.5 MB metadata-only optimizer state. The
`v4_compatible` warning was a red herring (config.json format only); the save
path itself is healthy once adapters exist (CPU-verified end to end).
**Fix shipped:** `'*.language_model.*.<proj>'` wildcards in both SFT YAMLs —
language-scoped because Qwen2.5-VL's VISION tower reuses gate/up/down_proj
names and unscoped wildcards would train it. Pinned by two tests; 3-way CPU
repro in `session/20260714_232247/repro_empty_adapter.py`.
**Still owed: a few-step paid re-run** with the fixed YAML to prove adapters
train + export (grep the log for `LinearLoRA`, expect val_loss to move and a
~60 MB adapter file). Do NOT start full five-adapter training on an
unvalidated export.
**Upstream-worthy findings (not filed):** silent 0-match in
`apply_lora_to_linear_modules`, and the empty-dict PEFT save writing a
16-byte safetensors without warning.

**The preflight gate caught a second latent defect on its first live run:** the
official image imports its own baked NeMo RL at `/opt/nemo-rl`, which shadows
the pinned+patched clone (yesterday's smoke ran the baked copy too). Fixes now
in `brev_setup.sh` (nemo_rl symlink through the payload dir + `/opt/nemo-rl/3rdparty`
symlink for Megatron + a `patched_nemo_rl_import_ok` assert). Never put the RL
repo ROOT on PYTHONPATH — NeMo's `tools/` regular package shadows our `tools/`
namespace package.

**Next:** the five-adapter SFT campaign is unblocked — remaining tasks
(`table_structure`, `alt_text_quality`, `reading_order`, `heading_hierarchy`)
use the same launch pattern, ~$39 of headroom to the $50 stop. The smoke
proves the PIPELINE, not the model: run the evaluation gates on the contrast
adapter before believing it.

## 2026-07-16 morning — the SFT dataloader blocker is FIXED (locally, $0)

The `IndexError: index 1 is out of bounds for dimension 0 with size 1` in Qwen2.5-VL
`image_grid_thw` is root-caused, reproduced on a Mac CPU, and fixed:

- **Root cause:** NeMo RL's pinned `datasets==4.4.1` None-pads heterogeneous multimodal
  `content` lists at `load_dataset("json", ...)` time — every text part gains a phantom
  `"image": None` key. Qwen2.5-VL's chat template tests key MEMBERSHIP
  (`'image' in content`), so corrupted text parts render as extra `<|image_pad|>`
  placeholders and their real text is silently dropped. 2 placeholders vs 1 loaded image
  crashes the HF processor's unbounded expansion loop on the first batch. All six
  variants tried on the paid smoke were downstream of the loader — that is why they
  failed identically.
- **Repro:** `session/20260714_232247/repro_image_grid_thw.py` runs the REAL pinned
  NeMo path (`OpenAIFormatDataset` -> `sft_processor` -> `Qwen2_5_VLProcessor`) on the
  first real row of our `sft/train.jsonl`. Unpatched tree: exact paid-smoke IndexError.
  Patched tree (`REPRO_EXPECT=fixed`): clean 2-turn log, exactly one vision block,
  prompt text intact.
- **Fix:** `tools/finetune/patches/nemo_rl_strip_none_multimodal_content.patch` —
  strips None-valued keys at READ time inside `sft_processor`. Read time is the ONLY
  correct placement: `Dataset.map` re-encodes through Arrow and re-injects the Nones,
  and `use_preserving_dataset: true` was tested and RULED OUT (`run_sft.py:92`
  `concatenate_datasets` type-rejects `PreservingDataset`). `brev_setup.sh` applies the
  patch idempotently right after the pinned checkout (HEAD stays `c339070…`, so the
  pinned-SHA assertions still pass).
- **Paid-run gate:** `tools/finetune/remedy_nemo_rl/dataloader_preflight.py` replays
  real rows through the real processing path inside the container in seconds;
  `campaign._run_sft` runs it BEFORE training and aborts on failure. The failure class
  that cost a $0.93 smoke now costs a log line.
- **Verification:** 359 unit tests passed, 1 skipped (includes
  `tests/unit/test_nemo_rl_vlm_dataloader_fix.py`); patch `git apply --check` clean
  against the pinned commit; RED->GREEN on the real row.
- **Trap for committers:** `.gitignore:150` (`tools/finetune/*`) ignores non-`.py`
  files — the `.patch` needs `git add -f` or it never ships in a payload.
- **Next paid step (needs fresh user authorization):** rerun the 1.5-hour Qwen2.5
  SFT smoke on `a100-80gb.1x`. Expected: preflight passes in seconds, training gets
  past the first batch, and the smoke either completes a checkpoint or surfaces the
  NEXT blocker much deeper in the run.
- **Qwen3.5-9B (still parked):** activation checkpointing was already on and a
  single-image forward/backward filled 79.16 GiB of 80. Single-GPU rescue options —
  8-bit/paged optimizer, CPU-offload of optimizer state, reduced `max_pixels` vision
  resolution, shorter sequence length — each trade fidelity or wall-time and none is
  proven for this model; 4x A100 escalation needs fresh user approval. Decide
  deliberately before spending anything on Qwen3.5.
- **GRPO watch-out for later:** the same `datasets` None-padding trap applies to ANY
  loader that round-trips heterogeneous multimodal messages through HF datasets. When
  the GRPO stage is built, preflight its data path the same way before paying.

## Resume From Here (2026-07-15 state, still accurate below)
Continue in `/Users/laccd/code/lamc_district_forms/remedy-server-nemo-rl-brev` on `codex/autoresearch/remedy-vlm-20260714/qwen25-vllm-serving`. The five-task corpus, shared verifier, Gym server, pinned recipes, compatibility spike, evaluation gates, and Brev budget watchdog are implemented. Dataset preflight passes. Integration commit `3f3f23d` has the baseline branch `codex/autoresearch/remedy-vlm-20260714/baseline`.

Three earlier paid Brev custom-container provisions failed before shell access across Crusoe and GCP. All were deleted, and no payload, inference, training, checkpoint, or evaluation result reached those instances.

A tiny known-good NVIDIA CUDA custom-container preflight was then run first, as requested. It confirmed the Brev host and A100 GPU came up, but the requested custom container did not start, no Docker container/image existed on the host, and the startup script did not run. That made custom-container mode a measured reject instead of a guess.

After that first targeted container failure, VM mode was used. `remedy-nemo-rl-vm-20260715` launched on a stoppable H100 80GB VM, accepted the payload archive, pulled `nvcr.io/nvidia/nemo-rl:v0.6.0`, and successfully completed the pinned NeMo RL/Gym setup inside the official image. The setup ended with `nemo_rl_and_gym_import_ok` using NeMo RL `c339070fa3bfa83a5ac58ff80d73518911e14b81` and Gym `25d471edfc6db9d783b31140a4e10e6194455f71`.

Provisioning follow-up commits on the current branch are `df57ea4`, `3295ba6`, `98f07b9`, `db4da1d`, and `f4f2f1d`. Fresh local verification previously finished with 347 unit tests passed and one skipped. Shell syntax, Python compilation, and campaign YAML parsing also passed.

`brev_state.json` records the H100 VM as stopped at 2026-07-15T09:54:37Z with $3.0055 VM cost and $5.3911 tracked campaign spend. After a short consistency lag, `brev ls` reports `remedy-nemo-rl-vm-20260715` as `STOPPED`. A short tiny-container preflight cost is not included in the numeric ledger because exact elapsed billing data was not captured; expected cost is below about $0.15 at the A100 rate.

The current measured blocker is not Brev VM provisioning anymore. The official NeMo RL training image does not include PEFT or vLLM; PEFT can be added in-run, but current vLLM installation attempts replace the NeMo-pinned Torch/Transformers stack and conflict with `nemo-rl==0.6.0`. Do not install vLLM into the NeMo RL training image.

The first split compatibility spike is complete. Commit `97251e4` added `--mode training|inference|both`. On `remedy-nemo-rl-compat-20260715`, Qwen3.5-9B failed the NeMo training-side gate with CUDA OOM on single H100 80GB during image forward/backward. Qwen2.5-VL-3B-Instruct passed the same gate with image forward/backward, 29,933,568 language-backbone LoRA trainables, 0 visual-tower trainables, and PEFT save/reload identity.

The serving-only follow-up is also complete. Commit `8c95465` added the reusable OpenAI-compatible one-image vLLM probe, and the current branch tightens it for vLLM versions that reject guided JSON on VLM requests. Fresh VM `remedy-qwen25-vllm-serving-20260715` used `a100-80gb.1x`, 128 GB disk, and a 1.25-hour watchdog. `vllm/vllm-openai:v0.25.1` was rejected on the Crusoe A100 host because the image required a newer NVIDIA driver path than the host exposed. `vllm/vllm-openai:v0.8.5` served `Qwen/Qwen2.5-VL-3B-Instruct` successfully after restarting with `--max-model-len 8192`; the 4096 attempt failed because the image prompt was 4863 tokens. Final probe report `qwen25_vllm_openai_probe_8192_raw_json_prompt.json` passed with `server_ready=true`, `one_image_chat_completions=true`, `zero_shot_json_valid=true`, and `technical_pass=true`.

Serving artifacts are copied locally under `session/20260714_232247/remote_artifacts/qwen25_vllm_serving/`. The serving VM was stopped through the guarded controller after artifact transfer. Delete was requested with `brev delete` by name, by ID, and through the documented stdin form. On 2026-07-15 the CLI still showed it as `STOPPED`, not `RUNNING`; the 2026-07-16 recovery check showed no instances, so the delete/state convergence eventually completed.

Tracked campaign spend in the local elapsed-time ledger is now $9.9713. The user's NVIDIA Billing screenshot is authoritative provider state and showed $7.28 total cost with $45.94 balance before the short serving-only and SFT-smoke reruns settled. The serving VM cost $0.5285 locally, and the SFT smoke cost $0.9347 locally. Earlier Brev stops/deletes had status lag, so always re-check `brev ls` before any paid restart.

The first low-cost Qwen2.5 SFT smoke is complete and stopped. `remedy-qwen25-sft-smoke-20260715` launched on `a100-80gb.1x` at $1.98/hour with a 1.5-hour watchdog. It cost $0.9347 in the local ledger and raised conservative tracked spend to $9.9713. It did not produce a checkpoint. It proved the A100 VM, official NeMo image, pinned RL/Gym setup, Qwen2.5 model load, and explicit language-module LoRA recipe, then failed at the first NeMo VLM SFT dataloader batch with `IndexError: index 1 is out of bounds for dimension 0 with size 1` in Qwen2.5-VL `image_grid_thw`. See `session/20260714_232247/qwen25_sft_smoke_20260715.md`.

Final Brev state from the CLI: no active compute in the local budget controller and, as of the 2026-07-16 recovery check, `brev ls` reports no instances in org `johnny-01be29-vebe`. `brev delete remedy-qwen25-sft-smoke-20260715` did not remove the stopped VM immediately on 2026-07-15, but Brev later converged and the instance no longer appears.

## Next Actions
- Do not repeat Brev custom-container mode; it failed across earlier full attempts and the tiny known-good NVIDIA preflight.
- Before any new paid command, run `brev ls` and confirm no instance is `RUNNING`.
- For training-side work, use Qwen2.5-VL-3B-Instruct as the measured fallback unless Qwen3.5 is deliberately re-tested with a different memory strategy.
- For serving-side work, use a separate serving runtime. The measured working path is `vllm/vllm-openai:v0.8.5` with `--max-model-len 8192` for Qwen2.5-VL-3B on a single A100 80GB. Do not co-locate the NeMo image, vLLM image, payload, and both model caches on a 100 GB root disk again.
- Keep the same $50 hard stop, $40 no-new-work threshold, one-GPU limit, and automatic wall-time watchdog.
- ~~Next work should be local/offline if possible: reproduce and fix the NeMo RL VLM `sft_processor` / `image_grid_thw` failure with a minimal Qwen2.5 row before another paid SFT run.~~ **DONE 2026-07-16 — see the update at the top of this file.** Do not spend more on Qwen3.5 unless intentionally testing a different memory strategy.

## Watch Outs
- Never train from or overwrite the three held-out catalogs.
- Do not modify `main` or the older dirty multitask worktree.
- No new paid job may start at $40 recorded/projected spend, all work stops at $50, and the first paid instance must stop within three hours.
- Do not use the 4x A100 escalation under the current allocation without fresh user approval.
- Keep Brev-generated artifacts under `/ephemeral` and never print secrets from `/home/ubuntu/RL/.env`.
- The $9.9713 spend is a local elapsed-time estimate, not a provider invoice. The user's NVIDIA Billing screenshot remains the authoritative provider view for actual settled spend.
- The SFT smoke logs were not copied before shutdown; SSH closed immediately after stop. The important failure traces are summarized in `qwen25_sft_smoke_20260715.md` from the captured terminal output.
