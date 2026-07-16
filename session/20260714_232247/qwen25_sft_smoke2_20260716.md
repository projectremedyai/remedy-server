# Qwen2.5 SFT Smoke 2 - 2026-07-16 - pipeline SUCCESS, adapter export DEFECTIVE

> **Post-retrieval audit correction (root cause found 2026-07-16 evening, $0):**
> `adapter_model.safetensors` is a 16-byte empty stub because **the run trained
> ZERO LoRA adapters**. The YAML's bare `target_modules: [q_proj, ...]` match
> NOTHING in NeMo Automodel's `ModuleMatcher` — it compares the FULL dotted
> module path with an anchored `re.match`, so only wildcard patterns like
> `*.q_proj` can hit nested modules, and `apply_lora_to_linear_modules`
> **silently accepts 0 matches**. Evidence: the printed model tree contains no
> `LinearLoRA` modules (all target projections are plain `nn.Linear`), val_loss
> is bit-identical (1.6127) at start/mid/end of "training", and the optimizer
> DCP state is 12.5 MB of metadata-only entries (no param ever received a
> gradient; activation checkpointing is why `backward()` didn't raise). The
> earlier `v4_compatible` suspicion was a red herring — that flag only affects
> config.json format. The checkpoint/save path itself is HEALTHY (CPU-verified:
> with adapters actually applied, ModelState+FSDP2+ignore_frozen returns all
> LoRA keys).
>
> **Fix (commit follows):** language-scoped wildcards in both SFT YAMLs —
> `'*.language_model.*.<proj>'`. Scoping matters: Qwen2.5-VL's VISION tower
> reuses `gate/up/down_proj` names, so unscoped `'*.gate_proj'` would train the
> vision tower and violate the 0-visual-trainables constraint. 3-way CPU repro:
> `repro_empty_adapter.py` (bare → 0 modules; unscoped → 1536 visual params
> leak; scoped → language-only, 12/12 adapter keys survive to the save).
>
> **Consequence for the smoke's claims:** the dataloader fix, preflight gate,
> import-shadowing fix, checkpoint mechanics, and budget tooling are all
> genuinely proven. The "training" itself was a frozen-model no-op — a
> few-step paid re-run with the fixed YAML is REQUIRED to prove adapters
> train and export (expect: `LinearLoRA` in the model tree, val_loss moves,
> adapter file ~60 MB).

## Scope

- Instance: `remedy-qwen25-sft-smoke2-20260716`
- GPU: single NVIDIA A100 80GB PCIe
- Brev mode: VM, official `nvcr.io/nvidia/nemo-rl:v0.6.0` container
- Hourly rate: $1.98/hour
- Guarded window: 1.5 hours (watchdog deadline 2026-07-16T19:09:58Z)
- Actual window: 2026-07-16T17:39Z to ~2026-07-16T18:57Z, $1.0673 local ledger
- Cumulative conservative local spend after stop: $11.0386
- Payload: `remedy-nemo-sft-payload-98ad504.tar.gz`, SHA-256 `00068b9771e2e944dc910ef2198290fdd75ed70a443d3f34ed0164bd8bb13549`
- Source commit: `98ad504` (strip-None patch + dataloader preflight gate)

## Result — the campaign's first trained checkpoint

`campaign sft --task contrast --model-role control` completed end to end:

- Dataloader preflight PASSED on both splits (4 rows each) — the exact code
  path that killed the 2026-07-15 smoke now processes real rows cleanly.
- Training ran 28 steps (2 epochs of 114 rows, global batch 8) at ~18.7s/step,
  GPU ~65% util / 25.5 GB. Loss ~1.42–1.50 across steps.
- Validation ran at step 0 and at end. Checkpoint saved at step 20 (pruned,
  keep_top_k=1) and step 28 (kept).
- `SFT_EXIT_CODE=0`.
- Artifacts SHA-verified and copied BEFORE stopping the VM:
  `session/20260714_232247/remote_artifacts/qwen25_sft_smoke2/` (26 MB) —
  includes `step_28/policy/weights/model/adapter_model.safetensors`,
  optimizer state, tensorboard events, and the full command log.

## The preflight gate paid for itself on the first attempt

Attempt 1 preflight FAILED with the same IndexError the fix targets. Cause:
the official image bakes its own NeMo RL copy at `/opt/nemo-rl`, which
shadows the pinned+patched clone at `/home/ubuntu/RL` — `import nemo_rl`
resolved to the UNPATCHED baked copy. The 2026-07-15 smoke ran that baked
copy too. Training never started; the defect cost a log line.

## Import-resolution fixes (now in brev_setup.sh + brev_vm_container_run.sh)

1. `ln -sfn /home/ubuntu/RL/nemo_rl /home/ubuntu/workspace/remedy-server/nemo_rl`
   — exposes the patched clone's package through the payload dir, which is
   first on the container PYTHONPATH.
2. Do NOT put the RL repo root on PYTHONPATH: NeMo's `tools/` is a REGULAR
   package (has `__init__.py`) and a regular package found at ANY sys.path
   entry beats a namespace package (our `tools/`) found earlier. This broke
   `tools.finetune.*` imports until reverted.
3. `ln -sfn /opt/nemo-rl/3rdparty /home/ubuntu/workspace/remedy-server/3rdparty`
   — `nemo_rl/__init__.py` injects Megatron-LM from `<parent>/3rdparty/…`,
   which resolves relative to the symlinked location; the shallow clone has no
   submodules, so borrow the image's baked Megatron.
4. `brev_setup.sh` now ASSERTS the imported `sft_processor` contains the patch
   (`patched_nemo_rl_import_ok`) so a future image change fails setup loudly
   instead of silently running unpatched code.

## Next

- The five-adapter SFT campaign is unblocked: same launch pattern per task
  (`contrast` done as smoke; `table_structure`, `alt_text_quality`,
  `reading_order`, `heading_hierarchy` pending) within the remaining budget
  (~$39 to the $50 hard stop, no-new-work at $40).
- Adapter quality is UNMEASURED — the smoke proves the pipeline, not the
  model. Run the evaluation gates before believing the adapter.
- Brev delete convergence: delete was requested after artifact transfer;
  `brev ls` lagged as usual. Confirm no instance remains before new work.
