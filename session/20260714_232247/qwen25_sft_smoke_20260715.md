# Qwen2.5 SFT Smoke - 2026-07-15

## Scope

- Instance: `remedy-qwen25-sft-smoke-20260715`
- GPU: single NVIDIA A100 80GB PCIe
- Brev mode: VM, not custom-container
- Hourly rate: $1.98/hour
- Guarded window: 1.5 hours
- Actual local window: 2026-07-15T18:45:07Z to 2026-07-15T19:13:26Z
- Local elapsed-time cost: $0.9347
- Final conservative local tracked campaign spend after stop: $9.9713
- Payload archive SHA-256: `29685583c1d51c5b439705541dcff36cfa59caf6d0df3bbb8efdf53a3c2f3f47`
- Source commit in payload: `4dcb5bb`

## Result

The low-cost Qwen2.5 SFT smoke did not produce a checkpoint. It did prove that the fresh A100 VM, official NeMo RL image, pinned RL/Gym setup, Qwen2.5 model load, and explicit language-module LoRA recipe all work up to the first dataloader batch.

## What passed

- Fresh A100 VM reached `RUNNING / COMPLETED / READY`.
- Payload checksum matched after upload.
- Official `nvcr.io/nvidia/nemo-rl:v0.6.0` image pulled successfully.
- NeMo RL/Gym setup completed with `nemo_rl_and_gym_import_ok`.
- `Qwen/Qwen2.5-VL-3B-Instruct` loaded inside NeMo RL on a single A100.
- Explicit LoRA target modules were accepted by NeMo Automodel:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`
- The previous invalid combination `match_all_linear=true` plus `exclude_modules=[visual,...]` is rejected by NeMo Automodel and has been replaced with explicit language-layer targets.

## Measured blockers

1. `uv run --project /home/ubuntu/RL ...` is not usable for this cloned NeMo RL layout. It fails because `nemo-gym` is referenced as a workspace source but is not a workspace member. The campaign launcher now calls `python /home/ubuntu/RL/examples/run_vlm_sft.py` directly.
2. The inherited recipe's `match_all_linear=true` with non-empty `exclude_modules` is invalid in NeMo Automodel. The recipe now uses explicit Qwen language-layer target modules with `match_all_linear=false`.
3. The generated payload contained macOS AppleDouble `._*` sidecar files when dataset files were copied from the local filesystem. `prepare_brev_payload.sh` now excludes these.
4. NeMo RL's VLM SFT dataloader still crashes on the first batch with:
   - `IndexError: index 1 is out of bounds for dimension 0 with size 1`
   - Location: `transformers/models/qwen2_5_vl/processing_qwen2_5_vl.py`, inside `image_grid_thw`

## Variants tried for the dataloader crash

- Original relative image paths, image-first content order: failed with `image_grid_thw` index error.
- Text-first content order matching NeMo examples: same `image_grid_thw` index error.
- Absolute image paths: same `image_grid_thw` index error.
- Native Qwen chat template via `policy.tokenizer.chat_template=null`: same `image_grid_thw` index error.
- Native Qwen chat template plus `data.add_bos=false` and `data.add_eos=false`: same `image_grid_thw` index error.
- Validation disabled with `sft.val_at_start=false`, `sft.val_at_end=false`, and `sft.val_period=999999`: training reaches epoch 1, then the train dataloader hits the same `image_grid_thw` error.

## Interpretation

This is now a NeMo RL VLM SFT data-processor compatibility blocker, not a Brev provisioning blocker, not a model-load blocker, and not a vLLM serving blocker. The next fix should be local/offline if possible: reproduce NeMo's `sft_processor` behavior against one dataset row, then either adapt the SFT JSONL shape to NeMo's Qwen2.5-VL processor expectations or add a campaign-specific processor before another paid SFT run.
