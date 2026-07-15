# Compatibility Results - 2026-07-15

## Scope

- Instance: `remedy-nemo-rl-compat-20260715`
- GPU: single NVIDIA H100 80GB
- Brev mode: VM, not custom-container
- Guarded window: 1.5 hours
- Final tracked campaign spend after guarded stop: $8.5081
- Payload archive SHA-256: `95143884f86935852edbed433c30b4e99790772fcfcfa636bd6df122b551cb6f`
- Official NeMo image digest: `sha256:336aa41391a99e01d018d17d327107fd6d1023ad4b2812c8d8c913dee95fd3f2`

## Training-side compatibility

### `Qwen/Qwen3.5-9B`

- Result: fail
- Gate: image forward/backward with language-backbone LoRA
- Error type: `OutOfMemoryError`
- Evidence: model loaded and reached image forward/backward, then failed with CUDA OOM on the H100 80GB.
- Key error excerpt: CUDA tried to allocate 20 MiB with only about 14 MiB free; PyTorch had about 78.11 GiB allocated and the process had about 79.16 GiB in use.
- Interpretation: Qwen3.5-9B does not pass the single-H100 training-side technical gate in the current NeMo/Transformers path. Per campaign policy, the control family becomes the practical fallback unless Qwen3.5 is revisited with a materially different memory strategy.

### `Qwen/Qwen2.5-VL-3B-Instruct`

- Result: pass
- Gate: image forward/backward with language-backbone LoRA, no visual-tower trainables, PEFT save/reload identity
- `technical_pass`: `true`
- `forward_backward`: `true`
- `adapter_save_reload_identical`: `true`
- `trainable_parameters`: `29933568`
- `gradient_parameters`: `29933568`
- `visual_trainable_parameters`: `0`
- `adapter_sha256`: `15ea97a265f605c47369bca7941a377de8beb162815ae9d80c198eac47a6f76b`
- Saved adapter files: `README.md`, `adapter_config.json`, `adapter_model.safetensors`
- Interpretation: the 3B control passed the NeMo training-side compatibility gate on single H100.

## Serving-side compatibility

### First H100 compatibility VM attempt

- Runtime attempted: separate `vllm/vllm-openai:v0.25.1` container, not installed into the NeMo training image.
- Image pull: succeeded.
- Server start: blocked.
- Evidence: `docker run -d` for the vLLM server stuck before visible container creation. Root disk was tight at about 16 GB free after pulling both NeMo RL and vLLM images plus model/cache data.
- Interpretation: this is a serving-runtime feasibility blocker, not a model-quality result. The next serving spike should use a larger disk or a fresh serving-only VM/image, then run the OpenAI-compatible `/v1/chat/completions` one-image gate for `Qwen/Qwen2.5-VL-3B-Instruct`.

### Fresh serving-only A100 VM

- Instance: `remedy-qwen25-vllm-serving-20260715`
- GPU: single NVIDIA A100 80GB PCIe
- Brev mode: VM, not custom-container
- Disk: 128 GB
- Hourly rate: $1.98/hour
- Guarded window: 1.25 hours
- Actual local window: 2026-07-15T18:20:41Z to 2026-07-15T18:36:42Z
- Local elapsed-time cost: $0.5285
- Final conservative local tracked campaign spend: $9.0366
- Provider billing screenshot before this rerun: $7.28 total cost, $45.94 balance
- Probe bundle SHA-256: `ab14067de81a38f037fc85a3a9bf12b520d38075a20a4ed7ab8ee1c695729879`
- Local artifact directory: `session/20260714_232247/remote_artifacts/qwen25_vllm_serving/`

#### `vllm/vllm-openai:v0.25.1`

- Image pull: succeeded.
- Image digest: `sha256:e4f88a835143cd22aee2397a26ec6bb80b3a4a6fe0c882bcbc63822904766089`
- Result: reject
- Evidence: the container rejected the host driver path with an NVIDIA driver/CUDA compatibility error. The host exposed driver `565.57.01` and CUDA `12.7`.
- Interpretation: do not use vLLM 0.25.1 on this Brev A100 host unless a newer driver/host image is intentionally selected.

#### `vllm/vllm-openai:v0.8.5`

- Image pull: succeeded.
- Image digest: `sha256:6cf9808ca8810fc6c3fd0451c2e7784fb224590d81f7db338e7eaf3c02a33d33`
- Initial 4096-token result: reject
- Evidence: the OpenAI server became ready, but `/v1/chat/completions` rejected the one-image request because the decoder prompt length was 4863 tokens, exceeding `--max-model-len 4096`.
- Final 8192-token result: pass
- Final server settings: `--max-model-len 8192`, `--gpu-memory-utilization 0.80`, BF16
- Final probe: `qwen25_vllm_openai_probe_8192_raw_json_prompt.json`
- `server_ready`: `true`
- `one_image_chat_completions`: `true`
- `zero_shot_json_valid`: `true`
- `technical_pass`: `true`
- Raw assistant content: `{"status": "pass", "findings": []}`
- Interpretation: Qwen2.5-VL-3B now passes both the NeMo training-side compatibility gate and the separate OpenAI-compatible one-image serving gate. Proceed with Qwen2.5-VL-3B for the low-cost baseline/SFT path unless Qwen3.5 is deliberately retested with a different memory strategy.

## Operational notes

- The Qwen3.5 OOM left a NeMo container lingering with no GPU memory in use; Docker required a VM reboot to recover cleanly.
- The 3B training pass also left a container lingering after writing the report, but GPU memory was clear.
- The VM was stopped immediately after the vLLM server start blocker to protect the user's $50 credit cap.
- `brev stop` reported success through the guarded controller. A follow-up direct stop reported Brev backend state was already `stopped`, while `brev ls` briefly displayed `STOPPING`, consistent with earlier Brev status lag.
- The serving-only VM was stopped immediately after the passing probe. Delete was requested after artifact transfer, but final `brev ls` still showed `STOPPED`, not `RUNNING`; compute is stopped, and the Brev UI should be checked if storage charges appear.
