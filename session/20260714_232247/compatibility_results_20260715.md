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

- Runtime attempted: separate `vllm/vllm-openai:v0.25.1` container, not installed into the NeMo training image.
- Image pull: succeeded.
- Server start: blocked.
- Evidence: `docker run -d` for the vLLM server stuck before visible container creation. Root disk was tight at about 16 GB free after pulling both NeMo RL and vLLM images plus model/cache data.
- Interpretation: this is a serving-runtime feasibility blocker, not a model-quality result. The next serving spike should use a larger disk or a fresh serving-only VM/image, then run the OpenAI-compatible `/v1/chat/completions` one-image gate for `Qwen/Qwen2.5-VL-3B-Instruct`.

## Operational notes

- The Qwen3.5 OOM left a NeMo container lingering with no GPU memory in use; Docker required a VM reboot to recover cleanly.
- The 3B training pass also left a container lingering after writing the report, but GPU memory was clear.
- The VM was stopped immediately after the vLLM server start blocker to protect the user's $50 credit cap.
- `brev stop` reported success through the guarded controller. A follow-up direct stop reported Brev backend state was already `stopped`, while `brev ls` briefly displayed `STOPPING`, consistent with earlier Brev status lag.
