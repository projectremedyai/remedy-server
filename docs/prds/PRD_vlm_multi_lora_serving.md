# PRD: High-Concurrency VLM Serving and Lower-Hardware Remediation Tier

**Status:** Draft for review
**Owner:** remedy-server engineering
**Prepared:** 2026-07-07
**Repo:** `remedy-server`
**Related spec:** `docs/prds/SPEC_vlm_multi_lora_serving.md`
**Related runbook:** `runbooks/lamission-router-remediation.md`

---

## 0. Bottom line up front

The current Remedy router can use the new Qwen3-VL-32B LoRA adapters as the
premium/high-accuracy remediation path. The problem is not that this model
family needs to be replaced; the problem is that the current PEFT fallback
server serializes generation while switching adapters. That makes it correct
enough for a safe remediation pass, but not the long-term throughput shape we
want for a 10,000+ page corpus.

The research conclusion is:

1. Do **not** remake the current adapters first. The current Remedy LoRAs are
   language-backbone PEFT adapters only, with no vision-tower weights. That is
   the preferred shape for vLLM/SGLang multi-LoRA serving.
2. vLLM is still the first engine to test for the existing Qwen3-VL-32B adapter
   set, but only on a current vLLM build. The pod environment observed during
   remediation had vLLM `0.10.2`, while upstream Qwen3-VL docs call for newer
   vLLM, and public upstream issues show Qwen3-VL plus LoRA failures in older
   or specific vLLM paths.
3. SGLang is the strongest alternate engine to test because its official LoRA
   docs explicitly target multi-LoRA batches with different adapters in the
   same batch.
4. Meta's Muse Spark write-up is useful as a training philosophy: data curation,
   post-training/RL, and test-time orchestration. It is not a recipe Remedy can
   reproduce exactly because Meta did foundation-model pretraining and large
   scale RL on private infrastructure and data.
5. For a separate lower-hardware remediation tier, the best trainable base
   model candidates are:
   - **Primary Qwen-family candidate:** `Qwen/Qwen3.5-9B`
   - **Primary non-Qwen candidate:** `mistralai/Ministral-3-8B-Instruct-2512`
     or the BF16 base variant for training
   - **Conservative mature fallback:** `Qwen/Qwen2.5-VL-7B-Instruct`
   - **Strong alternate:** `OpenGVLab/InternVL3_5-8B-HF`

This PRD turns that into two complementary engineering tracks:

- **Track A: premium-path acceleration.** Prove true multi-LoRA serving with
  the existing Qwen3-VL-32B adapters so the high-accuracy router can run with
  better concurrency.
- **Track B: lower-hardware remediation.** Build a separate 8B-14B VLM path
  that can remediate on cheaper GPUs or local-ish hardware with strong
  concurrency, with any quality tradeoff measured against the Qwen3-VL-32B
  premium path.

---

## 1. Problem statement

### 1.1 Current state

Remedy now has task-specific LoRA adapters for:

- alt text quality
- table structure
- contrast detection
- reading order
- heading hierarchy

The runtime router can select a task model alias through environment-driven
routing, e.g. `contrast:qwen3vl-32b-remedy-contrast-v1`. That is the right
application interface and should be preserved.

The serving problem is under the router:

- The PEFT fallback server keeps one base model loaded and switches adapters.
- To avoid adapter-switch races, it holds a process lock around
  `set_adapter(...)` and `generate(...)`.
- Multiple PDF files can still overlap local parsing, rendering, OCR,
  veraPDF, and HTTP wait time, but actual GPU generation is effectively queued.

This makes a long corpus pass cost and wall time much worse than a real
continuous-batching multi-LoRA server should.

### 1.2 Why this is not just "turn on more workers"

More corpus workers against the PEFT fallback can increase request pressure,
but it cannot create true GPU concurrency when the generation path is locked.
It can also increase timeouts if many requests wait behind a single generation
lock.

The product need is therefore a serving backend that can:

- keep one VLM base model resident,
- keep multiple task LoRAs available,
- choose the adapter per request,
- batch concurrent requests from different tasks,
- preserve OpenAI-compatible API behavior for `pdf_vision.py`.

### 1.3 Why this is not obviously solved by Qwen3-VL plus vLLM

vLLM's general multi-LoRA story is strong, but Qwen3-VL is a newer multimodal
architecture and public upstream issues show failures in Qwen3-VL LoRA
initialization/profile paths, including reports where the adapter contains zero
visual weights.

That does not mean vLLM cannot work. It means the next attempt must be a
controlled compatibility experiment on a current vLLM release, not a blind
production switch.

---

## 2. Goals

1. **G1 - True high-concurrency serving.** Serve multiple task LoRAs on one
   base VLM with per-request adapter selection and continuous batching.
2. **G2 - Preserve Remedy runtime semantics.** Keep the existing env-routed
   `VisionProvider` interface and model aliases; do not hard-code model names.
3. **G3 - Avoid unnecessary retraining.** First prove whether the current
   Qwen3-VL-32B LoRAs can be served by vLLM or SGLang.
4. **G4 - Establish a lower-hardware tier.** Identify and validate an 8B-14B
   vision-capable base that can be fine-tuned into Remedy task adapters for
   cheaper and more concurrent remediation runs.
5. **G5 - Keep verification task-specific.** Use the existing per-task eval
   gates for alt, table, contrast, reading order, and heading hierarchy.
6. **G6 - Keep deployment reversible.** If multi-LoRA serving fails, retain the
   PEFT fallback and current router path as the safe baseline.

---

## 3. Non-goals

- Do not merge all LoRAs into separate full model weights unless serving
  experiments prove multi-LoRA is infeasible.
- Do not retrain current Qwen3-VL adapters just to satisfy vLLM unless adapter
  inspection shows vision-tower weights or incompatible target modules.
- Do not replace the current Qwen3-VL remediation router during an active
  corpus run.
- Do not assume the lower-hardware tier must replace the Qwen3-VL-32B premium
  path. It can ship as a separate throughput/cost profile.
- Do not require live LaunchAgent runtime changes until serving and eval gates
  pass.
- Do not assume any model is safe for production based only on a model card.

---

## 4. Research findings

### 4.1 vLLM

Relevant primary sources:

- vLLM LoRA docs: https://docs.vllm.ai/en/latest/features/lora/
- vLLM supported models: https://docs.vllm.ai/en/latest/models/supported_models/
- vLLM Qwen3-VL recipe: https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3-VL.html
- Qwen3-VL LoRA issues: https://github.com/vllm-project/vllm/issues/26976,
  https://github.com/vllm-project/vllm/issues/27669,
  https://github.com/vllm-project/vllm/issues/28640

Findings:

- vLLM supports LoRA for models implementing `SupportsLoRA`.
- vLLM's latest supported-models page lists many multimodal architectures with
  LoRA support, including Qwen2.5-VL, Qwen3-VL, InternVL, Pixtral/Mistral3, and
  PaliGemma.
- vLLM docs distinguish language-backbone LoRA from experimental tower and
  connector LoRA for multimodal models.
- Qwen3-VL has upstream bug reports around LoRA profiling and multimodal
  encoder paths. These are model/engine compatibility risks, not proof that
  our adapters are wrong.
- The Qwen3-VL recipe includes image/video prompt limits; for Remedy, an
  image-only run should explicitly disable video prompt profiling where the
  engine allows it.

Implication:

vLLM remains the first experiment for current adapters, but the experiment must
use a current vLLM build and a minimal one-LoRA smoke before all adapters are
loaded.

### 4.2 SGLang

Relevant primary sources:

- SGLang LoRA docs: https://docs.sglang.io/docs/advanced_features/lora
- SGLang docs home: https://docs.sglang.ai/

Findings:

- SGLang explicitly supports multiple LoRA adapters for different sequences in
  a single batch using S-LoRA and Punica-derived techniques.
- It exposes `--enable-lora`, `--lora-paths`, `--max-loras-per-batch`,
  dynamic adapter loading, LoRA GPU pinning, and an OpenAI-compatible API using
  `base-model:adapter-name` model syntax.
- SGLang documents a `csgmv` LoRA backend intended for high-concurrency
  scenarios.

Implication:

SGLang is the best alternate engine to test if vLLM remains brittle for
Qwen3-VL. The Remedy router may need a small model-name mapping layer because
SGLang's OpenAI-compatible LoRA naming convention differs from vLLM's alias
style.

### 4.3 Hugging Face TGI

Relevant primary sources:

- TGI LoRA docs: https://huggingface.co/docs/text-generation-inference/en/conceptual/lora
- TGI multi-LoRA blog: https://huggingface.co/blog/multi-lora-serving

Findings:

- TGI supports multiple LoRA adapters using `LORA_ADAPTERS` and per-request
  adapter IDs.
- The public multi-LoRA docs are strongest around text generation use cases.

Implication:

TGI is a secondary fallback for multi-LoRA serving. It is less attractive for
Remedy unless its VLM + multi-LoRA path is proven with our exact model class.

### 4.4 LMDeploy

Relevant primary sources:

- LMDeploy supported models: https://lmdeploy.readthedocs.io/en/latest/supported_models/supported_models.html
- LMDeploy Qwen2.5-VL docs: https://lmdeploy.readthedocs.io/en/latest/multi_modal/qwen2_5_vl.html

Findings:

- LMDeploy supports many VLM families, including Qwen-VL, Qwen2.5-VL, Qwen3-VL,
  InternVL, Gemma3, and others.
- The support matrix shows engine and feature differences between TurboMind and
  PyTorchEngine; not every VLM has every quantization or engine feature.

Implication:

LMDeploy is a serving fallback worth tracking, especially for InternVL-family
models, but it is not the first multi-LoRA target unless vLLM/SGLang both fail.

### 4.5 Verifier-driven post-training lessons

Relevant primary sources:

- Meta Muse Spark blog: https://ai.meta.com/blog/introducing-muse-spark-msl/
- Meta Muse Spark eval methodology:
  https://ai.meta.com/static-resource/muse-spark-eval-methodology
- DeepSeek-R1 paper: https://arxiv.org/abs/2501.12948
- Direct Preference Optimization paper: https://arxiv.org/abs/2305.18290
- ORPO paper: https://arxiv.org/abs/2403.07691
- TRL GRPO Trainer docs: https://huggingface.co/docs/trl/en/grpo_trainer
- TRL ORPO Trainer docs: https://huggingface.co/docs/trl/en/orpo_trainer
- VLM-R1 paper: https://arxiv.org/abs/2504.07615
- Hugging Face VLM GRPO cookbook: https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_grpo_trl

Findings:

- Meta describes Muse Spark as a natively multimodal reasoning model with tool
  use, visual chain of thought, and multi-agent orchestration.
- The public training framing is three scaling axes: pretraining,
  reinforcement learning, and test-time reasoning.
- Meta describes rebuilding its pretraining stack, scaling RL compute, and
  using test-time reasoning controls such as thinking-time penalties and
  multi-agent orchestration.
- The blog is not a reproducible training recipe for outside teams; it does not
  provide model weights, training data, reward code, or compute budget.
- DeepSeek-R1 shows the broader lesson that reasoning behavior can be improved
  through RL on verifiable tasks without human-labeled reasoning trajectories.
- DPO and ORPO are simpler preference-optimization options when we can convert
  validator scores into chosen/rejected pairs.
- GRPO is the better first RL algorithm when we can score multiple generated
  candidates with deterministic reward functions.
- VLM-R1 is the closest open VLM precedent: it applies R1-style RL to visual
  understanding tasks with rule-based rewards and reports better
  out-of-domain generalization than SFT in some settings.
- Meta's eval methodology reinforces the same lesson: successful model work is
  paired with task-specific grading, tools, pass-rate metrics, and explicit
  benchmark methodology rather than a single generic quality label.

Implication:

Remedy should adapt the **shape** of the Muse Spark approach, not try to copy it
literally:

- keep supervised fine-tuning as the base post-training step,
- add verifier-driven rejection sampling before any RL step,
- use ORPO or DPO for the first low-risk preference pass where validator scores
  can create chosen/rejected pairs,
- use GRPO only for tasks where we can score groups of model outputs with
  deterministic rewards,
- use deterministic rewards from JSON validity, PDF structure-tree inspection,
  veraPDF outcomes, contrast ratios, heading tag exactness, table checks, and
  content-fidelity checks,
- use test-time orchestration only for hard pages, for example multiple
  candidate analyses followed by a validator/ranker,
- avoid training models to emit hidden or lengthy chain-of-thought in
  production responses; Remedy needs compact, auditable JSON outputs.

Copyable Remedy lessons:

- **Data curation discipline:** improve synthetic and real PDF examples, require
  inspected `/H1` through `/H6` structure tags for heading data, and never trust
  broad labels such as "Tagged: yes" or "accessible PDF" as training truth.
- **Verifier-first evaluation:** use task-specific graders before model
  promotion. For Remedy this means JSON schema checks, contrast ratio math,
  structure-tree inspection, table status metrics, veraPDF, content-fidelity
  checks, and pass false-positive rates.
- **Verifier-driven post-training:** generate multiple candidate JSON outputs,
  score them with deterministic validators, build chosen/rejected pairs, and
  run ORPO or DPO before attempting GRPO.
- **Test-time orchestration:** for hard pages only, sample multiple candidate
  analyses, run validators, and choose the best valid response instead of
  blindly accepting the first model output.
- **Efficiency pressure:** keep cheap non-vision remediation first, route to
  LoRAs only when needed, and evaluate smaller VLMs for concurrency per dollar.

Do not copy from Muse Spark:

- foundation pretraining from scratch,
- frontier-lab-scale RL runs,
- Meta's internal model architecture, optimizer, reward code, or private data,
- training on hidden or long chain-of-thought targets,
- model-judge-only reward signals where deterministic validators exist,
- open-ended multi-agent remediation loops that can modify production PDFs
  without explicit validator gates.

Recommended Remedy sequence:

1. Build a validator-score dataset from existing train/val examples.
2. Generate 2-4 candidate responses per example using the current SFT adapter.
3. Score each candidate with deterministic validators.
4. Build preference pairs where the chosen response clearly beats the rejected
   response and both are grounded in the same rendered page.
5. Run a small ORPO or DPO LoRA pilot on one task, preferably contrast or
   heading hierarchy.
6. If the pilot improves held-out metrics without pass false-positive
   regression, test a short GRPO pilot with the same validators as reward
   functions.
7. Keep the tuned adapter only if it beats SFT on held-out real pages and the
   production remediation smoke.

Reward design by task:

- **Contrast:** reward valid JSON, correct pass/fail status, correct ratio
  bucket, correct text/background RGB fields, and no false positives near
  4.5:1.
- **Heading hierarchy:** reward exact `/H1` through `/H6` tag correction,
  correct element index, no heading hallucination on pass pages, and parseable
  JSON.
- **Reading order:** reward correct issue/pass status, coherent issue summary,
  and corrected-order accuracy only when the prompt asks for explicit order IDs.
- **Table:** reward status accuracy, correct confusion-matrix class, and no
  false flags on pass tables.
- **Alt text:** reward valid JSON, no false flag on gold pages, grounded
  findings, and production evaluator improvement.

---

## 5. Candidate model decision matrix

| Candidate | Size | License | Why it is attractive | Multi-LoRA risk | Recommendation |
|---|---:|---|---|---|---|
| `Qwen/Qwen3.5-9B` | 9B | Apache-2.0 | Official 9B image-text model, vLLM and SGLang usage in model card, strong document/OCR benchmarks, close to Qwen training stack | Newer architecture, needs serving smoke | Best Qwen-family lower-hardware target |
| `mistralai/Ministral-3-8B-Instruct-2512` | 8.4B LM + 0.4B vision | Apache-2.0 | Small, vision capable, vLLM usage, designed for edge deployment, strong instruction and JSON claims | Newer model, LoRA fine-tune tooling must be verified | Best non-Qwen lower-hardware target |
| `mistralai/Ministral-3-8B-Base-2512` | 8.4B LM + 0.4B vision | Apache-2.0 | BF16 base variant intended for custom post-training | Needs instruction/task tuning | Best training base if Ministral wins |
| `Qwen/Qwen2.5-VL-7B-Instruct` | 7B | Apache-2.0 | Mature VLM, strong DocVQA/OCR/table/layout benchmarks, official vLLM and SGLang snippets | Smaller than requested 9B/12B, still Qwen VLM family | Conservative fallback |
| `OpenGVLab/InternVL3_5-8B-HF` | 8.5B total | OpenGVLab/MMPR-v1.2 | Strong VLM family, HF format, vLLM and SGLang snippets, efficient 0.3B vision encoder | License and fine-tune workflow need review | Strong alternate |
| `mistralai/Pixtral-12B-2409` | 12B + 0.4B vision | Apache-2.0 | Good 12B VLM, vLLM recommended in model card, high ChartQA | Mistral blog says Pixtral 12B is deprecated; public LoRA support issue existed | Do not choose as first lite base |
| `meta-llama/Llama-3.2-11B-Vision-Instruct` | 11B | Llama 3.2 Community License | Recognized 11B VLM, DocVQA use case in model card | Prior public vLLM issue requested LoRA support; custom license | Not preferred |
| `google/paligemma2-10b-ft-docci-448` | 10B | Gemma | Designed for fine-tuning and document-caption style transfer | Not a multi-turn chat model; research-use note; adapter serving path uncertain | Use only for narrow task-specific experiments |
| `Qwen/Qwen-VL-Chat` | 7B | Tongyi Qianwen custom license | Historical Qwen VLM with official LoRA/Q-LoRA scripts and document/OCR/task eval framing | Older architecture, 448px image path, custom license, weaker modern serving story | Reference only; do not pick as first lite base |

### Current recommended path

1. **Keep current Qwen3-VL-32B router as the production-quality premium model
   family.**
2. **Try latest vLLM and SGLang with the existing Qwen3-VL LoRAs before any
   retraining.**
3. **Run a lower-hardware bakeoff between Qwen3.5-9B, Ministral 3 8B,
   Qwen2.5-VL 7B, and InternVL3.5-8B.**
4. **Train a lightweight adapter family only after a small base passes
   zero-shot, serving-compatibility, and throughput gates.**
5. **Use the original Qwen-VL repo as historical training/eval reference, not
   as the target lite model.** Its LoRA/Q-LoRA data format and evaluation
   framing are useful, but the model family is too old to displace Qwen3.5,
   Qwen2.5-VL, Ministral, or InternVL in the bakeoff.

---

## 6. Product requirements

### 6.1 Serving

- The serving backend must expose an OpenAI-compatible `/v1/chat/completions`
  endpoint.
- The serving backend must expose `/v1/models` or equivalent model-listing
  health output.
- Requests must be able to select a task adapter through the model name or a
  deterministic router mapping.
- The service must support at least five resident task adapters:
  `alt_text_quality`, `table_structure`, `contrast`, `reading_order`, and
  `heading_hierarchy`.
- The service must not silently fall back to the wrong adapter during eval.
- The service must emit logs that identify selected model alias, adapter name,
  request duration, and generation token count.

### 6.2 Remedy integration

- Existing env configuration remains authoritative:
  - `VISION_BASE_URL`
  - `OLLAMA_VISION_MODEL`
  - `OLLAMA_VISION_TASK_MODELS`
  - `OLLAMA_VISION_TASK_BASE_URLS`
  - `OLLAMA_VISION_ROUTER_ALLOW_FALLBACK`
- No code path should hard-code the final model IDs.
- The runtime should continue to work with PEFT fallback, vLLM, SGLang, or
  separate endpoints as long as the endpoint is OpenAI-compatible.

### 6.3 Model selection

- Candidate models must support image-text input and text output.
- Candidate models must have an accessible license for Remedy's intended use.
- Candidate models must have a practical LoRA or QLoRA fine-tune path.
- Candidate models must either support vLLM/SGLang directly or have a credible
  fallback engine.
- Candidate models must be evaluated on the actual Remedy task prompts, not
  only general VQA benchmarks.

### 6.4 Evaluation

The chosen serving/model path must pass:

- valid JSON rate gates per task,
- pass false-positive gates per task,
- alt-text quality regression gate,
- table status and structure gate,
- contrast status and near-threshold gate,
- reading-order status and issue-summary gate,
- heading hierarchy exact correction gate,
- production remediation smoke on held-out LAMC PDFs.

---

## 7. Acceptance gates

### 7.1 Serving gate

The serving backend passes only if:

- `/v1/models` lists every expected task alias.
- One-image smoke requests succeed for every adapter.
- 20 concurrent mixed-task requests complete without adapter cross-talk.
- Same-prompt same-adapter deterministic responses are stable enough for eval.
- Mixed-adapter throughput is at least 1.8x the PEFT serialized baseline for a
  controlled batch of short Remedy prompts.
- No request for `contrast` is answered by the alt/table/default adapter.

### 7.2 Existing-adapter gate

For the current Qwen3-VL-32B adapters:

- vLLM or SGLang must serve all existing adapters without modifying adapter
  weights.
- If adapter key translation is required, it must be automated and reversible.
- If the engine requires vision-tower LoRA or merged full models, reject that
  engine path for the current adapter family.

### 7.3 Lower-hardware tier gate

A smaller model family is eligible as a lower-hardware remediation tier only if:

- zero-shot JSON validity is at least 0.80 on representative Remedy prompts,
- after LoRA training, per-task metrics beat the base model,
- contrast and table do not regress below current task-specific gates,
- throughput per dollar is materially better than Qwen3-VL-32B,
- quality/cost tradeoffs are documented clearly so callers can choose premium
  or lower-hardware remediation intentionally.

---

## 8. Phased plan

### Phase 0 - Preserve current research

- Save this PRD and the implementation spec.
- Do not change live runtime wiring.
- Keep current corpus remediation run on the PEFT router unless stopped for
  cost or throughput reasons.

### Phase 1 - Existing Qwen3-VL serving experiments

- Create a fresh latest-vLLM environment or container.
- Test one current Remedy LoRA with text-only prompt.
- Test one current Remedy LoRA with one image prompt.
- Test all current Remedy LoRAs with `/v1/models` and mixed concurrent
  requests.
- Repeat equivalent smoke in SGLang if vLLM fails.

### Phase 2 - Lower-hardware bakeoff

Run zero-shot and serving-compatibility tests for:

- `Qwen/Qwen3.5-9B`
- `mistralai/Ministral-3-8B-Instruct-2512`
- `Qwen/Qwen2.5-VL-7B-Instruct`
- `OpenGVLab/InternVL3_5-8B-HF`

Optional secondary tests:

- `mistralai/Ministral-3-14B-*`
- `google/paligemma2-10b-*` for narrow image-to-JSON tasks

### Phase 3 - Train lower-hardware adapters

- Train one pilot task adapter first, preferably contrast or heading.
- Serve the pilot adapter through the chosen engine.
- If serving and eval pass, train the full task adapter family.
- Preserve adapter naming parallel to the current aliases.

### Phase 4 - Production tiering

- Run full per-task evals.
- Run production remediation smoke.
- Update runbook with final engine/model profile.
- Add explicit runtime profiles for:
  - premium Qwen3-VL-32B remediation,
  - lower-hardware/high-concurrency remediation,
  - PEFT fallback.
- Update LaunchAgent runtime env only after the chosen profile's gates pass.

---

## 9. Open questions

- Does latest vLLM still fail Qwen3-VL language-only LoRA when image-only
  prompt limits disable video profiling?
- Can SGLang serve Qwen3-VL or Qwen3.5 VLM LoRAs with OpenAI-compatible image
  requests and adapter selection in the model field?
- Is Qwen3.5-9B mature enough for LoRA fine-tuning with our current HF trainer,
  or does it require model-specific training code?
- Does Ministral 3 8B support PEFT LoRA fine-tuning through our current
  training path, or do we need Axolotl/ms-swift updates?
- Which small model best preserves visual document-layout behavior for heading,
  table, and reading-order tasks?

---

## 10. Source index

- Meta Muse Spark blog: https://ai.meta.com/blog/introducing-muse-spark-msl/
- DeepSeek-R1 paper: https://arxiv.org/abs/2501.12948
- Direct Preference Optimization paper: https://arxiv.org/abs/2305.18290
- ORPO paper: https://arxiv.org/abs/2403.07691
- TRL GRPO Trainer docs: https://huggingface.co/docs/trl/en/grpo_trainer
- TRL ORPO Trainer docs: https://huggingface.co/docs/trl/en/orpo_trainer
- VLM-R1 paper: https://arxiv.org/abs/2504.07615
- Hugging Face VLM GRPO cookbook: https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_grpo_trl
- vLLM LoRA docs: https://docs.vllm.ai/en/latest/features/lora/
- vLLM supported models: https://docs.vllm.ai/en/latest/models/supported_models/
- vLLM Qwen3-VL recipe: https://docs.vllm.ai/projects/recipes/en/stable/Qwen/Qwen3-VL.html
- SGLang LoRA docs: https://docs.sglang.io/docs/advanced_features/lora
- Hugging Face TGI LoRA docs: https://huggingface.co/docs/text-generation-inference/en/conceptual/lora
- Hugging Face TGI multi-LoRA blog: https://huggingface.co/blog/multi-lora-serving
- LMDeploy supported models: https://lmdeploy.readthedocs.io/en/latest/supported_models/supported_models.html
- Qwen3.5-9B model card: https://huggingface.co/Qwen/Qwen3.5-9B
- Qwen2.5-VL-7B-Instruct model card: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- Qwen-VL repository: https://github.com/QwenLM/Qwen-VL
- Ministral 3 8B Instruct model card: https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512
- Ministral 3 8B Base model card: https://huggingface.co/mistralai/Ministral-3-8B-Base-2512
- InternVL3.5-8B-HF model card: https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF
- InternVL3-8B-HF model card: https://huggingface.co/OpenGVLab/InternVL3-8B-hf
- Pixtral-12B model card: https://huggingface.co/mistralai/Pixtral-12B-2409
- PaliGemma2 10B model card: https://huggingface.co/google/paligemma2-10b-ft-docci-448
- Llama 3.2 11B Vision model card: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
- Qwen3-VL vLLM LoRA issue: https://github.com/vllm-project/vllm/issues/26976
- Qwen3-VL vLLM LoRA issue: https://github.com/vllm-project/vllm/issues/27669
- Qwen3-VL vLLM LoRA issue: https://github.com/vllm-project/vllm/issues/28640
