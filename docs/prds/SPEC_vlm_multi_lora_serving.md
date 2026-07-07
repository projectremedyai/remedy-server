# Spec Sheet: VLM Multi-LoRA Serving and Lower-Hardware Bakeoff

**Status:** Draft implementation spec
**Prepared:** 2026-07-07
**Companion PRD:** `docs/prds/PRD_vlm_multi_lora_serving.md`

---

## 1. Objective

Build and validate high-concurrency VLM serving options for Remedy without
collapsing them into a single replacement path. There are two complementary
objectives:

1. accelerate the existing Qwen3-VL-32B premium Remedy LoRA router through
   true multi-LoRA batching, and
2. identify and train a smaller VLM base that can remediate on lower-end
   hardware while still supporting strong concurrency.

Any new runtime must remain OpenAI-compatible so `pdf_vision.py` can keep using
the existing provider and task-router abstractions.

This spec also captures a non-replacement post-training path:

- Meta Muse Spark's public write-up informs a verifier-driven post-training
  strategy, but Remedy will not attempt foundation pretraining at Meta scale.

---

## 2. Current Remedy adapter evidence

The current local Remedy LoRAs were inspected from their PEFT metadata and
safetensor headers.

Observed shape:

- `peft_type`: `LORA`
- `task_type`: `CAUSAL_LM`
- `r`: `16`
- `lora_alpha`: `32`
- `target_modules`: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, `down_proj`
- `modules_to_save`: `None`
- safetensor keys: `base_model.model.model.language_model.layers.*`
- visual-like keys: `0`

Adapters checked:

- `lamc-qwen3vl-32b-multitask-lora`
- `lamc-qwen3vl-32b-heading-lora`
- `lamc-qwen3vl-32b-reading-order-lora`
- `lamc-qwen3vl-32b-contrast-lora`
- `lamc-qwen3vl-32b-lora-v2`
- `lamc-qwen3vl-32b-table-lora`

Conclusion:

The adapters are already in the desired language-backbone-only shape for
multi-LoRA serving. Do not retrain or remake them unless a specific serving
engine proves it cannot map these PEFT target modules.

---

## 3. Verifier-Driven Post-Training Boundary

### 3.1 Muse Spark style training boundary

Meta's Muse Spark blog describes three useful axes: better pretraining,
reinforcement learning, and test-time reasoning/orchestration. Remedy can adopt
only the parts that fit our scale and compliance envelope.

Adopt:

- supervised fine-tuning on curated Remedy task data,
- verifier-driven rejection sampling,
- ORPO/DPO preference tuning where deterministic validators produce
  chosen/rejected pairs,
- GRPO-style RL only where reward signals are objective and stable,
- test-time candidate generation plus deterministic validators on hard pages,
- explicit eval-awareness checks so models do not learn to pass only the test
  format.

Do not adopt:

- foundation pretraining from scratch,
- training on hidden chain-of-thought targets,
- unbounded agentic loops during remediation,
- reward models based only on another model's opinion,
- automatic self-modification of remediation code.

### 3.2 Research-backed implementation stance

Use this order because it moves from safest and cheapest to riskiest and most
expensive:

1. **SFT baseline:** keep current LoRA training as the anchor.
2. **Verifier rejection sampling:** generate multiple candidate JSON responses
   and keep only validator-passing improvements.
3. **ORPO or DPO:** convert validator scores into chosen/rejected pairs and run
   a small preference LoRA pilot.
4. **GRPO:** only after pairwise preference tuning works, train with grouped
   sampled outputs and deterministic reward functions.
5. **Test-time orchestration:** for hard pages only, generate 2-4 candidates,
   score them with validators, and submit the best valid response.

The first pilot should be contrast or heading hierarchy:

- contrast has exact RGB/ratio labels and clear near-threshold rewards,
- heading hierarchy has exact `/H1` through `/H6` labels after structure-tree
  inspection.

Avoid open-ended reward models until deterministic rewards are exhausted. Remedy
already has better task validators than a generic learned evaluator for
contrast, heading, table, reading-order status, JSON validity, veraPDF results,
and content fidelity.

---

## 4. Serving backend matrix

| Engine | Strength | Risk | Initial role |
|---|---|---|---|
| vLLM | OpenAI-compatible server, native multi-LoRA docs, broad VLM support list, per-request model selection | Qwen3-VL LoRA bugs reported upstream; must use latest build and image-only limits | First test for premium Qwen3-VL LoRAs and lower-hardware candidates |
| SGLang | Explicit multi-LoRA batching for different adapters in one batch, OpenAI-compatible APIs, LoRA pinning and dynamic loading | Model naming differs; VLM+LoRA compatibility must be tested per candidate | Best fallback engine |
| TGI | Multi-LoRA startup loading and request adapter IDs; mature HF deployment story | Docs strongest for text generation; VLM+multi-LoRA path uncertain for Remedy | Secondary fallback |
| LMDeploy | Strong VLM support matrix, especially Qwen/InternVL families | Multi-LoRA serving path less directly aligned with Remedy needs | Track as engine fallback |
| PEFT fallback | Known to work with current adapters and aliases | Serializes generation behind a lock; limited throughput | Safety baseline only |

---

## 5. Candidate model matrix

### 5.1 Recommended primary candidates

#### Qwen/Qwen3.5-9B

- Model type: image-text-to-text, causal LM with vision encoder
- Size: 9B language model
- License: Apache-2.0
- Official serving snippets: Transformers, vLLM, SGLang
- Why test:
  - close to Qwen stack,
  - much smaller than Qwen3-VL-32B,
  - strong model-card document/OCR benchmark claims,
  - vLLM supported-models page lists Qwen3.5 multimodal architecture with LoRA.
- Risks:
  - new model family,
  - LoRA training path must be proven,
  - may share some Qwen multimodal serving complexity.
- Verdict:
  - primary Qwen-family lower-hardware candidate.

#### mistralai/Ministral-3-8B-Instruct-2512

- Model type: image-capable instruction model
- Size: 8.4B language model plus 0.4B vision encoder
- License: Apache-2.0
- Official serving snippets: vLLM, Transformers
- Why test:
  - small,
  - Apache,
  - vision-capable,
  - designed for edge deployment,
  - avoids Qwen3-VL-specific engine code paths.
- Risks:
  - FP8 instruct model may not be ideal for LoRA training,
  - model-specific tokenizer/dependency requirements,
  - multimodal LoRA support must be proven.
- Verdict:
  - primary non-Qwen serving candidate.

#### mistralai/Ministral-3-8B-Base-2512

- Model type: image-capable base model
- Size: 8.4B language model plus 0.4B vision encoder
- License: Apache-2.0
- Why test:
  - base/pretrained variant is better suited for custom post-training.
- Risks:
  - may need instruction tuning or multitask SFT before task LoRAs are useful.
- Verdict:
  - primary non-Qwen training base if Ministral serving works.

### 5.2 Conservative and alternate candidates

#### Qwen/Qwen2.5-VL-7B-Instruct

- Model type: image-text-to-text
- Size: 7B
- License: Apache-2.0
- Official serving snippets: Transformers, vLLM, SGLang
- Why test:
  - mature and widely used,
  - strong document/table/OCR results in model card,
  - likely easier than Qwen3-VL-32B.
- Risk:
  - below the requested 9B/12B band,
  - still Qwen VLM family.
- Verdict:
  - conservative fallback and sanity benchmark.

#### OpenGVLab/InternVL3_5-8B-HF

- Model type: image-text-to-text, HF Transformers format
- Size: 8.5B total
- License: OpenGVLab/MMPR-v1.2
- Official serving snippets: Transformers, vLLM, SGLang
- Why test:
  - strong VLM family,
  - efficient small vision encoder,
  - non-Qwen serving path but Qwen-derived language lineage.
- Risks:
  - license must be reviewed for Remedy use,
  - fine-tune path must be proven.
- Verdict:
  - strong alternate if license clears.

### 5.3 Not-first candidates

#### mistralai/Pixtral-12B-2409

- Size: 12B plus 0.4B vision encoder
- License: Apache-2.0
- Why not first:
  - Mistral's public blog marks Pixtral 12B as deprecated,
  - public vLLM Pixtral LoRA support request existed,
  - Ministral 3 is the newer Mistral-family path.

#### meta-llama/Llama-3.2-11B-Vision-Instruct

- Size: 11B
- License: Llama 3.2 Community License
- Why not first:
  - custom license,
  - public vLLM issue requested LoRA support for Llama 3.2 Vision,
  - not better aligned than Ministral/Qwen for Remedy.

#### google/paligemma2-10b-ft-docci-448

- Size: 10B
- License: Gemma
- Why not first:
  - model card says PaliGemma 2 is designed primarily for specialized
    fine-tuning and is not a multi-turn chatbot,
  - better suited for narrow image-to-JSON experiments than the full Remedy
    router.

#### Qwen/Qwen-VL-Chat

- Size: 7B
- License: Tongyi Qianwen custom license
- Local reference reviewed:
  `/Users/laccd/Desktop/qwenlm-qwen-vl-8a5edab282632443.txt`
- Why it matters:
  - official repo includes LoRA and Q-LoRA scripts,
  - training format uses image-tagged conversations with `<img>`, `<ref>`, and
    `<box>` tokens,
  - README frames useful document/OCR, table/chart, VQA, and grounding evals,
  - license text allows commercial use with conditions.
- Why not first:
  - older 2023-era VLM family,
  - Qwen-7B plus older visual stack rather than current Qwen3/Qwen3.5 VLMs,
  - 448px-focused image path is less attractive for rendered PDF pages,
  - custom license is less clean than Apache-2.0 candidates,
  - modern vLLM/SGLang multi-LoRA serving story is weaker than current model
    families.
- Verdict:
  - keep as historical reference for data format, LoRA scripts, and eval ideas;
    do not choose it as the primary lite remediation base.

---

## 6. Required adapter format

For any Remedy LoRA intended for multi-LoRA serving:

- Use PEFT LoRA unless the chosen engine requires a documented conversion.
- Keep rank <= 16 unless a specific task needs higher rank and serving memory
  is revalidated.
- Keep target modules consistent across adapters whenever possible.
- Prefer language-backbone modules only:
  - `q_proj`
  - `k_proj`
  - `v_proj`
  - `o_proj`
  - `gate_proj`
  - `up_proj`
  - `down_proj`
- Avoid vision-tower LoRA unless the serving engine explicitly supports it for
  the chosen model.
- Avoid `modules_to_save` unless the serving engine supports it.
- Publish LoRA adapters, not merged full models, unless a final deployment
  requires merged weights.

---

## 7. Experiment plan

### E1 - vLLM current-adapter single-LoRA smoke

Purpose:

Determine whether latest vLLM can serve one existing Qwen3-VL-32B Remedy LoRA.

Command shape:

```bash
vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt.image 1 \
  --limit-mm-per-prompt.video 0 \
  --enable-lora \
  --max-lora-rank 16 \
  --max-loras 2 \
  --lora-modules \
    qwen3vl-32b-remedy-contrast-v1=/workspace/remedy-adapters/contrast
```

Checks:

- server starts,
- `/v1/models` lists base and LoRA alias,
- text-only request succeeds,
- one-image contrast request succeeds,
- response is valid JSON for the contrast prompt.

Fail conditions:

- `lora_shrink_op` or profiling crash,
- engine tries to attach LoRA to visual encoder despite language-only adapter,
- OpenAI image request unsupported for the model.

### E2 - vLLM current-adapter multi-LoRA smoke

Purpose:

Determine whether latest vLLM can serve all existing Remedy LoRAs together.

Command shape:

```bash
vllm serve Qwen/Qwen3-VL-32B-Instruct \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt.image 1 \
  --limit-mm-per-prompt.video 0 \
  --enable-lora \
  --max-lora-rank 16 \
  --max-loras 6 \
  --max-cpu-loras 6 \
  --lora-modules \
    qwen3vl-32b-remedy=/workspace/remedy-adapters/alt-v2 \
    qwen3vl-32b-remedy-table-v1=/workspace/remedy-adapters/table \
    qwen3vl-32b-remedy-contrast-v1=/workspace/remedy-adapters/contrast \
    qwen3vl-32b-remedy-reading-order-v1=/workspace/remedy-adapters/reading-order \
    qwen3vl-32b-remedy-heading-v1=/workspace/remedy-adapters/heading
```

Checks:

- all aliases visible,
- one smoke per alias,
- 20 mixed concurrent requests,
- no adapter cross-talk,
- throughput better than PEFT fallback.

### E3 - SGLang current-adapter smoke

Purpose:

Test SGLang as the alternate multi-LoRA engine.

Command shape:

```bash
python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-VL-32B-Instruct \
  --host 0.0.0.0 \
  --port 30000 \
  --enable-lora \
  --lora-paths \
    contrast=/workspace/remedy-adapters/contrast \
    heading=/workspace/remedy-adapters/heading \
  --max-loras-per-batch 3 \
  --max-lora-rank 16 \
  --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
```

Checks:

- OpenAI-compatible image chat request works,
- adapter selection works using SGLang's model naming convention,
- two different adapters can be used in one concurrent batch.

Integration note:

If SGLang requires `base:adapter` model names, the Remedy task router should
map stable aliases to SGLang model strings internally through env config, not
hard-coded code.

### E4 - Lower-hardware zero-shot bakeoff

For each candidate:

- Start the model through vLLM if supported.
- Start the model through SGLang if vLLM fails or SGLang looks easier.
- Run 25 examples per task from held-out Remedy validation data.
- Use the production prompts and expected JSON schemas.

Candidate order:

1. `Qwen/Qwen3.5-9B`
2. `mistralai/Ministral-3-8B-Instruct-2512`
3. `Qwen/Qwen2.5-VL-7B-Instruct`
4. `OpenGVLab/InternVL3_5-8B-HF`

Reference-only:

- `Qwen/Qwen-VL-Chat`, because its official fine-tuning docs are useful but the
  model family is not the best target for a new Remedy lite profile.

Metrics:

- valid JSON rate,
- schema adherence,
- status accuracy where labels exist,
- median latency,
- GPU memory used,
- max stable concurrency,
- tokens/sec under mixed workload.

Pass to pilot training if:

- valid JSON >= 0.80 overall,
- no task is completely unusable,
- one-image OpenAI-compatible requests work,
- serving engine can start with LoRA enabled, even if no adapter is loaded yet.

### E5 - Pilot adapter training

Train a single pilot adapter before training the full family.

Recommended first task:

- contrast, because labels are exact and current task gate is clear.

Alternate first task:

- heading hierarchy, because exact H1-H6 correction is a strong structural
  signal.

Training settings:

```bash
python -u tools/finetune/train_lora_vision_hf.py \
  --model <candidate-base> \
  --train <task-train.jsonl> \
  --val <task-val.jsonl> \
  --out outputs/<candidate>-<task>-lora \
  --epochs 1 \
  --rank 16 \
  --alpha 32
```

Adapt this only if the chosen model requires a model-specific trainer.

Pass conditions:

- adapter serves through chosen engine,
- tuned beats base,
- valid JSON >= 0.90,
- pass false-positive <= 0.10,
- task-specific gate passes.

### E6 - Full lower-hardware adapter family

Train the full family only after E5 passes:

- alt text quality
- table structure
- contrast
- reading order
- heading hierarchy

Naming convention:

- `<base-short>-remedy-alt-v1`
- `<base-short>-remedy-table-v1`
- `<base-short>-remedy-contrast-v1`
- `<base-short>-remedy-reading-order-v1`
- `<base-short>-remedy-heading-v1`

Do not reuse `qwen3vl-32b-remedy` aliases for a lite base. Lower-hardware
aliases must make the profile obvious, for example:

- `qwen35-9b-remedy-lite`
- `qwen35-9b-remedy-lite-contrast-v1`
- `ministral3-8b-remedy-lite`
- `ministral3-8b-remedy-lite-heading-v1`

### E7 - Verifier-driven post-training pilot

Purpose:

Adapt the useful part of the Muse Spark training philosophy to Remedy: improve
models with objective feedback after SFT.

Recommended first task:

- contrast, because RGB ratios produce exact labels and rewards.

Secondary tasks:

- heading hierarchy, because exact `/H1` through `/H6` structure tags can be
  checked against inspected structure trees,
- table status, because target status and confusion-matrix labels are already
  available.

Pipeline:

1. Generate `n=2..4` candidate JSON responses per training example from the SFT
   adapter using varied temperatures, while preserving the same prompt and page
   image.
2. Score candidates with deterministic validators:
   - JSON validity and schema compliance,
   - task status correctness,
   - contrast ratio and RGB field correctness,
   - structure-tree tag exactness for headings,
   - table status/confusion class correctness,
   - pass false-positive penalty,
   - output compactness penalty,
   - content-fidelity penalty when the answer invents text not present on the
     page.
3. Build `chosen` / `rejected` pairs only when the score gap is large enough to
   be unambiguous. Drop ties and ambiguous examples.
4. Run a short ORPO or DPO LoRA pilot first, because it is cheaper and less
   operationally brittle than online RL.
5. Re-run base, SFT, and preference-tuned adapter evals side by side.
6. If preference tuning improves held-out metrics, run a short GRPO pilot on
   the same task with grouped samples and the validator score as the reward.
7. Keep test-time orchestration as a separate inference feature: generate
   multiple candidates only for pages marked hard or high-risk, then choose the
   best validator-passing output.

Reward weights for the first contrast pilot:

- `+0.30` valid JSON and exact schema,
- `+0.35` correct pass/fail status,
- `+0.20` correct near-threshold decision for 4.5:1 cases,
- `+0.10` correct RGB/ratio fields when an issue is reported,
- `+0.05` compact output with no extra prose,
- `-0.50` false positive on pass examples,
- `-1.00` invalid JSON.

Reward weights for the first heading pilot:

- `+0.25` valid JSON and exact schema,
- `+0.35` correct issue/pass status,
- `+0.25` exact `element_index` plus corrected `/H1` through `/H6` tag,
- `+0.10` per-level tag accuracy,
- `+0.05` compact output with no extra prose,
- `-0.50` false positive on pass examples,
- `-1.00` invalid JSON.

Pass conditions:

- post-trained adapter beats SFT on the targeted metric,
- valid JSON does not regress,
- pass false-positive rate does not regress,
- performance on hand-checked real LAMC pages improves or stays flat,
- alt/table/heading/reading-order tasks do not regress if the adapter is
  multitask.

---

## 8. Evaluation gates

### 8.1 Serving metrics

Collect for PEFT fallback, vLLM, SGLang, and any selected fallback engine:

- cold start time,
- VRAM after model load,
- VRAM after all adapters load,
- successful aliases,
- max stable request concurrency,
- median request latency,
- p95 request latency,
- tokens/sec,
- failures per 100 requests,
- adapter cross-talk incidents.

### 8.2 Task metrics

Use existing Remedy task metrics where available:

- table: status accuracy, confusion matrix, pass false-positive rate.
- heading: status accuracy, exact correction accuracy, per-level accuracy,
  pass false-positive rate.
- reading order: status accuracy, valid JSON, pass false-positive rate,
  corrected-order accuracy only if prompt asks for explicit corrected order IDs.
- contrast: status accuracy, near-threshold status accuracy, pass
  false-positive rate, valid JSON.
- alt: production eval win rate/status accuracy, valid JSON, false flags on
  gold/pass examples.

### 8.3 Runtime profile gates

A new serving stack is eligible for a named Remedy runtime profile only if:

- it passes all serving metrics,
- it passes the task metrics for its intended profile,
- any quality gap against the Qwen3-VL-32B premium path is documented,
- it completes one live remediation smoke,
- its env profile is documented in the runbook,
- rollback to PEFT fallback remains one env change away.

Runtime profiles should be explicit:

- **Premium:** Qwen3-VL-32B Remedy LoRAs, best quality, higher hardware cost.
- **Lite:** 8B-14B VLM LoRAs, lower-end hardware and stronger concurrency per
  dollar, quality verified against a documented gate.
- **Fallback:** PEFT server for correctness when optimized engines fail.
- **Verifier-enhanced:** SFT adapter plus local verifier-driven preference or
  RL post-training, only promoted after held-out task gates and production
  smoke pass.

---

## 9. Implementation tasks

### Serving tasks

- Add a `tools/serve_remedy_router_sglang.py` helper if SGLang smoke passes.
- Update `tools/serve_remedy_router_vllm.py` with latest known-good flags.
- Add a mixed-concurrency smoke script that sends task-specific requests and
  checks adapter identity.
- Add a small `/v1/models` alias validation helper.

### Eval tasks

- Add a compact bakeoff runner for candidate models.
- Save zero-shot outputs under `eval_runs/vlm_bakeoff_<date>/`.
- Add a summary script that compares valid JSON, task accuracy, latency, and
  VRAM across candidates.
- Add a validator-score export that records each candidate response, each
  reward component, the aggregate score, and chosen/rejected pair provenance.

### Training tasks

- Confirm current HF trainer supports the chosen candidate model.
- If not, add a model-specific training adapter path rather than changing the
  existing Qwen trainer blindly.
- Train one pilot task adapter before training the full family.
- Add a verifier-driven preference-data builder before attempting ORPO/DPO.
- Add GRPO only after the ORPO/DPO pilot proves useful on held-out data.

### Docs tasks

- Update `runbooks/lamission-router-remediation.md` after a serving stack is
  proven.
- Add final engine profile:
  - container/env,
  - model ID,
  - adapter IDs,
  - launch command,
  - env vars,
  - rollback command.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Latest vLLM still fails Qwen3-VL LoRA | Try SGLang; keep PEFT fallback for premium path and proceed separately to lower-hardware bakeoff |
| SGLang model naming does not match Remedy aliases | Add env-driven alias mapping |
| Small model loses document-layout accuracy | Require per-task gates and compare to current adapters |
| New model fine-tune path is unsupported | Pilot one task before full family |
| Multi-LoRA works but throughput gain is small | Measure PEFT baseline and mixed workload p95 before switching |
| Separate endpoints are required | Use `OLLAMA_VISION_TASK_BASE_URLS`, but expect higher VRAM/cost |
| Adapter cross-talk | Add deterministic mixed-adapter smoke with task-specific prompts |
| Meta-style RL overfits eval format | Use held-out pages, production smoke, and pass false-positive gates |
| Model-generated labels contain subtle PDF errors | Require deterministic validation and human spot checks before training |
| Reward hacking | Keep rewards simple, inspect samples, add negative tests, and reject improvements that only exploit the scorer |

---

## 11. Recommended next command for a future implementation thread

```text
/goal Read /Users/laccd/code/lamc_district_forms/remedy-server/docs/prds/PRD_vlm_multi_lora_serving.md and /Users/laccd/code/lamc_district_forms/remedy-server/docs/prds/SPEC_vlm_multi_lora_serving.md before acting. Objective: build two complementary Remedy serving profiles, not a forced replacement. First, try to accelerate the premium Qwen3-VL-32B Remedy LoRA router with latest vLLM multi-LoRA, then SGLang if vLLM fails. Do not retrain existing Qwen3-VL adapters unless serving evidence proves they are incompatible. Separately, run the lower-hardware bakeoff for Qwen3.5-9B, Ministral 3 8B, Qwen2.5-VL-7B, and InternVL3.5-8B, then recommend one base for a pilot lite LoRA adapter that can remediate on cheaper hardware with better concurrency per dollar. Also design and implement a local verifier-driven post-training pilot inspired by Muse Spark's pretraining/RL/test-time-reasoning axes without attempting foundation pretraining: start with contrast or heading, generate multiple candidate JSON outputs from the SFT adapter, score them with deterministic PDF/JSON/task validators, build chosen/rejected pairs, run ORPO or DPO first, and attempt GRPO only if the preference pilot improves held-out real-page metrics without pass false-positive regression.
```

---

## 12. Primary source links

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
- TGI LoRA docs: https://huggingface.co/docs/text-generation-inference/en/conceptual/lora
- LMDeploy supported models: https://lmdeploy.readthedocs.io/en/latest/supported_models/supported_models.html
- Qwen3.5-9B: https://huggingface.co/Qwen/Qwen3.5-9B
- Qwen2.5-VL-7B-Instruct: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- Qwen-VL repository export reviewed locally:
  `/Users/laccd/Desktop/qwenlm-qwen-vl-8a5edab282632443.txt`
- Ministral 3 8B Instruct: https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512
- Ministral 3 8B Base: https://huggingface.co/mistralai/Ministral-3-8B-Base-2512
- InternVL3.5-8B-HF: https://huggingface.co/OpenGVLab/InternVL3_5-8B-HF
- Pixtral 12B: https://huggingface.co/mistralai/Pixtral-12B-2409
- PaliGemma2 10B: https://huggingface.co/google/paligemma2-10b-ft-docci-448
- Llama 3.2 11B Vision: https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct
