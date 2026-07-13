# LAMission Corpus Remediation With the Qwen3-VL LoRA Router

This runbook is for re-remediating the downloaded `lamission.edu` PDF corpus with the default sampled vision path and the task-routed Qwen3-VL LoRA adapters.

## Current Decision

Use **default sampled router remediation** first.

Do not start with full all-page vision over the whole corpus. Large PDFs are sampled by `VISION_PAGE_SAMPLE_SIZE` in `src/project_remedy/pdf_vision.py`, which keeps cost and runtime sane while still letting the custom adapters resolve reading order, contrast, heading, table, and alt-text quality checks where the deterministic pipeline needs vision.

Use a second targeted pass for failures, priority PDFs, or documents whose sampling misses visually complex pages.

## Optional Pre-RunPod Deterministic Prep

While waiting for RunPod credits or pod setup, it is useful to process a conservative no-vision subset locally. This does not replace the router pass; it separates easy machine-verifiable L3 outputs from files that need vision, human review, or heavier remediation.

The corpus runner supports:

```bash
uv run python tools/remediate_pdf_corpus.py \
  --input-root /path/to/easy/pdf/subset \
  --output-root /path/to/no_vision_outputs \
  --manifest /path/to/no_vision_outputs/manifest.jsonl \
  --no-vision \
  --resume \
  --max-pages 5 \
  --max-mb 5 \
  --per-file-timeout 300
```

Use `--no-vision` for both fixing and acceptance so the local prep run does not spend Ollama Cloud calls or depend on the RunPod router.

For the current local workspace, a strict safe-easy subset was generated at:

```text
/Users/laccd/code/lamc_district_forms/lamc_no_vision_prep_20260706/safe_easy_inputs
```

The strict filter used pages <= 5, size <= 2 MB, objects <= 150, XObjects <= 4, and image XObjects <= 2. The initial inventory found 567 safe-easy PDFs covering 887 pages. These outputs should be treated as L3 machine-verified candidates, not final L5/human-certified accessibility deliverables.

## RunPod Requirement

Spin up a RunPod GPU pod before the router remediation run.

This run does not retrain models. The pod is needed to serve `Qwen/Qwen3-VL-32B-Instruct` plus the LoRA adapters behind an OpenAI-compatible endpoint. The PDF remediation process can run on the Mac or on the pod, but the custom LoRAs will not be used unless `VISION_BASE_URL` points at a running GPU model server.

Recommended pod shape:

- GPU: 1x H200 141GB VRAM. A100 80GB or A6000-class alternatives may work only if vLLM/PEFT can load the base model plus adapters reliably.
- Persistent storage: attach a network volume, ideally 250GB or larger. Do not rely on the small container disk for the base model cache, adapters, corpus outputs, and logs.
- Image/environment: CUDA-compatible PyTorch environment with Python 3.11+, `uv`, `git`, `huggingface_hub`, `transformers`, `peft`, `accelerate`, `pillow`, and either `vllm` for preferred serving or the PEFT fallback dependencies.
- Ports: expose or proxy port `8000` for the OpenAI-compatible server. Keep the endpoint private or reachable only through a trusted tunnel when possible.
- Workspace convention:
  - repo: `/workspace/remedy-server`
  - adapters: `/workspace/remedy-adapters`
  - optional input corpus copy: `/workspace/lamission_pdfs`
  - optional output directory: `/workspace/remediated_lamission`
  - model cache: put Hugging Face caches on the network volume if possible, for example `HF_HOME=/workspace/.cache/huggingface`

Minimum pod sanity checks:

```bash
nvidia-smi
df -h /workspace
python --version
git --version
```

From the Mac, verify the pod exists and stop it when finished:

```bash
runpodctl pod list
# after artifacts are copied or verified:
runpodctl pod stop <pod-id>
```

If no H200 pod is running, create one in the RunPod dashboard or with the available RunPod CLI/template flow, attach the persistent volume, and then verify it with `runpodctl pod list`.

## Model Topology

Serve the adapters behind one OpenAI-compatible endpoint:

- primary/stable model: `qwen3vl-32b-remedy`
- contrast route: `qwen3vl-32b-remedy-contrast-v1`
- reading-order route: `qwen3vl-32b-remedy-reading-order-v1`
- heading route: `qwen3vl-32b-remedy-heading-v1`
- table route: `qwen3vl-32b-remedy-table-v1`

The app selects routes through environment variables, not hard-coded names.

The adapter directories must contain `adapter_config.json`. If the adapters are pulled from private Hugging Face repos, download them into stable local directories and pass explicit paths to the serving helper when the names differ from the defaults.

Example adapter layout:

```text
/workspace/remedy-adapters/
  alt-v2/
  table-v1/
  contrast-v1/
  reading-order-v1/
  heading-v1/
```

Preferred serving path on RunPod is vLLM multi-LoRA:

```bash
cd /workspace/remedy-server
uv run python tools/serve_remedy_router_vllm.py \
  --adapter-root /workspace/remedy-adapters \
  --alt-adapter /workspace/remedy-adapters/alt-v2 \
  --table-adapter /workspace/remedy-adapters/table-v1 \
  --contrast-adapter /workspace/remedy-adapters/contrast-v1 \
  --reading-order-adapter /workspace/remedy-adapters/reading-order-v1 \
  --heading-adapter /workspace/remedy-adapters/heading-v1 \
  --host 0.0.0.0 \
  --port 8000 \
  --print-env
```

If vLLM multi-LoRA serving fails for Qwen3-VL on the pod, use the PEFT fallback server for a smoke or separate endpoints per adapter:

```bash
cd /workspace/remedy-server
uv run python tools/serve_remedy_router_peft.py \
  --adapter-root /workspace/remedy-adapters \
  --alt-adapter /workspace/remedy-adapters/alt-v2 \
  --table-adapter /workspace/remedy-adapters/table-v1 \
  --contrast-adapter /workspace/remedy-adapters/contrast-v1 \
  --reading-order-adapter /workspace/remedy-adapters/reading-order-v1 \
  --heading-adapter /workspace/remedy-adapters/heading-v1 \
  --host 0.0.0.0 \
  --port 8000 \
  --print-env
```

## Runtime Environment

Set these where the remediation process runs:

```bash
export OLLAMA_API_KEY=dummy
export VISION_BASE_URL=http://<runpod-host>:8000/v1
export OLLAMA_VISION_MODEL=qwen3vl-32b-remedy
export OLLAMA_VISION_TASK_MODELS=contrast:qwen3vl-32b-remedy-contrast-v1,reading_order:qwen3vl-32b-remedy-reading-order-v1,heading_hierarchy:qwen3vl-32b-remedy-heading-v2,table_structure:qwen3vl-32b-remedy-table-v1
export OLLAMA_VISION_TASK_BASE_URLS=
export OLLAMA_VISION_ROUTER_ALLOW_FALLBACK=0
export OLLAMA_VISION_MAX_INFLIGHT=8
export OLLAMA_ESCALATION_MAX_INFLIGHT=8
export OLLAMA_VISION_GATE_TIMEOUT_SECONDS=600
export OLLAMA_VISION_MAX_TOKENS=768
# heading-v2 (2026-07-11): trained on LAMC delivered-arbitration data; false-flag
# 0.634->0.0 on LAMC val. t=0 makes the heading verify deterministic (fixed
# files stay fixed); the adapter alias is served from
# johnnyrobotai/remedy-server-qwen3vl-32b-heading-v2-lora (HF, private).
export OLLAMA_VISION_TEMPERATURE=0
```

Leave `VISION_PAGE_SAMPLE_SIZE` unset for the first full pass. That uses the code default. For a targeted all-page pass on selected PDFs only, set:

```bash
export VISION_PAGE_SAMPLE_SIZE=0
```

## Smoke Test

First prove the model endpoint is alive:

```bash
curl -s http://<runpod-host>:8000/v1/models | jq .
```

The response must list at least:

- `qwen3vl-32b-remedy`
- `qwen3vl-32b-remedy-contrast-v1`
- `qwen3vl-32b-remedy-reading-order-v1`
- `qwen3vl-32b-remedy-heading-v1`
- `qwen3vl-32b-remedy-table-v1`

Then run a small corpus smoke:

```bash
cd /Users/laccd/code/lamc_district_forms/remedy-server

uv run python tools/remediate_pdf_corpus.py \
  --input-root /path/to/lamission.edu/pdfs \
  --output-root /path/to/remediated_lamission_smoke \
  --manifest /path/to/remediated_lamission_smoke/manifest.jsonl \
  --limit 20 \
  --force \
  --per-file-timeout 1200
```

Inspect:

- `manifest.jsonl`
- `manifest_levels_summary.json`
- representative output PDFs
- veraPDF and acceptance failures in the manifest rows
- whether failures are real content issues, font-clause residue, oversized files, or model/endpoint errors

## Full Corpus Run

Use document-level sharding. Do not split PDFs into pages and merge them back together.

Two execution layouts are valid:

- **Mac runner, RunPod model server:** simplest when the corpus is already local. The Mac rewrites/verifies PDFs and sends sampled page images to the RunPod endpoint.
- **RunPod runner and model server:** usually better for a long overnight run if the corpus can be synced to the network volume. It avoids sending thousands of rendered page images back and forth over the public network, but requires copying input PDFs and final outputs.

Example with four local workers against one RunPod H200 model server:

```bash
BASE_OUT=/path/to/remediated_lamission
INPUT_ROOT=/path/to/lamission.edu/pdfs

mkdir -p "$BASE_OUT"

for SHARD in 0 1 2 3; do
  uv run python tools/remediate_pdf_corpus.py \
    --input-root "$INPUT_ROOT" \
    --output-root "$BASE_OUT" \
    --manifest "$BASE_OUT/manifest-shard-${SHARD}.jsonl" \
    --shard-index "$SHARD" \
    --shard-count 4 \
    --resume \
    --per-file-timeout 1200 \
    > "$BASE_OUT/shard-${SHARD}.log" 2>&1 &
done

wait
```

Start with `--shard-count 2` if the local machine is CPU-bound, disk-bound, or if the H200 endpoint shows request queueing. Increase only after observing stable throughput.

Current caveat: the PEFT fallback router (`tools/serve_remedy_router_peft.py`) serializes `generate()` behind a process lock so that adapter switching is safe. Multiple corpus shards can still overlap local PDF parsing, rendering, OCR, veraPDF, and network wait time, but model generations will queue one at a time. Do not assume PEFT fallback sharding equals true GPU-concurrent serving. For future high-throughput runs, either get vLLM multi-LoRA serving working for Qwen3-VL (`--enable-lora` / `--lora-modules`) or run separate OpenAI-compatible endpoints per adapter and route tasks across them.

## What Parallelizes

Safe:

- multiple PDFs in parallel via `--shard-count`
- multiple API jobs with `WORKER_CONCURRENCY`, if using the HTTP API
- concurrent vision calls up to `OLLAMA_VISION_MAX_INFLIGHT` when the serving backend supports true concurrency

Avoid:

- splitting one PDF into pages and reassembling it
- enabling full all-page vision for the entire 10,000+ page corpus as the first run
- using `tools/batch_remediate.py` for this objective, because that script calls `remedy-pdf fix ... --no-vision`
- increasing shard count aggressively when using the PEFT fallback router; it queues GPU generations and can create request timeouts without improving throughput much

## Expected Runtime

Use these only as planning ranges until the 20-file smoke has real timings:

- default sampled router pass: roughly 6-18 hours for a 10,000+ page corpus on one H200, depending on file count, average PDF complexity, OCR, veraPDF repair loops, and shard count
- full all-page vision pass: roughly 24-72 hours or more

The main cost risk is idle RunPod time. Stop the pod immediately after logs, manifests, and output artifacts are copied or verified.

## Cleanup

After the run:

```bash
runpodctl pod list
runpodctl pod stop <pod-id>
```

Keep any RunPod network volume only until:

- LoRA adapters are verified local or on Hugging Face
- corpus manifests/logs are copied locally
- output PDFs are copied or intentionally left in durable storage

## Pasteable Goal Prompt for Another Thread

```text
/goal Read /Users/laccd/code/lamc_district_forms/remedy-server/runbooks/lamission-router-remediation.md before acting. Use the current main branch of /Users/laccd/code/lamc_district_forms/remedy-server. Objective: run default sampled task-routed Qwen3-VL LoRA remediation on the downloaded lamission.edu PDF corpus, beginning with a 20-file smoke and then a sharded full corpus pass only if the smoke is healthy.

Constraints:
- Spin up or connect to a RunPod GPU pod first. Use 1x H200 141GB when available, attach a persistent network volume, and stop the pod as soon as model serving/remediation work is complete.
- Use RunPod only to serve the Qwen3-VL-32B LoRA router; do not retrain models.
- Verify the RunPod environment with nvidia-smi, disk space under /workspace, Python, git, and a Python env that can import vLLM or the PEFT fallback dependencies.
- Serve aliases: qwen3vl-32b-remedy, qwen3vl-32b-remedy-contrast-v1, qwen3vl-32b-remedy-reading-order-v1, qwen3vl-32b-remedy-heading-v1, qwen3vl-32b-remedy-table-v1.
- Configure routing via env vars only: VISION_BASE_URL, OLLAMA_VISION_MODEL, OLLAMA_VISION_TASK_MODELS, OLLAMA_VISION_TASK_BASE_URLS, OLLAMA_VISION_ROUTER_ALLOW_FALLBACK=0.
- Leave VISION_PAGE_SAMPLE_SIZE unset for the first full pass.
- Do not use tools/batch_remediate.py because it forces --no-vision.
- Do not split PDFs into individual pages and merge them.
- Use tools/remediate_pdf_corpus.py with --resume and document-level sharding.
- Start with --limit 20 and inspect manifest/logs before full run.
- Preserve outputs, manifests, and shard logs under a timestamped remediated_lamission directory.
- Stop the RunPod pod after serving/remediation is complete or if work cannot start promptly.

First actions:
1. Verify git status and current branch.
2. Locate the actual lamission.edu downloaded PDF corpus folder on this Mac; prefer the known lamc_district_forms data/visual_match download area if present, but verify on disk.
3. Verify local dependencies: uv, remedy-pdf, veraPDF, Ghostscript, ocrmypdf.
4. Verify RunPod CLI auth and current pods.
5. If no suitable pod is running, spin up a RunPod H200 pod with a persistent volume.
6. Start or connect to the H200 pod, sync the repo and adapters if needed, serve the router, and confirm /v1/models lists every alias.
7. Run the 20-file smoke with tools/remediate_pdf_corpus.py.
8. Summarize smoke throughput, failures, estimated full-runtime, and whether to proceed.
9. If healthy, run a sharded full default sampled pass and keep clear logs.
```
