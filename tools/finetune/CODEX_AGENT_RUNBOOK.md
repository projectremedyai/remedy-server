# Codex agent runbook — fine-tune the LAMC remediation vision model (RTX 4080)

You are the agent operating the Proxmox GPU box. **Mission: run a QLoRA fine-tune
DRY-RUN on the RTX 4080 to prove the train → eval → serve pipeline end-to-end.**
This is a *pipeline validation*, not a production model — the v1 dataset is small
(206 alt-text examples). Success = the loop runs, loss drops, the adapter saves,
and eval works. Do NOT over-train or ship this model. Report honestly.

## Concrete artifacts (already prepared)
- **Code:** GitHub `projectremedyai/remedy-server`, branch **`feat/finetune-scaffolding`**.
  Scripts live in `tools/finetune/` (this file is there too).
- **Data:** HF **private** dataset `johnnyrobotai/lamc-remediation-vlora`
  (`train.jsonl` 178, `val.jsonl` 28, `renders/` 206 PNGs, image paths are
  relative). Task = alt-text quality; prompts are the production prompts.
- **Box:** LXC CTID 100 `vllm-rtx4080` @ 192.168.68.64, GPU passthrough working.
  Full env setup details: `docs/FINETUNE_HANDOFF_PROXMOX_LXC.md` (in the repo history;
  if not present in the clone it's gitignored — the setup steps are inlined below).

## Preconditions (verify first, stop if any fail)
1. `nvidia-smi` inside the container shows the RTX 4080 + CUDA 12.x.
2. **Ollama is STOPPED** (`systemctl stop ollama`) — 16 GB holds ONE job; a served
   model + a training run will OOM.
3. You can auth to HF with read access to the private dataset (`hf auth login`,
   or `HF_TOKEN` env). If you can't reach the private repo, STOP and report — do
   not fabricate data.

## Steps

### 1. Training env (once)
```bash
apt-get update && apt-get install -y python3-venv git build-essential
mkdir -p /opt/finetune && cd /opt/finetune
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip wheel
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install "unsloth[cu124-torch260]" qwen-vl-utils pillow "huggingface_hub[cli]"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from unsloth import FastVisionModel; print('unsloth vision OK')"
```
If an Unsloth import/signature fails, the API changed — reconcile against the
current Unsloth **vision** fine-tuning notebook (unsloth.ai/docs, Qwen2.5-VL) and
pin the working versions to `requirements.lock`. Do not guess silently; report the
exact error if you can't resolve it.

### 2. Get the code + data
```bash
cd /opt && git clone -b feat/finetune-scaffolding https://github.com/projectremedyai/remedy-server.git
cd remedy-server
hf auth login            # or export HF_TOKEN=...
hf download johnnyrobotai/lamc-remediation-vlora --repo-type dataset --local-dir tools/finetune/data
# -> tools/finetune/data/{train.jsonl,val.jsonl,renders/}. Image paths are
#    relative ("renders/..."), so RUN TRAINING FROM tools/finetune/data OR pass
#    absolute --train/--val and cd there; simplest: cd tools/finetune/data first,
#    or symlink renders next to where you run. Verify one image opens:
python - <<'PY'
import json, os
r=json.loads(open('tools/finetune/data/train.jsonl').readline())
img=r['messages'][0]['content'][0]['image']
print('image ref:', img, '| exists:', os.path.exists('tools/finetune/data/'+img))
PY
```

### 3. Smoke test — prove the loop runs (30 steps)
```bash
cd /opt/remedy-server/tools/finetune/data      # so relative renders/ paths resolve
python /opt/remedy-server/tools/finetune/train_qlora_vision.py \
  --model unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit \
  --train train.jsonl --val val.jsonl \
  --out /opt/finetune/out/smoke --max-steps 30 --rank 8 --batch 1 --grad-accum 4
```
PASS if: loss prints and trends DOWN, VRAM stays < ~15 GB, an adapter is written
to `/opt/finetune/out/smoke`. If OOM: drop `--render-dpi` is not applicable here
(data is pre-rendered) → lower `--max-seq-len 1536`, keep `--batch 1`. If it
passes, continue.

### 4. Full dry-run (1 epoch)
```bash
python /opt/remedy-server/tools/finetune/train_qlora_vision.py \
  --model unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit \
  --train train.jsonl --val val.jsonl \
  --out /opt/finetune/out/lamc-alt-v1 --epochs 1 --rank 8 --batch 1 --grad-accum 4
```
(206 examples is tiny → 1 epoch is enough; more will overfit. Don't crank epochs.)

### 5. Eval — base vs adapter
```bash
python /opt/remedy-server/tools/finetune/eval_adapter.py \
  --model unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit \
  --adapter /opt/finetune/out/lamc-alt-v1 --val val.jsonl
```
Report the valid-JSON rate for BASE and ADAPTER (primary gate). Adapter should be
>= base on valid-JSON; on 206 examples don't expect a big exact-match jump.

### 6. Report back (to whoever dispatched you)
- Preconditions result; the exact Unsloth/torch versions that worked.
- Smoke: did loss drop? starting/ending loss; peak VRAM.
- Full run: final train/val loss; adapter path + size.
- Eval: base vs adapter valid-JSON rate.
- Any errors + how you resolved them (or where you got stuck).
- Verdict: is the pipeline PROVEN end-to-end (train → adapter → eval)? yes/no.

### 7. (Optional, only if asked) Serve the adapter back
Merge adapter → base, convert to GGUF, `ollama create lamc-qwen25vl -f Modelfile`,
restart ollama, and the Mac points `.env`: `VISION_BASE_URL=http://192.168.68.64:11434`,
`OLLAMA_VISION_MODEL=lamc-qwen25vl`. (Stop training first; one model in 16 GB.)

## Stop / escalate — do NOT push through these silently
- Can't reach the private HF dataset → STOP, report (don't fabricate data).
- Repeated OOM after lowering seq-len → report the config + VRAM, ask for guidance.
- Unsloth API mismatch you can't reconcile in ~2 tries → report the exact traceback.
- Loss is NaN / not decreasing → report; likely a data/format issue, not "train harder."

## Scope guardrails
- This is a DRY-RUN. The goal is a WORKING pipeline, not a good model. The v1 data
  is alt-text-only and small — a modest/overfit result is expected and fine.
- Never claim success you didn't observe; paste real loss/eval numbers.
- After the pipeline is proven, the next work (NOT yours unless asked) is expanding
  the dataset: heading/reading-order extractors + adapting the permissive public
  sets (see `tools/finetune/download_public_datasets.py`), then re-train, then the
  32B run on a rented H100 (same scripts, `--model` swap).
