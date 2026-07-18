# HANDOFF — alt_text + heading data-fix retrain (v3) — 2026-07-17

**STATUS: v3 experiment COMPLETE; BOTH ADAPTERS REJECTED; NO PROMOTION.** The two
one-epoch adapters trained and exported correctly, but both failed the frozen promotion
gates and regress their incumbents. The A100 was provider-stopped at
`2026-07-18T08:16:48Z`; deletion converged and the instance no longer appears in
`brev ls`. Conservative campaign spend is **$58.4188 of the approved $60 ceiling**.

Repo: `remedy-server-nemo-rl-brev`, branch
`codex/autoresearch/remedy-vlm-20260714/datafix-v3-sft`. Continuation of the
eval-gate work (commit 7b84f08). User chose **data-first cheap-SFT over GRPO**.

## Final outcome (authoritative)

- **Alt adapter:** 29/29 steps, end-validation loss `0.1680`, genuine language-only
  PEFT adapter (`119,809,056` bytes; 504 language tensors, 0 visual tensors; SHA-256
  `60d99cc6e222faca3372cbcbe32e1bace65d50120e3277568da3a96dd1ee7d36`). Frozen
  gate: status `0.7460`, valid JSON `0.8571`, **7** real-pass false positives,
  structured exact `0.5313`. **REJECT.** Nine outputs ran into the 384-token ceiling
  mid-JSON, and the adapter overcalled delivered pass pages as failures.
- **Heading adapter:** 144/144 steps, end-validation loss `0.0193`, genuine
  language-only PEFT adapter (`119,809,056` bytes; 504 language tensors, 0 visual
  tensors; SHA-256
  `ed0be594e8459dbff787702a8ff54194769d7da734312b9f845bf8671129bbaa`). Frozen gate:
  status `0.8207`, valid JSON `0.9448`, **8** real-pass false positives, exact
  correction `0.3333`. **REJECT.** This is below both the gate and the incumbent.
- **Artifacts:** both adapters, training logs/configs, all 208 predictions, scorer
  reports, and SHA manifests are local under `remote_artifacts/qwen25_v3_*`.
- **Data integrity:** exact Qwen processor filtering is now applied to task-specific
  and aggregate SFT files. Final task counts are alt train/val `234/54`, heading
  `1153/184`, reading `278/32`, table `224/28`; frozen tests are unchanged. The
  authoritative `datafix_v3_preflight.json` is green with zero leakage, missing media,
  schema failures, holdout leaks, balance errors, or hash mismatches.
- **Decision:** keep the prior alt and heading adapters/routes. Table structure remains
  the only promoted adapter. Do not use either v3 adapter in PDF remediation.
- **Final verification:** 381 tests passed, 1 skipped; JSON/TSV, Python compile, shell
  syntax, dataset preflight, and local adapter/evaluation SHA checks all pass.

---

## Superseded prelaunch preconditions (historical; do not execute)
1. **`brev ls` shows 0 running instances.** On 2026-07-17 an UNTRACKED idle A100
   (`brevp1sftr2`/`gjn2yleez`, default jupyter box, GPU 0%, ~$2.50+ burned) was found
   and stopped — an orphan from the concurrent session. `brev ls` is the truth, NOT
   `brev_state.json`.
2. **The concurrent session is done with Brev** (that's the whole reason this is paused —
   avoid a two-GPU collision + double budget tracking).
3. **Provider spend is reconciled (2026-07-17).** NVIDIA Billing shows $74.24 org total.
   Excluding the two parallel-workstream instances (`brevp1sftr2` $16.79 and
   `brevp1sft20260716r1` $6.63) leaves **$50.82** for `remedy-*` campaign instances.
   This was above the old $50 hard stop. **RESOLVED:** user raised the campaign hard
   ceiling to $60 and approved the exact v3 plan on 2026-07-17. Persist $60 in the
   controller state so the detached watchdog does not use its historical $50 default.
4. **Full v3 preflight is green (2026-07-17, $0).** The three omitted catalog holdout
   entries were restored only after independently rechecking each path, SHA-256, and
   size against the prior accepted manifest. Evidence:
   `session/20260714_232247/datafix_v3_preflight.json` (`passed=true`). Also verified:
   1,456 media files, exact SFT/Gym ID alignment for both tasks, zero subjective alt
   labels, and full unit suite 368 passed / 1 skipped.

## Superseded prelaunch steps (completed; do not execute)
1. Package the v3 payload:
   `REMEDY_DATASET_ROOT=tools/finetune/generated/nemo_campaign_dataset_v3 \`
   `  bash tools/finetune/prepare_brev_payload.sh` (v3 has sft/ + media/ + manifest.json —
   verified present, 1456 media pngs).
2. Launch ONE A100 (`a100-80gb.1x`) via the guarded controller; setup = `brev_setup.sh`
   (applies the strip-None patch + import shims). Watchdog ≤3h.
3. Retrain ONLY the two fixed tasks on the warm box, one complete epoch each with start
   validation disabled and end validation/checkpoint retained so evaluation fits the
   2.85-hour watchdog:
   `campaign sft --task alt_text_quality` then `campaign sft --task heading_hierarchy`
   (`--dataset-root /ephemeral/nemo-rl/datasets`). The other 3 adapters are unchanged —
   **table_structure is already PROMOTED**; contrast/reading_order are the separate MOVE3
   input-redesign track, NOT retrained here.
4. Re-run eval gates: 208 adapter generations (63 alt + 145 heading) scored by
   `tools/finetune/remedy_nemo_rl/evaluation.py` on the v3 **test** splits.
   Preflight: adapter is genuine PEFT + example_id alignment + max_pixels=12845056.
   Base-v2 reports remain the frozen reference; the promotion checks themselves only
   require complete adapter predictions.
5. Retrieve adapters + eval_reports BEFORE stop (`retrieve_brev_artifacts.sh`,
   SHA-verify), then STOP the box. Record spend in the ledger + timeline.

## ▶ PROMOTION GATES (evaluation.py — must pass ALL)
valid_json==1.0 AND real_pass_false_positives==0 (non-synthetic gold-pass→pred-fail) AND
status_accuracy≥threshold {alt 0.90 / heading **0.95**}. Heading ALSO needs
exact_correction_accuracy≥0.85. Old adapter eval: alt 0.883/6FP/0.983json;
heading 0.866/1FP/0.995json/0.681exact.

---

## What's staged (the v3 fix — all verified locally)
- **alt_text = OBJECTIVE LABELS.** Edited `_alt_target` in `build_delivered_dataset.py`
  (+ new `_alt_is_placeholder`): a figure fails ONLY if source alt is missing/placeholder/
  generic/filename or decorative; a present, human-reworded alt is now PASS. Removed the
  subjective `"alt text improved in remediation"` label that caused ALL 6 eval FPs.
  Regenerated over 385 delivered docs → `tools/finetune/data_v3/delivered_alt.jsonl`
  (350 drafts) → finalized → fed into the v3 gym build. Verified: 0 subjective fails
  remain; fail types only `missing_or_placeholder`+`decorative`; placeholder detection
  spot-checked correct.
- **heading = REBALANCE (data only).** Down-sampled synthetic+gov `corrupt_flattened`
  fails to 35% in `nemo_source/source_builder_outputs/heading_hierarchy/{train,val}.jsonl`
  (in place; backups `*.jsonl.orig`). LAMC records untouched (NOT duplicated — dataset.py
  splits by doc_id, so pre-split dup pollutes test). LAMC share of train fails **14%→28%**,
  all **10/10 LAMC true-fails preserved in test**.
- **Also:** stripped 45 stale subjective alt rows from `delivered_conversations/{train,val}`
  (backups `*.jsonl.orig`) so alt comes purely from the objective-label v3.
- **Rebuilt gym → `tools/finetune/generated/nemo_campaign_dataset_v3/`** (working
  `nemo_campaign_dataset/` untouched). Verified: ZERO doc-leak both tasks; alt row
  lengths safe (<9k chars, well under 8192 tok — local tokenizer check needs
  `transformers`, absent locally; Brev preflight runs the real filter).

## Reversibility
- `git checkout tools/finetune/build_delivered_dataset.py`
- Restore the 4 `.jsonl.orig` backups (heading train/val + delivered_conv train/val).
- v3 gym is a SEPARATE dir — delete `nemo_campaign_dataset_v3/` to fully revert.

## Hypothesis verdict

- **Alt objective-label hypothesis: rejected.** Correcting the source labels did not
  cure overcalling and introduced severe output-length/JSON regressions.
- **Heading rebalance hypothesis: rejected.** Low validation loss did not transfer to
  the real/frozen correction gate; exact correction fell to `0.3333`.
- The experiment confirms that SFT validation loss is not a safe promotion signal for
  these tasks. Future work must change output shaping/input evidence and pass the same
  frozen gates before any deployment decision.

## Pointers
- Memory: `~/.claude/.../memory/adapter-eval-gates-datafix.md` (gate thresholds, GRPO
  state, decision) + `nemo-vlm-datasets-nonepad-trap.md` (campaign history).
- MOVE3 (contrast+reading_order input redesign, separate track):
  `session/20260714_232247/MOVE3_task_input_redesign.md`.
- GRPO: attempted 3× on 2026-07-17 (launch-debug only, outcome unrecorded; `vllm` missing
  → `VLMEnvironment num_workers` → `max_val_samples None`). Held in reserve; NOT the plan.
