#!/usr/bin/env python3
"""Convert reviewed correctable records into Unsloth conversation train/val JSONL.

Reads drafts.jsonl (after a human filled in `target` and set reviewed=true) and
emits the message format train_qlora_vision.py expects. Skips unreviewed rows so
you can build the set incrementally.

Usage:
    uv run python tools/finetune/finalize_dataset.py \
        --in tools/finetune/data/drafts.jsonl \
        --out-dir tools/finetune/data --val-frac 0.15

For a pure PIPELINE smoke (prove the training loop runs before any human labels
exist), pass --use-drafts-as-target to treat the base model's draft as the target.
That trains the model on its own output — useless as a real model, but it lets you
verify the 4080 loop end-to-end. Never ship a model trained this way.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def to_conversation(rec: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": rec["image"]},
                {"type": "text", "text": rec["prompt"]},
            ]},
            {"role": "assistant", "content": [
                {"type": "text", "text": rec["target"]},
            ]},
        ],
        "meta": {"doc_id": rec.get("doc_id"), "page": rec.get("page"),
                 "task": rec.get("task")},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("tools/finetune/data"))
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--use-drafts-as-target", action="store_true",
                    help="Pipeline-smoke only: treat draft_target as target.")
    args = ap.parse_args()

    rows = []
    for line in args.inp.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        target = rec.get("target") or ""
        if args.use_drafts_as_target and not target:
            target = rec.get("draft_target") or ""
        if not target.strip():
            continue  # unreviewed / unlabeled -> skip
        if not args.use_drafts_as_target and not rec.get("reviewed"):
            continue  # not human-approved
        rec = dict(rec, target=target)
        rows.append(to_conversation(rec))

    if not rows:
        print("No usable rows. Have humans filled in `target` + reviewed=true? "
              "(or pass --use-drafts-as-target for a pipeline smoke).")
        return 1

    # deterministic split by hash of (doc_id,page,task) so the same page never
    # straddles train/val across rebuilds.
    def _bucket(m):
        key = f"{m['meta']['doc_id']}|{m['meta']['page']}|{m['meta']['task']}"
        digest = hashlib.sha256(key.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") / float(2**64)

    val = [r for r in rows if _bucket(r) < args.val_frac]
    train = [r for r in rows if _bucket(r) >= args.val_frac]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n", encoding="utf-8")
    (args.out_dir / "val.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val) + "\n", encoding="utf-8")
    print(f"train={len(train)}  val={len(val)}  -> {args.out_dir}/train.jsonl, val.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
