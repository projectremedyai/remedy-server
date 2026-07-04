#!/usr/bin/env python3
"""Download the curated public datasets for vision fine-tuning (from the 4-agent
HF/web recon, see memory: vision-finetune-datasets). Run on the box (or Mac) to
pull the SHIP-SAFE permissive sets by default; flagged/eval-only sets are listed
but require --include-flagged / --include-eval and a license decision.

Usage:
    hf auth login                       # once, so gated/large pulls work
    python tools/finetune/download_public_datasets.py --list          # just print the registry
    python tools/finetune/download_public_datasets.py --dest ~/ftdata # pull permissive sets
    python tools/finetune/download_public_datasets.py --dest ~/ftdata --only doclaynet pubtables1m

Downloads are the raw HF dataset repos (snapshot). Converting each to our
image->JSON conversation format is a per-dataset adapter step (not done here) —
this only fetches the bytes. Big sets (Docmatix ~0.5TB) are OPT-IN via --only.
"""
from __future__ import annotations

import argparse
import sys

# tier: permissive | flagged | eval   (train only on 'permissive' without a license call)
# task: layout | reading_order | tables | alt_text | headings | base_replay | eval
REGISTRY = [
    # ---- permissive, train-safe ----
    ("doclaynet",    "ds4sd/DocLayNet",                        "permissive", "layout",        "CDLA-Permissive", "80k pg human gold; Section-header class"),
    ("d4la",         "liferecords/D4LA",                       "permissive", "layout",        "verify(Apache?)", "11k scanned RVL-CDIP; domain match to forms"),
    ("pubtables1m",  "bsmock/pubtables-1m",                    "permissive", "tables",        "CDLA-Permissive", "575k; column-header/proj-row-header roles ~= /Scope"),
    ("pubtabnet",    "apoidea/pubtabnet-html",                 "permissive", "tables",        "CDLA-Permissive", "568k img->HTML; thead/th = header rows"),
    ("fintabnet_c",  "bsmock/FinTabNet.c",                     "permissive", "tables",        "CDLA-Permissive-2", "90k financial/dense; canonicalized"),
    ("charttotext",  "saadob12/chart-to-text",                 "permissive", "alt_text",      "mixed(open)",     "44k human chart summaries"),
    ("vistext",      "oroikon/vistext_chart_captioning",       "permissive", "alt_text",      "CC-BY-SA",        "12k two-tier (L1 struct / L2-3 insight) captions"),
    ("vizwiz_caps",  "lmms-lab/VizWiz-Caps",                   "permissive", "alt_text",      "CC-BY-4.0",       "accessibility REGISTER (photos, not doc figures)"),
    ("publaynet",    "jordanparker6/publaynet",                "permissive", "layout",        "CDLA-Permissive", "360k AUTO labels; bulk filler only"),
    ("docbank",      "maveriq/DocBank",                        "permissive", "layout",        "Apache-2.0",      "500k AUTO; token reading-order stream"),
    ("tablebank",    "liminghao1630/TableBank",                "permissive", "tables",        "Apache-2.0",      "417k weak labels; coarse, volume filler"),
    ("docmatix",     "HuggingFaceM4/Docmatix",                 "permissive", "base_replay",   "MIT",             "~0.5TB MODEL-GENERATED QA; low-weight replay only"),
    ("cauldron",     "HuggingFaceM4/the_cauldron",             "permissive", "base_replay",   "mixed-per-subset","filter by subset license; anti-forgetting replay"),
    # ---- flagged: verify license before shipping a trained model ----
    ("scicap",       "CrowdAILab/scicap",                      "flagged",    "alt_text",      "CC-BY-NC-SA",     "best figure content BUT non-commercial"),
    ("scitsr_pd",    "bevaya/SciTSR-pd",                       "flagged",    "tables",        "PD split",        "spanning/merged cells; use -pd split only"),
    # ---- eval-only: NEVER put in the train set ----
    ("pdf_a11y_bench","(github: PDF-Accessibility-Benchmark)", "eval",       "eval",          "MIT/CC0",         "125 docs, 7 criteria = our tasks; page PNG+JSON; few-shot seed + eval"),
    ("docvqa",       "lmms-lab/DocVQA",                        "eval",       "eval",          "research",        "human doc QA; held-out eval"),
    ("comtqa",       "ByteDance/ComTQA",                       "eval",       "eval",          "CC-BY-NC-4.0",    "human table-VQA; header/scope eval"),
]


def _print_registry() -> None:
    w = max(len(n) for n, *_ in REGISTRY)
    print(f"{'key':<{w}}  {'tier':<10} {'task':<13} {'license':<18} hf_id / note")
    for key, hid, tier, task, lic, note in REGISTRY:
        print(f"{key:<{w}}  {tier:<10} {task:<13} {lic:<18} {hid}  — {note}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="print the registry and exit")
    ap.add_argument("--dest", default=None, help="download dir")
    ap.add_argument("--only", nargs="*", default=None, help="download only these keys")
    ap.add_argument("--include-flagged", action="store_true", help="also non-commercial/flagged sets")
    ap.add_argument("--include-eval", action="store_true", help="also eval-only sets (do NOT train on these)")
    args = ap.parse_args()

    if args.list or not args.dest:
        _print_registry()
        if not args.dest:
            print("\n(no --dest given; nothing downloaded. Pass --dest to pull the permissive sets.)")
        return 0

    from huggingface_hub import snapshot_download

    def _wanted(key, tier):
        if args.only:
            return key in args.only
        if tier == "permissive":
            return True
        if tier == "flagged":
            return args.include_flagged
        if tier == "eval":
            return args.include_eval
        return False

    picked = [(k, h, t) for k, h, t, *_ in REGISTRY if _wanted(k, t)]
    print(f"Downloading {len(picked)} dataset(s) -> {args.dest}")
    for key, hid, tier in picked:
        if hid.startswith("("):
            print(f"  {key}: not on HF ({hid}) — fetch manually; skipping.")
            continue
        print(f"  {key} ({tier}) <- {hid}")
        try:
            snapshot_download(repo_id=hid, repo_type="dataset",
                              local_dir=f"{args.dest}/{key}")
        except Exception as e:
            print(f"    FAILED {hid}: {e}  (gated? run `hf auth login`, or accept terms on the HF page)")
    print("Done. Each dataset still needs a per-source adapter to our image->JSON "
          "conversation format before training (not done here).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
