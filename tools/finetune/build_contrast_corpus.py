#!/usr/bin/env python3
"""Build a synthetic contrast corpus with exact WCAG ratio labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from project_remedy.vision_prompts import contrast_detection_prompt  # noqa: E402


FAIL_RATIOS = [2.0, 2.4, 2.8, 3.1, 3.5, 3.9, 4.2, 4.35, 4.45]
PASS_RATIOS = [4.55, 4.7, 4.9, 5.3, 5.8, 6.4, 7.0, 7.6, 8.0]


def srgb_to_linear(channel: int) -> float:
    value = channel / 255.0
    if value <= 0.03928:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    r, g, b = (srgb_to_linear(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(fg: tuple[int, int, int], bg: tuple[int, int, int]) -> float:
    l1, l2 = sorted((relative_luminance(fg), relative_luminance(bg)), reverse=True)
    return (l1 + 0.05) / (l2 + 0.05)


def gray_for_ratio_on_white(ratio: float) -> tuple[int, int, int]:
    target_lum = (1.05 / ratio) - 0.05
    lo, hi = 0, 255
    for _ in range(16):
        mid = (lo + hi) // 2
        lum = relative_luminance((mid, mid, mid))
        if lum < target_lum:
            lo = mid + 1
        else:
            hi = mid
    return (hi, hi, hi)


def load_font(size: int):
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def target_for_ratio(ratio: float, fg: tuple[int, int, int], bg: tuple[int, int, int]) -> dict:
    rounded = round(ratio, 2)
    if ratio >= 4.5:
        return {"issues": []}
    return {
        "issues": [{
            "severity": "error" if ratio < 3.0 else "warning",
            "description": f"Body text contrast ratio is {rounded}:1, below WCAG AA 4.5:1.",
            "ratio": rounded,
            "text_rgb": list(fg),
            "bg_rgb": list(bg),
            "fix_rgb": [0, 0, 0],
        }]
    }


def render_sample(path: Path, *, ratio: float, fg: tuple[int, int, int], bg: tuple[int, int, int], index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (1240, 1754), bg)
    draw = ImageDraw.Draw(image)
    title_font = load_font(54)
    body_font = load_font(38)
    small_font = load_font(26)
    accent = (36, 91, 115)
    draw.rectangle([90, 90, 1150, 210], fill=(235, 241, 243))
    draw.text((120, 120), f"Contrast Verification Sample {index:03d}", fill=accent, font=title_font)
    draw.text((120, 310), "Student services notice", fill=(33, 37, 41), font=body_font)
    draw.rounded_rectangle([110, 400, 1130, 760], radius=18, fill=bg, outline=(180, 190, 196), width=2)
    draw.text((150, 450), "This paragraph is the labeled contrast target.", fill=fg, font=body_font)
    draw.text((150, 525), f"Measured ratio: {ratio:.2f}:1 against the card background.", fill=fg, font=body_font)
    draw.text((150, 610), "The model should decide whether this visible text passes WCAG AA.", fill=fg, font=small_font)
    draw.text((120, 900), "Reference text with strong contrast", fill=(0, 0, 0), font=body_font)
    draw.text((120, 980), "Decorative footer and labels are not the target.", fill=(80, 88, 96), font=small_font)
    image.save(path)


def split_bucket(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def to_conversation(rec: dict) -> dict:
    return {
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": rec["image"]},
                {"type": "text", "text": rec["prompt"]},
            ]},
            {"role": "assistant", "content": [{"type": "text", "text": rec["target"]}]},
        ],
        "meta": {
            "doc_id": rec["doc_id"],
            "page": rec["page"],
            "task": rec["task"],
            "variant": rec["variant"],
            "ratio": rec["ratio"],
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )


def build_records(out_dir: Path, count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    renders = out_dir / "renders"
    records = []
    ratios = []
    for i in range(math.ceil(count / 2)):
        ratios.append(FAIL_RATIOS[i % len(FAIL_RATIOS)])
        ratios.append(PASS_RATIOS[i % len(PASS_RATIOS)])
    ratios = ratios[:count]
    rng.shuffle(ratios)
    for index, requested_ratio in enumerate(ratios, start=1):
        bg = (255, 255, 255)
        fg = gray_for_ratio_on_white(requested_ratio)
        actual_ratio = contrast_ratio(fg, bg)
        variant = "pass" if actual_ratio >= 4.5 else "fail"
        doc_id = f"synthetic_contrast_{index:04d}"
        png_name = f"{doc_id}_p1_200dpi.png"
        render_sample(renders / png_name, ratio=actual_ratio, fg=fg, bg=bg, index=index)
        target = target_for_ratio(actual_ratio, fg, bg)
        records.append({
            "doc_id": doc_id,
            "page": 1,
            "task": "contrast",
            "variant": variant,
            "image": f"renders/{png_name}",
            "prompt": contrast_detection_prompt("AA"),
            "draft_target": "",
            "target": json.dumps(target, ensure_ascii=False),
            "reviewed": True,
            "provenance": "synthetic-contrast-render",
            "ratio": round(actual_ratio, 4),
            "text_rgb": list(fg),
            "bg_rgb": list(bg),
        })
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=Path("tools/finetune/data_contrast"))
    ap.add_argument("--count", type=int, default=160)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=20260705)
    args = ap.parse_args()

    records = build_records(args.out_dir, args.count, args.seed)
    conversations = [to_conversation(rec) for rec in records]
    train = [
        row for rec, row in zip(records, conversations, strict=False)
        if split_bucket(rec["doc_id"]) >= args.val_frac
    ]
    val = [
        row for rec, row in zip(records, conversations, strict=False)
        if split_bucket(rec["doc_id"]) < args.val_frac
    ]
    write_jsonl(args.out_dir / "drafts.jsonl", records)
    write_jsonl(args.out_dir / "train.jsonl", train)
    write_jsonl(args.out_dir / "val.jsonl", val)
    manifest = {
        "task": "contrast",
        "count": len(records),
        "train": len(train),
        "val": len(val),
        "fail": sum(1 for rec in records if rec["variant"] == "fail"),
        "pass": sum(1 for rec in records if rec["variant"] == "pass"),
        "ratio_min": min(rec["ratio"] for rec in records),
        "ratio_max": max(rec["ratio"] for rec in records),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
