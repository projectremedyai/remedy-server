"""Tests for campaign dataset acceptance preflight."""

from __future__ import annotations

import json
from pathlib import Path

from tools.finetune.remedy_nemo_rl.dataset import build_dataset
from tools.finetune.remedy_nemo_rl.preflight import REQUIRED_CATALOG_SHA256, validate_campaign_dataset
from tests.unit.test_nemo_rl_dataset import _write_sources


def test_preflight_accepts_rebuilt_grouped_dataset(tmp_path: Path) -> None:
    output = tmp_path / "campaign"
    build_dataset(_write_sources(tmp_path), output)
    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["holdout_files"] = {
        f"catalog-{index}.pdf": {"sha256": digest, "size_bytes": 1}
        for index, digest in enumerate(sorted(REQUIRED_CATALOG_SHA256))
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_campaign_dataset(output)

    assert report["passed"] is True
    assert report["document_leakage"] == []
    assert report["missing_images"] == []
    assert report["hash_mismatches"] == []
