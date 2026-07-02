"""FR-1: backend selector + typst timeout follow the two-part config pattern."""

from __future__ import annotations

from project_remedy.config import RebuildConfig, load_config


def test_rebuild_config_defaults():
    cfg = RebuildConfig()
    assert cfg.backend == "questpdf"
    assert cfg.typst_timeout_s == 120.0


def test_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("REBUILD_BACKEND", "typst")
    monkeypatch.setenv("REBUILD_TYPST_TIMEOUT_S", "45.5")
    cfg = load_config()
    assert cfg.rebuild.backend == "typst"
    assert cfg.rebuild.typst_timeout_s == 45.5
