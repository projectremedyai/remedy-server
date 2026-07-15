"""Tests for Brev launch commands and runtime cost accounting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tools.finetune.remedy_nemo_rl.brev_control import (
    BrevCampaignState,
    active_accrued_cost,
    build_create_command,
)


def test_create_command_pins_stoppable_h200_and_official_container() -> None:
    command = build_create_command(
        instance_name="remedy-nemo-rl-20260714",
        instance_type="gpu-h200-sxm.1gpu-16vcpu-200gb",
        startup_script="/tmp/remedy-startup.sh",
    )

    assert command == [
        "brev",
        "create",
        "remedy-nemo-rl-20260714",
        "--type",
        "gpu-h200-sxm.1gpu-16vcpu-200gb",
        "--stoppable",
        "--mode",
        "container",
        "--container-image",
        "nvcr.io/nvidia/nemo-rl:v0.6.0",
        "--startup-script",
        "@/tmp/remedy-startup.sh",
        "--timeout",
        "600",
    ]


def test_active_cost_uses_real_elapsed_wall_time() -> None:
    started = datetime(2026, 7, 14, 20, 0, tzinfo=timezone.utc)
    state = BrevCampaignState(
        recorded_spend_usd=2.0,
        active_instance="remedy",
        active_started_at=started.isoformat(),
        active_hourly_rate_usd=5.40,
        active_deadline=(started + timedelta(hours=3)).isoformat(),
    )

    assert active_accrued_cost(state, started + timedelta(minutes=90)) == pytest.approx(8.10)


def test_inactive_state_has_no_accrued_compute_cost() -> None:
    assert active_accrued_cost(BrevCampaignState(), datetime.now(timezone.utc)) == 0.0
