"""Tests for Brev launch commands and runtime cost accounting."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tools.finetune.remedy_nemo_rl.brev_control import (
    BrevCampaignState,
    active_accrued_cost,
    build_budget_policy,
    build_create_command,
    finalize_active_state,
    load_state,
    save_state,
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
        "--min-disk",
        "100",
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


def test_vm_command_omits_custom_container_but_keeps_startup_and_disk_floor() -> None:
    command = build_create_command(
        instance_name="remedy-vm",
        instance_type="a2-ultragpu-1g:nvidia-a100-80gb:1",
        startup_script="/tmp/remedy-startup.sh",
        mode="vm",
    )

    assert "--container-image" not in command
    assert command[command.index("--mode") + 1] == "vm"
    assert "--jupyter=false" in command
    assert command[command.index("--min-disk") + 1] == "100"


def test_inactive_state_has_no_accrued_compute_cost() -> None:
    assert active_accrued_cost(BrevCampaignState(), datetime.now(timezone.utc)) == 0.0


def test_deleted_unhealthy_build_can_be_reconciled_into_cost_history() -> None:
    started = datetime(2026, 7, 15, 7, 0, tzinfo=timezone.utc)
    state = BrevCampaignState(
        active_instance="failed-build",
        active_started_at=started.isoformat(),
        active_hourly_rate_usd=1.98,
        active_deadline=(started + timedelta(hours=3)).isoformat(),
    )

    finalize_active_state(state, started + timedelta(minutes=10), outcome="deleted_unhealthy_build")

    assert state.recorded_spend_usd == pytest.approx(0.33)
    assert state.active_instance is None
    assert state.history[-1]["outcome"] == "deleted_unhealthy_build"


def test_approved_budget_policy_round_trips_through_campaign_state(tmp_path) -> None:
    state_path = tmp_path / "brev_state.json"
    state = BrevCampaignState(
        recorded_spend_usd=50.82,
        hard_limit_usd=60.0,
        reserve_usd=0.6,
    )

    save_state(state_path, state)
    restored = load_state(state_path)
    policy = build_budget_policy(restored)

    assert restored.hard_limit_usd == 60.0
    assert restored.reserve_usd == 0.6
    assert policy.hard_limit_usd == 60.0
    assert policy.no_new_work_usd == pytest.approx(59.4)


def test_explicit_launch_limits_override_historical_state() -> None:
    state = BrevCampaignState(hard_limit_usd=50.0, reserve_usd=10.0)

    policy = build_budget_policy(
        state,
        hard_limit_usd=60.0,
        reserve_override_usd=0.6,
    )

    assert policy.hard_limit_usd == 60.0
    assert policy.reserve_usd == 0.6
    assert policy.no_new_work_usd == pytest.approx(59.4)
