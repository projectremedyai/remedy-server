"""Tests for the revised $50 Brev credit guard."""

from __future__ import annotations

import pytest

from tools.finetune.remedy_nemo_rl.budget import BudgetPolicy, BudgetViolation, authorize_launch


def test_three_hour_h200_window_is_authorized_from_zero_spend() -> None:
    decision = authorize_launch(
        policy=BudgetPolicy(),
        recorded_spend_usd=0.0,
        hourly_rate_usd=5.40,
        requested_hours=3.0,
        gpu_count=1,
    )

    assert decision.authorized is True
    assert decision.projected_run_cost_usd == pytest.approx(16.20)
    assert decision.projected_total_usd == pytest.approx(16.20)


@pytest.mark.parametrize(
    ("spend", "hours", "gpu_count", "message"),
    [
        (40.0, 1.0, 1, "no-new-work threshold"),
        (0.0, 3.01, 1, "three-hour instance limit"),
        (0.0, 1.0, 4, "single-GPU allocation"),
        (49.0, 1.0, 1, "hard credit limit"),
    ],
)
def test_unsafe_launches_are_rejected(spend: float, hours: float, gpu_count: int, message: str) -> None:
    with pytest.raises(BudgetViolation, match=message):
        authorize_launch(
            policy=BudgetPolicy(),
            recorded_spend_usd=spend,
            hourly_rate_usd=5.40,
            requested_hours=hours,
            gpu_count=gpu_count,
        )


def test_projected_run_cannot_consume_ten_dollar_reserve() -> None:
    with pytest.raises(BudgetViolation, match="reserved balance"):
        authorize_launch(
            policy=BudgetPolicy(),
            recorded_spend_usd=30.0,
            hourly_rate_usd=5.40,
            requested_hours=2.0,
            gpu_count=1,
        )
