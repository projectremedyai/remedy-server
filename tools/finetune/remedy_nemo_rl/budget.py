"""Budget authorization for the credit-limited NVIDIA Brev campaign."""

from __future__ import annotations

from dataclasses import dataclass


class BudgetViolation(RuntimeError):
    """Raised when a proposed Brev action would violate a campaign limit."""


@dataclass(frozen=True)
class BudgetPolicy:
    """Immutable limits for the current Brev allocation.

    Attributes:
        hard_limit_usd: Absolute credit limit where all paid work must stop.
        no_new_work_usd: Spend level at which no new experiment may start.
        reserve_usd: Credit retained for teardown and artifact retrieval.
        max_instance_hours: Maximum lifetime of the first paid instance.
        max_gpu_count: Maximum GPU count authorized by this allocation.
    """

    hard_limit_usd: float = 50.0
    no_new_work_usd: float = 40.0
    reserve_usd: float = 10.0
    max_instance_hours: float = 3.0
    max_gpu_count: int = 1

    def __post_init__(self) -> None:
        if self.hard_limit_usd <= 0:
            raise ValueError("hard_limit_usd must be positive")
        if self.no_new_work_usd != self.hard_limit_usd - self.reserve_usd:
            raise ValueError("no_new_work_usd must preserve reserve_usd")
        if self.max_instance_hours <= 0 or self.max_gpu_count < 1:
            raise ValueError("instance time and GPU limits must be positive")


@dataclass(frozen=True)
class LaunchDecision:
    """Cost projection returned for an authorized launch."""

    authorized: bool
    projected_run_cost_usd: float
    projected_total_usd: float
    remaining_hard_limit_usd: float
    deadline_hours: float


def authorize_launch(
    *,
    policy: BudgetPolicy,
    recorded_spend_usd: float,
    hourly_rate_usd: float,
    requested_hours: float,
    gpu_count: int,
) -> LaunchDecision:
    """Authorize a Brev launch only when every cost and shape guard passes.

    Args:
        policy: Active immutable campaign limits.
        recorded_spend_usd: Spend already incurred by the campaign.
        hourly_rate_usd: Total advertised hourly rate for the instance.
        requested_hours: Maximum requested paid wall time.
        gpu_count: GPUs attached to the proposed instance.

    Returns:
        A cost projection for the authorized run.

    Raises:
        BudgetViolation: If the launch crosses any campaign guard.
        ValueError: If a numeric input is negative or zero where invalid.
    """

    if recorded_spend_usd < 0 or hourly_rate_usd <= 0 or requested_hours <= 0:
        raise ValueError("spend must be nonnegative and rate/time must be positive")
    if requested_hours > policy.max_instance_hours:
        raise BudgetViolation("proposed run exceeds the three-hour instance limit")
    if gpu_count > policy.max_gpu_count:
        raise BudgetViolation("proposed run exceeds the single-GPU allocation")

    run_cost = hourly_rate_usd * requested_hours
    projected_total = recorded_spend_usd + run_cost
    if projected_total > policy.hard_limit_usd:
        raise BudgetViolation("projected spend exceeds the hard credit limit")
    if recorded_spend_usd >= policy.no_new_work_usd:
        raise BudgetViolation("recorded spend reached the no-new-work threshold")
    if projected_total > policy.no_new_work_usd:
        raise BudgetViolation("projected run would consume the reserved balance")

    return LaunchDecision(
        authorized=True,
        projected_run_cost_usd=round(run_cost, 4),
        projected_total_usd=round(projected_total, 4),
        remaining_hard_limit_usd=round(policy.hard_limit_usd - projected_total, 4),
        deadline_hours=requested_hours,
    )
