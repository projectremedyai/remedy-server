"""Guarded NVIDIA Brev lifecycle and cost accounting for the $50 campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .budget import BudgetPolicy, authorize_launch


NEMO_RL_IMAGE = "nvcr.io/nvidia/nemo-rl:v0.6.0"


@dataclass
class BrevCampaignState:
    """Durable spend and active-instance state used by the watchdog."""

    recorded_spend_usd: float = 0.0
    active_instance: str | None = None
    active_started_at: str | None = None
    active_hourly_rate_usd: float | None = None
    active_deadline: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value)


def load_state(path: Path) -> BrevCampaignState:
    """Load campaign state, returning a zero-spend state when absent."""

    if not path.exists():
        return BrevCampaignState()
    return BrevCampaignState(**json.loads(path.read_text(encoding="utf-8")))


def save_state(path: Path, state: BrevCampaignState) -> None:
    """Atomically persist campaign state."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(asdict(state), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def active_accrued_cost(state: BrevCampaignState, now: datetime | None = None) -> float:
    """Calculate unrecorded active compute cost from real elapsed wall time."""

    if not state.active_instance or not state.active_started_at or state.active_hourly_rate_usd is None:
        return 0.0
    elapsed = max(0.0, ((now or _now()) - _parse_time(state.active_started_at)).total_seconds())
    return elapsed / 3600.0 * state.active_hourly_rate_usd


def finalize_active_state(
    state: BrevCampaignState,
    ended_at: datetime,
    *,
    outcome: str,
) -> BrevCampaignState:
    """Record elapsed cost and clear an externally ended active instance."""

    if not state.active_instance:
        return state
    instance = state.active_instance
    run_cost = active_accrued_cost(state, ended_at)
    state.recorded_spend_usd = round(state.recorded_spend_usd + run_cost, 4)
    state.history.append(
        {
            "instance": instance,
            "started_at": state.active_started_at,
            "stopped_at": ended_at.isoformat(),
            "hourly_rate_usd": state.active_hourly_rate_usd,
            "cost_usd": round(run_cost, 4),
            "outcome": outcome,
        }
    )
    state.active_instance = None
    state.active_started_at = None
    state.active_hourly_rate_usd = None
    state.active_deadline = None
    return state


def build_create_command(
    *,
    instance_name: str,
    instance_type: str,
    startup_script: str,
    minimum_disk_gb: int = 100,
    mode: str = "container",
) -> list[str]:
    """Build the exact pinned, stoppable, single-container Brev create command."""

    if mode not in {"container", "vm"}:
        raise ValueError(f"unsupported Brev mode: {mode}")
    command = [
        "brev",
        "create",
        instance_name,
        "--type",
        instance_type,
        "--min-disk",
        str(minimum_disk_gb),
        "--stoppable",
        "--mode",
        mode,
    ]
    if mode == "container":
        command.extend(["--container-image", NEMO_RL_IMAGE])
    else:
        command.append("--jupyter=false")
    command.extend(
        [
        "--startup-script",
        f"@{startup_script}",
        "--timeout",
        "600",
        ]
    )
    return command


def _stop_instance(state_path: Path, expected_instance: str | None = None) -> BrevCampaignState:
    state = load_state(state_path)
    if not state.active_instance:
        return state
    if expected_instance and state.active_instance != expected_instance:
        raise RuntimeError(f"active instance is {state.active_instance}, not {expected_instance}")

    stopped_at = _now()
    subprocess.run(["brev", "stop", state.active_instance], check=True)
    finalize_active_state(state, stopped_at, outcome="stopped")
    save_state(state_path, state)
    return state


def _reconcile_deleted(args: argparse.Namespace) -> int:
    state = load_state(args.state)
    if not state.active_instance:
        print(json.dumps(asdict(state), indent=2, sort_keys=True))
        return 0
    if args.instance and state.active_instance != args.instance:
        raise SystemExit(f"active instance is {state.active_instance}, not {args.instance}")
    listing = subprocess.run(["brev", "ls"], check=True, capture_output=True, text=True)
    if state.active_instance in listing.stdout:
        raise SystemExit(f"instance still appears in Brev inventory: {state.active_instance}")
    finalize_active_state(state, _now(), outcome=args.outcome)
    save_state(args.state, state)
    print(json.dumps(asdict(state), indent=2, sort_keys=True))
    return 0


def _arm_watchdog(state_path: Path, instance: str) -> int:
    log_path = state_path.with_suffix(".watchdog.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_stream = log_path.open("ab")
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "tools.finetune.remedy_nemo_rl.brev_control",
            "watch",
            "--state",
            str(state_path),
            "--instance",
            instance,
        ],
        stdout=log_stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log_stream.close()
    return process.pid


def _launch(args: argparse.Namespace) -> int:
    state = load_state(args.state)
    if state.active_instance:
        raise SystemExit(f"campaign already has active instance {state.active_instance}")
    decision = authorize_launch(
        policy=(
            BudgetPolicy(
                reserve_usd=args.reserve_override_usd,
                no_new_work_usd=BudgetPolicy.hard_limit_usd - args.reserve_override_usd,
            )
            if getattr(args, "reserve_override_usd", None) is not None
            else BudgetPolicy()
        ),
        recorded_spend_usd=state.recorded_spend_usd,
        hourly_rate_usd=args.hourly_rate,
        requested_hours=args.hours,
        gpu_count=1,
    )
    command = build_create_command(
        instance_name=args.instance,
        instance_type=args.instance_type,
        startup_script=str(args.startup_script.resolve()),
        minimum_disk_gb=args.minimum_disk_gb,
        mode=args.mode,
    )
    preview = {"command": command, "decision": asdict(decision), "execute": args.execute}
    print(json.dumps(preview, indent=2, sort_keys=True))
    if not args.execute:
        return 0
    if not args.startup_script.is_file():
        raise SystemExit(f"startup script does not exist: {args.startup_script}")

    subprocess.run(command, check=True)
    started_at = _now()
    state.active_instance = args.instance
    state.active_started_at = started_at.isoformat()
    state.active_hourly_rate_usd = args.hourly_rate
    state.active_deadline = (started_at + timedelta(hours=args.hours)).isoformat()
    save_state(args.state, state)
    watchdog_pid = _arm_watchdog(args.state, args.instance)
    print(json.dumps({"instance": args.instance, "deadline": state.active_deadline, "watchdog_pid": watchdog_pid}))
    return 0


def _start_existing(args: argparse.Namespace) -> int:
    """Start a stopped Brev instance and arm the campaign watchdog."""

    state = load_state(args.state)
    if state.active_instance:
        raise SystemExit(f"campaign already has active instance {state.active_instance}")
    decision = authorize_launch(
        policy=BudgetPolicy(),
        recorded_spend_usd=state.recorded_spend_usd,
        hourly_rate_usd=args.hourly_rate,
        requested_hours=args.hours,
        gpu_count=1,
    )
    command = ["brev", "start", args.instance]
    preview = {"command": command, "decision": asdict(decision), "execute": args.execute}
    print(json.dumps(preview, indent=2, sort_keys=True))
    if not args.execute:
        return 0

    subprocess.run(command, check=True)
    started_at = _now()
    state.active_instance = args.instance
    state.active_started_at = started_at.isoformat()
    state.active_hourly_rate_usd = args.hourly_rate
    state.active_deadline = (started_at + timedelta(hours=args.hours)).isoformat()
    save_state(args.state, state)
    watchdog_pid = _arm_watchdog(args.state, args.instance)
    print(json.dumps({"instance": args.instance, "deadline": state.active_deadline, "watchdog_pid": watchdog_pid}))
    return 0


def _watch(args: argparse.Namespace) -> int:
    policy = BudgetPolicy()
    while True:
        state = load_state(args.state)
        if state.active_instance != args.instance:
            return 0
        now = _now()
        total = state.recorded_spend_usd + active_accrued_cost(state, now)
        deadline = _parse_time(state.active_deadline) if state.active_deadline else now
        if now >= deadline or total >= policy.hard_limit_usd:
            _stop_instance(args.state, args.instance)
            return 0
        time.sleep(30)


def _status(args: argparse.Namespace) -> int:
    state = load_state(args.state)
    accrued = active_accrued_cost(state)
    payload = {
        **asdict(state),
        "active_accrued_cost_usd": round(accrued, 4),
        "estimated_total_spend_usd": round(state.recorded_spend_usd + accrued, 4),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main() -> int:
    """Run guarded launch, watchdog, stop, or status operations."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    launch = subparsers.add_parser("launch")
    launch.add_argument("--state", type=Path, required=True)
    launch.add_argument("--instance", required=True)
    launch.add_argument("--instance-type", default="gpu-h200-sxm.1gpu-16vcpu-200gb")
    launch.add_argument("--hourly-rate", type=float, default=5.40)
    launch.add_argument("--minimum-disk-gb", type=int, default=100)
    launch.add_argument("--mode", choices=("container", "vm"), default="container")
    launch.add_argument("--hours", type=float, default=3.0)
    launch.add_argument("--startup-script", type=Path, required=True)
    launch.add_argument(
        "--reserve-override-usd",
        type=float,
        default=None,
        help=(
            "USER-AUTHORIZED one-time reserve reduction. Lowers reserve_usd and "
            "raises no_new_work_usd consistently. Record the authorization in "
            "the session ledger when using this."
        ),
    )
    launch.add_argument("--execute", action="store_true")
    launch.set_defaults(handler=_launch)

    start_existing = subparsers.add_parser("start-existing")
    start_existing.add_argument("--state", type=Path, required=True)
    start_existing.add_argument("--instance", required=True)
    start_existing.add_argument("--hourly-rate", type=float, required=True)
    start_existing.add_argument("--hours", type=float, default=1.0)
    start_existing.add_argument("--execute", action="store_true")
    start_existing.set_defaults(handler=_start_existing)

    watch = subparsers.add_parser("watch")
    watch.add_argument("--state", type=Path, required=True)
    watch.add_argument("--instance", required=True)
    watch.set_defaults(handler=_watch)

    stop = subparsers.add_parser("stop")
    stop.add_argument("--state", type=Path, required=True)
    stop.add_argument("--instance")
    stop.set_defaults(handler=lambda args: (print(json.dumps(asdict(_stop_instance(args.state, args.instance)), indent=2)), 0)[1])

    reconcile = subparsers.add_parser("reconcile-deleted")
    reconcile.add_argument("--state", type=Path, required=True)
    reconcile.add_argument("--instance")
    reconcile.add_argument("--outcome", default="deleted_unhealthy_build")
    reconcile.set_defaults(handler=_reconcile_deleted)

    status = subparsers.add_parser("status")
    status.add_argument("--state", type=Path, required=True)
    status.set_defaults(handler=_status)

    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
