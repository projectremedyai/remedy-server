"""Single-step NeMo Gym resource server for five Remedy PDF tasks."""

from __future__ import annotations

from statistics import mean
from typing import Any

from pydantic import Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from tools.finetune.remedy_nemo_rl.gym_adapter import verify_gym_response


class RemedyPdfConfig(BaseResourcesServerConfig):
    """Configuration for the deterministic Remedy verifier."""


class RemedyVerifyRequest(BaseVerifyRequest):
    """Verifier input containing model-hidden normalized gold metadata."""

    example_id: str
    task: str
    verifier_target: dict[str, Any]
    verifier_metadata: dict[str, Any] = Field(default_factory=dict)


class RemedyVerifyResponse(BaseVerifyResponse):
    """Verifier output with scalar and named reward components."""

    example_id: str
    task: str
    verifier_target: dict[str, Any]
    verifier_metadata: dict[str, Any]
    passed: bool
    components: dict[str, float]
    verifier_error: str | None = None


class RemedyPdfResourcesServer(SimpleResourcesServer):
    """Dispatch all five tasks to one deterministic single-step verifier."""

    config: RemedyPdfConfig

    async def verify(self, body: RemedyVerifyRequest) -> RemedyVerifyResponse:
        """Verify one response without using self-reported confidence."""

        result = verify_gym_response(
            task=body.task,
            response=body.response,
            verifier_target=body.verifier_target,
        )
        return RemedyVerifyResponse(
            **body.model_dump(),
            reward=result.reward,
            passed=result.passed,
            components=result.components,
            verifier_error=result.error,
        )

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, float]:
        """Aggregate reward and named components across completed rollouts."""

        rows = [row for rollouts in tasks for row in rollouts]
        metrics = {"mean/reward": mean(float(row.get("reward", 0.0)) for row in rows)} if rows else {}
        component_names = sorted(
            {
                name
                for row in rows
                for name in (row.get("components") or {})
            }
        )
        for name in component_names:
            values = [float(row["components"][name]) for row in rows if name in (row.get("components") or {})]
            if values:
                metrics[f"mean/{name}"] = mean(values)
        return metrics


if __name__ == "__main__":
    RemedyPdfResourcesServer.run_webserver()
