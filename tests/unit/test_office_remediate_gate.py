"""FR5: Office remediation output is validated before being reported done."""

from __future__ import annotations

import json
from types import SimpleNamespace

from backend.app.engine_service import _remediate_office
from backend.app.jobs import JOB_KIND_REMEDIATE_OFFICE, Job
from project_remedy.office_acceptance import evaluate_office_acceptance, summarize_office_acceptance
from tests.unit.office_fixtures import make_docx


def test_summarize_office_acceptance(tmp_path):
    path = make_docx(tmp_path / "bad.docx")  # no title/language/headings
    summary = summarize_office_acceptance(evaluate_office_acceptance(path))
    assert summary["passed"] is False
    assert "docx-title" in summary["failed_rule_ids"]
    assert summary["package_valid"] is True
    assert isinstance(summary["manual_check_rule_ids"], list)


class _FakeStore:
    def __init__(self):
        self.updates: list[dict] = []

    async def update(self, job_id, **kwargs):
        self.updates.append({"job_id": job_id, **kwargs})


async def test_remediate_office_attaches_acceptance_metadata(tmp_path):
    input_path = make_docx(tmp_path / "input.docx",
                           body_paragraphs=["Some body text for the remediator to work with."])
    job = Job(
        id="job-test-1", kind=JOB_KIND_REMEDIATE_OFFICE, status="running", stage="",
        progress=0.0, input_path=str(input_path), output_path="", report_path="",
        error="", created_at="", updated_at="", metadata_json="{}",
    )
    store = _FakeStore()
    settings = SimpleNamespace(job_dir=tmp_path / "jobs")

    await _remediate_office(job, store, settings)

    final = store.updates[-1]
    assert final.get("status") == "done"
    meta = json.loads(final["metadata_json"])
    assert "acceptance" in meta
    assert set(meta["acceptance"]) >= {"passed", "failed_rule_ids", "summary"}
    # remediation sets title/language/headings, so those must not be in the failures
    assert "docx-title" not in meta["acceptance"]["failed_rule_ids"]
    stages = [u.get("stage") for u in store.updates]
    assert "evaluating_acceptance" in stages
