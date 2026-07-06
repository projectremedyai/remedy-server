from __future__ import annotations

import asyncio
from pathlib import Path

from PIL import Image

from project_remedy.contrast.detector import ContrastDetector
from project_remedy.pdf_wcag_verifier import PageTriageResult, WCAGVisionVerifier


class _TaskRecordingProvider:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def analyze_image(
        self,
        image_path,
        prompt,
        *,
        max_tokens=4096,
        response_format=None,
        task=None,
    ):
        self.calls.append({
            "image_path": image_path,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "response_format": response_format,
            "task": task,
        })
        if task == "table_structure":
            return (
                '{"status":"fail","confidence":0.91,"summary":"Missing table headers",'
                '"findings":[{"issue_id":"missing_table_headers","severity":"error",'
                '"message":"The visual table has column headers but the tags do not expose TH cells.",'
                '"fixer":"fix_table_headers"}]}'
            )
        if task == "contrast":
            return '{"status":"pass","confidence":0.88,"summary":"No low contrast found","findings":[]}'
        return '{"status":"pass","confidence":0.8,"findings":[]}'


async def test_focused_wcag_verifier_routes_table_and_contrast_tasks(tmp_path, monkeypatch):
    import project_remedy.pdf_vision as pdf_vision
    import project_remedy.pdf_wcag_verifier as verifier_mod

    image_path = tmp_path / "page.png"
    Image.new("RGB", (24, 24), (255, 255, 255)).save(image_path)

    def fake_render(_pdf_path: Path, _page_num: int, _dpi: int = 150) -> Path:
        return image_path

    monkeypatch.setattr(pdf_vision, "render_page_to_image", fake_render)
    monkeypatch.setattr(
        verifier_mod,
        "_extract_page_structure_context",
        lambda _pdf_path, _page_idx: (
            "1. /Table\n2. /TR\n3. /TD: Name\n4. /TD: Value",
            "(no heading context)",
        ),
    )

    provider = _TaskRecordingProvider()
    verifier = WCAGVisionVerifier(provider, vision_concurrency=1, render_concurrency=1)
    triage = PageTriageResult(focus_queue=["table_structure", "contrast"])

    result = await verifier._verify_page_focused(
        tmp_path / "dummy.pdf",
        0,
        triage,
        asyncio.Semaphore(1),
        asyncio.Semaphore(1),
    )

    assert [call["task"] for call in provider.calls] == ["table_structure", "contrast"]
    assert result.criteria["table_structure"].status == "fail"
    assert result.criteria["table_structure"].findings[0].fixer == "fix_table_headers"
    assert result.criteria["color_contrast"].status == "pass"


async def test_contrast_detector_routes_analyze_image_to_contrast_task():
    provider = _TaskRecordingProvider()
    detector = ContrastDetector(provider)

    parsed = await detector._call_vision(
        b"not actually decoded by the fake provider",
        "Find contrast issues",
        {"type": "object"},
    )

    assert parsed == {"status": "pass", "confidence": 0.88, "summary": "No low contrast found", "findings": []}
    assert provider.calls[0]["task"] == "contrast"
