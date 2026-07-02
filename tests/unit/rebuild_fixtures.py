"""Shared RebuildRequest builders for typst-backend tests (no binary blobs)."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from project_remedy.rebuild.ast import (
    ArtifactBlock,
    AssetRef,
    Conformance,
    FigureBlock,
    HeadingBlock,
    ListBlock,
    ListItem,
    Margin,
    Metadata,
    PageSettings,
    ParagraphBlock,
    RebuildRequest,
    Run,
    SimpleTableBlock,
    TableCell,
    TableRow,
)

# 1x1 transparent PNG
TINY_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def write_assets(asset_dir: Path) -> dict[str, str]:
    """Write the two standard fixture images; return {ref: absolute path}."""
    asset_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for ref in ("img-1", "img-2"):
        p = asset_dir / f"{ref}.png"
        p.write_bytes(TINY_PNG)
        paths[ref] = str(p)
    return paths


def default_content() -> list[Any]:
    return [
        HeadingBlock(level=1, runs=[Run(text="Sample Form")]),
        ParagraphBlock(runs=[Run(text="Intro text with "), Run(text="bold", bold=True), Run(text=" words.")]),
        ListBlock(
            ordered=False,
            items=[
                ListItem(label_runs=[Run(text="•")], body=[ParagraphBlock(runs=[Run(text="first item")])]),
                ListItem(label_runs=[Run(text="•")], body=[ParagraphBlock(runs=[Run(text="second item")])]),
            ],
        ),
        SimpleTableBlock(
            rows=[
                TableRow(cells=[TableCell(text="Name", header="col"), TableCell(text="Age", header="col")]),
                TableRow(cells=[TableCell(text="Alice"), TableCell(text="30")]),
            ]
        ),
        FigureBlock(asset_ref="img-1", alt="A tiny dot", caption=[Run(text="The caption")]),
        ArtifactBlock(asset_ref="img-2"),
    ]


def make_request(
    *,
    asset_dir: Path,
    content: list[Any] | None = None,
    assets: dict[str, AssetRef] | None = None,
    title: str = "Sample Form",
    language: str = "en-US",
) -> RebuildRequest:
    write_assets(asset_dir)
    if assets is None:
        assets = {
            "img-1": AssetRef(path=str(asset_dir / "img-1.png"), mime="image/png"),
            "img-2": AssetRef(path=str(asset_dir / "img-2.png"), mime="image/png"),
        }
    return RebuildRequest(
        metadata=Metadata(title=title, language=language),
        page=PageSettings(size="Letter", margin=Margin(top=0.75, right=0.75, bottom=0.75, left=0.75, unit="in")),
        conformance=Conformance(pdfua="PDFUA_1", pdfa=None),
        content=default_content() if content is None else content,
        assets=assets,
    )
