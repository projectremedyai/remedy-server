"""Typst rebuild renderer (PRD_typst_backend.md §5.2).

Same effective contract as QuestPdfSidecar.render(): RebuildRequest -> PDF
bytes. Unlike the sidecar's stdin/stdout protocol, Typst compiles generated
source from a job-scoped temp directory. `typst compile --pdf-standard ua-1`
is itself a hard accessibility gate (missing image/equation alt text fails
the build) — surfaced verbatim via TypstCompileError (NFR-4).
"""

from __future__ import annotations

import asyncio
import pathlib
import shutil
import tempfile
from dataclasses import dataclass

from project_remedy.rebuild.ast import RebuildRequest
from project_remedy.rebuild.typst_generator import GeneratorError, generate


class TypstError(RuntimeError):
    """Base class for Typst renderer failures."""


class TypstNotAvailable(TypstError):
    """No typst binary on PATH / configured."""


class TypstCompileError(TypstError):
    """typst compile exited non-zero; message carries stderr verbatim."""


class TypstTimeout(TypstError):
    """typst compile exceeded the configured timeout."""


class TypstUnsupportedConstruct(TypstError):
    """The RebuildRequest needs constructs the Typst backend cannot yet render safely (FR-6)."""


_EXT_BY_MIME = {"image/png": ".png", "image/jpeg": ".jpg"}


def resolve_typst_binary() -> pathlib.Path | None:
    found = shutil.which("typst")
    return pathlib.Path(found) if found else None


@dataclass
class TypstRenderer:
    binary_path: pathlib.Path
    timeout_s: float = 120.0

    async def render(self, request: RebuildRequest) -> bytes:
        with tempfile.TemporaryDirectory(prefix="typst-rebuild-") as tmp:
            tmpdir = pathlib.Path(tmp)
            asset_paths: dict[str, str] = {}
            for ref, asset in request.assets.items():
                filename = f"{ref}{_EXT_BY_MIME[asset.mime]}"
                shutil.copyfile(asset.path, tmpdir / filename)
                asset_paths[ref] = filename
            try:
                source = generate(request, asset_paths=asset_paths)
            except GeneratorError as exc:
                raise TypstUnsupportedConstruct(str(exc)) from exc
            (tmpdir / "main.typ").write_text(source, encoding="utf-8")
            return await self._compile(tmpdir)

    async def _compile(self, tmpdir: pathlib.Path) -> bytes:
        argv = [str(self.binary_path), "compile", "main.typ", "out.pdf", "--pdf-standard", "ua-1"]
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=str(tmpdir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            _, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_s)
        except asyncio.TimeoutError as exc:
            proc.kill()
            await proc.wait()
            raise TypstTimeout(f"typst compile timed out after {self.timeout_s}s") from exc
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace").strip()
            raise TypstCompileError(f"typst exited {proc.returncode}: {err or '<no stderr>'}")
        out = tmpdir / "out.pdf"
        if not out.exists():
            raise TypstCompileError("typst exited 0 but produced no out.pdf")
        pdf = out.read_bytes()
        if not pdf.startswith(b"%PDF"):
            raise TypstCompileError("typst output is not a PDF (missing %PDF magic bytes)")
        return pdf
