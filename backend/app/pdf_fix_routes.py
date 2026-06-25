"""/v1/pdf/* mutation endpoints (Phase D).

These run the engine's fix/rebuild/redistill/ocr/contrast paths. Short
operations return the modified PDF inline; longer ones return a job
``id`` instead.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse
from slowapi import Limiter
from starlette.background import BackgroundTask


# Modes wired up in faithful_rebuild today. Keep in sync with
# project_remedy.faithful_rebuild.pipeline.faithful_rebuild.
_REBUILD_MODES: frozenset[str] = frozenset({"preserving"})


def _cleanup(path: Path) -> BackgroundTask:
    """BackgroundTask that deletes ``path`` after the response is sent.

    Use on FileResponse for inline outputs the client doesn't re-fetch by token
    (otherwise the file accumulates under job_dir).
    """
    return BackgroundTask(lambda: path.unlink(missing_ok=True))

from backend.app.auth import require_api_key_dependency
from backend.app.config import Settings


_PDF_MAGIC = b"%PDF-"


def _serialize_issue(issue: Any) -> dict[str, Any]:
    if hasattr(issue, "model_dump"):
        return issue.model_dump(mode="json")
    if hasattr(issue, "dict"):
        return issue.dict()
    return asdict(issue)


async def _stage_pdf(file: UploadFile, settings: Settings) -> Path:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix != ".pdf":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .pdf accepted.",
        )
    max_bytes = settings.max_upload_mb * 1024 * 1024
    contents = await file.read(max_bytes + 1)
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"File exceeds max upload size ({settings.max_upload_mb} MB).",
        )
    if not contents.startswith(_PDF_MAGIC):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Not a valid PDF.",
        )
    settings.job_dir.mkdir(parents=True, exist_ok=True)
    staging = settings.job_dir / f"_pdf-{uuid.uuid4().hex}.pdf"
    staging.write_bytes(contents)
    return staging


def build_router(settings: Settings, limiter: Limiter, upload_rate_limit: str) -> APIRouter:
    router = APIRouter(prefix="/v1/pdf")
    require_key = Depends(require_api_key_dependency(settings))
    upload_limit = limiter.limit(upload_rate_limit)

    # ------------------------------------------------------------------
    # /fix — apply all auto-fixes, return JSON report + download URL
    # ------------------------------------------------------------------

    @router.post("/fix", dependencies=[require_key])
    @upload_limit
    async def fix_all_rules(request: Request, file: UploadFile = File(...)) -> JSONResponse:
        from project_remedy.config import load_config
        from project_remedy.pdf_fixer import fix_and_verify

        src = await _stage_pdf(file, settings)
        dst = src.with_suffix(".fixed.pdf")
        try:
            cfg = load_config()
            # TODO(REMEDY-57): thread a vision_result here so vision-aware
            # checks (reading order, contrast) don't silently degrade to
            # "Manual Check Needed". Requires either accepting a precomputed
            # vision result on the request or invoking the configured vision
            # provider internally — both involve coordination with
            # project_remedy.pdf_fixer.fix_and_verify's signature.
            report = await asyncio.to_thread(
                fix_and_verify, src, dst, config=cfg, original_path=src,
            )
            return JSONResponse({
                "changes": report.changes,
                "fixes_applied": len(report.changes),
                "output_file_size": dst.stat().st_size if dst.exists() else 0,
                # Client does a follow-up fetch via /v1/pdf/fix/download?token=<name>
                "download_token": dst.name,
            })
        finally:
            # Keep dst for a subsequent download call; routed via /fix/download.
            src.unlink(missing_ok=True)

    @router.get("/fix/download/{token}", dependencies=[require_key])
    async def fix_download(token: str) -> FileResponse:
        """Download the PDF produced by the last /fix call, identified by token."""
        # Only serve from within settings.job_dir, and only if it ends in .fixed.pdf.
        if not token.endswith(".fixed.pdf") or "/" in token or "\\" in token:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Bad token.")
        path = settings.job_dir / token
        if not path.exists():
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No such fixed PDF.")
        return FileResponse(path, media_type="application/pdf", filename=token)

    # ------------------------------------------------------------------
    # /fix/{rule_id} — apply a single fix rule, return PDF inline
    # ------------------------------------------------------------------

    @router.post("/fix/{rule_id}", dependencies=[require_key])
    @upload_limit
    async def fix_single_rule(
        request: Request,
        rule_id: str,
        file: UploadFile = File(...),
    ) -> FileResponse:
        import pikepdf
        from project_remedy.pdf_fixer import ALL_FIXES

        # Normalize rule lookup
        fix_fn = None
        for entry in ALL_FIXES:
            if entry[0] == rule_id:
                fix_fn = entry[1]
                break
        if fix_fn is None:
            known = [e[0] for e in ALL_FIXES]
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown rule_id. Known: {', '.join(known[:10])}...",
            )

        src = await _stage_pdf(file, settings)
        dst = src.with_suffix(".fixed.pdf")
        try:
            def _apply():
                with pikepdf.open(src) as pdf:
                    fix_fn(pdf)
                    pdf.save(dst)
            await asyncio.to_thread(_apply)
            return FileResponse(
                dst, media_type="application/pdf",
                filename=f"{Path(file.filename or 'out').stem}_{rule_id}.pdf",
                background=_cleanup(dst),
            )
        finally:
            src.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # /vision/alt-text — generate alt text for a single uploaded image
    # ------------------------------------------------------------------

    @router.post("/vision/alt-text", dependencies=[require_key])
    @upload_limit
    async def vision_alt_text(request: Request, file: UploadFile = File(...)) -> JSONResponse:
        from project_remedy.config import load_config
        from project_remedy.pdf_vision import create_provider_from_config

        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Accepts .png/.jpg/.jpeg/.webp/.gif.",
            )
        max_bytes = settings.max_upload_mb * 1024 * 1024
        img_bytes = await file.read(max_bytes + 1)
        if len(img_bytes) > max_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"File exceeds max upload size ({settings.max_upload_mb} MB).",
            )
        if not img_bytes:
            raise HTTPException(status_code=400, detail="Empty upload.")

        cfg = load_config()
        provider = create_provider_from_config(cfg)
        if provider is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No vision provider configured. Set OLLAMA_* env.",
            )
        settings.job_dir.mkdir(parents=True, exist_ok=True)
        img_path = settings.job_dir / f"_vision-{uuid.uuid4().hex}{suffix}"
        img_path.write_bytes(img_bytes)
        try:
            from project_remedy.vision_prompts import figure_alt_prompt
            result = await provider.analyze_image(img_path, prompt=figure_alt_prompt())
            return JSONResponse({
                "alt_text": result if isinstance(result, str) else str(result),
            })
        except HTTPException:
            raise
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Vision provider error: {exc}",
            )
        finally:
            img_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # /contrast/audit + /contrast/fix
    # ------------------------------------------------------------------

    @router.post("/contrast/audit", dependencies=[require_key])
    @upload_limit
    async def contrast_audit(request: Request, file: UploadFile = File(...)) -> JSONResponse:
        from project_remedy.config import load_config
        from project_remedy.contrast.detector import ContrastDetector
        from project_remedy.pdf_vision import create_provider_from_config

        src = await _stage_pdf(file, settings)
        try:
            cfg = load_config()
            provider = create_provider_from_config(cfg)
            if provider is None:
                raise HTTPException(
                    status_code=503,
                    detail="No vision provider configured.",
                )
            detector = ContrastDetector(provider, dpi=cfg.contrast.dpi)
            issues = await detector.detect_document(str(src), level=cfg.contrast.level)
            return JSONResponse({
                "issues": [_serialize_issue(i) for i in issues],
                "count": len(issues),
                "level": cfg.contrast.level,
            })
        finally:
            src.unlink(missing_ok=True)

    @router.post("/contrast/fix", dependencies=[require_key])
    @upload_limit
    async def contrast_fix(request: Request, file: UploadFile = File(...)) -> FileResponse:
        from project_remedy.config import load_config
        from project_remedy.contrast.remediator import ContrastRemediator
        from project_remedy.pdf_vision import create_provider_from_config

        src = await _stage_pdf(file, settings)
        dst = src.with_suffix(".contrast.pdf")
        try:
            cfg = load_config()
            provider = create_provider_from_config(cfg)
            if provider is None:
                raise HTTPException(status_code=503, detail="No vision provider configured.")
            remediator = ContrastRemediator(provider, dpi=cfg.contrast.dpi)
            await remediator.remediate_document(str(src), str(dst), level=cfg.contrast.level)
            if not dst.exists():
                raise HTTPException(status_code=500, detail="Contrast fix did not produce output.")
            return FileResponse(dst, media_type="application/pdf", background=_cleanup(dst))
        finally:
            src.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # /rebuild — hybrid faithful rebuild (mode A / B / simple-font)
    # ------------------------------------------------------------------

    @router.post("/rebuild", dependencies=[require_key])
    @upload_limit
    async def rebuild(
        request: Request,
        mode: str = Form("preserving"),
        file: UploadFile = File(...),
    ) -> JSONResponse:
        """Faithful PDF rebuild. ``mode`` accepts the values in ``_REBUILD_MODES``."""
        from project_remedy.config import load_config
        from project_remedy.faithful_rebuild.pipeline import faithful_rebuild

        if mode not in _REBUILD_MODES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported rebuild mode. Known: {', '.join(sorted(_REBUILD_MODES))}.",
            )

        src = await _stage_pdf(file, settings)
        dst = src.with_suffix(f".rebuilt.{mode}.pdf")
        try:
            cfg = load_config()
            result = await asyncio.to_thread(
                faithful_rebuild, src, dst, config=cfg, force_mode=mode,
            )
            # TODO(REMEDY-69): after rebuild, run the Tier-1 acceptance + ACR
            # workflow against ``dst`` and include the post-rebuild acceptance
            # state in the response so callers don't have to chain a second
            # call to /fix to learn whether the rebuilt artifact passed.
            return JSONResponse({
                "mode": getattr(result, "mode", mode),
                "success": bool(getattr(result, "success", dst.exists())),
                "error": getattr(result, "error", "") or "",
                "output_file_size": dst.stat().st_size if dst.exists() else 0,
                "download_token": dst.name if dst.exists() else "",
            })
        finally:
            src.unlink(missing_ok=True)

    @router.get("/rebuild/download/{token}", dependencies=[require_key])
    async def rebuild_download(token: str) -> FileResponse:
        # Only accept tokens produced by /rebuild: ``_pdf-<uuid>.rebuilt.<mode>.pdf``.
        # The narrower shape blocks downloads of unrelated PDFs that happen to
        # sit in job_dir (e.g. another route's *.fixed.pdf / *.contrast.pdf).
        if (
            "/" in token
            or "\\" in token
            or not token.startswith("_pdf-")
            or ".rebuilt." not in token
            or not token.endswith(".pdf")
        ):
            raise HTTPException(status_code=400, detail="Bad token.")
        path = settings.job_dir / token
        # Defense-in-depth: verify the resolved path is still inside job_dir.
        try:
            path.resolve(strict=False).relative_to(settings.job_dir.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="Bad token.")
        if not path.exists():
            raise HTTPException(status_code=404, detail="No such rebuilt PDF.")
        return FileResponse(path, media_type="application/pdf")

    # ------------------------------------------------------------------
    # /redistill — Ghostscript redistillation
    # ------------------------------------------------------------------

    @router.post("/redistill", dependencies=[require_key])
    @upload_limit
    async def redistill(
        request: Request,
        use_ocr: bool = Form(False),
        file: UploadFile = File(...),
    ) -> FileResponse:
        """Ghostscript redistill. Set ``use_ocr=true`` to enable OCR salvage."""
        from project_remedy.config import load_config
        from project_remedy.pdf_ghostscript import redistill_pdf

        src = await _stage_pdf(file, settings)
        dst = src.with_suffix(".redistilled.pdf")
        try:
            cfg = load_config()
            await asyncio.to_thread(
                redistill_pdf, src, dst, config=cfg, use_ocr=use_ocr,
            )
            if not dst.exists():
                raise HTTPException(
                    status_code=500,
                    detail="Redistill did not produce output. Check GHOSTSCRIPT_ENABLED + GHOSTSCRIPT_PATH.",
                )
            return FileResponse(dst, media_type="application/pdf", background=_cleanup(dst))
        finally:
            src.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # /ocr — OCR escalation path
    # ------------------------------------------------------------------

    @router.post("/ocr", dependencies=[require_key])
    @upload_limit
    async def ocr(request: Request, file: UploadFile = File(...)) -> FileResponse:
        """OCR via ``ocrmypdf`` subprocess. Requires ``ocrmypdf`` on PATH."""
        import shutil as _shutil
        import subprocess

        src = await _stage_pdf(file, settings)  # validate PDF first
        if _shutil.which("ocrmypdf") is None:
            src.unlink(missing_ok=True)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="ocrmypdf not installed. `brew install ocrmypdf` or "
                       "`apt install ocrmypdf`, then retry.",
            )


        dst = src.with_suffix(".ocr.pdf")
        try:
            def _run():
                return subprocess.run(
                    ["ocrmypdf", "--skip-text", "--output-type", "pdf",
                     str(src), str(dst)],
                    capture_output=True, text=True, timeout=600,
                )
            proc = await asyncio.to_thread(_run)
            if proc.returncode != 0 or not dst.exists():
                raise HTTPException(
                    status_code=500,
                    detail=f"ocrmypdf failed (exit {proc.returncode}): {proc.stderr[:500]}",
                )
            return FileResponse(dst, media_type="application/pdf", background=_cleanup(dst))
        finally:
            src.unlink(missing_ok=True)

    return router
