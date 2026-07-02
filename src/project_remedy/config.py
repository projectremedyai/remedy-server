"""Configuration for the PDF accessibility remediation engine.

Loads settings from a .env file and an optional config.yaml, exposing
them as typed dataclasses. Used by the HTTP API backend and any
direct consumers of the engine.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BrandingConfig:
    """Visual branding applied to generated HTML output.

    Generic HTML branding. Set name/colors/start_url per deployment.
    Callers supply any combination of fields; unset fields fall back
    to neutral defaults in ``campus_html_template``.
    """

    name: str = "Remedy Server"
    start_url: str = ""
    brand_primary: str = "#0b5ed7"
    brand_accent: str = "#fd7e14"
    brand_neutral: str = "#6c757d"
    # URL used as the href for the "Report an Accessibility Issue" link in
    # generated HTML. Typically a mailto: address; may also be a contact-form
    # URL. Empty string renders an inert link rather than raising KeyError.
    accessibility_email: str = ""


@dataclass(frozen=True)
class ProcessingConfig:
    """Knobs for the HTML conversion path."""

    html_workflow_enabled: bool = True
    max_concurrent_calls: int = 5
    max_retries: int = 3
    retry_backoff_base: float = 2.0


@dataclass(frozen=True)
class ValidationConfig:
    """HTML validation / remediation loop settings."""

    max_remediation_cycles: int = 3
    fail_on_serious: bool = True
    wave_api_key: str = ""
    wave_report_type: int = 3


@dataclass(frozen=True)
class APIConfig:
    """Settings for LLM / vision backends (Ollama Cloud or local Ollama)."""

    api_key: str = ""                        # OLLAMA_API_KEY (required for cloud)
    base_url: str = "https://ollama.com/v1"
    cluster_nodes: tuple[str, ...] = ()      # Additional Ollama node URLs
    vision_base_url: str = ""                # Dedicated vision endpoint
    vision_cluster_nodes: tuple[str, ...] = ()  # Additional vision node URLs
    vision_model: str = "kimi-k2.7-code:cloud"
    text_model: str = "kimi-k2.7-code:cloud"
    max_concurrent_calls: int = 5
    max_retries: int = 3
    retry_backoff_base: float = 2.0
    llm_backend: str = "ollama"
    liteparse_enabled: bool = False
    liteparse_bin: str = "lit"
    liteparse_timeout_seconds: float = 30.0
    liteparse_sample_pages: int = 3
    liteparse_text_rich_min_chars: int = 800
    liteparse_sparse_max_chars: int = 200
    escalation_backend: str = "ollama"
    escalation_base_url: str = ""
    escalation_model: str = "kimi-k2.7-code:cloud"
    quality_judge_backend: str = "ollama"
    quality_judge_base_url: str = ""
    quality_judge_model: str = "mistral-large-3:675b"
    behavioral_test_backend: str = "ollama"
    behavioral_test_model: str = "gemma4:31b-cloud"
    behavioral_test_cache_path: str = ""
    ollama_stream: bool = False
    ollama_reasoning_effort: str = "none"


@dataclass(frozen=True)
class OutputConfig:
    """Filesystem paths for engine artefacts."""

    output_dir: Path = Path("./output")
    log_dir: Path = Path("./logs")


@dataclass(frozen=True)
class PDFRemediationConfig:
    """Settings for the PDF-to-PDF remediation path."""

    enabled: bool = True
    verapdf_path: str = "/usr/local/bin/verapdf"
    use_programmatic_fixes: bool = True
    ghostscript_enabled: bool = False
    ghostscript_path: str = ""  # auto-detect via shutil.which("gs") if empty
    redistill_visual_tolerance: float = 0.05
    vision_planner_as_fallback: bool = False
    # Mode B residual font repair
    font_mode_b_enabled: bool = False
    font_mode_b_trigger_rules: tuple[str, ...] = (
        "7.21.4.1-1", "7.21.4.2-2", "7.21.7-1",
    )
    font_mode_b_use_checker_signals: bool = True
    # Simple-font replacement track
    simple_font_replacement_enabled: bool = False
    simple_font_replacement_trigger_rules: tuple[str, ...] = ("7.21.4.1-1",)
    simple_font_encoding_repair_enabled: bool = False
    # Mode A faithful-rebuild stage
    font_mode_a_enabled: bool = False
    font_mode_a_trigger_rules: tuple[str, ...] = (
        "7.1-1", "7.1-2", "7.1-3",
        "7.2-11", "7.2-14", "7.2-42", "7.2-43",
    )
    font_mode_a_visual_diff_threshold: float = 0.10


@dataclass(frozen=True)
class ContrastConfig:
    """Settings for PDF color-contrast remediation."""

    enabled: bool = True
    level: str = "AA"              # "AA" or "AAA"
    max_iterations: int = 3
    dpi: int = 150
    auto_fix: bool = True


@dataclass(frozen=True)
class RebuildConfig:
    """Full-rebuild tier configuration (semantic rebuild via QuestPDF sidecar)."""

    enabled: bool = True
    text_similarity_threshold: float = 0.85
    vision_concurrency: int = 4
    sidecar_timeout_s: float = 120.0
    markdown_parser: str = "markdown-it-py"
    backend: str = "questpdf"          # "questpdf" | "typst" (FR-1)
    typst_timeout_s: float = 120.0     # NFR-2: budgeted like sidecar_timeout_s


@dataclass
class PipelineConfig:
    """Top-level engine configuration."""

    api: APIConfig = field(default_factory=APIConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    pdf_remediation: PDFRemediationConfig = field(default_factory=PDFRemediationConfig)
    contrast: ContrastConfig = field(default_factory=ContrastConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    branding: BrandingConfig = field(default_factory=BrandingConfig)
    rebuild: RebuildConfig = field(default_factory=RebuildConfig)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.environ.get(key)
    return int(raw) if raw is not None else default


def _env_float(key: str, default: float) -> float:
    raw = os.environ.get(key)
    return float(raw) if raw is not None else default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.environ.get(key)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes")


def load_config(
    env_path: Path | None = None,
    yaml_path: Path | None = None,
) -> PipelineConfig:
    """Build a ``PipelineConfig`` by merging .env and YAML sources.

    Resolution order (last wins): dataclass defaults → config.yaml → env vars.
    """
    load_dotenv(env_path or Path(".env"), override=False)

    yml: dict[str, Any] = _load_yaml(yaml_path or Path("config.yaml"))

    api_yml = yml.get("api", {})
    output_yml = yml.get("output", {})
    pdf_rem_yml = yml.get("pdf_remediation", {})
    contrast_yml = yml.get("contrast", {})

    api = APIConfig(
        api_key=_env("OLLAMA_API_KEY", api_yml.get("api_key", "")),
        base_url=_env("OLLAMA_BASE_URL", api_yml.get("base_url", "https://ollama.com/v1")),
        cluster_nodes=tuple(
            _env("OLLAMA_CLUSTER_NODES", "").split(",")
            if _env("OLLAMA_CLUSTER_NODES")
            else api_yml.get("cluster_nodes", [])
        ),
        vision_base_url=_env("VISION_BASE_URL", api_yml.get("vision_base_url", "")),
        vision_cluster_nodes=tuple(
            _env("VISION_CLUSTER_NODES", "").split(",")
            if _env("VISION_CLUSTER_NODES")
            else api_yml.get("vision_cluster_nodes", [])
        ),
        vision_model=_env("OLLAMA_VISION_MODEL", api_yml.get("vision_model", "kimi-k2.7-code:cloud")),
        text_model=_env(
            "OLLAMA_MODEL",
            _env("OLLAMA_TEXT_MODEL", api_yml.get("text_model", "kimi-k2.7-code:cloud")),
        ),
        max_concurrent_calls=_env_int("MAX_CONCURRENT_API_CALLS", api_yml.get("max_concurrent_calls", 5)),
        max_retries=_env_int("MAX_RETRIES", api_yml.get("max_retries", 3)),
        retry_backoff_base=_env_float("RETRY_BACKOFF_BASE", api_yml.get("retry_backoff_base", 2.0)),
        llm_backend=_env("LLM_BACKEND", api_yml.get("llm_backend", "ollama")),
        liteparse_enabled=_env_bool("LITEPARSE_ENABLED", api_yml.get("liteparse_enabled", False)),
        liteparse_bin=_env("LITEPARSE_BIN", api_yml.get("liteparse_bin", "lit")),
        liteparse_timeout_seconds=_env_float("LITEPARSE_TIMEOUT_SECONDS", api_yml.get("liteparse_timeout_seconds", 30.0)),
        liteparse_sample_pages=_env_int("LITEPARSE_SAMPLE_PAGES", api_yml.get("liteparse_sample_pages", 3)),
        liteparse_text_rich_min_chars=_env_int("LITEPARSE_TEXT_RICH_MIN_CHARS", api_yml.get("liteparse_text_rich_min_chars", 800)),
        liteparse_sparse_max_chars=_env_int("LITEPARSE_SPARSE_MAX_CHARS", api_yml.get("liteparse_sparse_max_chars", 200)),
        escalation_backend=_env("OLLAMA_ESCALATION_BACKEND", _env("ESCALATION_BACKEND", api_yml.get("escalation_backend", "ollama"))),
        escalation_base_url=_env("ESCALATION_BASE_URL", api_yml.get("escalation_base_url", "")),
        escalation_model=_env(
            "OLLAMA_ESCALATION_MODEL",
            _env("ESCALATION_MODEL", api_yml.get("escalation_model", "kimi-k2.7-code:cloud")),
        ),
        quality_judge_backend=_env(
            "QUALITY_JUDGE_BACKEND",
            api_yml.get("quality_judge_backend", "ollama"),
        ),
        quality_judge_base_url=_env(
            "QUALITY_JUDGE_BASE_URL",
            api_yml.get("quality_judge_base_url", ""),
        ),
        quality_judge_model=_env(
            "QUALITY_JUDGE_MODEL",
            api_yml.get("quality_judge_model", "mistral-large-3:675b"),
        ),
        behavioral_test_backend=_env(
            "BEHAVIORAL_TEST_BACKEND",
            api_yml.get("behavioral_test_backend", "ollama"),
        ),
        behavioral_test_model=_env(
            "BEHAVIORAL_TEST_MODEL",
            api_yml.get("behavioral_test_model", "gemma4:31b-cloud"),
        ),
        behavioral_test_cache_path=_env(
            "BEHAVIORAL_TEST_CACHE_PATH",
            api_yml.get("behavioral_test_cache_path", ""),
        ),
        ollama_stream=_env_bool("OLLAMA_STREAM", api_yml.get("ollama_stream", False)),
        ollama_reasoning_effort=_env("OLLAMA_REASONING_EFFORT", api_yml.get("ollama_reasoning_effort", "low")),
    )

    output = OutputConfig(
        output_dir=Path(_env("OUTPUT_DIR", str(output_yml.get("output_dir", "./output")))),
        log_dir=Path(_env("LOG_DIR", str(output_yml.get("log_dir", "./logs")))),
    )

    pdf_remediation = PDFRemediationConfig(
        enabled=_env_bool("PDF_REMEDIATION_ENABLED", pdf_rem_yml.get("enabled", True)),
        verapdf_path=_env("VERAPDF_PATH", pdf_rem_yml.get("verapdf_path", "/usr/local/bin/verapdf")),
        use_programmatic_fixes=_env_bool("PDF_REMEDIATION_USE_PROGRAMMATIC_FIXES", pdf_rem_yml.get("use_programmatic_fixes", True)),
        ghostscript_enabled=_env_bool("GHOSTSCRIPT_ENABLED", pdf_rem_yml.get("ghostscript_enabled", False)),
        ghostscript_path=_env("GHOSTSCRIPT_PATH", pdf_rem_yml.get("ghostscript_path", "")),
        redistill_visual_tolerance=_env_float("REDISTILL_VISUAL_TOLERANCE", pdf_rem_yml.get("redistill_visual_tolerance", 0.05)),
        vision_planner_as_fallback=_env_bool("VISION_PLANNER_AS_FALLBACK", pdf_rem_yml.get("vision_planner_as_fallback", False)),
        font_mode_b_enabled=_env_bool("FONT_MODE_B_ENABLED", pdf_rem_yml.get("font_mode_b_enabled", False)),
        font_mode_b_trigger_rules=tuple(
            _env("FONT_MODE_B_TRIGGER_RULES", "").split(",")
            if _env("FONT_MODE_B_TRIGGER_RULES")
            else pdf_rem_yml.get("font_mode_b_trigger_rules", ("7.21.4.1-1", "7.21.4.2-2", "7.21.7-1"))
        ),
        font_mode_b_use_checker_signals=_env_bool("FONT_MODE_B_USE_CHECKER_SIGNALS", pdf_rem_yml.get("font_mode_b_use_checker_signals", True)),
        simple_font_replacement_enabled=_env_bool("SIMPLE_FONT_REPLACEMENT_ENABLED", pdf_rem_yml.get("simple_font_replacement_enabled", False)),
        simple_font_replacement_trigger_rules=tuple(
            _env("SIMPLE_FONT_REPLACEMENT_TRIGGER_RULES", "").split(",")
            if _env("SIMPLE_FONT_REPLACEMENT_TRIGGER_RULES")
            else pdf_rem_yml.get("simple_font_replacement_trigger_rules", ("7.21.4.1-1",))
        ),
        simple_font_encoding_repair_enabled=_env_bool("SIMPLE_FONT_ENCODING_REPAIR_ENABLED", pdf_rem_yml.get("simple_font_encoding_repair_enabled", False)),
        font_mode_a_enabled=_env_bool("FONT_MODE_A_ENABLED", pdf_rem_yml.get("font_mode_a_enabled", False)),
        font_mode_a_trigger_rules=tuple(
            _env("FONT_MODE_A_TRIGGER_RULES", "").split(",")
            if _env("FONT_MODE_A_TRIGGER_RULES")
            else pdf_rem_yml.get("font_mode_a_trigger_rules", (
                "7.1-1", "7.1-2", "7.1-3",
                "7.2-11", "7.2-14", "7.2-42", "7.2-43",
            ))
        ),
        font_mode_a_visual_diff_threshold=_env_float("FONT_MODE_A_VISUAL_DIFF_THRESHOLD", pdf_rem_yml.get("font_mode_a_visual_diff_threshold", 0.10)),
    )

    contrast = ContrastConfig(
        enabled=_env_bool("CONTRAST_ENABLED", contrast_yml.get("enabled", True)),
        level=_env("CONTRAST_LEVEL", contrast_yml.get("level", "AA")),
        max_iterations=_env_int("CONTRAST_MAX_ITERATIONS", contrast_yml.get("max_iterations", 3)),
        dpi=_env_int("CONTRAST_DPI", contrast_yml.get("dpi", 150)),
        auto_fix=_env_bool("CONTRAST_AUTO_FIX", contrast_yml.get("auto_fix", True)),
    )

    processing_yml = yml.get("processing", {})
    processing = ProcessingConfig(
        html_workflow_enabled=_env_bool("HTML_WORKFLOW_ENABLED", processing_yml.get("html_workflow_enabled", True)),
        max_concurrent_calls=_env_int("MAX_CONCURRENT_API_CALLS", processing_yml.get("max_concurrent_calls", 5)),
        max_retries=_env_int("MAX_RETRIES", processing_yml.get("max_retries", 3)),
        retry_backoff_base=_env_float("RETRY_BACKOFF_BASE", processing_yml.get("retry_backoff_base", 2.0)),
    )

    validation_yml = yml.get("validation", {})
    validation = ValidationConfig(
        max_remediation_cycles=_env_int("VALIDATION_MAX_REMEDIATION_CYCLES", validation_yml.get("max_remediation_cycles", 3)),
        fail_on_serious=_env_bool("VALIDATION_FAIL_ON_SERIOUS", validation_yml.get("fail_on_serious", True)),
        wave_api_key=_env("WAVE_API_KEY", validation_yml.get("wave_api_key", "")),
        wave_report_type=_env_int("WAVE_REPORT_TYPE", validation_yml.get("wave_report_type", 3)),
    )

    branding_yml = yml.get("branding", {})
    branding = BrandingConfig(
        name=_env("BRAND_NAME", branding_yml.get("name", "Remedy Server")),
        start_url=_env("BRAND_START_URL", branding_yml.get("start_url", "")),
        brand_primary=_env("BRAND_PRIMARY", branding_yml.get("brand_primary", "#0b5ed7")),
        brand_accent=_env("BRAND_ACCENT", branding_yml.get("brand_accent", "#fd7e14")),
        brand_neutral=_env("BRAND_NEUTRAL", branding_yml.get("brand_neutral", "#6c757d")),
    )

    rebuild_yml = yml.get("rebuild", {})
    rebuild = RebuildConfig(
        enabled=_env_bool("REBUILD_ENABLED", rebuild_yml.get("enabled", True)),
        text_similarity_threshold=_env_float(
            "REBUILD_TEXT_SIMILARITY_THRESHOLD",
            rebuild_yml.get("text_similarity_threshold", 0.85),
        ),
        vision_concurrency=_env_int(
            "REBUILD_VISION_CONCURRENCY",
            rebuild_yml.get("vision_concurrency", 4),
        ),
        sidecar_timeout_s=_env_float(
            "REBUILD_SIDECAR_TIMEOUT_S",
            rebuild_yml.get("sidecar_timeout_s", 120.0),
        ),
        markdown_parser=_env(
            "REBUILD_MARKDOWN_PARSER",
            rebuild_yml.get("markdown_parser", "markdown-it-py"),
        ),
        backend=_env("REBUILD_BACKEND", rebuild_yml.get("backend", "questpdf")),
        typst_timeout_s=_env_float(
            "REBUILD_TYPST_TIMEOUT_S",
            rebuild_yml.get("typst_timeout_s", 120.0),
        ),
    )

    return PipelineConfig(
        api=api,
        output=output,
        pdf_remediation=pdf_remediation,
        contrast=contrast,
        processing=processing,
        validation=validation,
        branding=branding,
        rebuild=rebuild,
    )
