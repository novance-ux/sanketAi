"""Central configuration. Every path, provider and credential comes from the environment.

Nothing here is machine-specific: relative paths are resolved against the project root
(the directory that contains this package), and all values can be overridden in `.env`.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# Bump when extraction/normalisation logic changes so that ingestion knows the stored
# database is stale even if the PDFs themselves are unchanged.
PIPELINE_VERSION = "1.0.0"

LLM_PROVIDERS = ("offline", "anthropic", "openai", "ollama")
# Providers that send the evidence bundle to a third-party service.
EXTERNAL_PROVIDERS = ("anthropic", "openai")


def _path(value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def _flag(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # --- data locations -------------------------------------------------
    pdf_dirs: tuple[Path, ...] = field(
        default_factory=lambda: tuple(
            _path(p) for p in os.getenv("SANKET_PDF_DIRS", "data/raw;.").split(";") if p.strip()
        )
    )
    processed_dir: Path = field(
        default_factory=lambda: _path(os.getenv("SANKET_PROCESSED_DIR", "data/processed"))
    )
    context_dir: Path = field(
        default_factory=lambda: _path(os.getenv("SANKET_CONTEXT_DIR", "data/context"))
    )
    alias_file: Path = field(
        default_factory=lambda: _path(os.getenv("SANKET_ALIAS_FILE", "config/mine_aliases.json"))
    )
    # --- retrieval ------------------------------------------------------
    # "onnx-minilm" = local all-MiniLM-L6-v2 via onnxruntime (no data leaves the machine;
    # the model file is downloaded once). "none" disables the vector index.
    embedding_backend: str = field(
        default_factory=lambda: os.getenv("SANKET_EMBEDDING_BACKEND", "onnx-minilm").strip().lower()
    )
    # --- LLM ------------------------------------------------------------
    llm_provider: str = field(
        default_factory=lambda: os.getenv("SANKET_LLM_PROVIDER", "offline").strip().lower()
    )
    llm_model: str = field(default_factory=lambda: os.getenv("SANKET_LLM_MODEL", "").strip())
    # Name of the environment variable that holds the API key (never the key itself).
    llm_api_key_env: str = field(
        default_factory=lambda: os.getenv("SANKET_LLM_API_KEY_ENV", "").strip()
    )
    llm_base_url: str = field(default_factory=lambda: os.getenv("SANKET_LLM_BASE_URL", "").strip())
    llm_max_tokens: int = field(
        default_factory=lambda: int(os.getenv("SANKET_LLM_MAX_TOKENS", "8000"))
    )
    llm_timeout_s: int = field(default_factory=lambda: int(os.getenv("SANKET_LLM_TIMEOUT_S", "180")))
    # Explicit opt-in before any Sanket text is sent to a third-party provider.
    allow_external_llm: bool = field(default_factory=lambda: _flag("SANKET_ALLOW_EXTERNAL_LLM"))
    # Strip victims' personal names from the evidence sent to the model.
    redact_person_names: bool = field(
        default_factory=lambda: _flag("SANKET_REDACT_PERSON_NAMES", True)
    )
    log_level: str = field(default_factory=lambda: os.getenv("SANKET_LOG_LEVEL", "INFO").upper())

    # --- derived paths --------------------------------------------------
    @property
    def db_path(self) -> Path:
        return _path(os.getenv("SANKET_DB_PATH", str(self.processed_dir / "sanket.sqlite3")))

    @property
    def vector_dir(self) -> Path:
        return _path(os.getenv("SANKET_VECTOR_DIR", str(self.processed_dir / "chroma")))

    @property
    def report_json(self) -> Path:
        return self.processed_dir / "extraction_report.json"

    @property
    def report_md(self) -> Path:
        return self.processed_dir / "extraction_audit.md"

    @property
    def llm_is_external(self) -> bool:
        return self.llm_provider in EXTERNAL_PROVIDERS

    def default_model(self) -> str:
        if self.llm_model:
            return self.llm_model
        return {
            "anthropic": "claude-opus-5-5",
            "openai": "gpt-4o",
            "ollama": "llama3.1:8b",
        }.get(self.llm_provider, "")

    def api_key(self) -> str | None:
        env = self.llm_api_key_env or {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
        }.get(self.llm_provider, "")
        return os.getenv(env) if env else None


def get_settings() -> Settings:
    """Build settings from the current environment (cheap; no caching so tests can patch env)."""
    return Settings()


def configure_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=getattr(logging, level or get_settings().log_level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "chromadb", "onnxruntime", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
