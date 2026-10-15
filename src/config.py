"""Central configuration. Every path, provider and credential comes from the environment.

Nothing here is machine-specific: relative paths are resolved against the project root
(the directory that contains this package), and all values can be overridden in `.env`.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# Bump when extraction/normalisation logic changes so that ingestion knows the stored
# database is stale even if the PDFs themselves are unchanged.
PIPELINE_VERSION = "1.1.0"  # 1.1.0: page-level index of both volumes for the document chat

LLM_PROVIDERS = ("offline", "groq", "openrouter", "anthropic", "openai", "ollama")
# Providers that send the evidence bundle to a third-party service.
EXTERNAL_PROVIDERS = ("groq", "openrouter", "anthropic", "openai")

DEFAULT_MODELS = {
    "groq": "openai/gpt-oss-120b",
    "openrouter": "nvidia/nemotron-3-super-120b-a12b:free",
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-5.5",
    "ollama": "llama3.1:8b",
}
DEFAULT_KEY_ENVS = {
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}
DEFAULT_BASE_URLS = {"openrouter": "https://openrouter.ai/api/v1"}


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
    # Providers tried in order when the primary one fails, e.g. "openrouter" or "openrouter,ollama".
    llm_fallback_providers: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            p.strip().lower() for p in os.getenv("SANKET_LLM_FALLBACK_PROVIDERS", "").split(",") if p.strip()
        )
    )
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
    # --- document chat ---------------------------------------------------
    # Model for the chat page (defaults to SANKET_LLM_MODEL / the provider default).
    chat_model_name: str = field(default_factory=lambda: os.getenv("SANKET_CHAT_MODEL", "").strip())
    # Reasoning effort for reasoning-capable OpenAI models (none|low|medium|high); blank = provider default.
    chat_reasoning_effort: str = field(
        default_factory=lambda: os.getenv("SANKET_CHAT_REASONING_EFFORT", "low").strip().lower()
    )
    # Largest share of an answer that may be general knowledge rather than the PDFs.
    chat_max_general_share: float = field(
        default_factory=lambda: float(os.getenv("SANKET_CHAT_MAX_GENERAL_SHARE", "0.20"))
    )
    chat_passages: int = field(default_factory=lambda: int(os.getenv("SANKET_CHAT_PASSAGES", "8")))
    # Chat persistence, cache, verification and abuse limits
    chat_dir: Path = field(default_factory=lambda: _path(os.getenv("SANKET_CHAT_DIR", "data/chat")))
    redis_url: str = field(default_factory=lambda: os.getenv("SANKET_REDIS_URL", "").strip())
    cache_ttl_s: int = field(default_factory=lambda: int(os.getenv("SANKET_CACHE_TTL_S", "86400")))
    # fast     = instant checks after the answer streams: citations must exist and every number must occur
    #            in the cited passage (no extra model call; label "Citations checked").
    # thorough = fast checks plus an independent model check of every statement, with one redraft
    #            if too little survives (slower; label "Sources verified").
    # off      = no checks (label "Not checked").
    chat_verify_mode: str = field(
        default_factory=lambda: os.getenv("SANKET_CHAT_VERIFY_MODE", "fast").strip().lower())
    # Internet search for secondary facts (DuckDuckGo; only the question is sent, never PDF text).
    chat_web: bool = field(default_factory=lambda: _flag("SANKET_CHAT_WEB", False))
    chat_web_results: int = field(default_factory=lambda: int(os.getenv("SANKET_CHAT_WEB_RESULTS", "4")))
    chat_web_timeout_s: int = field(default_factory=lambda: int(os.getenv("SANKET_CHAT_WEB_TIMEOUT_S", "7")))
    chat_max_regenerations: int = field(default_factory=lambda: int(os.getenv("SANKET_CHAT_MAX_REGENERATIONS", "1")))
    chat_rate_limit_per_min: int = field(default_factory=lambda: int(os.getenv("SANKET_CHAT_RATE_LIMIT_PER_MIN", "12")))
    chat_max_message_chars: int = field(default_factory=lambda: int(os.getenv("SANKET_CHAT_MAX_MESSAGE_CHARS", "2000")))
    # Shared secret required by the HTTP API (src/api.py). The API refuses requests when unset.
    api_token: str = field(default_factory=lambda: os.getenv("SANKET_API_TOKEN", "").strip())
    log_level: str = field(default_factory=lambda: os.getenv("SANKET_LOG_LEVEL", "INFO").upper())

    @property
    def chat_verify(self) -> bool:
        """True when the slower model-based check runs (thorough mode)."""
        return self.chat_verify_mode == "thorough"

    def chat_model(self) -> str:
        return self.chat_model_name or self.default_model()

    # --- derived paths --------------------------------------------------
    @property
    def db_path(self) -> Path:
        return _path(os.getenv("SANKET_DB_PATH", str(self.processed_dir / "sanket.sqlite3")))

    @property
    def vector_dir(self) -> Path:
        return _path(os.getenv("SANKET_VECTOR_DIR", str(self.processed_dir / "chroma")))

    @property
    def chat_db_path(self) -> Path:
        return self.chat_dir / "chat.sqlite3"

    @property
    def cache_path(self) -> Path:
        return self.chat_dir / "cache.sqlite3"

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
        return self.llm_model or DEFAULT_MODELS.get(self.llm_provider, "")

    def api_key(self) -> str | None:
        env = self.llm_api_key_env or DEFAULT_KEY_ENVS.get(self.llm_provider, "")
        return os.getenv(env) if env else None

    def base_url(self) -> str | None:
        return self.llm_base_url or DEFAULT_BASE_URLS.get(self.llm_provider)

    def for_provider(self, provider: str) -> "Settings":
        """Settings for a fallback provider: its own model, key variable and endpoint.

        Per-provider overrides: SANKET_<PROVIDER>_MODEL, SANKET_<PROVIDER>_API_KEY_ENV,
        SANKET_<PROVIDER>_BASE_URL (for example SANKET_OPENROUTER_MODEL).
        """
        p = provider.strip().lower()
        prefix = f"SANKET_{p.upper()}_"
        return replace(
            self, llm_provider=p,
            llm_model=os.getenv(prefix + "MODEL", "").strip(),
            llm_api_key_env=os.getenv(prefix + "API_KEY_ENV", "").strip(),
            llm_base_url=os.getenv(prefix + "BASE_URL", "").strip(),
            chat_model_name="",
        )

    def provider_chain(self) -> list["Settings"]:
        """Primary provider followed by the fallbacks in SANKET_LLM_FALLBACK_PROVIDERS."""
        chain = [self]
        for name in self.llm_fallback_providers:
            if name and name != self.llm_provider and name in LLM_PROVIDERS and name != "offline":
                chain.append(self.for_provider(name))
        return chain


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
