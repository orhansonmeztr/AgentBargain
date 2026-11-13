"""
AgentBargain - Centralized configuration via pydantic-settings.

All environment variables are managed from this module.
Values are loaded from a .env file or system environment variables.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root: 3 levels up from this file (src/agent_bargain/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

logger = logging.getLogger(__name__)

LLMProvider = Literal["google", "openai", "deepseek", "ollama"]

#: ``provider/`` prefixes accepted in a model string, mapped to their provider.
#: Single source of truth for both routing (``detect_provider``) and for
#: stripping the prefix before the name is handed to a provider SDK.
PROVIDER_PREFIXES: dict[str, LLMProvider] = {
    "ollama/": "ollama",
    "google/": "google",
    "gemini/": "google",
    "openai/": "openai",
    # Explicit prefix only. Bare "deepseek-..." names stay on Ollama, where
    # models like "deepseek-r1:1.5b" run locally under that very name.
    "deepseek/": "deepseek",
}


class Settings(BaseSettings):
    """Application-wide settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        # NOTE: ``populate_by_name`` must stay OFF. With it enabled
        # pydantic-settings *also* reads each field's own name from the
        # environment, which would re-expose the bare ``DEBUG`` variable that
        # ``debug``'s validation_alias exists to avoid.
    )

    # -- LLM Providers -------------------------------------------------
    google_api_key: str = Field(default="", description="Google Gemini API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    deepseek_api_key: str = Field(default="", description="DeepSeek API key")
    deepseek_base_url: str = Field(
        default="https://api.deepseek.com",
        description="DeepSeek API base URL (OpenAI-compatible)",
    )
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama server base URL",
    )

    # -- Model Configuration -------------------------------------------
    # Defaults are kept in sync with .env.example so the app behaves the
    # same whether or not a .env file is present.
    negotiation_model: str = Field(
        default="ollama/gemma3:1b",
    )
    negotiation_temperature: float = Field(
        default=0.4,
        description="Temperature for negotiation (0.2-0.5 is recommended for structured outputs)",
    )
    evaluation_model: str = Field(
        default="ollama/qwen3:0.6b",
    )
    evaluation_temperature: float = Field(
        default=0.0,
        description="Temperature for evaluation (0.0 is recommended for deterministic analysis)",
    )
    fallback_model: str = Field(
        default="ollama/llama3.2:1b",
    )
    fallback_temperature: float = Field(
        default=0.0,
        description="Temperature for fallback model",
    )

    # -- Database ------------------------------------------------------
    sqlite_db_path: str = Field(
        default="data/agent_bargain.db",
        description="Path to SQLite database file",
    )
    chroma_persist_dir: str = Field(
        default="data/chroma_data",
        description="ChromaDB persistence directory",
    )
    vector_backend: Literal["chroma"] = Field(
        default="chroma",
        description=(
            "Vector store backend. Only 'chroma' ships with this release; "
            "additional backends must implement the VectorStore protocol "
            "and be registered in create_vector_store()."
        ),
    )

    # -- Server --------------------------------------------------------
    #: Loopback by default: combined with authentication being off by
    #: default, ``0.0.0.0`` exposed an open write API on every interface. The
    #: Docker image sets ``0.0.0.0`` itself, where the container boundary and
    #: the startup check below apply.
    server_host: str = Field(default="127.0.0.1", description="Server bind host")
    server_port: int = Field(default=8000, description="Server bind port")
    # Deliberately NOT read from bare ``DEBUG``. That name is claimed by build
    # toolchains (e.g. ``DEBUG=release``), and because ``settings`` is built at
    # import time such a value used to make the whole package unimportable.
    debug: bool = Field(
        default=False,
        validation_alias="APP_DEBUG",
        description="Enable debug mode (hot reload). Env var: APP_DEBUG.",
    )
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins",
    )

    # -- Authentication ------------------------------------------------
    #: Shared API keys accepted on ``/api/v1``. Env var ``API_KEYS`` is a JSON
    #: list, e.g. ``API_KEYS='["secret-1","secret-2"]'``.
    #:
    #: **Empty means authentication is disabled.** That is deliberate so local
    #: development needs no setup, but it also means anyone who can reach the
    #: port can open negotiations and thereby write to the reputation ledger.
    #: ``create_app`` logs a warning at startup when this is empty, and a
    #: stronger one when the server is also bound to a non-loopback host.
    api_keys: list[str] = Field(
        default_factory=list,
        description="API keys accepted on /api/v1; empty disables authentication",
    )
    #: Operator keys. Accepted everywhere a regular key is, and additionally
    #: required by the per-negotiation analytics routes, which reveal both
    #: parties' private limits (seller floor, buyer budget). Env var
    #: ``ADMIN_API_KEYS``, a JSON list like ``API_KEYS``.
    #:
    #: Fails closed: with authentication on and this empty, those routes answer
    #: 403 to everyone rather than falling back to the shared keys.
    admin_api_keys: list[str] = Field(
        default_factory=list,
        description="Operator keys; required for analytics that reveal private limits",
    )
    #: Explicit opt-in to serving without keys on a non-loopback host. Without
    #: it, startup fails rather than exposing an open API (see
    #: ``api.auth.require_auth_when_exposed``).
    allow_unauthenticated: bool = Field(
        default=False,
        description="Allow a non-loopback SERVER_HOST with authentication disabled",
    )

    # -- Negotiation Parameters ----------------------------------------
    max_negotiation_rounds: int = Field(
        default=10,
        # Same bounds as the per-request ``max_rounds``. 0 timed every
        # negotiation out before a single exchange; negative was nonsense.
        ge=1,
        le=1000,
        description="Maximum negotiation rounds before timeout",
    )
    negotiation_timeout_seconds: int = Field(
        default=120,
        # 0 or less made asyncio.wait_for fail every negotiation immediately.
        ge=1,
        description="Negotiation timeout in seconds",
    )
    #: Generous on purpose. A warm local evaluator answers in ~20s, but the
    #: first evaluation after a negotiation often has to *load* the model
    #: (Ollama swaps out the negotiation model), and 60s timed that out —
    #: silently dropping the feedback for a deal that was struck.
    evaluation_timeout_seconds: int = Field(
        default=180,
        ge=1,
        description="Post-trade evaluation LLM timeout in seconds; on expiry no feedback is recorded",
    )

    # -- Observability -------------------------------------------------
    #: Root log level. Without explicit configuration Python's default is
    #: WARNING, which silently discarded every ``logger.info`` in the app —
    #: including the startup line confirming whether authentication is on.
    log_level: str = Field(default="INFO", description="Root logging level")
    langsmith_api_key: str = Field(default="", description="LangSmith API key")
    langsmith_project: str = Field(
        default="agent-bargain",
        description="LangSmith project name",
    )
    langsmith_tracing: bool = Field(
        default=False,
        description="Enable LangSmith tracing",
    )

    @property
    def sqlite_db_full_path(self) -> str:
        """Resolve the SQLite DB path relative to project root.

        Returns a string because ``:memory:`` is a special SQLite token
        that must be passed as-is (not wrapped in a ``Path``).
        """
        if self.sqlite_db_path == ":memory:":
            return ":memory:"
        path = Path(self.sqlite_db_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    @property
    def chroma_persist_full_path(self) -> Path:
        """Resolve the ChromaDB persist dir relative to project root."""
        path = Path(self.chroma_persist_dir)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def detect_provider(model_name: str) -> LLMProvider:
        """Determine the LLM provider from a model name string.

        Supports explicit prefix format (e.g., "ollama/gemma3:1b") as well as
        bare model names where the provider is inferred from the name itself.
        """
        name = model_name.lower()

        # Explicit provider prefix takes highest priority
        for prefix, provider in PROVIDER_PREFIXES.items():
            if name.startswith(prefix):
                return provider

        # Infer from model name patterns (bare names, no prefix)
        if name.startswith("gemini"):
            return "google"
        if name.startswith(("gpt-", "o1-", "o3-", "o4-")):
            return "openai"

        # Default to Ollama for unknown names
        return "ollama"

    @staticmethod
    def strip_provider_prefix(model_name: str) -> str:
        """Remove the ``provider/`` routing prefix, keeping the name's own case.

        The prefix is an AgentBargain routing convention; provider SDKs expect
        the bare model id. Matching is case-insensitive (``detect_provider``
        lowercases too) but the remainder is returned verbatim.
        """
        lowered = model_name.lower()
        for prefix in PROVIDER_PREFIXES:
            if lowered.startswith(prefix):
                return model_name[len(prefix) :]
        return model_name


# Singleton instance - import this throughout the app
settings = Settings()


def configure_logging() -> None:
    """Install a root handler at ``settings.log_level``.

    Call once at startup (from the FastAPI lifespan), not at import time, so
    tests keep control of log capture.

    Without this the root logger has no handler and Python falls back to
    "last resort" WARNING output — every ``logger.info`` in the application was
    silently discarded, including the startup line that confirms whether API
    authentication is enabled and the per-negotiation reputation/engine lines.

    A handler is added only when the root has none, and an existing handler is
    never removed: ``logging.basicConfig(force=True)`` would strip the handler
    pytest's ``caplog`` installs, and would override a supervisor (gunicorn and
    friends) that deliberately configured logging itself. Uvicorn's own
    ``uvicorn.*`` loggers are left alone either way.
    """
    level = getattr(logging, settings.log_level.upper(), None)
    if not isinstance(level, int):
        logger.warning("Unknown LOG_LEVEL %r — falling back to INFO.", settings.log_level)
        level = logging.INFO

    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(levelname)-8s %(asctime)s %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)
    root.setLevel(level)

    # Uvicorn logs the full request target, so a WebSocket's ?api_key=<key>
    # landed in plain text in every access log line (and `docker logs`).
    for name in _URL_LOGGING_LOGGERS:
        target = logging.getLogger(name)
        if not any(isinstance(f, RedactSecretsFilter) for f in target.filters):
            target.addFilter(RedactSecretsFilter())


#: Loggers that print request URLs: uvicorn's access log, and uvicorn.error,
#: which writes the ``"WebSocket <path>" [accepted]`` line.
_URL_LOGGING_LOGGERS = ("uvicorn.access", "uvicorn.error")

#: ``api_key=<value>`` in a query string, up to the next separator.
_API_KEY_IN_URL = re.compile(r"(?i)([?&]api_key=)[^&\s\"']+")


class RedactSecretsFilter(logging.Filter):
    """Mask ``api_key=`` query values in a log record before any handler sees it."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = _API_KEY_IN_URL.sub(r"\1***", message)
        if redacted != message:
            # Freeze the rendered text; the args held the secret.
            record.msg, record.args = redacted, None
        return True


def _clear_langsmith_env_caches() -> None:
    """Drop ``langsmith``'s cached environment reads, if the SDK exposes them.

    ``get_env_var`` is ``lru_cache``-wrapped, so a value read during import
    would survive the environment changes below. Best-effort and defensive: the
    function is private SDK surface and may not exist in every version.
    """
    import sys

    langsmith_utils = sys.modules.get("langsmith.utils")
    if langsmith_utils is None:
        return
    for fn_name in ("get_env_var", "get_tracer_project"):
        cache_clear = getattr(getattr(langsmith_utils, fn_name, None), "cache_clear", None)
        if cache_clear is not None:
            cache_clear()


def configure_langsmith() -> None:
    """Propagate LangSmith settings to the environment variables LangChain reads.

    Call this once at application startup (in the FastAPI lifespan), not at
    import time, to avoid side effects during testing.

    Both the primary ``LANGSMITH_*`` names (preferred by current ``langsmith``
    releases) and the legacy ``LANGCHAIN_*`` names are set, so tracing works
    regardless of which SDK version is installed.

    Disabling is *active*, not a no-op: the SDK reads the ambient environment,
    so an inherited ``LANGSMITH_TRACING=true`` (a shell profile, a container
    orchestrator) would keep tracing on no matter what ``.env`` said. Setting
    ``LANGSMITH_TRACING=false`` used to be impossible to honour.

    .. important::
       ``langsmith.utils.get_env_var`` is ``lru_cache``-wrapped, so these
       variables must be set **before** the first traced LLM/LangGraph call.
       The caches are cleared below to make that hold even if something read
       the environment during import.
    """
    _clear_langsmith_env_caches()

    if not settings.langsmith_tracing:
        # Explicitly off — do not merely leave the environment alone.
        for prefix in ("LANGSMITH_", "LANGCHAIN_"):
            os.environ[f"{prefix}TRACING_V2"] = "false"
            os.environ[f"{prefix}TRACING"] = "false"
        logger.info("LangSmith tracing disabled.")
        return

    if not settings.langsmith_api_key:
        logger.warning(
            "LANGSMITH_TRACING is enabled but LANGSMITH_API_KEY is empty — "
            "tracing will stay disabled. Set LANGSMITH_API_KEY or set "
            "LANGSMITH_TRACING=false."
        )
        for prefix in ("LANGSMITH_", "LANGCHAIN_"):
            os.environ[f"{prefix}TRACING_V2"] = "false"
            os.environ[f"{prefix}TRACING"] = "false"
        return

    values = {
        "TRACING_V2": "true",
        "API_KEY": settings.langsmith_api_key,
        "PROJECT": settings.langsmith_project,
    }
    for name, value in values.items():
        # Primary namespace, then legacy namespace for older SDK versions.
        os.environ[f"LANGSMITH_{name}"] = value
        os.environ[f"LANGCHAIN_{name}"] = value

    logger.info("LangSmith tracing enabled (project=%r).", settings.langsmith_project)
