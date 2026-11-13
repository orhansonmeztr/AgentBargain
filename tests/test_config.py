"""Unit tests for LangSmith tracing configuration."""

from __future__ import annotations

import logging
import sys

import pytest

import agent_bargain.config as config_module

_TRACING_ENV_VARS = (
    "LANGSMITH_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGSMITH_API_KEY",
    "LANGSMITH_PROJECT",
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_TRACING",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
)


def _clear_langsmith_caches() -> None:
    """Drop langsmith's environment caches so freshly-set vars are re-read."""
    langsmith_utils = sys.modules.get("langsmith.utils")
    if langsmith_utils is None:
        return
    for fn_name in ("get_env_var", "get_tracer_project"):
        fn = getattr(langsmith_utils, fn_name, None)
        cache_clear = getattr(fn, "cache_clear", None)
        if cache_clear is not None:
            cache_clear()


@pytest.fixture
def clean_env(monkeypatch):
    """Remove every tracing env var so assertions are not affected by the host.

    ``configure_langsmith`` writes to ``os.environ`` directly, which
    ``monkeypatch`` does not track, so the variables are also cleared afterwards
    to keep the rest of the test session hermetic.
    """
    for name in _TRACING_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    yield monkeypatch

    import os

    for name in _TRACING_ENV_VARS:
        os.environ.pop(name, None)

    # Drop langsmith's caches so later tests re-read the real environment.
    _clear_langsmith_caches()


def _configure(monkeypatch, *, tracing: bool, api_key: str, project: str = "agent-bargain"):
    settings = config_module.settings
    monkeypatch.setattr(settings, "langsmith_tracing", tracing)
    monkeypatch.setattr(settings, "langsmith_api_key", api_key)
    monkeypatch.setattr(settings, "langsmith_project", project)
    return settings


def test_configure_sets_primary_and_legacy_env_vars(clean_env):
    """Both LANGSMITH_* (primary) and LANGCHAIN_* (legacy) names are populated."""
    settings = _configure(clean_env, tracing=True, api_key="lsv2_test_key")

    config_module.configure_langsmith()

    import os

    assert os.environ["LANGSMITH_TRACING_V2"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "lsv2_test_key"
    assert os.environ["LANGSMITH_PROJECT"] == settings.langsmith_project
    # Legacy aliases kept for older SDK versions.
    assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    assert os.environ["LANGCHAIN_API_KEY"] == "lsv2_test_key"
    assert os.environ["LANGCHAIN_PROJECT"] == settings.langsmith_project


def test_configure_warns_when_tracing_enabled_without_api_key(clean_env, caplog):
    """Tracing without a key is a silent no-op in the SDK — we must warn about it."""
    _configure(clean_env, tracing=True, api_key="")

    with caplog.at_level(logging.WARNING, logger=config_module.__name__):
        config_module.configure_langsmith()

    import os

    assert "LANGSMITH_API_KEY is empty" in caplog.text
    # Not merely unset: explicitly disabled, so an inherited value cannot win.
    assert os.environ["LANGSMITH_TRACING_V2"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"


def test_disabling_overrides_an_inherited_env_var(clean_env):
    """``LANGSMITH_TRACING=false`` in .env must actually turn tracing off.

    Regression: the SDK reads the ambient environment, and the disabled path
    used to return without touching it. With LANGSMITH_TRACING=true exported by
    a shell profile or an orchestrator, tracing stayed on no matter what the
    project configuration said — and there was no way to switch it off.
    """
    import os

    import langsmith.utils as langsmith_utils

    os.environ["LANGSMITH_TRACING"] = "true"  # inherited from the shell
    os.environ["LANGCHAIN_TRACING"] = "true"
    assert langsmith_utils.tracing_is_enabled() is True

    _configure(clean_env, tracing=False, api_key="lsv2_test_key")
    config_module.configure_langsmith()
    _clear_langsmith_caches()

    assert os.environ["LANGSMITH_TRACING_V2"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    assert langsmith_utils.tracing_is_enabled() is False


def test_no_tracing_vars_are_set_when_disabled_and_none_inherited(clean_env):
    """With nothing inherited, disabling must not turn tracing on either."""
    import os

    _configure(clean_env, tracing=False, api_key="lsv2_test_key")
    config_module.configure_langsmith()

    assert os.environ.get("LANGSMITH_TRACING_V2") == "false"
    assert "LANGSMITH_API_KEY" not in os.environ
    assert "LANGSMITH_PROJECT" not in os.environ


def test_tracing_actually_enabled_via_langsmith_sdk(clean_env):
    """End-to-end: after configure(), the langsmith SDK reports tracing as enabled."""
    langsmith_utils = pytest.importorskip("langsmith.utils")
    _configure(clean_env, tracing=True, api_key="lsv2_test_key")

    config_module.configure_langsmith()
    _clear_langsmith_caches()

    assert langsmith_utils.tracing_is_enabled() is True
    assert langsmith_utils.get_tracer_project() == config_module.settings.langsmith_project


# -- logging configuration --------------------------------------------------


@pytest.fixture
def restore_root_logger():
    """Undo any handler/level configure_logging() installs on the root logger."""
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    yield root
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


def test_configure_logging_sets_the_level(restore_root_logger, monkeypatch):
    """INFO lines must actually reach a handler.

    With no root handler Python's last-resort output is WARNING-only, so every
    logger.info in the app — including the startup line reporting whether API
    authentication is on — was silently discarded.
    """
    monkeypatch.setattr(config_module.settings, "log_level", "INFO")

    config_module.configure_logging()

    assert restore_root_logger.level == logging.INFO
    assert restore_root_logger.handlers, "no handler installed — INFO would be dropped"
    assert restore_root_logger.isEnabledFor(logging.INFO)


def test_configure_logging_honours_log_level(restore_root_logger, monkeypatch):
    monkeypatch.setattr(config_module.settings, "log_level", "WARNING")
    config_module.configure_logging()
    assert restore_root_logger.level == logging.WARNING

    monkeypatch.setattr(config_module.settings, "log_level", "debug")  # case-insensitive
    config_module.configure_logging()
    assert restore_root_logger.level == logging.DEBUG


def test_configure_logging_falls_back_on_a_bad_level(restore_root_logger, monkeypatch, caplog):
    """A typo must degrade to INFO, not to a crash or to silent WARNING."""
    monkeypatch.setattr(config_module.settings, "log_level", "VERBOSE")

    with caplog.at_level(logging.WARNING, logger=config_module.__name__):
        config_module.configure_logging()

    assert restore_root_logger.level == logging.INFO
    assert "Unknown LOG_LEVEL" in caplog.text


def test_configure_logging_does_not_steal_existing_handlers(restore_root_logger, monkeypatch):
    """Must not clobber a handler installed by pytest's caplog or a supervisor."""
    monkeypatch.setattr(config_module.settings, "log_level", "INFO")
    sentinel = logging.NullHandler()
    restore_root_logger.handlers[:] = [sentinel]

    config_module.configure_logging()

    assert restore_root_logger.handlers == [sentinel], "an existing handler was removed"
    assert restore_root_logger.level == logging.INFO


def test_info_log_is_emitted_after_configuration(restore_root_logger, monkeypatch, capsys):
    """End to end: a logger.info call produces visible output."""
    monkeypatch.setattr(config_module.settings, "log_level", "INFO")
    restore_root_logger.handlers.clear()

    config_module.configure_logging()
    logging.getLogger("agent_bargain.probe").info("visible-info-line")

    assert "visible-info-line" in capsys.readouterr().err


def test_env_example_documents_every_setting():
    """``.env.example`` is the reference config; a setting missing from it is
    one nobody can discover (the temperatures were, for a while)."""
    import re

    from agent_bargain.config import PROJECT_ROOT, Settings

    documented = {
        m.group(1)
        for line in (PROJECT_ROOT / ".env.example").read_text().splitlines()
        if (m := re.match(r"\s*([A-Z_][A-Z0-9_]*)=", line))
    }
    settings_vars = {
        str(field.validation_alias or name).upper() for name, field in Settings.model_fields.items()
    }
    assert settings_vars - documented == set(), "add these to .env.example"
    assert documented - settings_vars == set(), "these are not settings any more"


def test_uvicorn_logs_never_contain_the_websocket_api_key():
    """Uvicorn logs the full target, so ``?api_key=`` must be masked."""
    import logging

    from agent_bargain.config import RedactSecretsFilter, configure_logging

    configure_logging()
    configure_logging()  # idempotent: one filter, not two
    for name in ("uvicorn.access", "uvicorn.error"):
        filters = logging.getLogger(name).filters
        assert sum(isinstance(f, RedactSecretsFilter) for f in filters) == 1

    record = logging.LogRecord(
        "uvicorn.error",
        logging.INFO,
        __file__,
        1,
        '%s - "WebSocket %s" [accepted]',
        ("172.19.0.1:1", "/api/v1/ws/events?api_key=s3cr3t-VALUE&x=1"),
        None,
    )
    assert RedactSecretsFilter().filter(record) is True
    message = record.getMessage()
    assert "s3cr3t-VALUE" not in message
    assert "/api/v1/ws/events?api_key=***&x=1" in message

    untouched = logging.LogRecord(
        "uvicorn.access", logging.INFO, __file__, 1, "%s %s", ("GET", "/health"), None
    )
    RedactSecretsFilter().filter(untouched)
    assert untouched.args == ("GET", "/health")
