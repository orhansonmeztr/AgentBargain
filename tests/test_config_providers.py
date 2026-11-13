"""Regression tests for configuration and LLM provider routing.

Both of these were silent, total-failure bugs:

* a stray ``DEBUG`` environment variable made ``agent_bargain.config``
  unimportable, so the whole package (and pytest collection) died;
* the ``provider/model`` prefix was only stripped for Ollama, so Google and
  OpenAI were handed ``"google/gemini-2.0-flash"`` as a model id.
"""

from __future__ import annotations

import pytest

from agent_bargain.config import Settings, settings
from agent_bargain.llm.provider import _create_llm

# -- DEBUG collision -------------------------------------------------------


def test_ambient_debug_env_var_does_not_break_settings(monkeypatch):
    """``DEBUG=release`` (set by build toolchains) must be ignored entirely."""
    monkeypatch.setenv("DEBUG", "release")
    monkeypatch.delenv("APP_DEBUG", raising=False)

    # Would raise bool_parsing ValidationError before the validation_alias fix.
    assert Settings().debug is False


def test_app_debug_controls_debug_flag(monkeypatch):
    monkeypatch.setenv("DEBUG", "release")  # must stay irrelevant

    monkeypatch.setenv("APP_DEBUG", "true")
    assert Settings().debug is True

    monkeypatch.setenv("APP_DEBUG", "false")
    assert Settings().debug is False


def test_invalid_app_debug_still_fails_loudly(monkeypatch):
    """Fail-fast on genuinely bad config must be preserved."""
    monkeypatch.setenv("APP_DEBUG", "not-a-bool")
    with pytest.raises(ValueError, match="valid boolean"):
        Settings()


# -- provider prefix stripping --------------------------------------------


@pytest.mark.parametrize(
    ("model_name", "expected"),
    [
        ("ollama/gemma3:1b", "gemma3:1b"),
        ("google/gemini-2.0-flash", "gemini-2.0-flash"),
        ("gemini/gemini-2.0-flash", "gemini-2.0-flash"),
        ("openai/gpt-4o-mini", "gpt-4o-mini"),
        # Bare names pass through untouched.
        ("gpt-4o-mini", "gpt-4o-mini"),
        ("llama3.2:1b", "llama3.2:1b"),
        ("gemini-2.0-flash", "gemini-2.0-flash"),
    ],
)
def test_strip_provider_prefix(model_name, expected):
    assert settings.strip_provider_prefix(model_name) == expected


def test_strip_provider_prefix_preserves_model_case():
    """Matching is case-insensitive, but the model id itself must not be folded.

    Provider APIs are case-sensitive about model ids.
    """
    assert settings.strip_provider_prefix("GEMINI/Gemini-2.0-Flash") == "Gemini-2.0-Flash"


def test_no_prefix_leaks_into_any_provider_client(monkeypatch):
    """The model id handed to each SDK must never contain the routing prefix."""
    monkeypatch.setattr(settings, "google_api_key", "dummy-key")
    monkeypatch.setattr(settings, "openai_api_key", "dummy-key")

    cases = {
        "google/gemini-2.0-flash": "gemini-2.0-flash",
        "gemini/gemini-2.0-flash": "gemini-2.0-flash",
        "openai/gpt-4o-mini": "gpt-4o-mini",
        "ollama/gemma3:1b": "gemma3:1b",
    }
    for model_name, expected in cases.items():
        llm = _create_llm(model_name, 0.2)
        sent = getattr(llm, "model", None) or getattr(llm, "model_name", None)
        assert sent == expected, f"{model_name}: SDK received {sent!r}"


def test_detect_provider_and_strip_agree():
    """Every recognised prefix must route to a provider *and* be stripped."""
    from agent_bargain.config import PROVIDER_PREFIXES

    for prefix, provider in PROVIDER_PREFIXES.items():
        name = f"{prefix}some-model"
        assert settings.detect_provider(name) == provider
        assert settings.strip_provider_prefix(name) == "some-model"


# -- DeepSeek ------------------------------------------------------------------


def test_deepseek_prefix_routes_to_deepseek():
    assert Settings.detect_provider("deepseek/deepseek-flash") == "deepseek"
    assert Settings.strip_provider_prefix("deepseek/deepseek-flash") == "deepseek-flash"


def test_bare_deepseek_names_stay_on_ollama():
    """A local "deepseek-r1:1.5b" must not be sent to the DeepSeek cloud."""
    assert Settings.detect_provider("deepseek-r1:1.5b") == "ollama"


def test_deepseek_client_targets_the_deepseek_api(monkeypatch):
    monkeypatch.setattr(settings, "deepseek_api_key", "sk-test")
    monkeypatch.setattr(settings, "deepseek_base_url", "https://api.deepseek.com")
    llm = _create_llm("deepseek/deepseek-flash", 0.3)
    assert llm.model_name == "deepseek-flash"
    assert llm.openai_api_base == "https://api.deepseek.com"
    assert llm.temperature == 0.3


def test_deepseek_structured_output_uses_tool_calling(monkeypatch):
    """DeepSeek rejects OpenAI's json_schema response format; tools work."""
    from langchain_openai import ChatOpenAI

    from agent_bargain.llm.schemas import NegotiationDecision

    seen: dict = {}
    real = ChatOpenAI.with_structured_output

    def _spy(self, schema=None, *, method="json_schema", **kwargs):
        seen["method"] = method
        return real(self, schema, method=method, **kwargs)

    monkeypatch.setattr(ChatOpenAI, "with_structured_output", _spy)
    monkeypatch.setattr(settings, "deepseek_api_key", "sk-test")
    _create_llm("deepseek/deepseek-flash", 0.3).with_structured_output(NegotiationDecision)
    assert seen["method"] == "function_calling"


def test_deepseek_without_a_key_fails_at_construction(monkeypatch):
    """Raised inside the node's guard, so the fallback LLM takes over."""
    monkeypatch.setattr(settings, "deepseek_api_key", "")
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        _create_llm("deepseek/deepseek-flash", 0.3)


def test_deepseek_requests_disable_thinking_mode(monkeypatch):
    """Thinking mode answers a forced tool choice with 400, and structured
    output forces one — every call failed over to the local fallback model."""
    monkeypatch.setattr(settings, "deepseek_api_key", "sk-test")
    llm = _create_llm("deepseek/deepseek-flash", 0.3)
    assert llm.extra_body == {"thinking": {"type": "disabled"}}
