"""Multi-provider LLM factory with unified routing logic."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from agent_bargain.config import settings


def _create_llm(model_name: str, temperature: float) -> BaseChatModel:
    """Internal factory that routes to the correct LLM provider."""
    provider = settings.detect_provider(model_name)
    # The "provider/" prefix is our routing convention — the SDKs want the bare id.
    clean_name = settings.strip_provider_prefix(model_name)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=clean_name,
            temperature=temperature,
            google_api_key=settings.google_api_key,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=clean_name,
            temperature=temperature,
            api_key=settings.openai_api_key,
        )

    if provider == "deepseek":
        # Fail here rather than send an unauthenticated request: the node's
        # guard turns this into the fallback LLM, like any other client error.
        if not settings.deepseek_api_key:
            raise ValueError(
                f"{model_name!r} needs DEEPSEEK_API_KEY; it is empty in the environment."
            )
        return _deepseek_chat_class()(
            model=clean_name,
            temperature=temperature,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            # deepseek-flash thinks by default, and thinking mode rejects a
            # forced tool choice with 400 — which is how structured output
            # works here. Every call failed and fell back to the local model.
            extra_body={"thinking": {"type": "disabled"}},
        )

    # Default: Ollama (local)
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=clean_name,
        temperature=temperature,
        base_url=settings.ollama_base_url,
    )


def get_negotiation_llm(temperature: float | None = None) -> BaseChatModel:
    """Returns the LLM configured for negotiation."""
    temp = temperature if temperature is not None else settings.negotiation_temperature
    return _create_llm(settings.negotiation_model, temp)


def get_evaluation_llm(temperature: float | None = None) -> BaseChatModel:
    """Returns the LLM configured for post-trade evaluation."""
    temp = temperature if temperature is not None else settings.evaluation_temperature
    return _create_llm(settings.evaluation_model, temp)


def get_fallback_llm(temperature: float | None = None) -> BaseChatModel:
    """Returns the fallback local LLM."""
    temp = temperature if temperature is not None else settings.fallback_temperature
    return _create_llm(settings.fallback_model, temp)


@lru_cache(maxsize=1)
def _deepseek_chat_class() -> type[BaseChatModel]:
    """``ChatOpenAI`` pointed at DeepSeek, with a structured-output default it supports.

    DeepSeek speaks the OpenAI chat API, but not its ``json_schema`` response
    format, which is ``ChatOpenAI.with_structured_output``'s default. Every
    caller here asks for structured output without naming a method, so the
    default is switched to tool calling instead of patching each call site.
    Built lazily so ``langchain_openai`` is only imported when DeepSeek is used.
    """
    from langchain_openai import ChatOpenAI

    class ChatDeepSeek(ChatOpenAI):
        def with_structured_output(self, schema=None, *, method="function_calling", **kwargs):
            return super().with_structured_output(schema, method=method, **kwargs)

    return ChatDeepSeek
