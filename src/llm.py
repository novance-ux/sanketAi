"""LLM provider adapter. The UI and the agent never import a provider SDK directly.

Providers (SANKET_LLM_PROVIDER):

* ``offline``    - no model call at all; the agent uses its deterministic rule engine.
* ``groq``       - Groq Cloud (langchain-groq). Default model openai/gpt-oss-120b.
* ``openrouter`` - OpenRouter, an OpenAI-compatible gateway (langchain-openai).
* ``anthropic``  - Claude through the official Anthropic SDK (structured outputs).
* ``openai``     - OpenAI or any OpenAI-compatible endpoint (langchain-openai).
* ``ollama``     - a model served on this machine or network (langchain-ollama). Data stays local.

External providers send retrieved Sanket text to a third party, so they are refused unless
SANKET_ALLOW_EXTERNAL_LLM=true has been set deliberately.

Fallback: SANKET_LLM_FALLBACK_PROVIDERS lists providers to try, in order, when the primary one
fails before producing any output (bad key, no credits, rate limit, outage).
"""
from __future__ import annotations

import logging
from typing import Iterator

from .config import LLM_PROVIDERS, Settings, get_settings
from .schemas import PlanDraft

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """No configured model could be used; the caller falls back to the offline behaviour."""


def data_flow_notice(settings: Settings | None = None) -> str:
    """Plain-language statement of where Sanket text goes under the current configuration."""
    s = settings or get_settings()
    if s.llm_provider == "offline":
        return ("Offline mode: no language model is called. Plans are assembled by a rule engine "
                "from the extracted records. No data leaves this machine.")
    chain = " -> fallback ".join(f"'{c.llm_provider}' ({c.chat_model()})" for c in s.provider_chain())
    if s.llm_provider == "ollama" and len(s.provider_chain()) == 1:
        return (f"Local model '{s.default_model()}' via Ollama at {s.llm_base_url or 'http://localhost:11434'}. "
                "Retrieved text is sent to that server only.")
    state = "ENABLED" if s.allow_external_llm else "BLOCKED (set SANKET_ALLOW_EXTERNAL_LLM=true to permit)"
    names = "victims' names are removed from plan evidence" if s.redact_person_names else "victims' names are included"
    return (f"Language model: {chain}. External providers: {state}. When enabled, passages retrieved from the "
            f"PDFs for each question or plan are sent to the provider's API; {names}.")


def is_llm_enabled(settings: Settings | None = None) -> bool:
    s = settings or get_settings()
    return any(c.llm_provider != "offline" and (not c.llm_is_external or c.allow_external_llm)
               for c in s.provider_chain())


def _check_allowed(s: Settings) -> None:
    if s.llm_provider not in LLM_PROVIDERS:
        raise LLMUnavailable(f"Unknown SANKET_LLM_PROVIDER '{s.llm_provider}'. Use one of {LLM_PROVIDERS}.")
    if s.llm_provider == "offline":
        raise LLMUnavailable("No language model is configured (SANKET_LLM_PROVIDER=offline).")
    if s.llm_is_external and not s.allow_external_llm:
        raise LLMUnavailable(
            f"Provider '{s.llm_provider}' would send Sanket text to an external service. "
            "This is blocked until SANKET_ALLOW_EXTERNAL_LLM=true is set."
        )


def _combine(errors: list[str]) -> str:
    return errors[0] if len(errors) == 1 else " | ".join(errors)


# --------------------------------------------------------------------------- plan drafts
def generate_plan_draft(system_prompt: str, user_prompt: str, settings: Settings | None = None) -> PlanDraft:
    """Ask the provider chain for a PlanDraft. Raises LLMUnavailable if every provider fails."""
    s = settings or get_settings()
    errors = []
    for cfg in s.provider_chain():
        try:
            _check_allowed(cfg)
            if cfg.llm_provider == "anthropic":
                return _anthropic(system_prompt, user_prompt, cfg)
            return _langchain(system_prompt, user_prompt, cfg)
        except LLMUnavailable as exc:
            log.warning("Plan generation via %s failed: %s", cfg.llm_provider, exc)
            errors.append(str(exc))
    raise LLMUnavailable(_combine(errors))


def _anthropic(system_prompt: str, user_prompt: str, s: Settings) -> PlanDraft:
    try:
        import anthropic
    except ImportError as exc:
        raise LLMUnavailable("The 'anthropic' package is not installed.") from exc
    kwargs = {"timeout": float(s.llm_timeout_s)}
    if key := s.api_key():
        kwargs["api_key"] = key
    try:
        # Without an explicit key the SDK resolves credentials from the environment/profile.
        client = anthropic.Anthropic(**kwargs)
        response = client.messages.parse(
            model=s.default_model(),
            max_tokens=max(s.llm_max_tokens, 16000),
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_format=PlanDraft,
        )
    except anthropic.AuthenticationError as exc:
        raise LLMUnavailable("Anthropic rejected the credentials. Check the API key variable.") from exc
    except anthropic.PermissionDeniedError as exc:
        raise LLMUnavailable("The Anthropic credentials lack permission for this model.") from exc
    except anthropic.NotFoundError as exc:
        raise LLMUnavailable(f"Model '{s.default_model()}' was not found.") from exc
    except anthropic.RateLimitError as exc:
        raise LLMUnavailable("Anthropic rate limit reached. Try again shortly.") from exc
    except anthropic.BadRequestError as exc:
        raise LLMUnavailable(f"Anthropic rejected the request: {exc.message}") from exc
    except anthropic.APIStatusError as exc:
        raise LLMUnavailable(f"Anthropic API error {exc.status_code}.") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMUnavailable("Could not reach the Anthropic API (network).") from exc
    except Exception as exc:  # credential resolution and client-side schema validation
        raise LLMUnavailable(f"Anthropic call failed: {exc}") from exc
    if response.stop_reason == "refusal":
        raise LLMUnavailable("The model declined to produce this plan.")
    if response.stop_reason == "max_tokens":
        raise LLMUnavailable("The model's answer was cut off (max_tokens). Raise SANKET_LLM_MAX_TOKENS.")
    if response.parsed_output is None:
        raise LLMUnavailable("The model returned no structured plan.")
    return response.parsed_output


def _langchain(system_prompt: str, user_prompt: str, s: Settings) -> PlanDraft:
    from langchain_core.messages import HumanMessage, SystemMessage

    try:
        model = _langchain_chat_model(s, s.default_model())
        result = model.with_structured_output(PlanDraft).invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
    except LLMUnavailable:
        raise
    except Exception as exc:  # provider SDKs raise unrelated exception hierarchies
        raise LLMUnavailable(_explain_provider_error(s.llm_provider, exc)) from exc
    if result is None:
        raise LLMUnavailable(f"{s.llm_provider} returned no structured plan.")
    return result if isinstance(result, PlanDraft) else PlanDraft.model_validate(result)


# --------------------------------------------------------------------------- short completions
def complete_text(system_prompt: str, messages: list[tuple[str, str]], settings: Settings | None = None,
                  *, json_mode: bool = False, used: dict | None = None) -> str:
    """Non-streaming completion through the provider chain (verifier, query rewriting).

    ``json_mode`` asks providers that support it for a JSON object; callers must still parse defensively.
    """
    s = settings or get_settings()
    errors: list[str] = []
    for cfg in s.provider_chain():
        try:
            _check_allowed(cfg)
            text = _complete_one(system_prompt, messages, cfg, json_mode)
        except LLMUnavailable as exc:
            log.warning("Completion via %s failed: %s", cfg.llm_provider, exc)
            errors.append(str(exc))
            continue
        if not text.strip():
            errors.append(f"{cfg.llm_provider} returned an empty response.")
            continue
        if used is not None:
            used.update(provider=cfg.llm_provider, model=cfg.chat_model(), skipped=list(errors))
        return text
    raise LLMUnavailable(_combine(errors) if errors else "No language model is configured.")


def _complete_one(system_prompt: str, messages: list[tuple[str, str]], s: Settings, json_mode: bool) -> str:
    if s.llm_provider == "anthropic":
        return "".join(_anthropic_stream(system_prompt, messages, s, s.chat_model()))
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    model = _langchain_chat_model(s, s.chat_model())
    if json_mode and s.llm_provider in ("groq", "openai", "openrouter"):
        model = model.bind(response_format={"type": "json_object"})
    lc_messages = [SystemMessage(content=system_prompt)] + [
        HumanMessage(content=t) if r == "user" else AIMessage(content=t) for r, t in messages
    ]
    try:
        result = model.invoke(lc_messages)
    except Exception as exc:  # provider SDKs raise unrelated exception hierarchies
        raise LLMUnavailable(_explain_provider_error(s.llm_provider, exc)) from exc
    content = result.content
    if isinstance(content, list):
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
    return content or ""


# --------------------------------------------------------------------------- chat streaming
def stream_chat(system_prompt: str, messages: list[tuple[str, str]], settings: Settings | None = None,
                used: dict | None = None) -> Iterator[str]:
    """Stream a plain-text answer. ``messages`` is [(role, text), ...] with role 'user' or 'assistant'.

    Providers are tried in order until one produces output; ``used`` (if given) receives the
    provider and model that answered and any errors from providers tried before it.
    Raises LLMUnavailable if no provider produces output, or if the answering one fails mid-stream.
    """
    s = settings or get_settings()
    errors: list[str] = []
    for cfg in s.provider_chain():
        try:
            _check_allowed(cfg)
            stream = _stream_one(system_prompt, messages, cfg)
            first = next(stream, None)  # fail over only while nothing has been shown yet
        except LLMUnavailable as exc:
            log.warning("Chat via %s failed: %s", cfg.llm_provider, exc)
            errors.append(str(exc))
            continue
        if first is None:
            errors.append(f"{cfg.llm_provider} returned an empty answer.")
            continue
        if used is not None:
            used.update(provider=cfg.llm_provider, model=cfg.chat_model(), skipped=list(errors))
        yield first
        yield from stream
        return
    raise LLMUnavailable(_combine(errors))


def _stream_one(system_prompt: str, messages: list[tuple[str, str]], s: Settings) -> Iterator[str]:
    model_name = s.chat_model()
    if s.llm_provider == "anthropic":
        yield from _anthropic_stream(system_prompt, messages, s, model_name)
        return
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

    model = _langchain_chat_model(s, model_name, streaming=True)
    lc_messages = [SystemMessage(content=system_prompt)] + [
        HumanMessage(content=t) if r == "user" else AIMessage(content=t) for r, t in messages
    ]
    try:
        for chunk in model.stream(lc_messages):
            text = chunk.content if isinstance(chunk.content, str) else "".join(
                p.get("text", "") for p in chunk.content if isinstance(p, dict) and p.get("type") == "text")
            if text:
                yield text
    except LLMUnavailable:
        raise
    except Exception as exc:  # provider SDKs raise unrelated exception hierarchies
        raise LLMUnavailable(_explain_provider_error(s.llm_provider, exc)) from exc


def _explain_provider_error(provider: str, exc: Exception) -> str:
    """Turn common provider failures into an instruction the operator can act on."""
    text = str(exc)
    status = getattr(exc, "status_code", None)
    if "insufficient_quota" in text or "credit_balance" in text or "no credits" in text.lower():
        return (f"The {provider} account has no API credits left. Add credits in the provider's billing "
                "settings and try again.")
    if status == 401 or "invalid_api_key" in text or "Incorrect API key" in text or "Invalid API Key" in text:
        return f"The {provider} API key was rejected. Check the key in .env."
    if status == 404 or "model_not_found" in text or "does not exist" in text:
        return f"The configured model is not available on {provider}. Change the model setting in .env."
    if status == 429 or "rate limit" in text.lower() or "rate-limited" in text:
        return f"{provider}: rate limit reached. Wait a moment and try again."
    if status == 413 or "too large" in text.lower() or "context_length" in text:
        return f"{provider}: the request was too large for this model."
    if "Connection" in type(exc).__name__ or "connect" in text.lower():
        return f"Could not reach the {provider} API (network)."
    return f"{provider} request failed: {text[:300]}"


def _anthropic_stream(system_prompt: str, messages: list[tuple[str, str]], s: Settings, model_name: str) -> Iterator[str]:
    try:
        import anthropic
    except ImportError as exc:
        raise LLMUnavailable("The 'anthropic' package is not installed.") from exc
    kwargs = {"timeout": float(s.llm_timeout_s)}
    if key := s.api_key():
        kwargs["api_key"] = key
    try:
        client = anthropic.Anthropic(**kwargs)
        with client.messages.stream(
            model=model_name, max_tokens=max(s.llm_max_tokens, 16000), system=system_prompt,
            messages=[{"role": r, "content": t} for r, t in messages],
        ) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            raise LLMUnavailable("The model declined to answer this question.")
    except LLMUnavailable:
        raise
    except anthropic.AuthenticationError as exc:
        raise LLMUnavailable("Anthropic rejected the credentials. Check the API key variable.") from exc
    except anthropic.RateLimitError as exc:
        raise LLMUnavailable("Anthropic rate limit reached. Try again shortly.") from exc
    except anthropic.APIStatusError as exc:
        raise LLMUnavailable(f"Anthropic API error {exc.status_code}.") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMUnavailable("Could not reach the Anthropic API (network).") from exc


# --------------------------------------------------------------------------- model construction
def _langchain_chat_model(s: Settings, model_name: str, *, streaming: bool = False):
    """LangChain chat model for the groq / openrouter / openai / ollama providers."""
    if s.llm_provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=model_name, temperature=0, base_url=s.base_url(), num_predict=s.llm_max_tokens)
    key = s.api_key()
    if not key:
        env = s.llm_api_key_env or {"groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY"}.get(
            s.llm_provider, "OPENAI_API_KEY")
        raise LLMUnavailable(f"No API key found for {s.llm_provider} in the environment variable '{env}'.")
    effort = s.chat_reasoning_effort if s.chat_reasoning_effort and _is_reasoning_model(model_name) else None
    if s.llm_provider == "groq":
        from langchain_groq import ChatGroq

        kwargs = dict(model=model_name, api_key=key, max_tokens=s.llm_max_tokens, timeout=s.llm_timeout_s,
                      streaming=streaming, temperature=0.2, max_retries=0)
        if effort:
            kwargs["reasoning_effort"] = "low" if effort == "none" else effort
        return ChatGroq(**kwargs)
    from langchain_openai import ChatOpenAI

    kwargs = dict(model=model_name, api_key=key, base_url=s.base_url(), timeout=s.llm_timeout_s,
                  max_tokens=s.llm_max_tokens, streaming=streaming,
                  max_retries=0 if s.llm_provider == "openrouter" else 1)
    if s.llm_provider == "openrouter":
        kwargs["default_headers"] = {"X-Title": "Inundation"}
        if s.chat_reasoning_effort:
            # OpenRouter's unified reasoning control; models without reasoning ignore it
            effort = "low" if s.chat_reasoning_effort == "none" else s.chat_reasoning_effort
            kwargs["extra_body"] = {"reasoning": {"effort": effort}}
    elif effort:
        # OpenAI reasoning models (o-series, gpt-5 and later) take an effort setting instead of temperature.
        kwargs["reasoning_effort"] = effort
    return ChatOpenAI(**kwargs)


def _is_reasoning_model(name: str) -> bool:
    n = name.lower().split("/")[-1]
    return (n.startswith(("o1", "o3", "o4", "gpt-oss"))
            or (n.startswith("gpt-") and n[4:5].isdigit() and int(n[4]) >= 5))
