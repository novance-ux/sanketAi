"""LLM provider adapter. The UI and the agent never import a provider SDK directly.

Providers (SANKET_LLM_PROVIDER):

* ``offline``   - no model call at all; the agent uses its deterministic rule engine.
* ``ollama``    - a model served on this machine or network (langchain-ollama). Data stays local.
* ``anthropic`` - Claude through the official Anthropic SDK (structured outputs).
* ``openai``    - OpenAI or any OpenAI-compatible endpoint (langchain-openai).

``anthropic`` and ``openai`` send the mine's evidence bundle to a third party, so they are
refused unless SANKET_ALLOW_EXTERNAL_LLM=true has been set deliberately.
"""
from __future__ import annotations

import logging

from .config import LLM_PROVIDERS, Settings, get_settings
from .schemas import PlanDraft

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """The configured model cannot be used; the caller falls back to the offline engine."""


def data_flow_notice(settings: Settings | None = None) -> str:
    """Plain-language statement of where Sanket text goes under the current configuration."""
    s = settings or get_settings()
    if s.llm_provider == "offline":
        return ("Offline mode: no language model is called. Plans are assembled by a rule engine "
                "from the extracted records. No data leaves this machine.")
    if s.llm_provider == "ollama":
        return (f"Local model '{s.default_model()}' via Ollama at {s.llm_base_url or 'http://localhost:11434'}. "
                "The evidence for the selected mine is sent to that server only.")
    state = "ENABLED" if s.allow_external_llm else "BLOCKED (set SANKET_ALLOW_EXTERNAL_LLM=true to permit)"
    names = "victims' names are removed before sending" if s.redact_person_names else "victims' names are included"
    return (f"External provider '{s.llm_provider}' (model {s.default_model()}): {state}. When enabled, the "
            f"extracted records of the selected mine are sent to the provider's API; {names}.")


def is_llm_enabled(settings: Settings | None = None) -> bool:
    s = settings or get_settings()
    return s.llm_provider != "offline" and (not s.llm_is_external or s.allow_external_llm)


def generate_plan_draft(system_prompt: str, user_prompt: str, settings: Settings | None = None) -> PlanDraft:
    """Ask the configured model for a PlanDraft. Raises LLMUnavailable on any failure."""
    s = settings or get_settings()
    if s.llm_provider not in LLM_PROVIDERS:
        raise LLMUnavailable(f"Unknown SANKET_LLM_PROVIDER '{s.llm_provider}'. Use one of {LLM_PROVIDERS}.")
    if s.llm_provider == "offline":
        raise LLMUnavailable("No language model is configured (offline mode).")
    if s.llm_is_external and not s.allow_external_llm:
        raise LLMUnavailable(
            f"Provider '{s.llm_provider}' would send Sanket records to an external service. "
            "This is blocked until SANKET_ALLOW_EXTERNAL_LLM=true is set."
        )
    if s.llm_provider == "anthropic":
        return _anthropic(system_prompt, user_prompt, s)
    return _langchain(system_prompt, user_prompt, s)


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
        if s.llm_provider == "ollama":
            from langchain_ollama import ChatOllama

            model = ChatOllama(model=s.default_model(), temperature=0,
                               base_url=s.llm_base_url or None, num_predict=s.llm_max_tokens)
        else:
            from langchain_openai import ChatOpenAI

            key = s.api_key()
            if not key:
                raise LLMUnavailable("No API key found in the configured environment variable.")
            model = ChatOpenAI(model=s.default_model(), api_key=key, base_url=s.llm_base_url or None,
                               timeout=s.llm_timeout_s, max_tokens=s.llm_max_tokens)
        result = model.with_structured_output(PlanDraft).invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
    except LLMUnavailable:
        raise
    except Exception as exc:  # provider SDKs raise unrelated exception hierarchies
        raise LLMUnavailable(f"{s.llm_provider} call failed: {exc}") from exc
    if result is None:
        raise LLMUnavailable("The model returned no structured plan.")
    return result if isinstance(result, PlanDraft) else PlanDraft.model_validate(result)
