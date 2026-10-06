"""The provider chain: which LLM clients to try, in order, with their rate limits.

LLM_CHAIN is a comma-separated list of provider:model, e.g.
"groq:openai/gpt-oss-120b,gemini:gemini-3.8-flash,gemini:gemini-3.5-flash-lite".
Without it, the chain is GEMINI_MODEL then GEMINI_FALLBACK_MODEL (if set and different).
Adding an OpenAI-compatible provider (Ollama, OpenRouter) is one PROVIDERS entry plus
its settings.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass

from classifier.llm import GeminiClient, LLMClient, OpenAICompatibleClient
from classifier.ratelimit import RateLimiter, TokenRateLimiter, interval_for_rpm
from classifier.tokens import EXPECTED_OUTPUT_TOKENS_PER_ITEM
from config import Settings, get_settings

# Share of the tokens-per-minute limit one batch may use, leaving room for estimate error.
BATCH_TOKEN_SHARE = 0.8


@dataclass(frozen=True)
class Provider:
    key_setting: str          # Settings attribute holding the API key
    key_env: str              # its env var, for error messages
    rpm_setting: str
    tpm_setting: str | None = None
    otpm_setting: str | None = None  # output tokens per minute, if limited separately
    base_url: str | None = None  # OpenAI-compatible endpoint; None = Gemini SDK


PROVIDERS = {
    "gemini": Provider("gemini_api_key", "GEMINI_API_KEY", "gemini_rpm"),
    "groq": Provider(
        "groq_api_key", "GROQ_API_KEY", "groq_rpm", "groq_tpm", "groq_otpm",
        base_url="https://api.groq.com/openai/v1",
    ),
}


@dataclass
class ChainEntry:
    client: LLMClient
    limiter: RateLimiter
    # Most tokens one request may use (None = no token limit), for batch sizing.
    max_batch_tokens: int | None = None

    def batch_size(self, default: int) -> int:
        """Items per batch: `default`, fewer under an output-tokens-per-minute limit
        (configured or learned from a 429) so one batch's reply stays within its share."""
        otpm = self.limiter.otpm
        if not otpm:
            return default
        per_batch = int(otpm * BATCH_TOKEN_SHARE) // EXPECTED_OUTPUT_TOKENS_PER_ITEM
        return max(1, min(default, per_batch))


def parse_chain(text: str) -> list[tuple[str, str]]:
    """[(provider, model), ...] from "provider:model,provider:model"."""
    specs = []
    for raw in text.split(","):
        entry = raw.strip()
        if not entry:
            continue
        provider, sep, model = entry.partition(":")
        provider, model = provider.strip().lower(), model.strip()
        if not sep or not provider or not model:
            raise ValueError(f"LLM_CHAIN entry {entry!r} is not provider:model")
        if provider not in PROVIDERS:
            raise ValueError(
                f"LLM_CHAIN: unknown provider {provider!r} (known: {', '.join(PROVIDERS)})"
            )
        if (provider, model) not in specs:
            specs.append((provider, model))
    if not specs:
        raise ValueError("LLM_CHAIN is empty")
    return specs


def chain_specs(settings: Settings | None = None) -> list[tuple[str, str]]:
    settings = settings or get_settings()
    if settings.llm_chain:
        return parse_chain(settings.llm_chain)
    if settings.llm_provider != "gemini":
        raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
    specs = [("gemini", settings.gemini_model)]
    fallback = settings.gemini_fallback_model
    if fallback and fallback != settings.gemini_model:
        specs.append(("gemini", fallback))
    return specs


def chain_names(settings: Settings | None = None) -> list[str]:
    """provider:model for each chain entry (for messages); [] if the chain is invalid."""
    try:
        return [f"{p}:{m}" for p, m in chain_specs(settings)]
    except ValueError:
        return []


def reasoning_effort(settings: Settings, model: str) -> str | None:
    """GROQ_REASONING_EFFORT if set, else the least reasoning the model allows, since
    reasoning tokens count toward the per-minute limits: "none" (off) for qwen, "low"
    for gpt-oss (it can't be turned off). Not sent for other models."""
    if settings.groq_reasoning_effort:
        return settings.groq_reasoning_effort
    if "gpt-oss" in model:
        return "low"
    if "qwen" in model:
        return "none"
    return None


def make_client(provider: str, model: str, settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    spec = PROVIDERS.get(provider)
    if spec is None:
        raise ValueError(f"Unknown LLM provider: {provider!r}")
    key = getattr(settings, spec.key_setting)
    if not key:
        raise ValueError(f"{spec.key_env} is not set")
    if spec.base_url is None:
        return GeminiClient(key, model)
    return OpenAICompatibleClient(
        spec.base_url, key, model, provider,
        reasoning_effort=reasoning_effort(settings, model) if provider == "groq" else None,
    )


def make_entry(
    client: LLMClient,
    provider: str,
    settings: Settings | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ChainEntry:
    """A chain entry with its provider's own request and token limits."""
    settings = settings or get_settings()
    spec = PROVIDERS[provider]
    interval = interval_for_rpm(getattr(settings, spec.rpm_setting))
    tpm = getattr(settings, spec.tpm_setting) if spec.tpm_setting else None
    otpm = getattr(settings, spec.otpm_setting) if spec.otpm_setting else None
    if not tpm:
        return ChainEntry(client, RateLimiter(interval, sleep=sleep))
    return ChainEntry(
        client,
        TokenRateLimiter(interval, tpm, sleep=sleep, otpm=otpm),
        max_batch_tokens=int(tpm * BATCH_TOKEN_SHARE),
    )


def build_entry(
    provider: str,
    model: str,
    settings: Settings | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ChainEntry:
    settings = settings or get_settings()
    return make_entry(make_client(provider, model, settings), provider, settings, sleep)


def build_chain(
    settings: Settings | None = None, sleep: Callable[[float], None] = time.sleep
) -> list[ChainEntry]:
    """Every chain entry, each with its own limiters (quotas are per model)."""
    settings = settings or get_settings()
    return [build_entry(p, m, settings, sleep) for p, m in chain_specs(settings)]
