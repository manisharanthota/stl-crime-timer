"""LLM client interface and the Gemini implementation."""

import re
from abc import ABC, abstractmethod
from typing import Any

import httpx

from config import Settings, get_settings


class LLMError(Exception):
    """LLM request failed (API error, network); the items can be retried later."""


class RateLimitError(LLMError):
    """429. retry_after is the server's suggested wait in seconds, if it gave one."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ServiceUnavailableError(LLMError):
    """503: the model is overloaded; retry with backoff."""


class LLMClient(ABC):
    model: str

    @abstractmethod
    def generate_json(self, system: str, user: str, schema: Any) -> str:
        """Return the model's raw JSON text for the prompt. Raises LLMError."""


_DURATION = re.compile(r"^\s*([\d.]+)s\s*$")


def parse_retry_delay(details: Any) -> float | None:
    """Pull RetryInfo.retryDelay (e.g. "37s") out of a Google API error body."""
    if not isinstance(details, dict):
        return None
    for entry in (details.get("error") or {}).get("details") or []:
        if isinstance(entry, dict) and "retryDelay" in entry:
            match = _DURATION.match(str(entry["retryDelay"]))
            if match:
                return float(match.group(1))
    return None


class GeminiClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        from google import genai

        self.model = model
        self._client = genai.Client(api_key=api_key)

    def generate_json(self, system: str, user: str, schema: Any) -> str:
        from google.genai import errors, types

        config = types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            temperature=0,
            # No tools are passed; disabling AFC also silences the SDK's
            # "AFC is enabled" log on every request.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            response = self._client.models.generate_content(
                model=self.model, contents=user, config=config
            )
        except errors.APIError as exc:
            if exc.code == 429:
                raise RateLimitError(str(exc), parse_retry_delay(exc.details)) from exc
            if exc.code == 503:
                raise ServiceUnavailableError(str(exc)) from exc
            raise LLMError(str(exc)) from exc
        except httpx.HTTPError as exc:
            raise LLMError(str(exc)) from exc
        return response.text or ""


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    if settings.llm_provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        return GeminiClient(settings.gemini_api_key, settings.gemini_model)
    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
