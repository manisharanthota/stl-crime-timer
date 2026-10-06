"""LLM client interface, the Gemini client, and an OpenAI-compatible client (Groq)."""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import httpx

from config import Settings, get_settings
from classifier.tokens import count_items, output_tokens


class LLMError(Exception):
    """LLM request failed (API error, network); the items can be retried later."""


class RateLimitError(LLMError):
    """429. retry_after is the server's suggested wait in seconds, if it gave one.
    per_minute is True for a per-minute limit (wait, never a daily quota), False for a
    per-day limit, None when the provider doesn't say (Gemini)."""

    def __init__(
        self,
        message: str,
        retry_after: float | None = None,
        per_minute: bool | None = None,
        output_limit: int | None = None,
    ):
        super().__init__(message)
        self.retry_after = retry_after
        self.per_minute = per_minute
        # The output-tokens-per-minute limit the message names (Groq OTPM), if any.
        self.output_limit = output_limit


class ServiceUnavailableError(LLMError):
    """503: the model is overloaded; retry with backoff."""


class RequestTooLargeError(LLMError):
    """413: the request alone is over the provider's token limit; retrying won't help."""


class InvalidJSONResponse(LLMError):
    """The provider rejected the model's own output as invalid JSON (Groq's 400
    json_validate_failed); handled like a reply that isn't a JSON array."""


@dataclass
class Usage:
    """Token usage of the last response and the provider's token budget left, if the
    provider reports them (Groq does, Gemini doesn't)."""

    total_tokens: int | None = None
    remaining_tokens: int | None = None
    reset_seconds: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    # Part of completion_tokens spent on reasoning, if the provider reports it.
    reasoning_tokens: int | None = None


class LLMClient(ABC):
    # "provider:model", stored in classifications.model.
    model: str
    last_usage: Usage | None = None

    @abstractmethod
    def generate_json(self, system: str, user: str, schema: Any) -> str:
        """Return the model's raw JSON text for the prompt. Raises LLMError."""


_DURATION = re.compile(r"^\s*([\d.]+)s\s*$")
# Groq-style durations: "7.66s", "2m59.56s", "1h2m3s", "120ms".
_DURATION_PART = re.compile(r"([\d.]+)(ms|h|m|s)")
_TRY_AGAIN = re.compile(r"try again in ((?:[\d.]+(?:ms|h|m|s))+)", re.IGNORECASE)
_DURATION_SCALE = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
# Groq names the limit it hit: "... on tokens per minute (TPM): Limit 8000, ...".
_PER_MINUTE = re.compile(r"per minute|\((?:TPM|RPM)\)", re.IGNORECASE)
_PER_DAY = re.compile(r"per day|\((?:TPD|RPD)\)", re.IGNORECASE)
_OTPM_LIMIT = re.compile(r"\(OTPM\):\s*Limit\s+(\d+)", re.IGNORECASE)


def rate_limit_period(message: str | None) -> bool | None:
    """True if a 429 message names a per-minute limit, False if a per-day one, else None."""
    if not message:
        return None
    if _PER_DAY.search(message):
        return False
    if _PER_MINUTE.search(message):
        return True
    return None


def parse_duration(text: str | None) -> float | None:
    """Seconds in "12", "7.66s", "2m59.56s" or "120ms"; None if unparseable."""
    if not text:
        return None
    text = text.strip()
    try:
        return float(text)
    except ValueError:
        pass
    parts = _DURATION_PART.findall(text)
    if not parts or "".join(n + u for n, u in parts) != text:
        return None
    return sum(float(n) * _DURATION_SCALE[u] for n, u in parts)


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

        from google.genai import types

        self.model = f"gemini:{model}"
        self.model_name = model
        # One attempt per call: retries/backoff are classify.request_llm's job, and
        # SDK retries would stack on top of ours (and may spend quota).
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1)),
        )

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
                model=self.model_name, contents=user, config=config
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


# Appended to the system prompt in JSON mode, which only allows a top-level object.
JSON_OBJECT_NOTE = (
    '\n\nReply with one JSON object of the form {"results": [...]}, where "results" is '
    "the JSON array described above."
)
# Groq answers 498 when its capacity is exceeded: the same as an overloaded 503.
_OVERLOADED_STATUSES = (498, 503)


def unwrap_results(text: str) -> str:
    """The array inside {"results": [...]} (or an object's only list value) as JSON
    text. Anything else is returned unchanged, for parse_batch to reject."""
    try:
        data = json.loads(text)
    except ValueError:
        return text
    if isinstance(data, dict):
        if isinstance(data.get("results"), list):
            return json.dumps(data["results"])
        lists = [v for v in data.values() if isinstance(v, list)]
        if len(lists) == 1:
            return json.dumps(lists[0])
    return text


class OpenAICompatibleClient(LLMClient):
    """Chat Completions over plain httpx in JSON mode (Groq; later Ollama, OpenRouter).

    JSON mode has no schema, so results are only checked by Pydantic (parse_batch).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str | None,
        model: str,
        provider: str,
        reasoning_effort: str | None = None,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ):
        self.model = f"{provider}:{model}"
        self.model_name = model
        self.reasoning_effort = reasoning_effort
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout)

    def generate_json(self, system: str, user: str, schema: Any) -> str:
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system + JSON_OBJECT_NOTE},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            # Caps the reply at what the batch sizing reserved; Groq counts it against
            # the per-minute token limit.
            "max_tokens": output_tokens(count_items(user)),
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        self.last_usage = None
        try:
            response = self._client.post(self._url, json=body, headers=self._headers)
        except httpx.HTTPError as exc:
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        self.last_usage = _usage_from_headers(response)
        if response.status_code >= 400:
            _raise_for_status(response)
        try:
            data = response.json()
            text = data["choices"][0]["message"]["content"] or ""
            usage = data.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMError(f"unexpected response: {response.text[:200]}") from exc
        for field, value in (
            ("total_tokens", usage.get("total_tokens")),
            ("prompt_tokens", usage.get("prompt_tokens")),
            ("completion_tokens", usage.get("completion_tokens")),
            ("reasoning_tokens", details.get("reasoning_tokens")),
        ):
            if isinstance(value, int):
                setattr(self.last_usage, field, value)
        return unwrap_results(text)


def _usage_from_headers(response: httpx.Response) -> Usage:
    remaining = response.headers.get("x-ratelimit-remaining-tokens")
    try:
        remaining_tokens = int(float(remaining)) if remaining is not None else None
    except ValueError:
        remaining_tokens = None
    return Usage(
        remaining_tokens=remaining_tokens,
        reset_seconds=parse_duration(response.headers.get("x-ratelimit-reset-tokens")),
    )


def _raise_for_status(response: httpx.Response) -> None:
    status = response.status_code
    try:
        error = response.json().get("error")
    except (ValueError, AttributeError):
        error = None
    error = error if isinstance(error, dict) else {}
    message = error.get("message")
    text = f"{status} {message or response.text[:200]}"
    if status == 400 and error.get("code") == "json_validate_failed":
        raise InvalidJSONResponse(text)
    if status == 429:
        retry_after = parse_duration(response.headers.get("retry-after"))
        if retry_after is None and message:
            match = _TRY_AGAIN.search(message)
            retry_after = parse_duration(match.group(1)) if match else None
        otpm = _OTPM_LIMIT.search(message or "")
        raise RateLimitError(
            text, retry_after, rate_limit_period(message),
            int(otpm.group(1)) if otpm else None,
        )
    if status in _OVERLOADED_STATUSES:
        raise ServiceUnavailableError(text)
    if status == 413:
        raise RequestTooLargeError(text)
    raise LLMError(text)


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    """Client for the first model of the chain (LLM_CHAIN, else GEMINI_MODEL)."""
    from classifier.providers import chain_specs, make_client

    settings = settings or get_settings()
    provider, model = chain_specs(settings)[0]
    return make_client(provider, model, settings)
