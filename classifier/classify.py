"""Classify new raw_items: keyword prefilter, then batched LLM calls, then save results.

Tuned for free tiers (few requests or tokens per minute, small daily quotas): items go
up to BATCH_SIZE per request, fewer when a provider has a tokens-per-minute limit, and
each provider's requests are spaced by its own limiter. The providers form a chain
(LLM_CHAIN, see providers.py), tried in order for every batch:
- daily quota / rate limit that won't clear (QuotaExhausted): out for the rest of the run;
- overloaded (503): the batch goes to the next provider at once (no backoff, since
  failed attempts may count toward quota); after MAX_OVERLOADED_BATCHES such batches
  the provider is out for the rest of the run;
- request over the provider's token limit (413): the batch goes to the next provider.
Only the last provider still in the chain backs off on 503s. The run stops (remaining
items left new) only once every provider has hit its quota.
"""

import json
import logging
import time
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from classifier.llm import (
    InvalidJSONResponse,
    LLMClient,
    LLMError,
    RateLimitError,
    RequestTooLargeError,
    ServiceUnavailableError,
)
from classifier.prefilter import prefilter
from classifier.prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from classifier.providers import ChainEntry, build_chain
from classifier.ratelimit import RateLimiter, interval_for_rpm
from classifier.schema import BatchResult, ClassifierOutput
from classifier.tokens import count_items, expected_output, request_tokens
from config import get_settings
from models import Classification, RawItem

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
# A bad or missing result sends the item back as new this many times before failing it.
MAX_ITEM_RETRIES = 3
# 429s asking us to wait at least this long are the daily quota: stop the run.
MAX_RATE_LIMIT_WAIT_SECONDS = 120.0
MAX_RATE_LIMIT_RETRIES = 3
# Longest wait for a 429 the provider says is per-minute (window plus a margin).
PER_MINUTE_WAIT_SECONDS = 61.0
# Kept short: failed attempts may count toward the daily quota.
UNAVAILABLE_BACKOFF_SECONDS = (5.0, 15.0)
# A provider overloaded on this many batches in one run is skipped for the rest of it.
MAX_OVERLOADED_BATCHES = 2
PREFILTER_MODEL = "prefilter"
_MAX_ERROR_CHARS = 200


class QuotaExhausted(Exception):
    """Rate limited for longer than we're willing to wait; stop the run."""


class LLMUnavailable(Exception):
    """The LLM kept failing (503 or other API error) after backoff."""


class ModelOverloaded(LLMUnavailable):
    """The last failure was a 503: this model is overloaded, another one may answer."""


class BatchTooLarge(LLMUnavailable):
    """The request is over the provider's token limit; another provider may take it."""


def _short(exc: BaseException) -> str:
    """First line of an error message, capped, for one-line log messages."""
    lines = str(exc).strip().splitlines()
    text = lines[0] if lines else type(exc).__name__
    return text if len(text) <= _MAX_ERROR_CHARS else text[: _MAX_ERROR_CHARS - 3] + "..."


def request_llm(
    llm: LLMClient,
    user: str,
    limiter: RateLimiter,
    sleep: Callable[[float], None],
    backoff: tuple[float, ...] = UNAVAILABLE_BACKOFF_SECONDS,
    retry_overloaded: bool = True,
) -> str:
    """One LLM request with the free-tier retry policy:
    - 429 the provider calls per-minute: wait (retryDelay, at most 61s) and retry.
    - 429 the provider calls per-day: raise QuotaExhausted.
    - otherwise 429 with retryDelay < 2 min: wait it out and retry; longer or missing:
      raise QuotaExhausted. Still limited after MAX_RATE_LIMIT_RETRIES: QuotaExhausted.
    - 503: raise ModelOverloaded at once if not `retry_overloaded`, else back off like
      other errors.
    - 413: raise BatchTooLarge (retrying won't help).
    - other API errors: back off per `backoff` (5s, 15s), then raise ModelOverloaded if
      the last error was a 503, else LLMUnavailable.
    - provider-side invalid JSON: return "" (an invalid response).
    """
    rate_limit_retries = 0
    unavailable_retries = 0
    tokens = request_tokens(SYSTEM_PROMPT, user)
    output = expected_output(count_items(user))
    while True:
        limiter.wait(tokens, output)
        try:
            text = llm.generate_json(SYSTEM_PROMPT, user, list[BatchResult])
        except RateLimitError as exc:
            limiter.rate_limited(exc)
            delay = exc.retry_after
            if exc.per_minute:
                # A per-minute limit clears within a minute whatever the delay says.
                delay = min(delay if delay is not None else PER_MINUTE_WAIT_SECONDS,
                            PER_MINUTE_WAIT_SECONDS)
            elif exc.per_minute is False or delay is None or delay >= MAX_RATE_LIMIT_WAIT_SECONDS:
                raise QuotaExhausted(
                    f"rate limited, retryDelay={delay}: {_short(exc)}"
                ) from exc
            if rate_limit_retries >= MAX_RATE_LIMIT_RETRIES:
                raise QuotaExhausted(f"still rate limited after retries: {_short(exc)}") from exc
            rate_limit_retries += 1
            logger.warning(
                "%s rate limited (%s); retrying in %.1fs", llm.model, _short(exc), delay
            )
            sleep(delay)
        except InvalidJSONResponse as exc:
            logger.warning("%s returned invalid JSON (%s)", llm.model, _short(exc))
            return ""
        except RequestTooLargeError as exc:
            raise BatchTooLarge(_short(exc)) from exc
        except LLMError as exc:
            overloaded = isinstance(exc, ServiceUnavailableError)
            if overloaded and not retry_overloaded:
                raise ModelOverloaded(_short(exc)) from exc
            if unavailable_retries >= len(backoff):
                if overloaded:
                    raise ModelOverloaded(_short(exc)) from exc
                raise LLMUnavailable(_short(exc)) from exc
            delay = backoff[unavailable_retries]
            unavailable_retries += 1
            logger.warning("%s error (%s); retrying in %.0fs", llm.model, _short(exc), delay)
            sleep(delay)
        else:
            limiter.record(llm.last_usage)
            return text


def parse_batch(text: str, ids: set[int]) -> dict[int, ClassifierOutput] | None:
    """Validate each array element on its own. Returns {raw_item_id: result} for the
    good ones, or None if the response isn't a JSON array at all."""
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not isinstance(data, list):
        return None

    results: dict[int, ClassifierOutput] = {}
    for entry in data:
        try:
            result = BatchResult.model_validate(entry)
        except ValidationError as exc:
            logger.warning("Invalid result in batch: %s", exc)
            continue
        if result.raw_item_id not in ids or result.raw_item_id in results:
            logger.warning("Ignoring unexpected/duplicate raw_item_id %s", result.raw_item_id)
            continue
        results[result.raw_item_id] = result
    return results


def classify_batch(
    items: list[RawItem],
    llm: LLMClient,
    limiter: RateLimiter,
    sleep: Callable[[float], None] = time.sleep,
    backoff: tuple[float, ...] = UNAVAILABLE_BACKOFF_SECONDS,
    retry_overloaded: bool = True,
) -> dict[int, ClassifierOutput]:
    """Classify items in one request. Items missing from the result had a bad or
    missing entry. If the whole response is invalid, retry once; if it's still
    invalid, return {}. Raises QuotaExhausted or LLMUnavailable."""
    user = build_user_prompt(items)
    ids = {item.id for item in items}
    for attempt in range(2):
        text = request_llm(llm, user, limiter, sleep, backoff, retry_overloaded)
        results = parse_batch(text, ids)
        if results is not None:
            return results
        logger.warning("Batch response is not a JSON array (attempt %d)", attempt + 1)
    return {}


def take_batch(
    items: list[RawItem], start: int, size: int = BATCH_SIZE, max_tokens: int | None = None
) -> list[RawItem]:
    """Up to `size` items from `start`, fewer if the request would go over
    `max_tokens`. Always at least one item: one too big alone is sent alone (a 413
    then moves it to the next provider)."""
    batch = items[start : start + 1]
    for item in items[start + 1 : start + size]:
        if max_tokens is not None and (
            request_tokens(SYSTEM_PROMPT, build_user_prompt(batch + [item])) > max_tokens
        ):
            break
        batch.append(item)
    return batch


def make_limiter(sleep: Callable[[float], None] = time.sleep) -> RateLimiter:
    return RateLimiter(interval_for_rpm(get_settings().gemini_rpm), sleep=sleep)


def _save(session: Session, item: RawItem, out: ClassifierOutput, model: str) -> None:
    session.add(
        Classification(
            raw_item_id=item.id,
            is_crime=out.is_crime,
            crime_type=out.crime_type,
            was_shooting=out.was_shooting,
            in_stl=out.in_stl,
            occurred_at=out.occurred_at,
            time_precision=out.time_precision,
            location=out.location,
            neighborhood=out.neighborhood,
            is_followup=out.is_followup,
            confidence=out.confidence,
            model=model,
            prompt_version=PROMPT_VERSION,
        )
    )
    item.status = "classified"


class _Chain:
    """The chain's entries and which are still usable this run (see module docstring)."""

    def __init__(self, entries: list[ChainEntry]):
        self.entries = entries
        self.alive = [True] * len(entries)
        self.overloaded = [0] * len(entries)
        self.switched = False  # the first provider hit its quota and another took over

    def current(self) -> ChainEntry | None:
        """The first usable entry: the one the next batch is sized for."""
        return next((e for e, ok in zip(self.entries, self.alive) if ok), None)

    def classify(
        self, batch: list[RawItem], sleep: Callable[[float], None]
    ) -> tuple[dict[int, ClassifierOutput], ChainEntry]:
        """Results for the batch and the entry that produced them. Raises
        QuotaExhausted (stop the run) or LLMUnavailable (defer this batch)."""
        skipped: LLMUnavailable | None = None
        for i, entry in enumerate(self.entries):
            if not self.alive[i]:
                continue
            name = entry.client.model
            last = not any(self.alive[i + 1 :])
            try:
                results = classify_batch(
                    batch, entry.client, entry.limiter, sleep, retry_overloaded=last
                )
                return results, entry
            except QuotaExhausted as exc:
                self.alive[i] = False
                if i == 0 and any(self.alive):
                    self.switched = True
                logger.warning(
                    "%s hit its quota (%s); skipping it for the rest of the run",
                    name, _short(exc),
                )
            except ModelOverloaded as exc:
                if last:
                    raise
                skipped = exc
                self.overloaded[i] += 1
                if self.overloaded[i] >= MAX_OVERLOADED_BATCHES:
                    self.alive[i] = False
                    logger.warning(
                        "%s overloaded on %d batches (%s); skipping it for the rest of the run",
                        name, self.overloaded[i], _short(exc),
                    )
                else:
                    logger.warning("%s overloaded (%s); trying the next model", name, _short(exc))
            except BatchTooLarge as exc:
                if last:
                    raise
                skipped = exc
                logger.warning(
                    "Batch too large for %s (%s); trying the next model", name, _short(exc)
                )
        if skipped is not None:
            # A model that was skipped for this batch still has quota: defer the batch.
            raise LLMUnavailable(_short(skipped))
        raise QuotaExhausted("every model hit its daily quota")


def classify_pending(
    session: Session,
    llm: LLMClient | None = None,
    sleep: Callable[[float], None] = time.sleep,
    limiter: RateLimiter | None = None,
    limit: int | None = None,
    batch_size: int = BATCH_SIZE,
    fallback: LLMClient | None = None,
    chain: list[ChainEntry] | None = None,
) -> dict[str, Any]:
    """Classify raw_items with status=new, committing after each batch. Returns counts:
    prefiltered (rejected without the LLM), classified, fallback (how many of those a
    model other than the chain's first classified), failed, retry_later (bad or missing
    result, back to new), deferred (LLM unavailable, left new), switched_quota (1 if the
    first model hit its quota and another took over), stopped_quota (1 if the run
    stopped because every model hit its quota), and by_model ({provider:model: count}).

    The chain is `chain`, else `llm` (+ `fallback`) sharing `limiter`, else built from
    settings. `limiter` given with a settings chain replaces every entry's limiter."""
    counts: dict[str, Any] = {
        "prefiltered": 0, "classified": 0, "fallback": 0, "failed": 0,
        "retry_later": 0, "deferred": 0, "switched_quota": 0, "stopped_quota": 0,
        "by_model": {},
    }
    query = select(RawItem).where(RawItem.status == "new").order_by(RawItem.id)
    if limit is not None:
        query = query.limit(limit)

    eligible = []
    for item in session.scalars(query).all():
        if prefilter(item):
            eligible.append(item)
        else:
            not_crime = ClassifierOutput(is_crime=False, in_stl=False, confidence=1.0)
            _save(session, item, not_crime, PREFILTER_MODEL)
            counts["prefiltered"] += 1
    session.commit()
    if not eligible:
        return counts

    # Built lazily so a batch the prefilter fully rejects needs no API key.
    if chain is None:
        if llm is not None:
            limiter = limiter or make_limiter(sleep)
            chain = [ChainEntry(c, limiter) for c in (llm, fallback) if c is not None]
        else:
            chain = build_chain(sleep=sleep)
            if limiter is not None:
                for entry in chain:
                    entry.limiter = limiter
    models = _Chain(chain)
    first = chain[0]
    by_model = counts["by_model"]
    pos = 0
    while pos < len(eligible):
        entry = models.current()
        batch = take_batch(eligible, pos, entry.batch_size(batch_size), entry.max_batch_tokens)
        pos += len(batch)
        try:
            results, used = models.classify(batch, sleep)
        except QuotaExhausted:
            logger.warning("Daily quota exhausted; leaving remaining items new")
            counts["stopped_quota"] = 1
            break
        except LLMUnavailable as exc:
            logger.warning(
                "LLM unavailable (%s); leaving %d item(s) new", _short(exc), len(batch)
            )
            counts["deferred"] += len(batch)
            continue
        finally:
            counts["switched_quota"] = int(models.switched)

        for item in batch:
            if item.id in results:
                _save(session, item, results[item.id], used.client.model)
                counts["classified"] += 1
                by_model[used.client.model] = by_model.get(used.client.model, 0) + 1
                if used is not first:
                    counts["fallback"] += 1
            elif (item.retries or 0) >= MAX_ITEM_RETRIES:
                logger.error("Marking raw_item %s failed after %d retries", item.id, item.retries)
                item.status = "failed"
                counts["failed"] += 1
            else:
                item.retries = (item.retries or 0) + 1
                counts["retry_later"] += 1
        session.commit()
    return counts
