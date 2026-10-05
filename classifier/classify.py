"""Classify new raw_items: keyword prefilter, then batched LLM calls, then save results.

Tuned for the Gemini free tier (few requests per minute, small daily quota): items go
up to BATCH_SIZE per request and requests are spaced by a RateLimiter. With
GEMINI_FALLBACK_MODEL set: a batch whose primary is still overloaded (503) after
backoff gets one try on the fallback, and a primary daily-quota 429 switches the rest
of the run to the fallback. The run stops (remaining items left new) only once every
available model has hit its daily quota.
"""

import json
import logging
import time
from collections.abc import Callable, Iterable

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from classifier.llm import (
    LLMClient,
    LLMError,
    RateLimitError,
    ServiceUnavailableError,
    get_fallback_client,
    get_llm_client,
)
from classifier.prefilter import prefilter
from classifier.prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from classifier.ratelimit import RateLimiter, interval_for_rpm
from classifier.schema import BatchResult, ClassifierOutput
from config import get_settings
from models import Classification, RawItem

logger = logging.getLogger(__name__)

BATCH_SIZE = 10
# A bad or missing result sends the item back as new this many times before failing it.
MAX_ITEM_RETRIES = 3
# 429s asking us to wait at least this long are the daily quota: stop the run.
MAX_RATE_LIMIT_WAIT_SECONDS = 120.0
MAX_RATE_LIMIT_RETRIES = 3
# Kept short: failed attempts may count toward the daily quota.
UNAVAILABLE_BACKOFF_SECONDS = (5.0, 15.0)
PREFILTER_MODEL = "prefilter"
_MAX_ERROR_CHARS = 200


class QuotaExhausted(Exception):
    """Rate limited for longer than we're willing to wait; stop the run."""


class LLMUnavailable(Exception):
    """The LLM kept failing (503 or other API error) after backoff."""


class ModelOverloaded(LLMUnavailable):
    """The last failure was a 503: this model is overloaded, another one may answer."""


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
) -> str:
    """One LLM request with the free-tier retry policy:
    - 429 with retryDelay < 2 min: wait it out and retry (up to MAX_RATE_LIMIT_RETRIES).
    - 429 with a longer or missing delay: raise QuotaExhausted.
    - 503 / other API errors: back off per `backoff` (5s, 15s), then raise
      ModelOverloaded if the last error was a 503, else LLMUnavailable.
    """
    rate_limit_retries = 0
    unavailable_retries = 0
    while True:
        limiter.wait()
        try:
            return llm.generate_json(SYSTEM_PROMPT, user, list[BatchResult])
        except RateLimitError as exc:
            delay = exc.retry_after
            if delay is None or delay >= MAX_RATE_LIMIT_WAIT_SECONDS:
                raise QuotaExhausted(f"rate limited, retryDelay={delay}") from exc
            if rate_limit_retries >= MAX_RATE_LIMIT_RETRIES:
                raise QuotaExhausted("still rate limited after retries") from exc
            rate_limit_retries += 1
            logger.warning("Rate limited; retrying in %.0fs", delay)
            sleep(delay)
        except LLMError as exc:
            if unavailable_retries >= len(backoff):
                if isinstance(exc, ServiceUnavailableError):
                    raise ModelOverloaded(_short(exc)) from exc
                raise LLMUnavailable(_short(exc)) from exc
            delay = backoff[unavailable_retries]
            unavailable_retries += 1
            logger.warning("LLM error (%s); retrying in %.0fs", _short(exc), delay)
            sleep(delay)


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
) -> dict[int, ClassifierOutput]:
    """Classify items in one request. Items missing from the result had a bad or
    missing entry. If the whole response is invalid, retry once; if it's still
    invalid, return {}. Raises QuotaExhausted or LLMUnavailable."""
    user = build_user_prompt(items)
    ids = {item.id for item in items}
    for attempt in range(2):
        results = parse_batch(request_llm(llm, user, limiter, sleep, backoff), ids)
        if results is not None:
            return results
        logger.warning("Batch response is not a JSON array (attempt %d)", attempt + 1)
    return {}


def make_limiter(sleep: Callable[[float], None] = time.sleep) -> RateLimiter:
    return RateLimiter(interval_for_rpm(get_settings().gemini_rpm), sleep=sleep)


def _chunks(items: list[RawItem], size: int) -> Iterable[list[RawItem]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


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


class _Models:
    """The primary and fallback clients, and which are still usable this run.

    - Primary 503s after backoff (ModelOverloaded): one try on the fallback for that
      batch; the next batch starts on the primary again.
    - Primary daily quota (QuotaExhausted): the fallback takes over for the rest of the
      run, one attempt per batch (no 503 backoff).
    - Fallback daily quota: the fallback is dropped. The run stops (QuotaExhausted) only
      when neither model is usable.
    """

    def __init__(self, primary: LLMClient, fallback: LLMClient | None):
        self.primary = primary
        self.fallback = fallback
        self.primary_ok = True
        self.fallback_ok = fallback is not None
        self.switched = False  # primary hit its quota and the fallback took over

    def classify(
        self, batch: list[RawItem], limiter: RateLimiter, sleep: Callable[[float], None]
    ) -> tuple[dict[int, ClassifierOutput], LLMClient]:
        """Results for the batch and the client that produced them. Raises
        QuotaExhausted (stop the run) or LLMUnavailable (defer this batch)."""
        overloaded = None
        if self.primary_ok:
            try:
                return classify_batch(batch, self.primary, limiter, sleep), self.primary
            except QuotaExhausted as exc:
                self.primary_ok = False
                if self.fallback_ok:
                    self.switched = True
                    logger.warning(
                        "%s hit its daily quota (%s); using fallback %s for the rest of the run",
                        self.primary.model, _short(exc), self.fallback.model,
                    )
            except ModelOverloaded as exc:
                if not self.fallback_ok:
                    raise
                overloaded = exc
                logger.warning(
                    "%s still overloaded (%s); trying fallback %s once",
                    self.primary.model, _short(exc), self.fallback.model,
                )
        if self.fallback_ok:
            try:
                results = classify_batch(batch, self.fallback, limiter, sleep, backoff=())
                return results, self.fallback
            except QuotaExhausted as exc:
                self.fallback_ok = False
                logger.warning(
                    "Fallback %s hit its daily quota (%s)", self.fallback.model, _short(exc)
                )
                if overloaded is not None:
                    # The primary still has quota: defer this batch, keep going on it.
                    raise LLMUnavailable(_short(overloaded)) from exc
        raise QuotaExhausted("every model hit its daily quota")


def classify_pending(
    session: Session,
    llm: LLMClient | None = None,
    sleep: Callable[[float], None] = time.sleep,
    limiter: RateLimiter | None = None,
    limit: int | None = None,
    batch_size: int = BATCH_SIZE,
    fallback: LLMClient | None = None,
) -> dict[str, int]:
    """Classify raw_items with status=new, committing after each batch. Returns counts:
    prefiltered (rejected without the LLM), classified, fallback (how many of those the
    fallback model classified), failed, retry_later (bad or missing result, back to
    new), deferred (LLM unavailable, left new), switched_quota (1 if the primary hit its
    daily quota and the fallback took over), and stopped_quota (1 if the run stopped
    because every model hit its daily quota).

    With no `llm` given, both clients come from settings; an injected `llm` gets a
    fallback only if one is passed too."""
    counts = {
        "prefiltered": 0, "classified": 0, "fallback": 0, "failed": 0,
        "retry_later": 0, "deferred": 0, "switched_quota": 0, "stopped_quota": 0,
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

    # Created lazily so a batch the prefilter fully rejects needs no API key.
    if llm is None:
        llm = get_llm_client()
        fallback = fallback or get_fallback_client()
    limiter = limiter or make_limiter(sleep)
    models = _Models(llm, fallback)
    for batch in _chunks(eligible, batch_size):
        try:
            results, used = models.classify(batch, limiter, sleep)
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
                _save(session, item, results[item.id], used.model)
                counts["classified"] += 1
                if used is fallback:
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
