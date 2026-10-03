"""Classify new raw_items: keyword prefilter, then batched LLM calls, then save results.

Tuned for the Gemini free tier (few requests per minute, small daily quota): items go
up to BATCH_SIZE per request, requests are spaced by a RateLimiter, and a daily-quota
429 stops the run with the remaining items left new.
"""

import json
import logging
import time
from collections.abc import Callable, Iterable

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from classifier.llm import LLMClient, LLMError, RateLimitError, get_llm_client
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
UNAVAILABLE_BACKOFF_SECONDS = (5.0, 15.0, 45.0)
PREFILTER_MODEL = "prefilter"


class QuotaExhausted(Exception):
    """Rate limited for longer than we're willing to wait; stop the run."""


class LLMUnavailable(Exception):
    """The LLM kept failing (503 or other API error) after backoff."""


def request_llm(
    llm: LLMClient, user: str, limiter: RateLimiter, sleep: Callable[[float], None]
) -> str:
    """One LLM request with the free-tier retry policy:
    - 429 with retryDelay < 2 min: wait it out and retry (up to MAX_RATE_LIMIT_RETRIES).
    - 429 with a longer or missing delay: raise QuotaExhausted.
    - 503 / other API errors: back off 5s, 15s, 45s, then raise LLMUnavailable.
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
            if unavailable_retries >= len(UNAVAILABLE_BACKOFF_SECONDS):
                raise LLMUnavailable(str(exc)) from exc
            delay = UNAVAILABLE_BACKOFF_SECONDS[unavailable_retries]
            unavailable_retries += 1
            logger.warning("LLM error (%s); retrying in %.0fs", exc, delay)
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
) -> dict[int, ClassifierOutput]:
    """Classify items in one request. Items missing from the result had a bad or
    missing entry. If the whole response is invalid, retry once; if it's still
    invalid, return {}. Raises QuotaExhausted or LLMUnavailable."""
    user = build_user_prompt(items)
    ids = {item.id for item in items}
    for attempt in range(2):
        results = parse_batch(request_llm(llm, user, limiter, sleep), ids)
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
            in_stl=out.in_stl,
            occurred_at=out.occurred_at,
            location=out.location,
            confidence=out.confidence,
            model=model,
            prompt_version=PROMPT_VERSION,
        )
    )
    item.status = "classified"


def classify_pending(
    session: Session,
    llm: LLMClient | None = None,
    sleep: Callable[[float], None] = time.sleep,
    limiter: RateLimiter | None = None,
    limit: int | None = None,
    batch_size: int = BATCH_SIZE,
) -> dict[str, int]:
    """Classify raw_items with status=new, committing after each batch. Returns counts:
    prefiltered (rejected without the LLM), classified, failed, retry_later (bad or
    missing result, back to new), deferred (LLM unavailable, left new), and
    stopped_quota (1 if the run stopped on the daily quota)."""
    counts = {
        "prefiltered": 0, "classified": 0, "failed": 0,
        "retry_later": 0, "deferred": 0, "stopped_quota": 0,
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
    llm = llm or get_llm_client()
    limiter = limiter or make_limiter(sleep)
    for batch in _chunks(eligible, batch_size):
        try:
            results = classify_batch(batch, llm, limiter, sleep)
        except QuotaExhausted:
            logger.warning("Daily quota exhausted; leaving remaining items new")
            counts["stopped_quota"] = 1
            break
        except LLMUnavailable:
            logger.exception("LLM unavailable; leaving %d item(s) new", len(batch))
            counts["deferred"] += len(batch)
            continue

        for item in batch:
            if item.id in results:
                _save(session, item, results[item.id], llm.model)
                counts["classified"] += 1
            elif (item.retries or 0) >= MAX_ITEM_RETRIES:
                logger.error("Marking raw_item %s failed after %d retries", item.id, item.retries)
                item.status = "failed"
                counts["failed"] += 1
            else:
                item.retries = (item.retries or 0) + 1
                counts["retry_later"] += 1
        session.commit()
    return counts
