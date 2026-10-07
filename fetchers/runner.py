"""Run every active source's fetcher and store new items in raw_items."""

import logging
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit, urlunsplit

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from fetchers.base import REQUEST_TIMEOUT, USER_AGENT, BaseFetcher, FetchedItem
from fetchers.rss import RSSFetcher
from models import RawItem, Source, hash_url
from timeutil import to_utc

logger = logging.getLogger(__name__)

# Source.type -> fetcher class. Unknown types fall back to RSS.
FETCHERS: dict[str, type[BaseFetcher]] = {
    "news": RSSFetcher,
    "police": RSSFetcher,
    "facebook": RSSFetcher,
}


def normalize_url(url: str) -> str:
    """Drop query string, fragment, and trailing slash so variants of a link dedup."""
    parts = urlsplit(url.strip())
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip("/"), "", ""))


def get_fetcher(source: Source, client: httpx.Client) -> BaseFetcher:
    return FETCHERS.get(source.type, RSSFetcher)(client)


def make_client() -> httpx.Client:
    return httpx.Client(
        timeout=REQUEST_TIMEOUT,
        headers={"User-Agent": USER_AGENT},
        follow_redirects=True,
    )


def drop_stale(items: list[FetchedItem], cutoff: datetime | None) -> list[FetchedItem]:
    """Items published at or after cutoff. Undated items are kept (their age is
    unknown); cutoff None keeps everything."""
    if cutoff is None:
        return items
    return [i for i in items if i.published_at is None or to_utc(i.published_at) >= cutoff]


def store_items(session: Session, source: Source, items: list[FetchedItem]) -> int:
    """Add items not already in raw_items (by normalized url hash). Returns count added."""
    by_hash: dict[str, FetchedItem] = {}
    for item in items:
        by_hash.setdefault(hash_url(normalize_url(item.url)), item)
    if not by_hash:
        return 0

    existing = set(
        session.scalars(select(RawItem.url_hash).where(RawItem.url_hash.in_(by_hash)))
    )
    added = 0
    for url_hash, item in by_hash.items():
        if url_hash in existing:
            continue
        session.add(
            RawItem(
                source_id=source.id,
                url=item.url,
                url_hash=url_hash,
                title=item.title,
                body=item.body,
                published_at=item.published_at,
                status="new",
            )
        )
        added += 1
    return added


def run_fetchers(
    session: Session,
    client: httpx.Client | None = None,
    now: datetime | None = None,
    max_age_days: float | None = None,
) -> dict[str, int]:
    """Fetch all active sources, isolating failures. Items published more than
    max_age_days (default MAX_ITEM_AGE_DAYS; 0 = no limit) before now are skipped, so a
    stale feed can't flood the pipeline. Returns {source name: items added} for sources
    that succeeded."""
    if max_age_days is None:
        from config import get_settings

        max_age_days = get_settings().max_item_age_days
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max_age_days) if max_age_days > 0 else None
    own_client = client is None
    client = client or make_client()
    results: dict[str, int] = {}
    try:
        sources = session.scalars(select(Source).where(Source.active)).all()
        for source in sources:
            try:
                fetched = get_fetcher(source, client).fetch(source)
                items = drop_stale(fetched, cutoff)
                if len(items) < len(fetched):
                    logger.info(
                        "Skipped %d item(s) from %s older than %g day(s)",
                        len(fetched) - len(items), source.name, max_age_days,
                    )
                added = store_items(session, source, items)
                source.last_success_at = datetime.now(timezone.utc)
                source.fail_count = 0
                session.commit()
                results[source.name] = added
                logger.info("Fetched %s: %d new item(s)", source.name, added)
            except Exception:
                session.rollback()
                source.fail_count = (source.fail_count or 0) + 1
                session.commit()
                logger.exception(
                    "Fetch failed for %s (%s); fail_count=%d",
                    source.name, source.url, source.fail_count,
                )
    finally:
        if own_client:
            client.close()
    return results
