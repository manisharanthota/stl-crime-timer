"""RSS/Atom fetcher for news sites and RSS.app (Facebook) feeds."""

import calendar
import logging
from datetime import datetime, timezone

import feedparser

from fetchers.base import BaseFetcher, FetchedItem, FetchError
from models import Source

logger = logging.getLogger(__name__)


def _entry_datetime(entry) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    # feedparser normalizes to a UTC struct_time.
    return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)


def _entry_body(entry) -> str | None:
    if entry.get("summary"):
        return entry.summary
    content = entry.get("content")
    if content:
        return content[0].get("value")
    return None


class RSSFetcher(BaseFetcher):
    def fetch(self, source: Source) -> list[FetchedItem]:
        resp = self.client.get(source.url)
        resp.raise_for_status()
        parsed = feedparser.parse(resp.content)

        if parsed.bozo and not parsed.entries:
            raise FetchError(f"malformed feed: {parsed.get('bozo_exception')}")
        if parsed.bozo:
            logger.warning(
                "Feed %s parsed with errors: %s", source.url, parsed.get("bozo_exception")
            )

        items = []
        for entry in parsed.entries:
            link = entry.get("link")
            if not link:
                continue
            items.append(
                FetchedItem(
                    url=link,
                    title=entry.get("title", ""),
                    body=_entry_body(entry),
                    published_at=_entry_datetime(entry),
                )
            )
        return items
