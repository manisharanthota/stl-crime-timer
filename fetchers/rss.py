"""RSS/Atom fetcher for news sites and RSS.app (Facebook) feeds."""

import calendar
import html
import logging
import re
from datetime import datetime, timezone

import feedparser

from fetchers.base import BaseFetcher, FetchedItem, FetchError
from models import Source

logger = logging.getLogger(__name__)

# Enough for the opening paragraphs, where the time and place of a crime are stated,
# without letting a full article eat the LLM's per-minute token budget.
MAX_BODY_CHARS = 1500

_DROP = re.compile(r"<(script|style)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_BREAK = re.compile(r"<\s*(br|/p|/div|/li|/h\d)\b[^>]*>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_SPACE = re.compile(r"[ \t\r\f\v\xa0]+")
_LINES = re.compile(r"\s*\n\s*")


def _entry_datetime(entry) -> datetime | None:
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if not parsed:
        return None
    # feedparser normalizes to a UTC struct_time.
    return datetime.fromtimestamp(calendar.timegm(parsed), tz=timezone.utc)


def html_to_text(value: str) -> str:
    """Plain text from feed HTML: tags dropped, entities decoded, paragraphs kept as
    single newlines, other whitespace collapsed."""
    text = _BREAK.sub("\n", _DROP.sub(" ", value))
    text = html.unescape(_TAG.sub(" ", text))
    text = _SPACE.sub(" ", text)
    return _LINES.sub("\n", text).strip()


def _truncate(text: str, limit: int = MAX_BODY_CHARS) -> str:
    if len(text) <= limit:
        return text
    cut = text[: limit - 2]  # room for " …"
    space = cut.rfind(" ")
    return (cut[:space] if space > limit // 2 else cut).rstrip() + " …"


def _entry_body(entry) -> str | None:
    """The fullest text the feed gives: some feeds (KMOV, STLPR, SLMPD) put the whole
    article in content:encoded next to a one-line summary, and the time of a crime is
    usually only in the article. Plain text, capped at MAX_BODY_CHARS."""
    candidates = [entry.get("summary") or ""]
    candidates += [c.get("value") or "" for c in entry.get("content") or []]
    text = max((html_to_text(c) for c in candidates), key=len)
    return _truncate(text) if text else None


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
