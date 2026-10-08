"""Visitor feedback from the page, posted to a Discord webhook (POST /feedback)."""

import re
import threading
import time
from collections import deque
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

MAX_MESSAGE = 1000
MAX_CONTACT = 200
# A stripped name/email is cut to this (links removed first).
CONTACT_SHOWN = 100
RATE_LIMIT = 3
RATE_WINDOW = 600.0  # seconds

_EMAIL = re.compile(r"[^\s@]+@[^\s@]+\.[a-z]{2,}", re.IGNORECASE)
_LINKISH = re.compile(
    r"(?:[a-z][a-z0-9+.-]*://|www\.)"  # scheme or www.
    r"|[\w-]+(?:\.[\w-]+)*\.[a-z]{2,63}(?::\d+)?(?:[/?#]|$)",  # bare domain
    re.IGNORECASE,
)


class FeedbackIn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    message: str = Field(min_length=1, max_length=MAX_MESSAGE)
    contact: str | None = Field(default=None, max_length=MAX_CONTACT)
    # Honeypot: hidden on the page, so only bots fill it.
    website: str | None = None


def strip_links(text: str) -> str:
    """Drop every whitespace-separated word that is or holds a link (scheme, www., or
    a bare domain like evil.com/x); email addresses are kept. Whitespace collapses."""
    kept = []
    for word in text.split():
        bare = word.strip("()[]<>{}'\",;!")
        if _EMAIL.fullmatch(bare):
            kept.append(word)
        elif not _LINKISH.search(word) and not _LINKISH.search(bare):
            kept.append(word)
    return " ".join(kept)


def format_message(message: str, contact: str | None) -> str:
    name = strip_links(contact or "")[:CONTACT_SHOWN].strip()
    return f"**Feedback** from {name or 'anonymous'}\n{message}"


class RateLimiter:
    """At most `limit` hits per key in a sliding `window` of seconds. In memory, so it
    resets on restart; fine for one web instance."""

    def __init__(
        self,
        limit: int = RATE_LIMIT,
        window: float = RATE_WINDOW,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.limit = limit
        self.window = window
        self.clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str) -> float | None:
        """Record a hit for `key`. Returns None if allowed, else the seconds until the
        next one would be."""
        now = self.clock()
        with self._lock:
            if len(self._hits) > 10_000:
                self._prune(now)
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= now - self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return hits[0] + self.window - now
            hits.append(now)
            return None

    def _prune(self, now: float) -> None:
        for key in [k for k, h in self._hits.items() if not h or h[-1] <= now - self.window]:
            del self._hits[key]


def client_ip(forwarded_for: str | None, peer: str | None) -> str:
    """The visitor's IP. Behind Render's proxy the peer is the proxy, which appends
    the real client as the last X-Forwarded-For entry (earlier ones can be forged)."""
    if forwarded_for:
        last = forwarded_for.split(",")[-1].strip()
        if last:
            return last
    return peer or "unknown"
