"""In-memory cache for the public read endpoints.

Every visitor polls /timer, /stats, /incidents and /health once a minute. Without a
cache each poll is a database read, so database egress grows with the number of open
tabs. With it the database is read at most once per key per TTL, however many people
are watching. Only database results are cached: anything that depends on the current
time (seconds since, ongoing gaps, health staleness) is still computed per request.

Admin writes clear the cache so a confirm/reject/edit shows up at once. Pipeline
writes (another process) show up within the TTL.
"""

import threading
import time
from collections.abc import Callable, Hashable
from typing import Any, TypeVar

T = TypeVar("T")

READ_CACHE_TTL = 120.0  # seconds; also the public endpoints' Cache-Control max-age
CACHE_CONTROL = f"public, max-age={int(READ_CACHE_TTL)}"


class ReadCache:
    def __init__(self, ttl: float = READ_CACHE_TTL, clock: Callable[[], float] = time.monotonic):
        self.ttl = ttl
        self.clock = clock
        self._entries: dict[Hashable, tuple[float, Any]] = {}
        # One lock for loads too: concurrent misses on a key cost one query, not many.
        self._lock = threading.Lock()

    def get(self, key: Hashable, load: Callable[[], T]) -> T:
        with self._lock:
            entry = self._entries.get(key)
            now = self.clock()
            if entry is not None and now - entry[0] < self.ttl:
                return entry[1]
            value = load()  # an exception caches nothing
            self._entries[key] = (now, value)
            return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
