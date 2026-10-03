"""Minimum spacing between LLM requests, to stay under a requests-per-minute quota."""

import time
from collections.abc import Callable

# Extra seconds per request on top of 60/rpm, so clock skew doesn't trip the quota.
SAFETY_MARGIN_SECONDS = 1.0


def interval_for_rpm(rpm: float) -> float:
    """Seconds between requests for a quota of `rpm` (5 RPM -> 13s)."""
    return 60.0 / rpm + SAFETY_MARGIN_SECONDS


class RateLimiter:
    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        """Block until min_interval has passed since the previous wait() returned."""
        if self._last is not None:
            remaining = self.min_interval - (self._clock() - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()
