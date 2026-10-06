"""Request spacing (requests per minute) and token budgets (tokens per minute)."""

import time
from collections import deque
from collections.abc import Callable

# Extra seconds per request on top of 60/rpm, so clock skew doesn't trip the quota.
SAFETY_MARGIN_SECONDS = 1.0
TOKEN_WINDOW_SECONDS = 60.0


def interval_for_rpm(rpm: float) -> float:
    """Seconds between requests for a quota of `rpm` (5 RPM -> 13s)."""
    return 60.0 / rpm + SAFETY_MARGIN_SECONDS


class RateLimiter:
    """Minimum spacing between requests. The token arguments and hooks are only used
    by TokenRateLimiter."""

    otpm: int | None = None

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

    def wait(self, tokens: int = 0, output: int = 0) -> None:
        """Block until min_interval has passed since the previous wait() returned."""
        if self._last is not None:
            remaining = self.min_interval - (self._clock() - self._last)
            if remaining > 0:
                self._sleep(remaining)
        self._last = self._clock()

    def record(self, usage) -> None:
        """Called after each successful request with the client's last_usage."""

    def rate_limited(self, exc) -> None:
        """Called with each RateLimitError, to learn limits the provider names."""


class _Window:
    """Sliding 60-second window of [time, tokens] reservations."""

    def __init__(self, limit: int):
        self.limit = limit
        self.entries: deque[list] = deque()

    def used(self, now: float) -> int:
        while self.entries and now - self.entries[0][0] >= TOKEN_WINDOW_SECONDS:
            self.entries.popleft()
        return sum(tokens for _, tokens in self.entries)

    def wait_time(self, now: float, need: int) -> float:
        """Seconds until `need` more tokens fit (a request bigger than the whole
        limit waits for an empty window)."""
        need = min(need, self.limit)
        if not self.entries or self.used(now) + need <= self.limit:
            return 0.0
        return max(self.entries[0][0] + TOKEN_WINDOW_SECONDS - now, 0.01)


class TokenRateLimiter(RateLimiter):
    """Request spacing plus sliding 60-second windows of tokens: all tokens (`tpm`)
    and, when the provider limits them separately, output tokens (`otpm`).

    wait(tokens, output) reserves estimates once both fit; record(usage) swaps them for
    the tokens the provider says were used and remembers its remaining budget, so a
    budget the server says is spent is waited out too. An OTPM limit is also learned
    from a 429 that names it.
    """

    def __init__(
        self,
        min_interval: float,
        tpm: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        otpm: int | None = None,
    ):
        super().__init__(min_interval, clock, sleep)
        self.tpm = tpm
        self.otpm = otpm
        self._window = _Window(tpm)
        self._output = _Window(otpm or 0)
        self._server_remaining: int | None = None
        self._server_reset_at: float | None = None

    def used(self) -> int:
        return self._window.used(self._clock())

    def output_used(self) -> int:
        return self._output.used(self._clock())

    def _windows(self) -> list[tuple[_Window, int]]:
        return [(self._window, 0), (self._output, 1)] if self.otpm else [(self._window, 0)]

    def wait(self, tokens: int = 0, output: int = 0) -> None:
        super().wait()
        needs = (tokens, output)
        while True:
            delay = max(w.wait_time(self._clock(), needs[i]) for w, i in self._windows())
            if delay <= 0:
                break
            self._sleep(delay)
        if (
            self._server_reset_at is not None
            and self._server_remaining is not None
            and tokens > self._server_remaining
        ):
            remaining = self._server_reset_at - self._clock()
            if remaining > 0:
                self._sleep(remaining)
        self._server_remaining = self._server_reset_at = None
        now = self._clock()
        self._window.entries.append([now, tokens])
        self._output.entries.append([now, output])
        self._last = now

    def record(self, usage) -> None:
        if usage is None:
            return
        if usage.total_tokens is not None and self._window.entries:
            self._window.entries[-1][1] = usage.total_tokens
        if usage.completion_tokens is not None and self._output.entries:
            self._output.entries[-1][1] = usage.completion_tokens
        if usage.remaining_tokens is not None and usage.reset_seconds is not None:
            self._server_remaining = usage.remaining_tokens
            self._server_reset_at = self._clock() + usage.reset_seconds

    def rate_limited(self, exc) -> None:
        limit = getattr(exc, "output_limit", None)
        if limit and (self.otpm is None or limit < self.otpm):
            self.otpm = self._output.limit = limit
