import pytest

from api.cache import CACHE_CONTROL, READ_CACHE_TTL, ReadCache


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def counter():
    calls = {"n": 0}

    def load():
        calls["n"] += 1
        return calls["n"]

    return calls, load


def test_loads_once_per_ttl():
    clock = Clock()
    cache = ReadCache(ttl=120, clock=clock)
    calls, load = counter()
    assert cache.get("k", load) == 1
    clock.t = 119.9
    assert cache.get("k", load) == 1
    clock.t = 120
    assert cache.get("k", load) == 2
    assert calls["n"] == 2


def test_keys_are_separate():
    cache = ReadCache(clock=Clock())
    assert cache.get(("incidents", 20, 0), lambda: "a") == "a"
    assert cache.get(("incidents", 20, 20), lambda: "b") == "b"
    assert cache.get(("incidents", 20, 0), lambda: "c") == "a"


def test_clear_forces_reload():
    cache = ReadCache(clock=Clock())
    calls, load = counter()
    cache.get("k", load)
    cache.clear()
    assert cache.get("k", load) == 2


def test_failed_load_is_not_cached():
    cache = ReadCache(clock=Clock())

    def boom():
        raise RuntimeError("db down")

    with pytest.raises(RuntimeError):
        cache.get("k", boom)
    assert cache.get("k", lambda: "ok") == "ok"


def test_header_matches_ttl():
    assert READ_CACHE_TTL == 120
    assert CACHE_CONTROL == "public, max-age=120"
