import logging
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest
from sqlalchemy import select

import fetchers.__main__ as cli
from fetchers.base import REQUEST_TIMEOUT, USER_AGENT
from fetchers.runner import make_client, normalize_url, run_fetchers
from models import RawItem, Source, hash_url

FEEDS = Path(__file__).parent / "fixtures" / "feeds"

GOOD_URL = "https://good.example/feed"
BAD_URL = "https://bad.example/feed"
SLOW_URL = "https://slow.example/feed"
DUP_URL = "https://dup.example/feed"


def feed_bytes(name: str) -> bytes:
    return (FEEDS / name).read_bytes()


def handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if url == GOOD_URL:
        return httpx.Response(200, content=feed_bytes("good.xml"))
    if url == BAD_URL:
        return httpx.Response(200, content=feed_bytes("malformed.xml"))
    if url == DUP_URL:
        return httpx.Response(200, content=feed_bytes("duplicate.xml"))
    if url == SLOW_URL:
        raise httpx.ReadTimeout("timed out", request=request)
    raise AssertionError(f"unexpected request to {url}")


@pytest.fixture
def client():
    with httpx.Client(transport=httpx.MockTransport(handler)) as c:
        yield c


def add_source(session, url, name=None, active=True, fail_count=0):
    source = Source(
        name=name or url, url=url, type="news", active=active, fail_count=fail_count
    )
    session.add(source)
    session.commit()
    return source


def raw_items(session):
    return session.scalars(select(RawItem).order_by(RawItem.id)).all()


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://a.example/story/?utm_source=rss", "https://a.example/story"),
        ("https://a.example/story#comments", "https://a.example/story"),
        ("https://a.example/story", "https://a.example/story"),
        ("https://a.example/", "https://a.example"),
    ],
)
def test_normalize_url(url, expected):
    assert normalize_url(url) == expected


def test_good_feed_inserts_new_items(session, client):
    source = add_source(session, GOOD_URL, fail_count=3)

    assert run_fetchers(session, client) == {GOOD_URL: 3}

    items = raw_items(session)
    assert [i.title for i in items] == [
        "Man shot in north St. Louis",
        "Burglary reported in Soulard",
        "City council passes budget",
    ]
    assert all(i.status == "new" and i.source_id == source.id for i in items)
    assert items[0].body == "Police say a man was shot Tuesday night."
    assert items[2].body is None
    # Original link is stored; hash is of the normalized link.
    assert items[1].url == "https://news.example/2026/10/01/soulard-burglary?utm_source=rss"
    assert items[1].url_hash == hash_url("https://news.example/2026/10/01/soulard-burglary")

    published = [i.published_at for i in items[:2]]
    assert published == [
        datetime(2026, 10, 1, 3, 15, tzinfo=timezone.utc),
        datetime(2026, 10, 1, 17, 0, tzinfo=timezone.utc),
    ]
    assert items[2].published_at is None

    session.refresh(source)
    assert source.fail_count == 0
    assert source.last_success_at is not None


def test_malformed_feed_counts_as_failure(session, client, caplog):
    source = add_source(session, BAD_URL)

    with caplog.at_level(logging.ERROR):
        assert run_fetchers(session, client) == {}

    assert raw_items(session) == []
    session.refresh(source)
    assert source.fail_count == 1
    assert source.last_success_at is None
    assert "malformed feed" in caplog.text


def test_timeout_counts_as_failure_and_is_logged(session, client, caplog):
    source = add_source(session, SLOW_URL, fail_count=2)

    with caplog.at_level(logging.ERROR):
        assert run_fetchers(session, client) == {}

    session.refresh(source)
    assert source.fail_count == 3
    assert source.last_success_at is None
    assert SLOW_URL in caplog.text
    assert "ReadTimeout" in caplog.text


def test_duplicates_are_skipped(session, client):
    add_source(session, GOOD_URL)
    add_source(session, DUP_URL)

    # duplicate.xml repeats a good.xml story (different query string) and lists
    # one new story twice (trailing slash vs query string).
    assert run_fetchers(session, client) == {GOOD_URL: 3, DUP_URL: 1}
    assert len(raw_items(session)) == 4
    assert [i.title for i in raw_items(session)][-1] == "Homicide in Dutchtown"

    # Second run: everything already stored.
    assert run_fetchers(session, client) == {GOOD_URL: 0, DUP_URL: 0}
    assert len(raw_items(session)) == 4


def test_failing_source_does_not_stop_others(session, client):
    bad = add_source(session, SLOW_URL)
    good = add_source(session, GOOD_URL)

    assert run_fetchers(session, client) == {GOOD_URL: 3}

    session.refresh(bad)
    session.refresh(good)
    assert bad.fail_count == 1
    assert good.fail_count == 0
    assert good.last_success_at is not None
    assert len(raw_items(session)) == 3


def test_inactive_sources_are_skipped(session, client):
    add_source(session, "https://inactive.example/feed", active=False)

    # The handler raises on unknown URLs, so any request would fail the source.
    assert run_fetchers(session, client) == {}
    source = session.scalars(select(Source)).one()
    assert source.fail_count == 0


def test_default_client_has_timeout_and_user_agent():
    with make_client() as c:
        assert c.timeout == httpx.Timeout(REQUEST_TIMEOUT)
        assert c.headers["User-Agent"] == USER_AGENT


def test_cli_prints_counts(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_fetchers", lambda session: {"KSDK 5": 4, "Fox2": 0})
    cli.main()
    out = capsys.readouterr().out
    assert "KSDK 5: 4 new item(s)" in out
    assert "Fox2: 0 new item(s)" in out
    assert "2 source(s) fetched successfully" in out
