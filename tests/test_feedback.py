"""POST /feedback and the page's feedback box. The Discord webhook is always an
httpx.MockTransport; nothing is posted for real."""

import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from api.feedback import RateLimiter, client_ip, format_message, strip_links
from api.main import (
    app,
    get_feedback_client,
    get_feedback_limiter,
    get_feedback_webhook_url,
)

ROOT = Path(__file__).resolve().parent.parent
WEBHOOK = "https://discord.com/api/webhooks/123/secret-token"


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def hook():
    """Records webhook posts; set `status` to make Discord fail."""
    state = {"posts": [], "status": 204}

    def handler(request):
        state["posts"].append(json.loads(request.content))
        return httpx.Response(state["status"])

    state["client"] = httpx.Client(transport=httpx.MockTransport(handler))
    yield state
    state["client"].close()


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def setup(hook, clock):
    limiter = RateLimiter(clock=clock)
    cfg = {"url": WEBHOOK}
    app.dependency_overrides[get_feedback_webhook_url] = lambda: cfg["url"]
    app.dependency_overrides[get_feedback_client] = lambda: hook["client"]
    app.dependency_overrides[get_feedback_limiter] = lambda: limiter
    yield cfg
    app.dependency_overrides.clear()


@pytest.fixture
def client(setup):
    return TestClient(app)


def post(client, ip="1.1.1.1", **body):
    body.setdefault("message", "Love the site")
    return client.post("/feedback", json=body, headers={"X-Forwarded-For": ip})


def test_sends_to_webhook_with_mentions_off(client, hook):
    r = post(client, message="Great work @everyone", contact="Jane")
    assert r.status_code == 200 and r.json() == {"ok": True}
    (payload,) = hook["posts"]
    assert payload["allowed_mentions"] == {"parse": []}
    assert payload["content"] == "**Feedback** from Jane\nGreat work @everyone"


def test_anonymous_and_whitespace_trimmed(client, hook):
    assert post(client, message="  hi there \n", contact="   ").status_code == 200
    assert hook["posts"][0]["content"] == "**Feedback** from anonymous\nhi there"


def test_links_stripped_from_name(client, hook):
    post(client, contact="Bob https://evil.com/x www.spam.io [x](http://a.b) jane@example.com")
    assert hook["posts"][0]["content"].startswith("**Feedback** from Bob jane@example.com\n")


@pytest.mark.parametrize("message", ["", "   \n ", "x" * 1001])
def test_rejects_empty_or_too_long(client, hook, message):
    assert post(client, message=message).status_code == 422
    assert hook["posts"] == []


def test_accepts_exactly_max_length(client, hook):
    assert post(client, message="x" * 1000).status_code == 200


def test_rejects_too_long_contact(client, hook):
    assert post(client, contact="a" * 201).status_code == 422
    assert hook["posts"] == []


def test_honeypot_fakes_success_and_sends_nothing(client, hook):
    r = post(client, website="http://spam.example")
    assert r.status_code == 200 and r.json() == {"ok": True}
    assert hook["posts"] == []
    # And doesn't use up the visitor's quota.
    for _ in range(3):
        assert post(client).status_code == 200


def test_rate_limit_per_ip(client, hook, clock):
    for _ in range(3):
        assert post(client).status_code == 200
    r = post(client)
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) == 600
    assert len(hook["posts"]) == 3
    # Another visitor is unaffected.
    assert post(client, ip="2.2.2.2").status_code == 200
    clock.now += 599
    assert post(client).status_code == 429
    clock.now += 1
    assert post(client).status_code == 200


def test_invalid_messages_dont_use_quota(client, hook):
    for _ in range(5):
        assert post(client, message="").status_code == 422
    for _ in range(3):
        assert post(client).status_code == 200


def test_missing_webhook_is_a_clear_error(client, hook, setup):
    setup["url"] = None
    r = post(client)
    assert r.status_code == 503
    assert r.json() == {"detail": "Feedback is not configured"}
    assert hook["posts"] == []


def test_missing_webhook_from_settings(hook):
    """Without overrides, the URL comes from FEEDBACK_WEBHOOK_URL (blank in tests)."""
    with TestClient(app) as c:
        assert post(c).status_code == 503


@pytest.mark.parametrize("status", [400, 500])
def test_discord_failure(client, hook, status):
    hook["status"] = status
    r = post(client)
    assert r.status_code == 502
    assert WEBHOOK not in r.text and "secret-token" not in r.text


def test_network_error(setup, clock):
    def boom(request):
        raise httpx.ConnectError("down")

    with httpx.Client(transport=httpx.MockTransport(boom)) as broken:
        app.dependency_overrides[get_feedback_client] = lambda: broken
        r = post(TestClient(app))
    assert r.status_code == 502


def test_webhook_url_never_in_responses(client):
    for r in (post(client), client.get("/"), post(client, message="")):
        assert "secret-token" not in r.text


# --- unit tests -------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Jane Doe", "Jane Doe"),
        ("jane.doe@example.com", "jane.doe@example.com"),
        ("Visit https://x.com now", "Visit now"),
        ("HTTP://X.COM", ""),
        ("see www.example.org", "see"),
        ("evil.com/path, ok", "ok"),
        ("(evil.co.uk)", ""),
        ("<https://a.b/c>", ""),
        ("[click](https://a.bc)", ""),
        ("ftp://files.host", ""),
        ("  lots   of   space ", "lots of space"),
    ],
)
def test_strip_links(text, expected):
    assert strip_links(text) == expected


def test_format_message_caps_name():
    assert format_message("m", "a" * 150) == f"**Feedback** from {'a' * 100}\nm"
    assert format_message("m", "https://only.link") == "**Feedback** from anonymous\nm"


def test_rate_limiter_window(clock):
    limiter = RateLimiter(limit=2, window=10, clock=clock)
    assert limiter.hit("a") is None
    clock.now += 4
    assert limiter.hit("a") is None
    assert limiter.hit("a") == 6
    clock.now += 6  # the first hit leaves the window
    assert limiter.hit("a") is None
    assert limiter.hit("b") is None


def test_rate_limiter_prunes_idle_keys(clock):
    limiter = RateLimiter(limit=1, window=10, clock=clock)
    for i in range(10_001):
        limiter.hit(str(i))
    clock.now += 10
    limiter.hit("new")
    assert set(limiter._hits) == {"new"}


@pytest.mark.parametrize(
    "header, peer, expected",
    [
        ("9.9.9.9", "10.0.0.1", "9.9.9.9"),
        ("6.6.6.6, 9.9.9.9", "10.0.0.1", "9.9.9.9"),  # forged first entry ignored
        (None, "10.0.0.1", "10.0.0.1"),
        (" , ", "10.0.0.1", "10.0.0.1"),
        (None, None, "unknown"),
    ],
)
def test_client_ip(header, peer, expected):
    assert client_ip(header, peer) == expected


def test_page_has_feedback_form():
    html = (ROOT / "api/static/index.html").read_text(encoding="utf-8")
    assert "Feedback or suggestion" in html
    assert 'id="fb-message" name="message" maxlength="1000"' in html
    assert 'name="website" tabindex="-1"' in html  # honeypot
    assert 'fetch("/feedback"' in html
    assert "discord" not in html.lower()


def test_env_example_lists_feedback_webhook_without_value():
    lines = (ROOT / ".env.example").read_text(encoding="utf-8").splitlines()
    assert "FEEDBACK_WEBHOOK_URL=" in lines
