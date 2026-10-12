"""RSSFetcher item bodies: the fullest text the feed gives, as capped plain text."""

from pathlib import Path

import feedparser
import httpx

from fetchers.rss import MAX_BODY_CHARS, RSSFetcher, _entry_body, html_to_text
from models import Source

FEED = Path(__file__).parent / "fixtures" / "feeds" / "content.xml"
URL = "https://content.example/feed"


def fetch() -> dict[str, str | None]:
    client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=FEED.read_bytes())
    ))
    items = RSSFetcher(client).fetch(Source(name="Test", url=URL, type="news"))
    client.close()
    return {item.title: item.body for item in items}


def test_full_content_beats_summary():
    """A stated time often appears only in the article, not the one-line summary."""
    body = fetch()["Man shot to death at basketball court in park"]
    assert "around 6 p.m. Saturday at Loretta Hall Park along Carr Street." in body
    assert body.startswith("ST. LOUIS – Homicide detectives")


def test_body_is_plain_text():
    body = fetch()["Man shot to death at basketball court in park"]
    assert "<" not in body and "&nbsp;" not in body and "&#" not in body
    assert "trackView" not in body  # script contents dropped
    assert body.split("\n") == [
        "ST. LOUIS – Homicide detectives are investigating a fatal shooting at a basketball court.",
        "The shooting took place around 6 p.m. Saturday at Loretta Hall Park along Carr Street.",
        "Anyone with information should call CrimeStoppers.",
    ]


def test_summary_only_and_content_only_and_none():
    bodies = fetch()
    assert bodies["Summary only"] == "Police say a man was shot & wounded Friday night."
    assert bodies["Content only"] == "A business on Gravois was broken into overnight."
    assert bodies["No text"] is None


def test_longer_summary_wins_over_shorter_content():
    entry = feedparser.FeedParserDict(
        summary="<p>A long summary that says the shooting was at 9:15 p.m. Friday.</p>",
        content=[{"value": "<p>Short.</p>"}],
    )
    assert _entry_body(entry) == "A long summary that says the shooting was at 9:15 p.m. Friday."


def test_long_body_capped_at_word_boundary():
    words = " ".join(f"word{n}" for n in range(1000))
    entry = feedparser.FeedParserDict(content=[{"value": f"<p>{words}</p>"}])
    body = _entry_body(entry)
    assert len(body) <= MAX_BODY_CHARS
    assert body.endswith(" …")
    assert body[:-2].split(" ")[-1] in words.split(" ")  # no half word
    assert body.startswith("word0 word1 ")


def test_short_body_not_truncated():
    entry = feedparser.FeedParserDict(summary="x" * MAX_BODY_CHARS)
    assert _entry_body(entry) == "x" * MAX_BODY_CHARS


def test_html_to_text():
    assert html_to_text("a<br>b<br/>c") == "a\nb\nc"
    assert html_to_text("<div> one </div>\n\n<div>two</div>") == "one\ntwo"
    assert html_to_text("<style>p{color:red}</style>x&amp;y&#8217;s") == "x&y’s"
    assert html_to_text("   ") == ""
