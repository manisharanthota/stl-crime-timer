from sqlalchemy import select

from models import Source
from seed import load_sources, seed_sources

TWO_SOURCES = """
sources:
  - name: A
    url: https://a.example/feed
    type: news
  - name: B
    url: https://b.example/feed
    type: news
"""

THREE_SOURCES = TWO_SOURCES + """
  - name: C
    url: https://c.example/feed
    type: facebook
"""


def test_seed_skips_duplicates_by_url(session, tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text(TWO_SOURCES)
    assert seed_sources(session, path) == 2
    assert seed_sources(session, path) == 0

    path.write_text(THREE_SOURCES)
    assert seed_sources(session, path) == 1
    urls = session.scalars(select(Source.url).order_by(Source.url)).all()
    assert urls == ["https://a.example/feed", "https://b.example/feed", "https://c.example/feed"]


def test_real_sources_yaml_parses():
    sources = load_sources()
    assert sources
    for entry in sources:
        assert {"name", "url", "type"} <= entry.keys()
