from datetime import datetime, timezone

import pytest
from sqlalchemy.exc import IntegrityError

from models import Classification, Incident, IncidentItem, RawItem, Source, hash_url


def make_source(session, url="https://example.com/feed"):
    source = Source(name="Example", url=url, type="news")
    session.add(source)
    session.flush()
    return source


def make_raw_item(session, source, url="https://example.com/a"):
    item = RawItem(source_id=source.id, url=url, url_hash=hash_url(url), title="Title")
    session.add(item)
    session.flush()
    return item


def test_create_full_chain(session):
    source = make_source(session)
    item = make_raw_item(session, source)
    occurred = datetime(2026, 9, 30, 22, 15, tzinfo=timezone.utc)
    session.add(
        Classification(
            raw_item_id=item.id,
            is_crime=True,
            crime_type="shooting",
            in_stl=True,
            occurred_at=occurred,
            location="Downtown",
            confidence=0.92,
            model="claude-test",
            prompt_version="v1",
        )
    )
    incident = Incident(crime_type="shooting", occurred_at=occurred, location="Downtown")
    session.add(incident)
    session.flush()
    session.add(IncidentItem(incident_id=incident.id, raw_item_id=item.id))
    session.commit()
    session.expire_all()

    assert source.active is True
    assert source.fail_count == 0
    assert item.status == "new"
    assert item.retries == 0
    assert item.created_at is not None
    assert item.source.name == "Example"
    assert source.raw_items == [item]
    assert incident.status == "review"
    assert incident.raw_items == [item]
    assert session.query(Classification).one().crime_type == "shooting"


def test_url_hash_is_unique(session):
    source = make_source(session)
    make_raw_item(session, source, url="https://example.com/a")
    with pytest.raises(IntegrityError):
        make_raw_item(session, source, url="https://example.com/a")


def test_source_url_is_unique(session):
    make_source(session, url="https://example.com/feed")
    with pytest.raises(IntegrityError):
        make_source(session, url="https://example.com/feed")


def test_hash_url_is_deterministic():
    assert hash_url("https://x.com/a") == hash_url("https://x.com/a")
    assert hash_url("https://x.com/a") != hash_url("https://x.com/b")
    assert len(hash_url("https://x.com/a")) == 64


def test_invalid_status_rejected(session):
    source = make_source(session)
    item = RawItem(
        source_id=source.id, url="u", url_hash=hash_url("u"), title="t", status="bogus"
    )
    session.add(item)
    with pytest.raises(IntegrityError):
        session.flush()


def test_raw_item_requires_existing_source(session):
    session.add(RawItem(source_id=999, url="u", url_hash=hash_url("u"), title="t"))
    with pytest.raises(IntegrityError):
        session.flush()
