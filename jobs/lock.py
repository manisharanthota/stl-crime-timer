"""A named lock stored as a job_locks row, so overlapping runs are impossible even
across processes. Each hold has an expiry; a lock past it is stale (its run crashed)
and the next run takes it over."""

import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy import delete, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from models import JobLock

logger = logging.getLogger(__name__)

LOCK_TTL = timedelta(minutes=30)


def _now() -> datetime:
    return datetime.now(timezone.utc)


def acquire(session: Session, name: str, owner: str, ttl: timedelta = LOCK_TTL) -> bool:
    """Take the lock for owner. Returns False if another owner holds a live lock."""
    now = _now()
    try:
        session.add(JobLock(name=name, owner=owner, acquired_at=now, expires_at=now + ttl))
        session.commit()
        return True
    except IntegrityError:
        session.rollback()

    # The row exists; take it over only if it has expired. The WHERE makes this
    # atomic: of two runs racing for a stale lock, only one UPDATE matches.
    result = session.execute(
        update(JobLock)
        .where(JobLock.name == name, JobLock.expires_at < now)
        .values(owner=owner, acquired_at=now, expires_at=now + ttl)
        .execution_options(synchronize_session=False)
    )
    session.commit()
    if result.rowcount == 1:
        logger.warning("Took over stale lock %r", name)
        return True
    return False


def refresh(session: Session, name: str, owner: str, ttl: timedelta = LOCK_TTL) -> bool:
    """Push the expiry forward. Returns False if owner no longer holds the lock."""
    result = session.execute(
        update(JobLock)
        .where(JobLock.name == name, JobLock.owner == owner)
        .values(expires_at=_now() + ttl)
        .execution_options(synchronize_session=False)
    )
    session.commit()
    return result.rowcount == 1


def release(session: Session, name: str, owner: str) -> None:
    """Drop the lock if owner still holds it (never someone else's takeover)."""
    session.execute(
        delete(JobLock)
        .where(JobLock.name == name, JobLock.owner == owner)
        .execution_options(synchronize_session=False)
    )
    session.commit()
