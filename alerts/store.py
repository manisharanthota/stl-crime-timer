"""Alert cooldowns and open/resolved state, kept in the alerts_sent table.

A key can be sent again once its cooldown has passed, or right away after it was
resolved (a recovery went out), so a new outage is never muted by the last one."""

from datetime import datetime, timedelta

from sqlalchemy.orm import Session

from config import get_settings
from models import AlertSent


def _cooldown(hours: float | None) -> timedelta:
    return timedelta(hours=get_settings().alert_cooldown_hours if hours is None else hours)


def should_send(session: Session, key: str, now: datetime, cooldown_hours: float | None = None) -> bool:
    row = session.get(AlertSent, key)
    if row is None or row.resolved_at is not None:
        return True
    return now - row.last_sent_at >= _cooldown(cooldown_hours)


def was_sent(session: Session, key: str) -> bool:
    """For one-shot alerts (a specific incident): sent ever, regardless of cooldown."""
    return session.get(AlertSent, key) is not None


def is_open(session: Session, key: str) -> bool:
    """True if the key was alerted and no recovery has gone out since."""
    row = session.get(AlertSent, key)
    return row is not None and row.resolved_at is None


def mark_sent(session: Session, key: str, now: datetime) -> None:
    row = session.get(AlertSent, key)
    if row is None:
        session.add(AlertSent(key=key, last_sent_at=now))
    else:
        row.last_sent_at = now
        row.resolved_at = None
    session.commit()


def mark_resolved(session: Session, key: str, now: datetime) -> None:
    row = session.get(AlertSent, key)
    if row is not None:
        row.resolved_at = now
        session.commit()
