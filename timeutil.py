"""Timezone rules: store and compute in UTC; St. Louis time is for display only.

Datetimes without a timezone from outside the app (LLM output, feeds, fixtures) are
taken to be St. Louis local time (America/Chicago, DST-aware).
"""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("America/Chicago")


def to_utc(value: datetime) -> datetime:
    """Aware UTC datetime. Naive input is interpreted as America/Chicago."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=LOCAL_TZ)
    return value.astimezone(timezone.utc)


def to_local(value: datetime) -> datetime:
    """St. Louis time for display. Naive input is assumed to already be UTC (as read
    from the database)."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(LOCAL_TZ)
