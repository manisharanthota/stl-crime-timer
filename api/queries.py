"""Read-side queries, computed from confirmed incidents.

The public endpoints cache what these load (api/cache.py), so each query selects only
the columns the page shows and is split from the parts that depend on the current
time (load_* -> build_*).
"""

from datetime import datetime

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, load_only, selectinload

from api.schemas import (
    AdminIncidentOut,
    ArticleOut,
    Gap,
    GapEntry,
    IncidentOut,
    LastIncident,
    StatsResponse,
    TimerEntry,
    TimerResponse,
)
from models import CRIME_TYPES, Incident, RawItem, Source

CONFIRMED = Incident.status == "confirmed"


def _counts_as_sql(crime_type: str):
    """Filter for incidents that count toward a crime type's timer. A fatal shooting
    is stored as a homicide with was_shooting=True and counts as a shooting too."""
    if crime_type == "shooting":
        return or_(Incident.crime_type == "shooting", Incident.was_shooting.is_(True))
    return Incident.crime_type == crime_type


def _counts_as(crime_type: str, incident_type: str, was_shooting: bool) -> bool:
    """Python twin of _counts_as_sql(), for rows already loaded."""
    if crime_type == "shooting":
        return incident_type == "shooting" or was_shooting
    return incident_type == crime_type


def _seconds(start: datetime, end: datetime) -> int:
    # Clamped: a bad extraction can put occurred_at in the future.
    return max(0, int((end - start).total_seconds()))


def load_latest(session: Session) -> dict[str, LastIncident | None]:
    """Latest confirmed incident per crime type (seconds_since left at 0; build_timer
    fills it in). One row per type, only the columns the timer shows."""
    latest: dict[str, LastIncident | None] = {}
    for crime_type in CRIME_TYPES:
        row = session.execute(
            select(
                Incident.id, Incident.crime_type, Incident.occurred_at, Incident.location,
                Incident.time_estimated, Incident.was_shooting,
            )
            .where(CONFIRMED, _counts_as_sql(crime_type))
            .order_by(Incident.occurred_at.desc(), Incident.id.desc())
            .limit(1)
        ).first()
        latest[crime_type] = None if row is None else LastIncident(
            incident_id=row.id,
            crime_type=row.crime_type,
            occurred_at=row.occurred_at,
            seconds_since=0,
            location=row.location,
            time_estimated=row.time_estimated,
            was_shooting=row.was_shooting,
        )
    return latest


def build_timer(latest: dict[str, LastIncident | None], now: datetime) -> TimerResponse:
    def since(last: LastIncident | None) -> LastIncident | None:
        if last is None:
            return None
        return last.model_copy(update={"seconds_since": _seconds(last.occurred_at, now)})

    by_type = [TimerEntry(crime_type=t, last=since(latest.get(t))) for t in CRIME_TYPES]
    lasts = [e.last for e in by_type if e.last is not None]
    overall = max(lasts, key=lambda l: (l.occurred_at, l.incident_id), default=None)
    return TimerResponse(
        now=now, overall=TimerEntry(crime_type=None, last=overall), by_type=by_type
    )


def timer(session: Session, now: datetime) -> TimerResponse:
    return build_timer(load_latest(session), now)


def incident_out(incident: Incident) -> IncidentOut:
    items = sorted(
        incident.raw_items,
        key=lambda i: (i.published_at is None, i.published_at, i.id),
    )
    return IncidentOut(
        id=incident.id,
        crime_type=incident.crime_type,
        occurred_at=incident.occurred_at,
        time_estimated=incident.time_estimated,
        was_shooting=incident.was_shooting,
        location=incident.location,
        neighborhood=incident.neighborhood,
        status=incident.status,
        articles=[
            ArticleOut(
                title=i.title,
                url=i.url,
                source_name=i.source.name,
                published_at=i.published_at,
            )
            for i in items
        ],
    )


def admin_incident_out(incident: Incident) -> AdminIncidentOut:
    return AdminIncidentOut(
        **incident_out(incident).model_dump(),
        manual_occurred_at=incident.manual_occurred_at,
        manual_location=incident.manual_location,
        manual_neighborhood=incident.manual_neighborhood,
    )


def _with_articles():
    """Linked articles without their bodies (up to 1500 chars each, never shown)."""
    return (
        selectinload(Incident.raw_items)
        .load_only(RawItem.id, RawItem.title, RawItem.url, RawItem.published_at, RawItem.source_id)
        .selectinload(RawItem.source)
        .load_only(Source.id, Source.name)
    )


def get_incident(session: Session, incident_id: int) -> Incident | None:
    return session.scalars(
        select(Incident).where(Incident.id == incident_id).options(_with_articles())
    ).first()


def recent_incidents(session: Session, limit: int, offset: int = 0) -> list[IncidentOut]:
    """One page of confirmed incidents, newest first, with their articles."""
    incidents = session.scalars(
        select(Incident)
        .where(CONFIRMED)
        .options(
            load_only(
                Incident.id, Incident.crime_type, Incident.occurred_at,
                Incident.time_estimated, Incident.was_shooting, Incident.location,
                Incident.neighborhood, Incident.status,
            ),
            _with_articles(),
        )
        .order_by(Incident.occurred_at.desc(), Incident.id.desc())
        .offset(offset)
        .limit(limit)
    ).all()
    return [incident_out(i) for i in incidents]


def review_incidents(session: Session) -> list[AdminIncidentOut]:
    incidents = session.scalars(
        select(Incident)
        .where(Incident.status == "review")
        .options(_with_articles())
        .order_by(Incident.occurred_at, Incident.id)
    ).all()
    return [admin_incident_out(i) for i in incidents]


def estimated_incidents(session: Session, limit: int) -> list[AdminIncidentOut]:
    """Newest review/confirmed incidents whose time is still an estimate (not set by
    hand), for /admin."""
    incidents = session.scalars(
        select(Incident)
        .where(
            Incident.status.in_(("review", "confirmed")),
            Incident.time_estimated.is_(True),
            Incident.manual_occurred_at.is_(False),
        )
        .options(_with_articles())
        .order_by(Incident.occurred_at.desc(), Incident.id.desc())
        .limit(limit)
    ).all()
    return [admin_incident_out(i) for i in incidents]


def _longest_gap(rows: list[tuple[int, datetime]], now: datetime) -> Gap | None:
    """Longest gap between consecutive incidents, counting the open gap up to now."""
    if not rows:
        return None
    best = None
    for (prev_id, prev_at), (next_id, next_at) in zip(rows, rows[1:]):
        seconds = _seconds(prev_at, next_at)
        if best is None or seconds > best.seconds:
            best = Gap(
                seconds=seconds,
                start_incident_id=prev_id,
                end_incident_id=next_id,
                start=prev_at,
                end=next_at,
                ongoing=False,
            )
    last_id, last_at = rows[-1]
    seconds = _seconds(last_at, now)
    if best is None or seconds > best.seconds:
        best = Gap(
            seconds=seconds,
            start_incident_id=last_id,
            end_incident_id=None,
            start=last_at,
            end=now,
            ongoing=True,
        )
    return best


StatsRow = tuple[int, str, bool, datetime]  # id, crime_type, was_shooting, occurred_at


def load_stats_rows(session: Session) -> list[StatsRow]:
    """Every confirmed incident, oldest first: the longest gap needs them all, so this
    selects four small columns and the endpoint caches the result."""
    return [
        tuple(r)
        for r in session.execute(
            select(Incident.id, Incident.crime_type, Incident.was_shooting, Incident.occurred_at)
            .where(CONFIRMED)
            .order_by(Incident.occurred_at, Incident.id)
        ).all()
    ]


def build_stats(rows: list[StatsRow], now: datetime) -> StatsResponse:
    overall = [(id_, at) for id_, _, _, at in rows]
    by_type = [
        GapEntry(
            crime_type=t,
            longest=_longest_gap(
                [(id_, at) for id_, type_, shot, at in rows if _counts_as(t, type_, shot)],
                now,
            ),
        )
        for t in CRIME_TYPES
    ]
    return StatsResponse(
        now=now,
        overall=GapEntry(crime_type=None, longest=_longest_gap(overall, now)),
        by_type=by_type,
    )


def stats(session: Session, now: datetime) -> StatsResponse:
    return build_stats(load_stats_rows(session), now)
