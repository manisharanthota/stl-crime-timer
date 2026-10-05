"""Read-side queries. Everything is computed from confirmed incidents on each call."""

from datetime import datetime

from sqlalchemy import or_, select
from sqlalchemy.orm import Session, selectinload

from api.schemas import (
    ArticleOut,
    Gap,
    GapEntry,
    IncidentOut,
    LastIncident,
    StatsResponse,
    TimerEntry,
    TimerResponse,
)
from models import CRIME_TYPES, Incident, RawItem

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


def timer(session: Session, now: datetime) -> TimerResponse:
    by_type = []
    for crime_type in CRIME_TYPES:
        incident = session.scalars(
            select(Incident)
            .where(CONFIRMED, _counts_as_sql(crime_type))
            .order_by(Incident.occurred_at.desc(), Incident.id.desc())
            .limit(1)
        ).first()
        last = None
        if incident is not None:
            last = LastIncident(
                incident_id=incident.id,
                crime_type=incident.crime_type,
                occurred_at=incident.occurred_at,
                seconds_since=_seconds(incident.occurred_at, now),
                location=incident.location,
                time_estimated=incident.time_estimated,
                was_shooting=incident.was_shooting,
            )
        by_type.append(TimerEntry(crime_type=crime_type, last=last))

    lasts = [e.last for e in by_type if e.last is not None]
    overall = max(lasts, key=lambda l: (l.occurred_at, l.incident_id), default=None)
    return TimerResponse(
        now=now, overall=TimerEntry(crime_type=None, last=overall), by_type=by_type
    )


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


def _with_articles():
    return selectinload(Incident.raw_items).selectinload(RawItem.source)


def get_incident(session: Session, incident_id: int) -> Incident | None:
    return session.scalars(
        select(Incident).where(Incident.id == incident_id).options(_with_articles())
    ).first()


def recent_incidents(session: Session, limit: int) -> list[IncidentOut]:
    incidents = session.scalars(
        select(Incident)
        .where(CONFIRMED)
        .options(_with_articles())
        .order_by(Incident.occurred_at.desc(), Incident.id.desc())
        .limit(limit)
    ).all()
    return [incident_out(i) for i in incidents]


def review_incidents(session: Session) -> list[IncidentOut]:
    incidents = session.scalars(
        select(Incident)
        .where(Incident.status == "review")
        .options(_with_articles())
        .order_by(Incident.occurred_at, Incident.id)
    ).all()
    return [incident_out(i) for i in incidents]


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


def stats(session: Session, now: datetime) -> StatsResponse:
    rows = session.execute(
        select(Incident.id, Incident.crime_type, Incident.was_shooting, Incident.occurred_at)
        .where(CONFIRMED)
        .order_by(Incident.occurred_at, Incident.id)
    ).all()
    overall = [(r.id, r.occurred_at) for r in rows]
    by_type = [
        GapEntry(
            crime_type=t,
            longest=_longest_gap(
                [
                    (r.id, r.occurred_at)
                    for r in rows
                    if _counts_as(t, r.crime_type, r.was_shooting)
                ],
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
