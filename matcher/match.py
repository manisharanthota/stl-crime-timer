"""Group classified crime items into incidents."""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from config import get_settings
from matcher.location import location_similarity
from models import Classification, Incident, IncidentItem, RawItem
from timeutil import to_utc

logger = logging.getLogger(__name__)

MATCH_WINDOW = timedelta(hours=6)
CONFIRM_THRESHOLD = 0.8
# A shooting victim can later die, so these two may describe the same incident.
_COMPATIBLE_TYPES = {
    "shooting": ("shooting", "homicide"),
    "homicide": ("shooting", "homicide"),
    "burglary": ("burglary",),
}


@dataclass
class _Pending:
    classification: Classification
    occurred_at: datetime
    estimated: bool


def _pending(session: Session) -> tuple[list[_Pending], int]:
    """Newest classification of each unlinked item that is a St. Louis crime.

    Returns the pending items sorted oldest first, plus how many were skipped
    for having no usable time.
    """
    latest = (
        select(func.max(Classification.id))
        .group_by(Classification.raw_item_id)
        .scalar_subquery()
    )
    linked = select(IncidentItem.raw_item_id)
    rows = session.execute(
        select(Classification, RawItem.published_at)
        .join(RawItem, RawItem.id == Classification.raw_item_id)
        .where(
            Classification.id.in_(latest),
            Classification.is_crime.is_(True),
            Classification.in_stl.is_(True),
            Classification.crime_type.is_not(None),
            Classification.raw_item_id.not_in(linked),
        )
    ).all()

    pending, skipped = [], 0
    for c, published_at in rows:
        if c.occurred_at is not None:
            pending.append(_Pending(c, to_utc(c.occurred_at), False))
        elif published_at is not None:
            pending.append(_Pending(c, to_utc(published_at), True))
        else:
            logger.warning("raw_item %s has no occurred_at or published_at", c.raw_item_id)
            skipped += 1
    pending.sort(key=lambda p: (p.occurred_at, p.classification.raw_item_id))
    return pending, skipped


def _best_match(
    session: Session, item: _Pending, threshold: float
) -> Incident | None:
    """Best incident (including rejected ones) of a compatible type, time, and place."""
    c = item.classification
    candidates = session.scalars(
        select(Incident).where(
            Incident.crime_type.in_(_COMPATIBLE_TYPES[c.crime_type]),
            Incident.occurred_at >= item.occurred_at - MATCH_WINDOW,
            Incident.occurred_at <= item.occurred_at + MATCH_WINDOW,
        )
    ).all()
    scored = []
    for incident in candidates:
        score = location_similarity(c.location, incident.location)
        if score >= threshold:
            gap = abs(to_utc(incident.occurred_at) - item.occurred_at)
            # Prefer a live incident; a rejected one only matters if nothing else fits.
            scored.append((incident.status == "rejected", -score, gap, incident.id, incident))
    return min(scored)[-1] if scored else None


def _merge(incident: Incident, item: _Pending) -> None:
    c = item.classification
    if {incident.crime_type, c.crime_type} == {"shooting", "homicide"}:
        incident.crime_type = "homicide"
    current = to_utc(incident.occurred_at)
    if incident.time_estimated and not item.estimated:
        # A reported time beats a published_at fallback, even if later.
        incident.occurred_at = item.occurred_at
        incident.time_estimated = False
    elif incident.time_estimated == item.estimated and item.occurred_at < current:
        incident.occurred_at = item.occurred_at
    if incident.location is None and c.location:
        incident.location = c.location
    if c.confidence >= CONFIRM_THRESHOLD:
        incident.status = "confirmed"


def match_pending(session: Session, *, threshold: float | None = None) -> dict[str, int]:
    """Link every unlinked St. Louis crime classification to an incident.

    Items matching a rejected incident are left unlinked so a manually rejected
    crime doesn't come back as a new incident. Safe to run repeatedly.
    """
    if threshold is None:
        threshold = get_settings().match_location_threshold
    pending, skipped = _pending(session)
    counts = {"created": 0, "merged": 0, "skipped_rejected": 0, "skipped_no_time": skipped}

    for item in pending:
        c = item.classification
        incident = _best_match(session, item, threshold)
        if incident is not None and incident.status == "rejected":
            counts["skipped_rejected"] += 1
            continue
        if incident is None:
            incident = Incident(
                crime_type=c.crime_type,
                occurred_at=item.occurred_at,
                location=c.location,
                time_estimated=item.estimated,
                status="confirmed" if c.confidence >= CONFIRM_THRESHOLD else "review",
            )
            session.add(incident)
            session.flush()
            counts["created"] += 1
        else:
            _merge(incident, item)
            counts["merged"] += 1
        session.add(IncidentItem(incident_id=incident.id, raw_item_id=c.raw_item_id))
        session.flush()

    session.commit()
    return counts
