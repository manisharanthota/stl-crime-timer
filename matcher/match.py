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
# Used when the incident's or the item's time is a published_at fallback.
ESTIMATED_MATCH_WINDOW = timedelta(hours=24)
CONFIRM_THRESHOLD = 0.8
LIVE_STATUSES = ("confirmed", "review")
# A shooting victim can later die, so these two may describe the same incident.
COMPATIBLE_TYPES = {
    "shooting": ("shooting", "homicide"),
    "homicide": ("shooting", "homicide"),
    "burglary": ("burglary",),
}


def _was_shooting(c: Classification) -> bool:
    return bool(c.was_shooting) or c.crime_type == "shooting"


def _reported(c: Classification) -> bool:
    """occurred_at is a stated time of day, not a guess from a date."""
    return c.occurred_at is not None and c.time_precision == "exact"


def _latest_ids():
    """Subquery: the newest classification id of each raw_item."""
    return (
        select(func.max(Classification.id))
        .group_by(Classification.raw_item_id)
        .scalar_subquery()
    )


@dataclass
class _Pending:
    classification: Classification
    occurred_at: datetime
    estimated: bool


def _pending(session: Session) -> tuple[list[_Pending], int, int]:
    """Newest classification of each unlinked item that is a St. Louis crime.

    Returns the pending items sorted oldest first, plus how many were skipped for
    having no usable time, and how many follow-ups were skipped for having no
    occurred_at (their published_at can be days after the crime).
    """
    linked = select(IncidentItem.raw_item_id)
    rows = session.execute(
        select(Classification, RawItem.published_at)
        .join(RawItem, RawItem.id == Classification.raw_item_id)
        .where(
            Classification.id.in_(_latest_ids()),
            Classification.is_crime.is_(True),
            Classification.in_stl.is_(True),
            Classification.crime_type.is_not(None),
            Classification.raw_item_id.not_in(linked),
        )
    ).all()

    pending, skipped, skipped_followup = [], 0, 0
    for c, published_at in rows:
        if c.occurred_at is not None:
            # A date-only time is an estimate, like a published_at fallback.
            pending.append(_Pending(c, to_utc(c.occurred_at), not _reported(c)))
        elif c.is_followup:
            skipped_followup += 1
        elif published_at is not None:
            pending.append(_Pending(c, to_utc(published_at), True))
        else:
            logger.warning("raw_item %s has no occurred_at or published_at", c.raw_item_id)
            skipped += 1
    pending.sort(key=lambda p: (p.occurred_at, p.classification.raw_item_id))
    return pending, skipped, skipped_followup


def _best_match(
    session: Session, item: _Pending, threshold: float
) -> Incident | None:
    """Best incident (including rejected ones, never merged ones) of a compatible
    type, close enough in time, in the same neighborhood or a similar location."""
    c = item.classification
    candidates = session.scalars(
        select(Incident).where(
            Incident.status != "merged",
            Incident.crime_type.in_(COMPATIBLE_TYPES[c.crime_type]),
            Incident.occurred_at >= item.occurred_at - ESTIMATED_MATCH_WINDOW,
            Incident.occurred_at <= item.occurred_at + ESTIMATED_MATCH_WINDOW,
        )
    ).all()
    scored = []
    for incident in candidates:
        gap = abs(to_utc(incident.occurred_at) - item.occurred_at)
        estimated = item.estimated or incident.time_estimated
        if gap > (ESTIMATED_MATCH_WINDOW if estimated else MATCH_WINDOW):
            continue
        score = location_similarity(c.location, incident.location)
        same_neighborhood = bool(c.neighborhood) and c.neighborhood == incident.neighborhood
        if same_neighborhood or score >= threshold:
            # Prefer a live incident; a rejected one only matters if nothing else fits.
            scored.append((incident.status == "rejected", -score, gap, incident.id, incident))
    return min(scored)[-1] if scored else None


def _merge(incident: Incident, item: _Pending) -> None:
    c = item.classification
    if {incident.crime_type, c.crime_type} == {"shooting", "homicide"}:
        incident.crime_type = "homicide"
    if _was_shooting(c):
        incident.was_shooting = True
    current = to_utc(incident.occurred_at)
    if incident.time_estimated and not item.estimated:
        # A reported time beats a published_at fallback, even if later.
        incident.occurred_at = item.occurred_at
        incident.time_estimated = False
    elif incident.time_estimated == item.estimated and item.occurred_at < current:
        incident.occurred_at = item.occurred_at
    if incident.location is None and c.location:
        incident.location = c.location
    if incident.neighborhood is None and c.neighborhood:
        incident.neighborhood = c.neighborhood
    if c.confidence >= CONFIRM_THRESHOLD:
        incident.status = "confirmed"


def refresh_incidents(session: Session) -> int:
    """Re-apply linked items' newest classifications to live incidents.

    The matcher only looks at unlinked items, so a re-classified linked item would
    otherwise never reach its incident. From each linked item's newest St. Louis
    crime classification:
    - if any has an exact occurred_at, the incident takes the earliest exact one
      (even if later: a stated 00:40 replaces a guessed midnight) and stops being
      estimated; otherwise its time is left alone;
    - was_shooting is turned on, never off;
    - an empty location/neighborhood is filled.
    Rejected and merged incidents are left alone. Returns how many changed.
    """
    rows = session.execute(
        select(Incident, Classification)
        .join(IncidentItem, IncidentItem.incident_id == Incident.id)
        .join(Classification, Classification.raw_item_id == IncidentItem.raw_item_id)
        .where(
            Incident.status.in_(LIVE_STATUSES),
            Classification.id.in_(_latest_ids()),
            Classification.is_crime.is_(True),
            Classification.in_stl.is_(True),
        )
        .order_by(Incident.id, Classification.id)
    ).all()
    by_incident: dict[int, tuple[Incident, list[Classification]]] = {}
    for incident, c in rows:
        by_incident.setdefault(incident.id, (incident, []))[1].append(c)

    changed = 0
    for incident, classifications in by_incident.values():
        before = (
            incident.occurred_at, incident.time_estimated, incident.was_shooting,
            incident.location, incident.neighborhood,
        )
        reported = [to_utc(c.occurred_at) for c in classifications if _reported(c)]
        if reported:
            incident.occurred_at = min(reported)
            incident.time_estimated = False
        if any(_was_shooting(c) for c in classifications):
            incident.was_shooting = True
        if incident.location is None:
            incident.location = next((c.location for c in classifications if c.location), None)
        if incident.neighborhood is None:
            incident.neighborhood = next(
                (c.neighborhood for c in classifications if c.neighborhood), None
            )
        after = (
            incident.occurred_at, incident.time_estimated, incident.was_shooting,
            incident.location, incident.neighborhood,
        )
        if after != before:
            changed += 1
    return changed


def match_pending(session: Session, *, threshold: float | None = None) -> dict[str, int]:
    """Link every unlinked St. Louis crime classification to an incident, then
    refresh live incidents from their linked items' newest classifications.

    Items matching a rejected incident are left unlinked so a manually rejected
    crime doesn't come back as a new incident. Safe to run repeatedly.
    """
    if threshold is None:
        threshold = get_settings().match_location_threshold
    pending, skipped, skipped_followup = _pending(session)
    counts = {
        "created": 0, "merged": 0, "skipped_rejected": 0, "skipped_no_time": skipped,
        "skipped_followup_no_time": skipped_followup, "refreshed": 0,
    }

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
                neighborhood=c.neighborhood,
                time_estimated=item.estimated,
                was_shooting=_was_shooting(c),
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

    counts["refreshed"] = refresh_incidents(session)
    session.commit()
    return counts
