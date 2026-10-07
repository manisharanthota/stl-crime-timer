"""Manually fold one incident into another (admin merge)."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from matcher.match import COMPATIBLE_TYPES
from models import Incident, IncidentItem
from timeutil import to_utc


class MergeError(Exception):
    """The merge isn't allowed; `status` is the HTTP status to report."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


def merge_incidents(session: Session, source_id: int, target_id: int) -> Incident:
    """Move every item linked to `source` onto `target` and mark `source` merged.

    The target keeps the earliest reported time (or the earliest estimated one if
    neither has a reported time), ORs was_shooting, becomes a homicide if a shooting
    and a homicide merge, is confirmed if either was (never downgraded), and fills an
    empty location/neighborhood from the source. A field an admin set by hand wins:
    the target's, else the source's (which then stays manual on the target).
    Commits and returns the target.
    """
    if source_id == target_id:
        raise MergeError("Can't merge an incident into itself", 400)
    source = session.get(Incident, source_id)
    target = session.get(Incident, target_id)
    for incident_id, incident in ((source_id, source), (target_id, target)):
        if incident is None:
            raise MergeError(f"Incident {incident_id} not found", 404)
    if source.status == "merged" or target.status == "merged":
        raise MergeError("Incident is already merged", 409)
    if target.status == "rejected":
        raise MergeError("Can't merge into a rejected incident", 409)
    if target.crime_type not in COMPATIBLE_TYPES[source.crime_type]:
        raise MergeError(
            f"Can't merge a {source.crime_type} into a {target.crime_type}", 409
        )

    target_items = set(
        session.scalars(
            select(IncidentItem.raw_item_id).where(IncidentItem.incident_id == target.id)
        )
    )
    for link in session.scalars(
        select(IncidentItem).where(IncidentItem.incident_id == source.id)
    ).all():
        session.delete(link)
        if link.raw_item_id not in target_items:
            session.add(IncidentItem(incident_id=target.id, raw_item_id=link.raw_item_id))
    session.flush()

    if {source.crime_type, target.crime_type} == {"shooting", "homicide"}:
        target.crime_type = "homicide"
    if target.manual_occurred_at:
        pass
    elif source.manual_occurred_at:
        target.occurred_at, target.time_estimated = source.occurred_at, source.time_estimated
        target.manual_occurred_at = True
    else:
        # Reported times beat estimated ones; within the same kind, the earliest wins.
        target.occurred_at, target.time_estimated = min(
            (to_utc(i.occurred_at), i.time_estimated) for i in (source, target)
            if i.time_estimated == min(source.time_estimated, target.time_estimated)
        )
    target.was_shooting = target.was_shooting or source.was_shooting
    if "confirmed" in (source.status, target.status):
        target.status = "confirmed"
    if target.location is None and not target.manual_location:
        target.location, target.manual_location = source.location, source.manual_location
    if target.neighborhood is None and not target.manual_neighborhood:
        target.neighborhood = source.neighborhood
        target.manual_neighborhood = source.manual_neighborhood

    source.status = "merged"
    source.merged_into_id = target.id
    session.commit()
    session.refresh(target)  # reload raw_items with the moved links
    return target
