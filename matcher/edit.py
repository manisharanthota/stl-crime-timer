"""Manually correct an incident (admin edit)."""

from datetime import datetime

from sqlalchemy.orm import Session

from matcher.merge import MergeError
from matcher.neighborhoods import normalize_neighborhood
from models import Incident

EDITABLE = ("occurred_at", "time_estimated", "location", "neighborhood")


class EditError(MergeError):
    """The edit isn't allowed; `status` is the HTTP status to report."""


def edit_incident(session: Session, incident_id: int, changes: dict) -> Incident:
    """Apply `changes` (a subset of EDITABLE; None clears location/neighborhood) and
    mark each changed field manual, so the matcher and refresh_incidents never
    overwrite it. occurred_at and time_estimated are one field: setting either locks
    the time. A naive occurred_at is St. Louis time. Commits and returns the incident.
    """
    unknown = set(changes) - set(EDITABLE)
    if unknown:
        raise EditError(f"Can't edit {', '.join(sorted(unknown))}", 422)
    if not changes:
        raise EditError("Nothing to change", 422)
    if "occurred_at" in changes and not isinstance(changes["occurred_at"], datetime):
        raise EditError("occurred_at must be a datetime", 422)
    if "time_estimated" in changes and not isinstance(changes["time_estimated"], bool):
        raise EditError("time_estimated must be true or false", 422)
    neighborhood = changes.get("neighborhood")
    if neighborhood is not None:
        official = normalize_neighborhood(neighborhood)
        if official is None:
            raise EditError(f"Unknown neighborhood: {neighborhood!r}", 422)
        neighborhood = official

    incident = session.get(Incident, incident_id)
    if incident is None:
        raise EditError(f"Incident {incident_id} not found", 404)
    if incident.status == "merged":
        raise EditError(
            f"Incident {incident_id} is merged into {incident.merged_into_id}; edit that one",
            409,
        )

    if "occurred_at" in changes:
        incident.occurred_at = changes["occurred_at"]
    if "time_estimated" in changes:
        incident.time_estimated = changes["time_estimated"]
    if {"occurred_at", "time_estimated"} & set(changes):
        incident.manual_occurred_at = True
    if "location" in changes:
        incident.location = (changes["location"] or "").strip() or None
        incident.manual_location = True
    if "neighborhood" in changes:
        incident.neighborhood = neighborhood
        incident.manual_neighborhood = True
    session.commit()
    session.refresh(incident)  # read occurred_at back as aware UTC
    return incident
