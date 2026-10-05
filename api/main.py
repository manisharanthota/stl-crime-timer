import secrets
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from api import queries
from api.schemas import (
    HealthResponse,
    IncidentOut,
    LastRun,
    StatsResponse,
    TimerResponse,
)
from config import get_settings
from matcher.merge import MergeError, merge_incidents
from models import PipelineRun

INDEX_HTML = Path(__file__).parent / "static" / "index.html"

app = FastAPI(title="STL Crime Tracker")


def get_session() -> Iterator[Session]:
    from db import SessionLocal

    with SessionLocal() as session:
        yield session


def get_now() -> datetime:
    return datetime.now(timezone.utc)


def get_admin_token() -> str | None:
    return get_settings().admin_token


def require_admin(
    x_admin_token: str | None = Header(default=None),
    expected: str | None = Depends(get_admin_token),
) -> None:
    if not expected:
        raise HTTPException(503, "Admin is disabled: ADMIN_TOKEN is not set")
    if not x_admin_token or not secrets.compare_digest(
        x_admin_token.encode(), expected.encode()
    ):
        raise HTTPException(401, "Missing or invalid X-Admin-Token")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/health", response_model=HealthResponse)
def health(session: Session = Depends(get_session)) -> HealthResponse:
    run = session.scalars(
        select(PipelineRun).order_by(PipelineRun.started_at.desc(), PipelineRun.id.desc())
    ).first()
    last_run = None
    if run is not None:
        last_run = LastRun(
            started_at=run.started_at, finished_at=run.finished_at, status=run.status
        )
    return HealthResponse(status="ok", last_run=last_run)


@app.get("/timer", response_model=TimerResponse)
def timer(
    session: Session = Depends(get_session), now: datetime = Depends(get_now)
) -> TimerResponse:
    return queries.timer(session, now)


@app.get("/incidents", response_model=list[IncidentOut])
def incidents(
    limit: int = Query(20, ge=1, le=100), session: Session = Depends(get_session)
) -> list[IncidentOut]:
    return queries.recent_incidents(session, limit)


@app.get("/stats", response_model=StatsResponse)
def stats(
    session: Session = Depends(get_session), now: datetime = Depends(get_now)
) -> StatsResponse:
    return queries.stats(session, now)


@app.get(
    "/admin/review",
    response_model=list[IncidentOut],
    dependencies=[Depends(require_admin)],
)
def admin_review(session: Session = Depends(get_session)) -> list[IncidentOut]:
    return queries.review_incidents(session)


def _set_status(session: Session, incident_id: int, status: str) -> IncidentOut:
    incident = queries.get_incident(session, incident_id)
    if incident is None:
        raise HTTPException(404, f"Incident {incident_id} not found")
    incident.status = status
    session.commit()
    return queries.incident_out(incident)


@app.post(
    "/admin/incidents/{incident_id}/confirm",
    response_model=IncidentOut,
    dependencies=[Depends(require_admin)],
)
def admin_confirm(incident_id: int, session: Session = Depends(get_session)) -> IncidentOut:
    return _set_status(session, incident_id, "confirmed")


@app.post(
    "/admin/incidents/{incident_id}/reject",
    response_model=IncidentOut,
    dependencies=[Depends(require_admin)],
)
def admin_reject(incident_id: int, session: Session = Depends(get_session)) -> IncidentOut:
    return _set_status(session, incident_id, "rejected")


@app.post(
    "/admin/incidents/{incident_id}/merge",
    response_model=IncidentOut,
    dependencies=[Depends(require_admin)],
)
def admin_merge(
    incident_id: int,
    into: int = Query(..., description="id of the incident to merge into"),
    session: Session = Depends(get_session),
) -> IncidentOut:
    """Fold incident_id into `into`; incident_id becomes status=merged."""
    try:
        target = merge_incidents(session, incident_id, into)
    except MergeError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    return queries.incident_out(queries.get_incident(session, target.id))
