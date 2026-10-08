import logging
import math
import secrets
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from alerts import notify
from alerts.checks import WATCHDOG_WINDOW, last_success
from api import queries
from api.feedback import FeedbackIn, RateLimiter, client_ip, format_message
from api.schemas import (
    HealthResponse,
    IncidentOut,
    IncidentPatch,
    LastRun,
    StatsResponse,
    TimerResponse,
)
from config import get_settings
from matcher.edit import edit_incident
from matcher.merge import MergeError, merge_incidents
from models import PipelineRun

INDEX_HTML = Path(__file__).parent / "static" / "index.html"

logger = logging.getLogger(__name__)

app = FastAPI(title="STL Crime Tracker")
feedback_limiter = RateLimiter()


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


def get_feedback_webhook_url() -> str | None:
    return get_settings().feedback_webhook_url


def get_feedback_client() -> httpx.Client | None:
    """None = notify.send makes its own client."""
    return None


def get_feedback_limiter() -> RateLimiter:
    return feedback_limiter


def get_client_ip(
    request: Request, x_forwarded_for: str | None = Header(default=None)
) -> str:
    return client_ip(x_forwarded_for, request.client.host if request.client else None)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html")


# HEAD too: uptime monitors (UptimeRobot) check with HEAD by default.
@app.api_route(
    "/health",
    methods=["GET", "HEAD"],
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Pipeline is stale"}},
)
def health(
    response: Response,
    session: Session = Depends(get_session),
    now: datetime = Depends(get_now),
) -> HealthResponse:
    """503 when no pipeline run has succeeded in the last 30 minutes (or ever), so an
    external uptime monitor notices when the scheduled runs stop."""
    run = session.scalars(
        select(PipelineRun).order_by(PipelineRun.started_at.desc(), PipelineRun.id.desc())
    ).first()
    last_run = None
    if run is not None:
        last_run = LastRun(
            started_at=run.started_at, finished_at=run.finished_at, status=run.status
        )
    success_at = last_success(session)
    fresh = success_at is not None and now - success_at < WATCHDOG_WINDOW
    if not fresh:
        response.status_code = 503
    return HealthResponse(
        status="ok" if fresh else "stale", last_run=last_run, last_success_at=success_at
    )


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


@app.patch(
    "/admin/incidents/{incident_id}",
    response_model=IncidentOut,
    dependencies=[Depends(require_admin)],
)
def admin_edit(
    incident_id: int, patch: IncidentPatch, session: Session = Depends(get_session)
) -> IncidentOut:
    """Correct occurred_at/time_estimated, location, or neighborhood. Edited fields
    become manual: the matcher never overwrites them."""
    try:
        edit_incident(session, incident_id, patch.model_dump(exclude_unset=True))
    except MergeError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    return queries.incident_out(queries.get_incident(session, incident_id))


@app.post("/feedback")
def feedback(
    body: FeedbackIn,
    ip: str = Depends(get_client_ip),
    url: str | None = Depends(get_feedback_webhook_url),
    client: httpx.Client | None = Depends(get_feedback_client),
    limiter: RateLimiter = Depends(get_feedback_limiter),
) -> dict:
    # Honeypot filled: pretend it worked so the bot learns nothing.
    if body.website:
        return {"ok": True}
    if not url:
        logger.error("Feedback received but FEEDBACK_WEBHOOK_URL is not set")
        raise HTTPException(503, "Feedback is not configured")
    retry_after = limiter.hit(ip)
    if retry_after is not None:
        raise HTTPException(
            429,
            "Too many messages; please try again later",
            headers={"Retry-After": str(max(1, math.ceil(retry_after)))},
        )
    if not notify.send(format_message(body.message, body.contact), url=url, client=client):
        raise HTTPException(502, "Couldn't deliver feedback")
    return {"ok": True}
