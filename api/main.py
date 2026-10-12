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
    AdminIncidentOut,
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
ADMIN_HTML = Path(__file__).parent / "static" / "admin.html"
# /admin page: never indexed, cached, framed, or leaked as a referrer.
ADMIN_PAGE_HEADERS = {
    "X-Robots-Tag": "noindex, nofollow",
    "Cache-Control": "no-store",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}
# Per IP: all /admin/* API calls, and wrong or missing tokens (brute force). An IP
# over the failure limit gets 429 even with the right token until the window passes.
ADMIN_RATE_LIMIT, ADMIN_RATE_WINDOW = 120, 60.0
ADMIN_FAIL_LIMIT, ADMIN_FAIL_WINDOW = 5, 900.0

logger = logging.getLogger(__name__)

app = FastAPI(title="STL Crime Tracker")
feedback_limiter = RateLimiter()
admin_limiter = RateLimiter(ADMIN_RATE_LIMIT, ADMIN_RATE_WINDOW)
admin_fail_limiter = RateLimiter(ADMIN_FAIL_LIMIT, ADMIN_FAIL_WINDOW)


def get_session() -> Iterator[Session]:
    from db import SessionLocal

    with SessionLocal() as session:
        yield session


def get_now() -> datetime:
    return datetime.now(timezone.utc)


def get_admin_token() -> str | None:
    return get_settings().admin_token


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


def get_admin_limiters() -> tuple[RateLimiter, RateLimiter]:
    """(all admin calls, failed tokens)."""
    return admin_limiter, admin_fail_limiter


def _too_many(retry_after: float, detail: str) -> HTTPException:
    return HTTPException(
        429, detail, headers={"Retry-After": str(max(1, math.ceil(retry_after)))}
    )


def require_admin(
    x_admin_token: str | None = Header(default=None),
    expected: str | None = Depends(get_admin_token),
    ip: str = Depends(get_client_ip),
    limiters: tuple[RateLimiter, RateLimiter] = Depends(get_admin_limiters),
) -> None:
    if not expected:
        raise HTTPException(503, "Admin is disabled: ADMIN_TOKEN is not set")
    requests, failures = limiters
    if (retry := failures.blocked(ip)) is not None:
        raise _too_many(retry, "Too many wrong admin tokens; try again later")
    if (retry := requests.hit(ip)) is not None:
        raise _too_many(retry, "Too many admin requests; slow down")
    if not x_admin_token:
        failures.hit(ip)
        raise HTTPException(401, "Missing X-Admin-Token")
    if not secrets.compare_digest(x_admin_token.encode(), expected.encode()):
        failures.hit(ip)
        raise HTTPException(401, "Wrong admin token")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/admin", include_in_schema=False)
def admin_page() -> FileResponse:
    """Static page; it holds no secrets. The token is typed in and sent as a header."""
    return FileResponse(ADMIN_HTML, media_type="text/html", headers=ADMIN_PAGE_HEADERS)


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
    response_model=list[AdminIncidentOut],
    dependencies=[Depends(require_admin)],
)
def admin_review(session: Session = Depends(get_session)) -> list[AdminIncidentOut]:
    return queries.review_incidents(session)


@app.get(
    "/admin/estimated",
    response_model=list[AdminIncidentOut],
    dependencies=[Depends(require_admin)],
)
def admin_estimated(
    limit: int = Query(50, ge=1, le=200), session: Session = Depends(get_session)
) -> list[AdminIncidentOut]:
    """Review/confirmed incidents whose time is an estimate nobody has fixed."""
    return queries.estimated_incidents(session, limit)


def _get_or_404(session: Session, incident_id: int):
    incident = queries.get_incident(session, incident_id)
    if incident is None:
        raise HTTPException(404, f"Incident {incident_id} not found")
    return incident


@app.get(
    "/admin/incidents/{incident_id}",
    response_model=AdminIncidentOut,
    dependencies=[Depends(require_admin)],
)
def admin_incident(incident_id: int, session: Session = Depends(get_session)) -> AdminIncidentOut:
    return queries.admin_incident_out(_get_or_404(session, incident_id))


def _set_status(session: Session, incident_id: int, status: str) -> AdminIncidentOut:
    incident = _get_or_404(session, incident_id)
    incident.status = status
    session.commit()
    return queries.admin_incident_out(incident)


@app.post(
    "/admin/incidents/{incident_id}/confirm",
    response_model=AdminIncidentOut,
    dependencies=[Depends(require_admin)],
)
def admin_confirm(incident_id: int, session: Session = Depends(get_session)) -> AdminIncidentOut:
    return _set_status(session, incident_id, "confirmed")


@app.post(
    "/admin/incidents/{incident_id}/reject",
    response_model=AdminIncidentOut,
    dependencies=[Depends(require_admin)],
)
def admin_reject(incident_id: int, session: Session = Depends(get_session)) -> AdminIncidentOut:
    return _set_status(session, incident_id, "rejected")


@app.post(
    "/admin/incidents/{incident_id}/merge",
    response_model=AdminIncidentOut,
    dependencies=[Depends(require_admin)],
)
def admin_merge(
    incident_id: int,
    into: int = Query(..., description="id of the incident to merge into"),
    session: Session = Depends(get_session),
) -> AdminIncidentOut:
    """Fold incident_id into `into`; incident_id becomes status=merged."""
    try:
        target = merge_incidents(session, incident_id, into)
    except MergeError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    return queries.admin_incident_out(queries.get_incident(session, target.id))


@app.patch(
    "/admin/incidents/{incident_id}",
    response_model=AdminIncidentOut,
    dependencies=[Depends(require_admin)],
)
def admin_edit(
    incident_id: int, patch: IncidentPatch, session: Session = Depends(get_session)
) -> AdminIncidentOut:
    """Correct occurred_at/time_estimated, location, or neighborhood. Edited fields
    become manual: the matcher never overwrites them."""
    try:
        edit_incident(session, incident_id, patch.model_dump(exclude_unset=True))
    except MergeError as exc:
        raise HTTPException(exc.status, str(exc)) from exc
    return queries.admin_incident_out(queries.get_incident(session, incident_id))


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
        raise _too_many(retry_after, "Too many messages; please try again later")
    if not notify.send(format_message(body.message, body.contact), url=url, client=client):
        raise HTTPException(502, "Couldn't deliver feedback")
    return {"ok": True}
