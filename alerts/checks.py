"""Alert conditions. check_after_run() runs after each pipeline run; run_watchdog() is
its own scheduler job. Each check is isolated: one that raises is logged and skipped,
and nothing here ever propagates into the pipeline."""

import logging
from collections.abc import Callable
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from alerts import notify, store
from config import get_settings
from models import Incident, PipelineRun, RawItem, Source
from timeutil import to_local

logger = logging.getLogger(__name__)

Sender = Callable[[str], bool]

SOURCE_FAIL_THRESHOLD = 5
WATCHDOG_WINDOW = timedelta(minutes=30)
STUCK_MIN_ITEMS = 20  # alert when more than this many are stuck
STUCK_AGE = timedelta(hours=2)

WATCHDOG_KEY = "watchdog"
QUOTA_KEY = "llm_quota"
STUCK_KEY = "stuck_new"


def _fmt(dt: datetime | None) -> str:
    return to_local(dt).strftime("%b %d %I:%M %p") if dt is not None else "never"


def _alert(session: Session, key: str, text: str, now: datetime, send: Sender) -> bool:
    """Send unless the key is cooling down; record it only if the send worked."""
    if not store.should_send(session, key, now):
        logger.debug("Alert %s suppressed by cooldown", key)
        return False
    if not send(text):
        return False
    store.mark_sent(session, key, now)
    return True


def _recover(session: Session, key: str, text: str, now: datetime, send: Sender) -> bool:
    """Send a recovery message if the key has an open alert."""
    if not store.is_open(session, key) or not send(text):
        return False
    store.mark_resolved(session, key, now)
    return True


def check_sources(session: Session, now: datetime, send: Sender) -> None:
    for source in session.scalars(select(Source).where(Source.active)).all():
        key = f"source_down:{source.id}"
        fails = source.fail_count or 0
        if fails >= SOURCE_FAIL_THRESHOLD:
            _alert(
                session, key,
                f"🔴 Source down: **{source.name}** has failed {fails} fetches in a row.\n"
                f"Last success: {_fmt(source.last_success_at)}\n<{source.url}>",
                now, send,
            )
        elif fails == 0:
            _recover(
                session, key, f"🟢 Source recovered: **{source.name}** is fetching again.",
                now, send,
            )


def check_quota(session: Session, run: PipelineRun, now: datetime, send: Sender) -> None:
    counts = run.classify_counts or {}
    if counts.get("stopped_quota"):
        settings = get_settings()
        models = ", ".join(m for m in (settings.gemini_model, settings.gemini_fallback_model) if m)
        _alert(
            session, QUOTA_KEY,
            f"🟠 LLM quota: every model ({models}) hit its daily quota. "
            "Items stay new until the quota resets.",
            now, send,
        )
    elif counts.get("classified"):
        _recover(session, QUOTA_KEY, "🟢 LLM quota: classifying again.", now, send)


def check_stuck(session: Session, now: datetime, send: Sender) -> None:
    stuck = session.scalar(
        select(func.count()).select_from(RawItem).where(
            RawItem.status == "new", RawItem.created_at < now - STUCK_AGE
        )
    )
    if stuck > STUCK_MIN_ITEMS:
        _alert(
            session, STUCK_KEY,
            f"🟠 Backlog: {stuck} items have been waiting to be classified for over "
            f"{STUCK_AGE.total_seconds() / 3600:g} hours.",
            now, send,
        )


def _article_url(incident: Incident) -> str | None:
    items = sorted(incident.raw_items, key=lambda i: (i.published_at is None, i.published_at, i.id))
    return items[0].url if items else None


def _incident_line(incident: Incident) -> str:
    kind = incident.crime_type + (" (shooting)" if incident.was_shooting and incident.crime_type == "homicide" else "")
    place = incident.location or "location unknown"
    if incident.neighborhood:
        place += f" ({incident.neighborhood})"
    when = _fmt(incident.occurred_at) + (" (estimated)" if incident.time_estimated else "")
    url = _article_url(incident)
    return f"#{incident.id} **{kind}**: {place}, {when}" + (f"\n{url}" if url else "")


def check_new_incidents(session: Session, run: PipelineRun, now: datetime, send: Sender) -> None:
    """Alert on incidents this run created: review always, confirmed if
    ALERT_ON_NEW_INCIDENT. Each incident is announced once (keyed by id)."""
    # created_at can be stored at whole-second precision (SQLite CURRENT_TIMESTAMP).
    since = run.started_at.replace(microsecond=0)
    new = session.scalars(
        select(Incident).where(Incident.created_at >= since).order_by(Incident.id)
    ).all()

    groups = [("review", "review", "🟡 Needs your review")]
    if get_settings().alert_on_new_incident:
        groups.append(("confirmed", "incident", "🚨 New incident"))
    for status, prefix, title in groups:
        pending = [
            i for i in new
            if i.status == status and not store.was_sent(session, f"{prefix}:{i.id}")
        ]
        if not pending:
            continue
        text = f"{title} ({len(pending)}):\n" + "\n".join(_incident_line(i) for i in pending)
        if status == "review":
            text += "\nConfirm or reject in /admin/review."
        if send(text):
            for incident in pending:
                store.mark_sent(session, f"{prefix}:{incident.id}", now)


def last_success(session: Session) -> datetime | None:
    """When the newest successful pipeline run finished (also used by /health)."""
    return session.scalar(
        select(func.max(PipelineRun.finished_at)).where(PipelineRun.status == "success")
    )


def recover_watchdog(session: Session, now: datetime, send: Sender) -> None:
    last = last_success(session)
    if last is not None and now - last < WATCHDOG_WINDOW:
        _recover(
            session, WATCHDOG_KEY,
            f"🟢 Pipeline healthy again: successful run at {_fmt(last)}.", now, send,
        )


def check_watchdog(
    session: Session, now: datetime, send: Sender, started_at: datetime | None = None
) -> None:
    """Alert if no run has succeeded in WATCHDOG_WINDOW, counted from the last success
    or `started_at` (when the scheduler started), whichever is later."""
    last = last_success(session)
    reference = max((t for t in (last, started_at) if t is not None), default=None)
    if reference is not None and now - reference < WATCHDOG_WINDOW:
        recover_watchdog(session, now, send)
        return

    latest = session.scalars(
        select(PipelineRun).order_by(PipelineRun.started_at.desc()).limit(1)
    ).first()
    text = (
        f"🔴 No successful pipeline run in {WATCHDOG_WINDOW.total_seconds() / 60:g} minutes.\n"
        f"Last success: {_fmt(last)}"
    )
    if latest is not None:
        text += f"\nLatest run: #{latest.id} {latest.status}, started {_fmt(latest.started_at)}"
        errors = [
            f"{step}: {err}"
            for step in ("fetch", "classify", "match")
            if (err := getattr(latest, f"{step}_error"))
        ]
        if errors:
            text += "\n" + "\n".join(e[:300] for e in errors)
    _alert(session, WATCHDOG_KEY, text, now, send)


def _now(now: datetime | None) -> datetime:
    return now or datetime.now(timezone.utc)


def _run_checks(session: Session, checks: list[tuple[str, Callable[[], None]]]) -> None:
    for name, check in checks:
        try:
            check()
        except Exception:
            session.rollback()
            logger.exception("Alert check %s failed; continuing", name)


def check_after_run(
    session: Session, run: PipelineRun, *, now: datetime | None = None,
    send: Sender = notify.send,
) -> None:
    now = _now(now)
    checks = [
        ("sources", lambda: check_sources(session, now, send)),
        ("quota", lambda: check_quota(session, run, now, send)),
        ("stuck", lambda: check_stuck(session, now, send)),
        ("incidents", lambda: check_new_incidents(session, run, now, send)),
    ]
    if run.status == "success":
        checks.append(("watchdog", lambda: recover_watchdog(session, now, send)))
    _run_checks(session, checks)


def run_watchdog(
    session_factory=None, *, started_at: datetime | None = None,
    now: datetime | None = None, send: Sender = notify.send,
) -> None:
    """The scheduler's watchdog job. Never raises."""
    try:
        if session_factory is None:
            from db import SessionLocal

            session_factory = SessionLocal
        with session_factory() as session:
            _run_checks(
                session,
                [("watchdog", lambda: check_watchdog(session, _now(now), send, started_at))],
            )
    except Exception:
        logger.exception("Watchdog failed")
