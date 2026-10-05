"""One pipeline run: fetch -> classify -> match, under a DB lock, recorded in
pipeline_runs (the heartbeat)."""

import logging
import threading
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime, timezone

from sqlalchemy.orm import Session, sessionmaker

from classifier.classify import classify_pending
from fetchers.runner import run_fetchers
from jobs import lock
from matcher.match import match_pending
from models import PipelineRun

logger = logging.getLogger(__name__)

LOCK_NAME = "pipeline"

Step = Callable[[Session], dict]
STEPS: list[tuple[str, Step]] = [
    ("fetch", run_fetchers),
    ("classify", classify_pending),
    ("match", match_pending),
]


def _status(errors: dict[str, str], step_count: int) -> str:
    if not errors:
        return "success"
    return "failed" if len(errors) == step_count else "partial"


def run_pipeline(
    session_factory: Callable[[], Session] | sessionmaker | None = None,
    steps: Sequence[tuple[str, Step]] | None = None,
    stop_event: threading.Event | None = None,
) -> PipelineRun | None:
    """Run each step in order, each with its own session. A failed step is logged and
    recorded; later steps still run. If stop_event is set, the current step finishes
    and the rest are skipped. Returns the finished PipelineRun, or None if another run
    holds the lock (nothing is recorded then)."""
    if session_factory is None:
        from db import SessionLocal

        session_factory = SessionLocal
    steps = STEPS if steps is None else steps
    owner = uuid.uuid4().hex

    with session_factory() as meta:
        if not lock.acquire(meta, LOCK_NAME, owner):
            logger.info("Another pipeline run holds the lock; skipping")
            return None

        try:
            run = PipelineRun(started_at=datetime.now(timezone.utc), status="running")
            meta.add(run)
            meta.commit()
            logger.info("Pipeline run %d started", run.id)

            counts: dict[str, dict] = {}
            errors: dict[str, str] = {}
            for name, step in steps:
                if stop_event is not None and stop_event.is_set():
                    errors[name] = "skipped: shutdown"
                    logger.info("Skipping %s: shutting down", name)
                    continue
                if not lock.refresh(meta, LOCK_NAME, owner):
                    errors[name] = "skipped: lock lost"
                    logger.error("Lost the pipeline lock; skipping %s", name)
                    continue
                with session_factory() as session:
                    try:
                        counts[name] = step(session)
                        logger.info("Step %s done: %s", name, counts[name])
                    except Exception as exc:
                        session.rollback()
                        errors[name] = f"{type(exc).__name__}: {exc}"
                        logger.exception("Step %s failed; continuing", name)

            for name, _ in steps:
                setattr(run, f"{name}_counts", counts.get(name))
                setattr(run, f"{name}_error", errors.get(name))
            run.status = _status(errors, len(steps))
            run.finished_at = datetime.now(timezone.utc)
            meta.commit()
            logger.info("Pipeline run %d finished: %s", run.id, run.status)
            # Detach loaded, so the lock release's commit can't expire it.
            meta.refresh(run)
            meta.expunge(run)
            return run
        except BaseException:
            meta.rollback()
            raise
        finally:
            # Runs even if recording the run itself blew up, so the lock never leaks.
            lock.release(meta, LOCK_NAME, owner)
