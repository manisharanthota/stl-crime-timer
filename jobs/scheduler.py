"""Run the pipeline once or on an APScheduler interval, with graceful Ctrl+C.

Pipeline work runs off the main thread so Ctrl+C never lands mid-step: the first
Ctrl+C sets stop_event (the current step finishes, the rest are skipped, the lock is
released); a second one exits immediately.
"""

import logging
import os
import threading
import time
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler

from config import get_settings
from jobs.pipeline import run_pipeline

logger = logging.getLogger(__name__)

JOB_ID = "pipeline"


def _run_job(stop_event: threading.Event) -> None:
    if stop_event.is_set():
        return
    try:
        run_pipeline(stop_event=stop_event)
    except Exception:
        logger.exception("Pipeline run crashed")


def build_scheduler(
    stop_event: threading.Event, interval_minutes: float | None = None
) -> BackgroundScheduler:
    if interval_minutes is None:
        interval_minutes = get_settings().pipeline_interval_minutes
    scheduler = BackgroundScheduler(timezone=timezone.utc)
    scheduler.add_job(
        _run_job,
        "interval",
        minutes=interval_minutes,
        args=[stop_event],
        id=JOB_ID,
        max_instances=1,
        coalesce=True,
        next_run_time=datetime.now(timezone.utc),
    )
    return scheduler


def _wait(thread_alive, stop_event: threading.Event) -> None:
    """Block until thread_alive() is False, turning Ctrl+C into a graceful stop."""
    while thread_alive():
        try:
            # Short sleeps keep the main thread responsive to Ctrl+C on Windows.
            time.sleep(0.5)
        except KeyboardInterrupt:
            if stop_event.is_set():
                logger.warning("Second Ctrl+C: exiting without waiting")
                os._exit(1)
            logger.info("Ctrl+C: finishing the current step (Ctrl+C again to force)")
            stop_event.set()


def run_once() -> None:
    stop_event = threading.Event()
    worker = threading.Thread(target=_run_job, args=(stop_event,), name="pipeline")
    worker.start()
    _wait(worker.is_alive, stop_event)


def run_scheduled() -> None:
    stop_event = threading.Event()
    scheduler = build_scheduler(stop_event)
    scheduler.start()
    logger.info(
        "Scheduler started: pipeline every %s min", get_settings().pipeline_interval_minutes
    )
    _wait(lambda: not stop_event.is_set(), stop_event)
    logger.info("Shutting down scheduler; waiting for the current run")
    shutdown = threading.Thread(target=scheduler.shutdown, kwargs={"wait": True})
    shutdown.start()
    _wait(shutdown.is_alive, stop_event)
    logger.info("Scheduler stopped")
