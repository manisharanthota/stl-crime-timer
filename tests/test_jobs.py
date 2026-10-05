import logging
import threading
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select

from jobs import lock
from jobs.logsetup import LOG_FILE, UTCFormatter, setup_logging
from jobs.pipeline import LOCK_NAME, STEPS, run_pipeline
from jobs.scheduler import JOB_ID, _run_job, build_scheduler
from models import JobLock, PipelineRun


class Recorder:
    """Fake pipeline steps that record call order; no network, no LLM."""

    def __init__(self, fail: tuple[str, ...] = ()):
        self.calls: list[str] = []
        self.fail = fail

    def step(self, name: str, counts: dict):
        def run(session):
            self.calls.append(name)
            if name in self.fail:
                raise RuntimeError(f"{name} broke")
            return counts

        return run

    def steps(self):
        return [
            ("fetch", self.step("fetch", {"KSDK": 3})),
            ("classify", self.step("classify", {"classified": 2, "stopped_quota": 1})),
            ("match", self.step("match", {"created": 1, "merged": 0})),
        ]


def _locks(factory) -> list[JobLock]:
    with factory() as s:
        return s.scalars(select(JobLock)).all()


def _runs(factory) -> list[PipelineRun]:
    with factory() as s:
        return s.scalars(select(PipelineRun).order_by(PipelineRun.id)).all()


def _hold_lock(factory, expires_at: datetime) -> None:
    with factory() as s:
        s.add(JobLock(
            name=LOCK_NAME, owner="other", acquired_at=expires_at - lock.LOCK_TTL,
            expires_at=expires_at,
        ))
        s.commit()


def test_default_steps_order():
    assert [name for name, _ in STEPS] == ["fetch", "classify", "match"]


def test_full_success(session_factory):
    rec = Recorder()
    run = run_pipeline(session_factory, rec.steps())

    assert rec.calls == ["fetch", "classify", "match"]
    assert run.status == "success"
    assert run.finished_at is not None and run.finished_at >= run.started_at
    assert run.fetch_counts == {"KSDK": 3}
    assert run.classify_counts == {"classified": 2, "stopped_quota": 1}
    assert run.match_counts == {"created": 1, "merged": 0}
    assert (run.fetch_error, run.classify_error, run.match_error) == (None, None, None)
    assert _locks(session_factory) == []


def test_classifier_fails_matcher_still_runs(session_factory):
    rec = Recorder(fail=("classify",))
    run = run_pipeline(session_factory, rec.steps())

    assert rec.calls == ["fetch", "classify", "match"]
    assert run.status == "partial"
    assert run.classify_error == "RuntimeError: classify broke"
    assert run.classify_counts is None
    assert run.match_counts == {"created": 1, "merged": 0}
    assert run.match_error is None
    assert _locks(session_factory) == []


def test_all_steps_fail(session_factory):
    rec = Recorder(fail=("fetch", "classify", "match"))
    run = run_pipeline(session_factory, rec.steps())

    assert rec.calls == ["fetch", "classify", "match"]
    assert run.status == "failed"
    assert run.fetch_error and run.classify_error and run.match_error


def test_failed_step_changes_are_rolled_back(session_factory):
    def bad_fetch(session):
        session.add(PipelineRun(started_at=datetime.now(timezone.utc), status="success"))
        session.flush()
        raise RuntimeError("boom")

    rec = Recorder()
    run_pipeline(session_factory, [("fetch", bad_fetch)] + rec.steps()[1:])
    assert len(_runs(session_factory)) == 1  # only the run's own row survived


def test_lock_prevents_overlap(session_factory):
    _hold_lock(session_factory, datetime.now(timezone.utc) + timedelta(minutes=20))
    rec = Recorder()

    assert run_pipeline(session_factory, rec.steps()) is None
    assert rec.calls == []
    assert _runs(session_factory) == []
    [held] = _locks(session_factory)
    assert held.owner == "other"


def test_stale_lock_is_taken_over(session_factory, caplog):
    _hold_lock(session_factory, datetime.now(timezone.utc) - timedelta(minutes=1))
    rec = Recorder()

    with caplog.at_level(logging.WARNING):
        run = run_pipeline(session_factory, rec.steps())

    assert run.status == "success"
    assert rec.calls == ["fetch", "classify", "match"]
    assert "stale lock" in caplog.text
    assert _locks(session_factory) == []


def test_lock_held_during_run_blocks_second_run(session_factory):
    inner: list = []

    def fetch(session):
        inner.append(run_pipeline(session_factory, Recorder().steps()))
        return {}

    run = run_pipeline(session_factory, [("fetch", fetch)])
    assert inner == [None]
    assert run.status == "success"
    assert len(_runs(session_factory)) == 1


def test_lock_release_keeps_takeover(session_factory):
    with session_factory() as s:
        assert lock.acquire(s, "x", "a", ttl=timedelta(minutes=-1))  # already stale
        assert lock.acquire(s, "x", "b")
        assert not lock.acquire(s, "x", "c")
        lock.release(s, "x", "a")  # a's lock was taken over; must not drop b's
        assert not lock.refresh(s, "x", "a")
        assert lock.refresh(s, "x", "b")
    [held] = _locks(session_factory)
    assert held.owner == "b"


def test_row_written_for_every_run(session_factory):
    run_pipeline(session_factory, Recorder().steps())
    run_pipeline(session_factory, Recorder(fail=("fetch",)).steps())
    run_pipeline(session_factory, Recorder(fail=("fetch", "classify", "match")).steps())

    runs = _runs(session_factory)
    assert [r.status for r in runs] == ["success", "partial", "failed"]
    assert all(r.finished_at is not None for r in runs)


def test_run_row_is_running_while_steps_execute(session_factory):
    seen = []

    def fetch(session):
        seen.append(session.scalar(select(PipelineRun.status)))
        return {}

    run_pipeline(session_factory, [("fetch", fetch)])
    assert seen == ["running"]


def test_stop_event_finishes_current_step_and_skips_rest(session_factory):
    stop = threading.Event()
    rec = Recorder()
    steps = rec.steps()
    fetch = steps[0][1]

    def fetch_then_ctrl_c(session):
        result = fetch(session)
        stop.set()
        return result

    run = run_pipeline(session_factory, [("fetch", fetch_then_ctrl_c)] + steps[1:], stop)

    assert rec.calls == ["fetch"]
    assert run.fetch_counts == {"KSDK": 3}
    assert run.classify_error == "skipped: shutdown"
    assert run.match_error == "skipped: shutdown"
    assert run.status == "partial"
    assert _locks(session_factory) == []


def test_lock_released_if_recording_fails(session_factory, monkeypatch):
    monkeypatch.setattr("jobs.pipeline._status", lambda *a: 1 / 0)
    with pytest.raises(ZeroDivisionError):
        run_pipeline(session_factory, Recorder().steps())
    assert _locks(session_factory) == []


def test_run_job_skips_after_stop(monkeypatch):
    calls = []
    monkeypatch.setattr("jobs.scheduler.run_pipeline", lambda **kw: calls.append(kw))
    stop = threading.Event()
    _run_job(stop)
    stop.set()
    _run_job(stop)
    assert calls == [{"stop_event": stop}]


def test_run_job_logs_crash(monkeypatch, caplog):
    def crash(**kw):
        raise RuntimeError("db down")

    monkeypatch.setattr("jobs.scheduler.run_pipeline", crash)
    with caplog.at_level(logging.ERROR):
        _run_job(threading.Event())
    assert "Pipeline run crashed" in caplog.text


def test_scheduler_config(monkeypatch):
    monkeypatch.setenv("PIPELINE_INTERVAL_MINUTES", "7")
    from config import get_settings

    get_settings.cache_clear()
    try:
        scheduler = build_scheduler(threading.Event())
    finally:
        get_settings.cache_clear()

    job = scheduler.get_job(JOB_ID)
    assert job.trigger.interval == timedelta(minutes=7)
    assert job.max_instances == 1
    assert job.coalesce is True


def test_interval_defaults_to_10(monkeypatch):
    monkeypatch.delenv("PIPELINE_INTERVAL_MINUTES", raising=False)
    monkeypatch.setattr("config.load_dotenv", lambda: None)
    from config import get_settings

    get_settings.cache_clear()
    try:
        assert get_settings().pipeline_interval_minutes == 10
    finally:
        get_settings.cache_clear()


def test_logging_writes_rotating_file_in_utc(tmp_path):
    root = logging.getLogger()
    before = list(root.handlers)
    try:
        setup_logging(tmp_path / "logs")
        logging.getLogger("jobs.test").info("hello pipeline")
        for h in root.handlers:
            h.flush()
        text = (tmp_path / "logs" / LOG_FILE).read_text(encoding="utf-8")
    finally:
        for h in root.handlers[len(before):]:
            h.close()
        root.handlers[:] = before

    assert "INFO jobs.test: hello pipeline" in text


def test_utc_formatter():
    record = logging.LogRecord("x", logging.INFO, "", 0, "m", None, None)
    record.created = datetime(2026, 1, 15, 12, 30, tzinfo=timezone.utc).timestamp()
    assert UTCFormatter("%(asctime)s").format(record) == "2026-01-15T12:30:00Z"



def test_formatter_redacts_webhook_token():
    record = logging.LogRecord(
        "httpx", logging.INFO, "", 0,
        'HTTP Request: POST %s "HTTP/1.1 204"',
        ("https://discord.com/api/webhooks/123/sEcr3t-Tok_en",), None,
    )
    line = UTCFormatter("%(message)s").format(record)
    assert "sEcr3t" not in line
    assert "/api/webhooks/123/[redacted]" in line
