import json
import logging
import threading
from datetime import datetime, timedelta, timezone
from functools import partial

import httpx
import pytest

from alerts import checks, notify, store
from alerts.__main__ import main as alerts_cli
from config import get_settings
from jobs.pipeline import run_pipeline
from jobs.scheduler import WATCHDOG_JOB_ID, build_scheduler
from models import AlertSent, Incident, IncidentItem, PipelineRun, RawItem, Source, hash_url

NOW = datetime(2026, 10, 5, 18, 0, tzinfo=timezone.utc)
HOOK = "https://discord.test/api/webhooks/1/abc"


class Webhook:
    """A mocked Discord webhook that records posted messages."""

    def __init__(self, status: int = 204, error: Exception | None = None):
        self.status = status
        self.error = error
        self.messages: list[str] = []
        self.payloads: list[dict] = []
        self.client = httpx.Client(transport=httpx.MockTransport(self._handle))

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if self.error is not None:
            raise self.error
        payload = json.loads(request.content)
        self.payloads.append(payload)
        self.messages.append(payload["content"])
        return httpx.Response(self.status)

    @property
    def send(self):
        return partial(notify.send, url=HOOK, client=self.client)


@pytest.fixture
def hook():
    webhook = Webhook()
    yield webhook
    webhook.client.close()


def set_env(monkeypatch, **values):
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    get_settings.cache_clear()


def add_source(session, fail_count=0, name="KSDK"):
    source = Source(name=name, url=f"https://{name.lower()}.test/rss", type="news",
                    fail_count=fail_count)
    session.add(source)
    session.commit()
    return source


def add_run(session, status="success", started_at=NOW - timedelta(minutes=5),
            finished_at=None, classify_counts=None, **errors):
    run = PipelineRun(
        started_at=started_at,
        finished_at=finished_at or started_at + timedelta(minutes=1),
        status=status,
        classify_counts=classify_counts,
        **errors,
    )
    session.add(run)
    session.commit()
    return run


_item_seq = iter(range(1, 10_000))


def add_item(session, source, created_at=NOW, status="new", url=None):
    n = next(_item_seq)
    url = url or f"https://news.test/story-{n}"
    item = RawItem(source_id=source.id, url=url, url_hash=hash_url(url), title=f"Story {n}",
                   status=status, created_at=created_at, published_at=created_at)
    session.add(item)
    session.commit()
    return item


def add_incident(session, source, status="review", created_at=NOW, url="https://ksdk.test/shot"):
    item = add_item(session, source, status="classified", url=url)
    incident = Incident(crime_type="shooting", occurred_at=NOW - timedelta(hours=1),
                        location="4500 Natural Bridge Ave", neighborhood="Penrose",
                        was_shooting=True, status=status, created_at=created_at)
    session.add(incident)
    session.flush()
    session.add(IncidentItem(incident_id=incident.id, raw_item_id=item.id))
    session.commit()
    return incident


# --- notify -----------------------------------------------------------------


def test_send_posts_content_without_mentions(hook):
    assert hook.send("hello @everyone") is True
    assert hook.messages == ["hello @everyone"]
    assert hook.payloads[0]["allowed_mentions"] == {"parse": []}


def test_send_truncates_to_discord_limit(hook):
    hook.send("x" * 5000)
    assert len(hook.messages[0]) == notify.MAX_LENGTH


def test_send_without_url_only_logs(caplog):
    with caplog.at_level(logging.WARNING):
        assert notify.send("log me") is True
    assert "log me" in caplog.text


@pytest.mark.parametrize("webhook", [
    Webhook(status=500),
    Webhook(status=429),
    Webhook(error=httpx.ConnectError("refused")),
    Webhook(error=httpx.ReadTimeout("slow")),
])
def test_send_failure_returns_false_and_logs(webhook, caplog):
    with caplog.at_level(logging.ERROR):
        assert webhook.send("boom") is False
    assert "Alert webhook" in caplog.text


# --- store / cooldown -------------------------------------------------------


def test_cooldown_blocks_repeat_until_it_passes(session, hook):
    assert checks._alert(session, "k", "first", NOW, hook.send)
    assert not checks._alert(session, "k", "again", NOW + timedelta(hours=5, minutes=59), hook.send)
    assert checks._alert(session, "k", "later", NOW + timedelta(hours=6), hook.send)
    assert hook.messages == ["first", "later"]


def test_cooldown_hours_from_settings(session, hook, monkeypatch):
    set_env(monkeypatch, ALERT_COOLDOWN_HOURS="1")
    checks._alert(session, "k", "first", NOW, hook.send)
    assert checks._alert(session, "k", "again", NOW + timedelta(hours=1), hook.send)


def test_failed_send_is_not_recorded(session):
    broken = Webhook(status=500)
    assert not checks._alert(session, "k", "x", NOW, broken.send)
    assert session.get(AlertSent, "k") is None


def test_resolved_key_can_alert_again_immediately(session, hook):
    checks._alert(session, "k", "down", NOW, hook.send)
    store.mark_resolved(session, "k", NOW)
    assert checks._alert(session, "k", "down again", NOW + timedelta(minutes=1), hook.send)


# --- sources ----------------------------------------------------------------


def test_source_alerts_at_five_failures(session, hook):
    source = add_source(session, fail_count=4)
    checks.check_sources(session, NOW, hook.send)
    assert hook.messages == []

    source.fail_count = 5
    session.commit()
    checks.check_sources(session, NOW, hook.send)
    assert len(hook.messages) == 1
    assert "KSDK" in hook.messages[0] and "5 fetches" in hook.messages[0]


def test_source_down_respects_cooldown(session, hook):
    add_source(session, fail_count=5)
    checks.check_sources(session, NOW, hook.send)
    checks.check_sources(session, NOW + timedelta(minutes=10), hook.send)
    assert len(hook.messages) == 1


def test_source_recovery_sent_once(session, hook):
    source = add_source(session, fail_count=7)
    checks.check_sources(session, NOW, hook.send)
    source.fail_count = 0
    session.commit()
    checks.check_sources(session, NOW + timedelta(minutes=10), hook.send)
    checks.check_sources(session, NOW + timedelta(minutes=20), hook.send)
    assert len(hook.messages) == 2
    assert "recovered" in hook.messages[1]


def test_healthy_source_never_sends_recovery(session, hook):
    add_source(session, fail_count=0)
    checks.check_sources(session, NOW, hook.send)
    assert hook.messages == []


def test_failed_recovery_send_is_retried(session, hook):
    source = add_source(session, fail_count=5)
    checks.check_sources(session, NOW, hook.send)
    source.fail_count = 0
    session.commit()
    checks.check_sources(session, NOW, Webhook(status=500).send)
    checks.check_sources(session, NOW, hook.send)
    assert "recovered" in hook.messages[-1]


# --- LLM quota --------------------------------------------------------------


def test_quota_alert_and_recovery(session, hook, monkeypatch):
    set_env(monkeypatch, LLM_CHAIN="", GEMINI_MODEL="primary-m", GEMINI_FALLBACK_MODEL="fallback-m")
    stopped = add_run(session, classify_counts={"stopped_quota": 1, "classified": 3})
    checks.check_quota(session, stopped, NOW, hook.send)
    assert "gemini:primary-m, gemini:fallback-m" in hook.messages[0]

    checks.check_quota(session, stopped, NOW + timedelta(minutes=10), hook.send)
    assert len(hook.messages) == 1  # cooldown

    idle = add_run(session, classify_counts={"stopped_quota": 0, "classified": 0})
    checks.check_quota(session, idle, NOW + timedelta(minutes=20), hook.send)
    assert len(hook.messages) == 1  # nothing classified: no proof the quota is back

    working = add_run(session, classify_counts={"stopped_quota": 0, "classified": 2})
    checks.check_quota(session, working, NOW + timedelta(hours=8), hook.send)
    assert "classifying again" in hook.messages[1]


def test_quota_alert_names_llm_chain(session, hook, monkeypatch):
    set_env(monkeypatch, LLM_CHAIN="groq:g-model, gemini:primary-m")
    run = add_run(session, classify_counts={"stopped_quota": 1})
    checks.check_quota(session, run, NOW, hook.send)
    assert "groq:g-model, gemini:primary-m" in hook.messages[0]


def test_primary_only_quota_does_not_alert(session, hook):
    run = add_run(session, classify_counts={"switched_quota": 1, "stopped_quota": 0, "classified": 5})
    checks.check_quota(session, run, NOW, hook.send)
    assert hook.messages == []


def test_quota_check_handles_failed_classify_step(session, hook):
    run = add_run(session, status="partial", classify_counts=None, classify_error="Boom")
    checks.check_quota(session, run, NOW, hook.send)
    assert hook.messages == []


# --- stuck items ------------------------------------------------------------


def test_stuck_items_alert_over_twenty(session, hook):
    source = add_source(session)
    for _ in range(20):
        add_item(session, source, created_at=NOW - timedelta(hours=3))
    add_item(session, source, created_at=NOW - timedelta(minutes=30))  # too recent
    add_item(session, source, created_at=NOW - timedelta(hours=3), status="classified")
    checks.check_stuck(session, NOW, hook.send)
    assert hook.messages == []

    add_item(session, source, created_at=NOW - timedelta(hours=2, minutes=1))
    checks.check_stuck(session, NOW, hook.send)
    assert len(hook.messages) == 1 and "21 items" in hook.messages[0]

    checks.check_stuck(session, NOW + timedelta(minutes=10), hook.send)
    assert len(hook.messages) == 1


# --- new incidents ----------------------------------------------------------


def test_new_review_incident_alerts_once(session, hook):
    source = add_source(session)
    run = add_run(session, started_at=NOW - timedelta(minutes=2))
    old = add_incident(session, source, created_at=NOW - timedelta(days=1), url="https://a.test/old")
    new = add_incident(session, source, created_at=NOW - timedelta(minutes=1), url="https://a.test/new")

    checks.check_new_incidents(session, run, NOW, hook.send)
    assert len(hook.messages) == 1
    text = hook.messages[0]
    assert f"#{new.id}" in text and f"#{old.id}" not in text
    assert "4500 Natural Bridge Ave (Penrose)" in text and "https://a.test/new" in text

    checks.check_new_incidents(session, run, NOW + timedelta(hours=7), hook.send)
    assert len(hook.messages) == 1


def test_several_review_incidents_share_one_message(session, hook):
    source = add_source(session)
    run = add_run(session, started_at=NOW - timedelta(minutes=2))
    a = add_incident(session, source, url="https://a.test/1")
    b = add_incident(session, source, url="https://a.test/2")
    checks.check_new_incidents(session, run, NOW, hook.send)
    assert len(hook.messages) == 1
    assert f"#{a.id}" in hook.messages[0] and f"#{b.id}" in hook.messages[0]


def test_incident_created_in_runs_start_second_counts(session, hook):
    """SQLite's CURRENT_TIMESTAMP drops microseconds."""
    source = add_source(session)
    run = add_run(session, started_at=NOW.replace(microsecond=700_000))
    add_incident(session, source, created_at=NOW)
    checks.check_new_incidents(session, run, NOW, hook.send)
    assert len(hook.messages) == 1


def test_confirmed_incident_not_alerted_by_default(session, hook):
    source = add_source(session)
    run = add_run(session, started_at=NOW - timedelta(minutes=2))
    add_incident(session, source, status="confirmed")
    checks.check_new_incidents(session, run, NOW, hook.send)
    assert hook.messages == []


def test_confirmed_incident_alert_when_enabled(session, hook, monkeypatch):
    set_env(monkeypatch, ALERT_ON_NEW_INCIDENT="true")
    source = add_source(session)
    run = add_run(session, started_at=NOW - timedelta(minutes=2))
    incident = add_incident(session, source, status="confirmed", url="https://ksdk.test/story")
    add_incident(session, source, status="rejected", url="https://ksdk.test/rejected")
    checks.check_new_incidents(session, run, NOW, hook.send)
    assert len(hook.messages) == 1
    text = hook.messages[0]
    assert "New incident" in text and f"#{incident.id}" in text
    assert "shooting" in text and "4500 Natural Bridge Ave" in text
    assert "https://ksdk.test/story" in text


def test_failed_incident_send_retried_next_run(session, hook):
    source = add_source(session)
    run = add_run(session, started_at=NOW - timedelta(minutes=2))
    add_incident(session, source)
    checks.check_new_incidents(session, run, NOW, Webhook(status=500).send)
    checks.check_new_incidents(session, run, NOW, hook.send)
    assert len(hook.messages) == 1


# --- watchdog ---------------------------------------------------------------


def test_watchdog_quiet_with_recent_success(session, hook):
    add_run(session, finished_at=NOW - timedelta(minutes=29))
    checks.check_watchdog(session, NOW, hook.send)
    assert hook.messages == []


def test_watchdog_alerts_without_success_in_30_min(session, hook):
    add_run(session, started_at=NOW - timedelta(hours=1), finished_at=NOW - timedelta(minutes=31))
    add_run(session, status="partial", started_at=NOW - timedelta(minutes=10),
            classify_error="RuntimeError: kaput")
    checks.check_watchdog(session, NOW, hook.send)
    assert len(hook.messages) == 1
    assert "No successful pipeline run" in hook.messages[0]
    assert "partial" in hook.messages[0] and "kaput" in hook.messages[0]

    checks.check_watchdog(session, NOW + timedelta(minutes=5), hook.send)
    assert len(hook.messages) == 1  # cooldown


def test_watchdog_alerts_when_no_run_ever(session, hook):
    checks.check_watchdog(session, NOW, hook.send)
    assert "Last success: never" in hook.messages[0]


def test_watchdog_grace_after_scheduler_start(session, hook):
    add_run(session, started_at=NOW - timedelta(days=1))
    checks.check_watchdog(session, NOW, hook.send, started_at=NOW - timedelta(minutes=10))
    assert hook.messages == []
    checks.check_watchdog(session, NOW, hook.send, started_at=NOW - timedelta(minutes=30))
    assert len(hook.messages) == 1


def test_watchdog_recovery(session, hook):
    checks.check_watchdog(session, NOW, hook.send)
    add_run(session, finished_at=NOW + timedelta(minutes=3))
    checks.check_watchdog(session, NOW + timedelta(minutes=5), hook.send)
    checks.check_watchdog(session, NOW + timedelta(minutes=10), hook.send)
    assert len(hook.messages) == 2
    assert "healthy again" in hook.messages[1]


def test_run_watchdog_never_raises(caplog):
    def broken_factory():
        raise RuntimeError("db gone")

    with caplog.at_level(logging.ERROR):
        checks.run_watchdog(broken_factory, now=NOW)
    assert "Watchdog failed" in caplog.text


def test_run_watchdog_uses_session_factory(session_factory, hook):
    checks.run_watchdog(session_factory, now=NOW, send=hook.send)
    assert len(hook.messages) == 1


def test_scheduler_has_watchdog_job(monkeypatch):
    scheduler = build_scheduler(threading.Event(), interval_minutes=10)
    job = scheduler.get_job(WATCHDOG_JOB_ID)
    assert job.trigger.interval == timedelta(minutes=5)
    assert job.max_instances == 1
    assert job.kwargs["started_at"] is not None


# --- check_after_run --------------------------------------------------------


def test_check_after_run_covers_all_conditions(session, hook):
    add_source(session, fail_count=5)
    checks.check_watchdog(session, NOW - timedelta(minutes=20), hook.send)  # open watchdog
    run = add_run(session, started_at=NOW - timedelta(minutes=2), finished_at=NOW,
                  classify_counts={"stopped_quota": 1})
    source = add_source(session, name="Other")
    for _ in range(21):
        add_item(session, source, created_at=NOW - timedelta(hours=3))
    add_incident(session, source)

    checks.check_after_run(session, run, now=NOW, send=hook.send)
    text = "\n".join(hook.messages)
    for expected in ("Source down", "LLM quota", "Backlog", "Needs your review", "healthy again"):
        assert expected in text


def test_one_failing_check_does_not_stop_the_others(session, hook, monkeypatch, caplog):
    add_source(session, fail_count=5)
    run = add_run(session, classify_counts={"stopped_quota": 1})

    def boom(*args):
        raise RuntimeError("check broke")

    monkeypatch.setattr(checks, "check_sources", boom)
    with caplog.at_level(logging.ERROR):
        checks.check_after_run(session, run, now=NOW, send=hook.send)
    assert "check broke" in caplog.text
    assert any("LLM quota" in m for m in hook.messages)


def test_webhook_failures_do_not_break_checks(session):
    add_source(session, fail_count=5)
    run = add_run(session, classify_counts={"stopped_quota": 1})
    down = Webhook(error=httpx.ConnectError("refused"))
    checks.check_after_run(session, run, now=NOW, send=down.send)
    assert session.query(AlertSent).count() == 0


# --- pipeline integration ---------------------------------------------------


def test_pipeline_runs_alert_check_after_releasing_lock(session_factory):
    seen = []

    def alert_check(session, run):
        from models import JobLock

        seen.append((run.status, session.query(JobLock).count()))

    run = run_pipeline(session_factory, [("fetch", lambda s: {})], alert_check=alert_check)
    assert run.status == "success"
    assert seen == [("success", 0)]


def test_pipeline_survives_alert_check_crash(session_factory, caplog):
    def alert_check(session, run):
        raise RuntimeError("alerts exploded")

    with caplog.at_level(logging.ERROR):
        run = run_pipeline(session_factory, [("fetch", lambda s: {})], alert_check=alert_check)
    assert run.status == "success"
    assert "alerts exploded" in caplog.text


def test_pipeline_with_unreachable_webhook_still_succeeds(session_factory, monkeypatch):
    set_env(monkeypatch, ALERT_WEBHOOK_URL=HOOK)
    down = Webhook(error=httpx.ConnectError("refused"))
    monkeypatch.setattr(notify, "make_client", lambda: down.client)

    def fetch(session):
        session.add(Source(name="Dead", url="https://dead.test/rss", type="news", fail_count=9))
        session.commit()
        return {}

    run = run_pipeline(session_factory, [("fetch", fetch)])
    assert run.status == "success"
    with session_factory() as s:
        assert s.query(AlertSent).count() == 0


def test_default_pipeline_alert_goes_to_webhook(session_factory, monkeypatch):
    set_env(monkeypatch, ALERT_WEBHOOK_URL=HOOK)
    webhook = Webhook()
    monkeypatch.setattr(notify, "make_client", lambda: webhook.client)

    def fetch(session):
        session.add(Source(name="Dead", url="https://dead.test/rss", type="news", fail_count=5))
        session.commit()
        return {}

    run_pipeline(session_factory, [("fetch", fetch)])
    assert len(webhook.messages) == 1 and "Dead" in webhook.messages[0]


def test_no_alert_check_when_lock_is_held(session_factory):
    from jobs import lock

    calls = []
    with session_factory() as s:
        lock.acquire(s, "pipeline", "someone-else")
    assert run_pipeline(session_factory, [], alert_check=lambda *a: calls.append(a)) is None
    assert calls == []


# --- CLI --------------------------------------------------------------------


def test_cli_test_sends_message(hook, monkeypatch, capsys):
    set_env(monkeypatch, ALERT_WEBHOOK_URL=HOOK)
    monkeypatch.setattr("alerts.__main__.setup_logging", lambda: None)
    assert alerts_cli(["test"], send=hook.send) == 0
    assert "test alert" in hook.messages[0]
    assert "sent" in capsys.readouterr().out


def test_cli_test_reports_failure(monkeypatch, capsys):
    monkeypatch.setattr("alerts.__main__.setup_logging", lambda: None)
    assert alerts_cli(["test"], send=Webhook(status=404).send) == 1
    assert "failed" in capsys.readouterr().out


def test_cli_test_log_only(monkeypatch, capsys):
    monkeypatch.setattr("alerts.__main__.setup_logging", lambda: None)
    assert alerts_cli(["test"]) == 0
    assert "only logged" in capsys.readouterr().out


def test_settings_parse_alert_flags(monkeypatch):
    set_env(monkeypatch, ALERT_ON_NEW_INCIDENT="Yes", ALERT_COOLDOWN_HOURS="2.5",
            ALERT_WEBHOOK_URL=HOOK)
    s = get_settings()
    assert s.alert_on_new_incident is True
    assert s.alert_cooldown_hours == 2.5
    assert s.alert_webhook_url == HOOK
    set_env(monkeypatch, ALERT_ON_NEW_INCIDENT="0")
    assert get_settings().alert_on_new_incident is False
