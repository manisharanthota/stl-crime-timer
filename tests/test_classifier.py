import json
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors
from pydantic import ValidationError
from sqlalchemy import select

import classifier.__main__ as cli
import classifier.eval as eval_script
from classifier.classify import _short, classify_pending
from classifier.llm import (
    GeminiClient,
    LLMClient,
    LLMError,
    RateLimitError,
    ServiceUnavailableError,
    get_llm_client,
    parse_retry_delay,
)
from classifier.prefilter import prefilter
from classifier.providers import ChainEntry
from classifier.prompt import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from classifier.ratelimit import RateLimiter, interval_for_rpm
from classifier.schema import BatchResult, ClassifierOutput
from config import Settings
from models import Classification, RawItem, Source

VALID = {
    "is_crime": True,
    "crime_type": "shooting",
    "in_stl": True,
    "occurred_at": "2026-09-27T21:00:00-05:00",
    "location": "5600 block of Riverview Boulevard",
    "confidence": 0.93,
}


def echo(overrides=None, drop=()):
    """Scripted response that returns a VALID result for every item in the request,
    with per-id overrides and some ids dropped."""
    overrides = overrides or {}

    def respond(user):
        return json.dumps([
            {**VALID, "raw_item_id": entry["raw_item_id"], **overrides.get(entry["raw_item_id"], {})}
            for entry in json.loads(user)
            if entry["raw_item_id"] not in drop
        ])

    return respond


class FakeLLM(LLMClient):
    """Returns (or raises) scripted responses in order and records each request.
    A callable response is called with the user prompt."""

    def __init__(self, *responses):
        self.model = "fake-flash"
        self.responses = list(responses)
        self.calls = []

    def generate_json(self, system, user, schema):
        self.calls.append((system, user, schema))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response(user) if callable(response) else response

    def batch_ids(self, call=0):
        return [e["raw_item_id"] for e in json.loads(self.calls[call][1])]


@pytest.fixture
def sleeps():
    return []


@pytest.fixture
def run(session, sleeps):
    """classify_pending with no rate-limit spacing and recorded sleeps."""

    def _run(llm, **kwargs):
        return classify_pending(
            session, llm=llm, sleep=sleeps.append, limiter=RateLimiter(0), **kwargs
        )

    return _run


@pytest.fixture
def make_item(session):
    source = Source(name="KSDK", url="https://ksdk.example/feed", type="news")
    session.add(source)
    session.commit()
    counter = iter(range(1, 10_000))

    def _make(title="Man shot in Walnut Park", body=None, retries=0):
        n = next(counter)
        item = RawItem(
            source_id=source.id,
            url=f"https://ksdk.example/{n}",
            url_hash=f"hash{n}",
            title=title,
            body=body,
            published_at=datetime(2026, 9, 28, 12, tzinfo=timezone.utc),
            retries=retries,
        )
        session.add(item)
        session.commit()
        return item

    return _make


def classifications(session):
    return session.scalars(select(Classification)).all()


# --- prefilter ---------------------------------------------------------------

@pytest.mark.parametrize(
    "title, body",
    [
        ("Man shot in north St. Louis", None),
        ("Police investigate shooting", None),
        ("Gunfire reported downtown", None),
        ("Break-in at Soulard bar", None),
        ("Burglaries rise in Tower Grove", None),
        ("Update on Walnut Park", "A woman was found dead inside a home."),
        ("Man STABBED near Fox Park", None),
        ("Teen killed Friday", None),
        ("Murder suspect sought", None),
        ("Human remains found in south St. Louis home after wall collapse", None),
        ("Remains were found in a vacant lot", None),
        ("Police: body found in Carondelet Park", None),
        ("Officers found a body near the riverfront", None),
        ("Bodies were found inside the home", None),
        ("Update", "Police opened a death investigation Tuesday."),
    ],
)
def test_prefilter_passes_crime_keywords(title, body):
    assert prefilter(RawItem(title=title, body=body))


@pytest.mark.parametrize(
    "title, body",
    [
        ("Cardinals win home opener", None),
        ("Filing deadline nears for city races", None),
        ("Screenshot of new stadium plan goes viral", None),
        ("Rain expected this weekend", "Bring an umbrella."),
        ("Mystery remains unsolved", None),
        ("Student body president elected", None),
    ],
)
def test_prefilter_rejects_non_crime(title, body):
    assert not prefilter(RawItem(title=title, body=body))


def test_prefilter_rejected_item_classified_without_llm(session, make_item, run):
    item = make_item("Cardinals win home opener")
    llm = FakeLLM()

    counts = run(llm)

    assert counts["prefiltered"] == 1
    assert llm.calls == []
    assert item.status == "classified"
    [c] = classifications(session)
    assert c.is_crime is False and c.crime_type is None and c.in_stl is False
    assert c.model == "prefilter"
    assert c.prompt_version == PROMPT_VERSION


def test_all_prefiltered_needs_no_llm_client(session, make_item, monkeypatch):
    make_item("Rain expected this weekend")

    def boom(*a, **k):
        raise AssertionError("LLM client should not be created")

    monkeypatch.setattr("classifier.classify.build_chain", boom)
    assert classify_pending(session)["prefiltered"] == 1


# --- batching ----------------------------------------------------------------

def test_valid_batch_saved(session, make_item, run):
    item = make_item("Man shot in Walnut Park", "Shot Saturday night on Riverview.")
    llm = FakeLLM(echo())

    counts = run(llm)

    assert counts["classified"] == 1
    assert item.status == "classified"
    [c] = classifications(session)
    assert c.raw_item_id == item.id
    assert c.is_crime is True
    assert c.crime_type == "shooting"
    assert c.in_stl is True
    session.expire_all()
    assert c.occurred_at == datetime(2026, 9, 28, 2, tzinfo=timezone.utc)
    assert c.location == "5600 block of Riverview Boulevard"
    assert c.confidence == pytest.approx(0.93)
    assert c.model == "fake-flash"
    assert c.prompt_version == PROMPT_VERSION == "v7"
    assert c.was_shooting is True  # crime_type=shooting implies it
    system, user, schema = llm.calls[0]
    assert "City of St. Louis" in system
    assert json.loads(user)[0]["title"] == "Man shot in Walnut Park"
    assert schema == list[BatchResult]


def test_items_sent_in_batches_of_ten(session, make_item, run):
    items = [make_item() for _ in range(12)]
    make_item("Cardinals win")  # prefiltered, never sent
    llm = FakeLLM(echo(), echo())

    counts = run(llm)

    assert len(llm.calls) == 2
    assert llm.batch_ids(0) == [i.id for i in items[:10]]
    assert llm.batch_ids(1) == [i.id for i in items[10:]]
    assert counts["classified"] == 12
    assert counts["prefiltered"] == 1


def test_bad_entry_only_affects_that_item(session, make_item, run):
    good, bad = make_item(), make_item()
    llm = FakeLLM(echo(overrides={bad.id: {"confidence": 7}}))

    counts = run(llm)

    assert good.status == "classified"
    assert bad.status == "new"
    assert bad.retries == 1
    assert counts["classified"] == 1 and counts["retry_later"] == 1
    assert [c.raw_item_id for c in classifications(session)] == [good.id]


def test_missing_entry_goes_back_to_new(make_item, run):
    good, missing = make_item(), make_item()
    llm = FakeLLM(echo(drop={missing.id}))

    run(llm)

    assert good.status == "classified"
    assert missing.status == "new"
    assert missing.retries == 1


def test_bad_entry_fails_after_three_retries(make_item, run):
    item = make_item(retries=3)
    llm = FakeLLM(echo(drop={item.id}))

    counts = run(llm)

    assert item.status == "failed"
    assert counts["failed"] == 1


def test_bad_entry_retried_in_later_runs_until_failed(make_item, run):
    item = make_item()
    for expected_retries in (1, 2, 3):
        run(FakeLLM(echo(drop={item.id})))
        assert (item.status, item.retries) == ("new", expected_retries)
    run(FakeLLM(echo(drop={item.id})))
    assert item.status == "failed"


def test_unexpected_and_duplicate_ids_ignored(make_item, run):
    item = make_item()
    body = json.dumps([
        {**VALID, "raw_item_id": item.id},
        {**VALID, "raw_item_id": item.id, "crime_type": "burglary"},
        {**VALID, "raw_item_id": 9999},
    ])

    counts = run(FakeLLM(body))

    assert counts["classified"] == 1
    assert item.status == "classified"


def test_invalid_batch_retried_once_then_succeeds(make_item, run):
    item = make_item()
    llm = FakeLLM("not json", echo())

    run(llm)

    assert len(llm.calls) == 2
    assert item.status == "classified"


def test_invalid_batch_twice_sends_items_back_to_new(session, make_item, run):
    a, b = make_item(), make_item()
    llm = FakeLLM("{oops", json.dumps({"not": "an array"}))

    counts = run(llm)

    assert len(llm.calls) == 2
    assert (a.status, a.retries) == ("new", 1)
    assert (b.status, b.retries) == ("new", 1)
    assert counts["retry_later"] == 2
    assert classifications(session) == []


def test_only_new_items_processed(make_item, run):
    done = make_item()
    done.status = "failed"
    llm = FakeLLM()

    assert run(llm)["classified"] == 0
    assert llm.calls == []


# --- rate limiting -----------------------------------------------------------

def test_interval_for_rpm():
    assert interval_for_rpm(5) == 13
    assert interval_for_rpm(10) == 7


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_spaces_requests():
    clock = FakeClock()
    limiter = RateLimiter(13, clock=clock, sleep=clock.sleep)

    limiter.wait()          # first request: no wait
    clock.now += 4          # request took 4s
    limiter.wait()
    clock.now += 20         # long gap: no wait needed
    limiter.wait()

    assert clock.sleeps == [9]


def test_requests_spaced_by_gemini_rpm(make_item, session, monkeypatch, sleeps):
    monkeypatch.setattr(
        "classifier.classify.get_settings", lambda: Settings(gemini_rpm=10)
    )
    for _ in range(11):
        make_item()

    classify_pending(session, llm=FakeLLM(echo(), echo()), sleep=sleeps.append)

    assert sleeps == [pytest.approx(7, abs=0.5)]


# --- 429 / 503 ---------------------------------------------------------------

def test_429_short_delay_waits_and_retries(make_item, run, sleeps):
    item = make_item()
    llm = FakeLLM(RateLimitError("429", retry_after=37), echo())

    counts = run(llm)

    assert sleeps == [37]
    assert item.status == "classified"
    assert item.retries == 0
    assert counts["stopped_quota"] == 0


@pytest.mark.parametrize("retry_after", [3600, 120, None])
def test_429_long_delay_stops_run(session, make_item, run, sleeps, retry_after):
    items = [make_item() for _ in range(15)]
    llm = FakeLLM(RateLimitError("429", retry_after=retry_after))

    counts = run(llm)

    assert counts["stopped_quota"] == 1
    assert len(llm.calls) == 1  # second batch never requested
    assert sleeps == []
    assert all(i.status == "new" and i.retries == 0 for i in items)
    assert classifications(session) == []


def test_repeated_short_429s_stop_run(make_item, run, sleeps):
    item = make_item()
    llm = FakeLLM(*[RateLimitError("429", retry_after=10)] * 4)

    counts = run(llm)

    assert sleeps == [10, 10, 10]
    assert counts["stopped_quota"] == 1
    assert (item.status, item.retries) == ("new", 0)


def test_503_backoff_then_leaves_items_new(make_item, run, sleeps):
    first = [make_item() for _ in range(10)]
    second = make_item()
    llm = FakeLLM(*[ServiceUnavailableError("503")] * 3, echo())

    counts = run(llm)

    assert sleeps == [5, 15]
    assert all(i.status == "new" and i.retries == 0 for i in first)
    assert counts["deferred"] == 10
    assert second.status == "classified"  # next batch still runs


def test_503_then_success(make_item, run, sleeps):
    item = make_item()
    llm = FakeLLM(ServiceUnavailableError("503"), echo())

    run(llm)

    assert sleeps == [5]
    assert item.status == "classified"


@pytest.fixture
def run_fb(session, sleeps):
    """classify_pending with an injected primary and fallback client."""

    def _run(llm, fallback, **kwargs):
        return classify_pending(
            session, llm=llm, fallback=fallback, sleep=sleeps.append,
            limiter=RateLimiter(0), **kwargs,
        )

    return _run


def fallback_llm(*responses):
    llm = FakeLLM(*responses)
    llm.model = "fake-fallback"
    return llm


def test_fallback_classifies_after_primary_503(session, make_item, run_fb, sleeps):
    item = make_item()
    primary = FakeLLM(ServiceUnavailableError("503"))
    fallback = fallback_llm(echo())

    counts = run_fb(primary, fallback)

    assert len(primary.calls) == 1  # no backoff while a later model is available
    assert len(fallback.calls) == 1
    assert sleeps == []
    assert item.status == "classified"
    [c] = classifications(session)
    assert c.model == "fake-fallback"
    assert (counts["classified"], counts["fallback"], counts["deferred"]) == (1, 1, 0)


def test_last_model_backs_off_then_defers(make_item, run_fb, sleeps):
    item = make_item()
    primary = FakeLLM(ServiceUnavailableError("503"))
    fallback = fallback_llm(*[ServiceUnavailableError("503")] * 3)

    counts = run_fb(primary, fallback)

    assert len(primary.calls) == 1
    assert len(fallback.calls) == 3  # the last model left gets the backoff
    assert sleeps == [5, 15]
    assert (item.status, item.retries) == ("new", 0)
    assert (counts["deferred"], counts["fallback"]) == (1, 0)


def test_fallback_not_used_for_non_503_errors(make_item, run_fb):
    item = make_item()
    primary = FakeLLM(*[LLMError("500 internal")] * 3)
    fallback = fallback_llm(echo())

    counts = run_fb(primary, fallback)

    assert fallback.calls == []
    assert counts["deferred"] == 1
    assert item.status == "new"


def test_503_after_other_error_moves_to_fallback(make_item, run_fb, sleeps):
    make_item()
    primary = FakeLLM(LLMError("500 internal"), ServiceUnavailableError("503"))
    fallback = fallback_llm(echo())
    assert run_fb(primary, fallback)["fallback"] == 1
    assert sleeps == [5]  # backoff for the 500 only
    assert len(fallback.calls) == 1


def test_fallback_not_called_when_primary_answers(session, make_item, run_fb):
    make_item()
    fallback = fallback_llm(echo())
    counts = run_fb(FakeLLM(echo()), fallback)
    assert fallback.calls == []
    assert counts["fallback"] == 0
    assert classifications(session)[0].model == "fake-flash"


QUOTA = RateLimitError("429 quota", retry_after=None)


@pytest.mark.parametrize("retry_after", [None, 600.0])
def test_primary_quota_switches_to_fallback_for_rest_of_run(
    session, make_item, run_fb, sleeps, retry_after
):
    items = [make_item() for _ in range(15)]  # two batches
    primary = FakeLLM(RateLimitError("429", retry_after=retry_after))
    fallback = fallback_llm(echo(), echo())

    counts = run_fb(primary, fallback)

    assert len(primary.calls) == 1  # never asked again after its quota ran out
    assert len(fallback.calls) == 2
    assert sleeps == []
    assert all(i.status == "classified" for i in items)
    assert {c.model for c in classifications(session)} == {"fake-fallback"}
    assert counts["classified"] == counts["fallback"] == 15
    assert (counts["switched_quota"], counts["stopped_quota"]) == (1, 0)


def test_quota_switch_mid_run(session, make_item, run_fb):
    for _ in range(15):
        make_item()
    primary = FakeLLM(echo(), QUOTA)  # batch 1 ok, batch 2 hits the quota
    fallback = fallback_llm(echo())

    counts = run_fb(primary, fallback)

    assert (counts["classified"], counts["fallback"], counts["switched_quota"]) == (15, 5, 1)
    models = [c.model for c in classifications(session)]
    assert models.count("fake-flash") == 10 and models.count("fake-fallback") == 5


def test_both_quotas_exhausted_stops_run(make_item, run_fb):
    items = [make_item() for _ in range(15)]
    primary = FakeLLM(QUOTA)
    fallback = fallback_llm(echo(), QUOTA)  # batch 1 on the fallback, then its quota

    counts = run_fb(primary, fallback)

    assert (counts["classified"], counts["fallback"]) == (10, 10)
    assert (counts["switched_quota"], counts["stopped_quota"]) == (1, 1)
    assert all((i.status, i.retries) == ("new", 0) for i in items[10:])


def test_primary_quota_without_fallback_stops_run(make_item, run_fb):
    item = make_item()
    counts = run_fb(FakeLLM(QUOTA), None)
    assert (counts["switched_quota"], counts["stopped_quota"]) == (0, 1)
    assert (item.status, item.retries) == ("new", 0)


def test_after_quota_switch_fallback_503_defers_batch_and_continues(make_item, run_fb, sleeps):
    items = [make_item() for _ in range(15)]
    primary = FakeLLM(QUOTA)
    fallback = fallback_llm(*[ServiceUnavailableError("503")] * 3, echo())

    counts = run_fb(primary, fallback)

    assert len(primary.calls) == 1
    assert len(fallback.calls) == 4  # now the last model: backoff, then batch 2
    assert sleeps == [5, 15]
    assert all(i.status == "new" for i in items[:10])
    assert all(i.status == "classified" for i in items[10:])
    assert (counts["deferred"], counts["fallback"], counts["stopped_quota"]) == (10, 5, 0)


def test_overloaded_primary_and_fallback_quota_defers_and_keeps_primary(make_item, run_fb):
    items = [make_item() for _ in range(15)]
    primary = FakeLLM(ServiceUnavailableError("503"), echo())
    fallback = fallback_llm(QUOTA)

    counts = run_fb(primary, fallback)

    # The primary still has quota: batch 1 waits, batch 2 runs on the primary.
    assert len(fallback.calls) == 1
    assert all(i.status == "new" for i in items[:10])
    assert all(i.status == "classified" for i in items[10:])
    assert (counts["deferred"], counts["stopped_quota"], counts["switched_quota"]) == (10, 0, 0)


def test_overloaded_primary_after_fallback_quota_gets_no_fallback(make_item, run_fb, sleeps):
    for _ in range(15):
        make_item()
    # Batch 1: 503 -> fallback quota. Batch 2: 503s again, fallback is gone, so the
    # primary is the last model and backs off.
    primary = FakeLLM(*[ServiceUnavailableError("503")] * 4)
    fallback = fallback_llm(QUOTA)

    counts = run_fb(primary, fallback)

    assert len(fallback.calls) == 1
    assert len(primary.calls) == 4
    assert sleeps == [5, 15]
    assert (counts["deferred"], counts["stopped_quota"]) == (15, 0)


def test_each_batch_tries_primary_first(make_item, run_fb):
    for _ in range(11):
        make_item()
    primary = FakeLLM(ServiceUnavailableError("503"), echo())
    fallback = fallback_llm(echo())

    counts = run_fb(primary, fallback)

    assert len(fallback.calls) == 1  # first batch only
    assert len(primary.calls) == 2  # second batch went to the primary
    assert (counts["classified"], counts["fallback"]) == (11, 10)


def test_no_fallback_configured_defers(make_item, run_fb):
    make_item()
    counts = run_fb(FakeLLM(*[ServiceUnavailableError("503")] * 3), None)
    assert counts["deferred"] == 1


def test_injected_llm_does_not_load_fallback_from_settings(make_item, run, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("fallback loaded from settings")

    monkeypatch.setattr("classifier.classify.build_chain", boom)
    make_item()
    assert run(FakeLLM(*[ServiceUnavailableError("503")] * 3))["deferred"] == 1


def test_default_clients_come_from_settings(session, make_item, monkeypatch, sleeps):
    primary = FakeLLM(ServiceUnavailableError("503"))
    fallback = fallback_llm(echo())
    monkeypatch.setattr(
        "classifier.classify.build_chain",
        lambda sleep: [ChainEntry(primary, RateLimiter(9)), ChainEntry(fallback, RateLimiter(9))],
    )
    make_item()

    counts = classify_pending(session, sleep=sleeps.append, limiter=RateLimiter(0))

    assert counts["fallback"] == 1


def test_llm_unavailable_logged_as_one_line_warning(make_item, run, caplog):
    make_item()
    long_error = "503 UNAVAILABLE. " + "x" * 500 + "\nsecond line"
    with caplog.at_level("WARNING", logger="classifier.classify"):
        run(FakeLLM(*[ServiceUnavailableError(long_error)] * 4))

    [record] = [r for r in caplog.records if "LLM unavailable" in r.getMessage()]
    assert record.levelname == "WARNING"
    assert record.exc_info is None
    message = record.getMessage()
    assert "\n" not in message
    assert "second line" not in message
    assert len(message) < 300
    assert "leaving 1 item(s) new" in message
    assert all(r.exc_info is None for r in caplog.records)


def test_short_error_text():
    assert _short(ValueError("one\ntwo")) == "one"
    assert _short(ValueError("")) == "ValueError"
    long = _short(ValueError("y" * 500))
    assert len(long) == 200 and long.endswith("...")


def test_other_api_error_uses_same_backoff(make_item, run, sleeps):
    item = make_item()
    llm = FakeLLM(*[LLMError("400 bad request")] * 3)

    counts = run(llm)

    assert sleeps == [5, 15]
    assert counts["deferred"] == 1
    assert item.status == "new"


# --- schema / prompt ---------------------------------------------------------

def test_schema_rejects_confidence_out_of_range():
    with pytest.raises(ValidationError):
        ClassifierOutput.model_validate({**VALID, "confidence": 1.5})


def test_schema_rejects_unknown_crime_type():
    with pytest.raises(ValidationError):
        ClassifierOutput.model_validate({**VALID, "crime_type": "robbery"})


def test_schema_requires_crime_type_when_crime():
    with pytest.raises(ValidationError):
        ClassifierOutput.model_validate({**VALID, "crime_type": None})


def test_schema_clears_crime_type_when_not_crime():
    out = ClassifierOutput.model_validate({**VALID, "is_crime": False})
    assert out.crime_type is None


def test_schema_naive_time_is_st_louis_local_converted_to_utc():
    out = ClassifierOutput.model_validate({**VALID, "occurred_at": "2026-09-27T21:00:00"})
    assert out.occurred_at == datetime(2026, 9, 28, 2, tzinfo=timezone.utc)
    assert out.occurred_at.utcoffset().total_seconds() == 0


def test_schema_offset_time_converted_to_utc():
    out = ClassifierOutput.model_validate({**VALID, "occurred_at": "2026-01-15T21:00:00-06:00"})
    assert out.occurred_at == datetime(2026, 1, 16, 3, tzinfo=timezone.utc)


def test_schema_shooting_implies_was_shooting():
    out = ClassifierOutput.model_validate({**VALID, "was_shooting": False})
    assert out.was_shooting is True


@pytest.mark.parametrize("was_shooting", [True, False])
def test_schema_homicide_keeps_was_shooting(was_shooting):
    out = ClassifierOutput.model_validate(
        {**VALID, "crime_type": "homicide", "was_shooting": was_shooting}
    )
    assert out.was_shooting is was_shooting


def test_schema_homicide_was_shooting_defaults_false():
    out = ClassifierOutput.model_validate({**VALID, "crime_type": "homicide"})
    assert out.was_shooting is False


@pytest.mark.parametrize(
    "overrides", [{"crime_type": "burglary"}, {"is_crime": False, "crime_type": None}]
)
def test_schema_clears_was_shooting_for_burglary_and_non_crime(overrides):
    out = ClassifierOutput.model_validate({**VALID, **overrides, "was_shooting": True})
    assert out.was_shooting is False


def test_fatal_shooting_saved_as_homicide_with_was_shooting(session, make_item, run):
    make_item("Teen killed in shooting", "Fatally shot on Arsenal Street.")
    llm = FakeLLM(echo({1: {"crime_type": "homicide", "was_shooting": True}}))
    run(llm)
    [c] = classifications(session)
    assert (c.crime_type, c.was_shooting) == ("homicide", True)
    assert "was_shooting" in llm.calls[0][0]


@pytest.mark.parametrize(
    "given, expected",
    [
        ("West End", "West End"),
        ("west end neighborhood", "West End"),
        ("Hill", "The Hill"),
        ("CWE", "Central West End"),
        ("Saint Louis Hills", "St. Louis Hills"),
        ("Forest Park", None),  # a park, not a neighborhood
        ("Ferguson", None),
        (None, None),
    ],
)
def test_schema_normalizes_neighborhood(given, expected):
    out = ClassifierOutput.model_validate({**VALID, "neighborhood": given})
    assert out.neighborhood == expected


def test_schema_followup_defaults_false_and_cleared_when_not_crime():
    assert ClassifierOutput.model_validate(VALID).is_followup is False
    assert ClassifierOutput.model_validate({**VALID, "is_followup": True}).is_followup is True
    out = ClassifierOutput.model_validate({**VALID, "is_crime": False, "is_followup": True})
    assert out.is_followup is False


def test_neighborhood_and_followup_saved(session, make_item, run):
    make_item("Police identify man killed in West End shooting")
    run(FakeLLM(echo({1: {
        "crime_type": "homicide", "was_shooting": True,
        "neighborhood": "west end", "is_followup": True,
    }})))
    [c] = classifications(session)
    assert (c.neighborhood, c.is_followup) == ("West End", True)


def test_prompt_has_followup_rule_and_neighborhoods():
    from matcher.neighborhoods import NEIGHBORHOODS

    assert "within 7 days" in SYSTEM_PROMPT
    assert "is_followup" in SYSTEM_PROMPT
    assert "charges filed" in SYSTEM_PROMPT
    assert all(f"- {name}\n" in SYSTEM_PROMPT for name in NEIGHBORHOODS)


def test_official_neighborhood_list():
    from matcher.neighborhoods import NEIGHBORHOODS, normalize_neighborhood

    assert len(NEIGHBORHOODS) == len(set(NEIGHBORHOODS)) == 79
    assert all(normalize_neighborhood(name) == name for name in NEIGHBORHOODS)


@pytest.mark.parametrize("precision", ["exact", "date_only", "unknown"])
def test_schema_keeps_time_precision(precision):
    out = ClassifierOutput.model_validate({**VALID, "time_precision": precision})
    assert out.time_precision == precision


def test_schema_time_precision_defaults_unknown_and_is_unknown_without_time():
    assert ClassifierOutput.model_validate(VALID).time_precision == "unknown"
    out = ClassifierOutput.model_validate(
        {**VALID, "occurred_at": None, "time_precision": "exact"}
    )
    assert out.time_precision == "unknown"


def test_schema_rejects_bad_time_precision():
    with pytest.raises(ValidationError):
        ClassifierOutput.model_validate({**VALID, "time_precision": "approximate"})


def test_time_precision_saved(session, make_item, run):
    make_item()
    run(FakeLLM(echo({1: {"time_precision": "date_only"}})))
    assert classifications(session)[0].time_precision == "date_only"


def test_prompt_explains_time_precision():
    assert "time_precision" in SYSTEM_PROMPT
    assert "date_only" in SYSTEM_PROMPT and "00:00" in SYSTEM_PROMPT


def test_prompt_v6_arrests_are_followups_with_crime_time():
    text = " ".join(SYSTEM_PROMPT.split())
    assert "arrest, a detention" in text and '"person of interest"' in text
    assert "is always a follow-up (is_followup=true)" in text
    assert "never when the arrest, detention, or identification happened" in text
    assert "doesn't say when the crime itself happened, occurred_at is null" in text


def test_prompt_v7_string_of_crimes_and_publish_cap():
    assert PROMPT_VERSION == "v7"
    text = " ".join(SYSTEM_PROMPT.split())
    assert "For a string or series of crimes" in text
    assert "occurred_at is the time of the latest one" in text
    assert "occurred_at is never later than the item's published_at" in text


def test_batch_result_requires_raw_item_id():
    with pytest.raises(ValidationError):
        BatchResult.model_validate(VALID)


def test_user_prompt_is_json_array_with_ids():
    items = [
        RawItem(id=7, title="A", body=None, published_at=datetime(2026, 9, 28, tzinfo=timezone.utc)),
        RawItem(id=8, title="B", body="b", published_at=None),
    ]
    data = json.loads(build_user_prompt(items))
    assert [d["raw_item_id"] for d in data] == [7, 8]
    # Stored UTC midnight is shown to the model as St. Louis time (CDT, UTC-5).
    assert data[0]["published_at"] == "2026-09-27T19:00:00-05:00"
    assert data[1]["published_at"] is None


# --- Gemini client -----------------------------------------------------------

class FakeModels:
    def __init__(self, result):
        self.result = result
        self.kwargs = None

    def generate_content(self, **kwargs):
        self.kwargs = kwargs
        if isinstance(self.result, Exception):
            raise self.result
        return SimpleNamespace(text=self.result)


def fake_genai(monkeypatch, result):
    models = FakeModels(result)

    def client(api_key, http_options=None):
        models.client_kwargs = {"api_key": api_key, "http_options": http_options}
        return SimpleNamespace(api_key=api_key, models=models)

    monkeypatch.setattr("google.genai.Client", client)
    return models


def api_error(code, retry_delay=None):
    details = []
    if retry_delay is not None:
        details.append({
            "@type": "type.googleapis.com/google.rpc.RetryInfo",
            "retryDelay": retry_delay,
        })
    return genai_errors.APIError(
        code, {"error": {"code": code, "message": "nope", "details": details}}
    )


def test_gemini_client_returns_text_and_requests_json(monkeypatch):
    models = fake_genai(monkeypatch, '[{"ok": true}]')
    client = GeminiClient("test-key", "gemini-test")

    assert client.generate_json("sys", "user", list[BatchResult]) == '[{"ok": true}]'
    assert models.kwargs["model"] == "gemini-test"
    assert models.kwargs["contents"] == "user"
    config = models.kwargs["config"]
    assert config.response_mime_type == "application/json"
    assert config.response_schema == list[BatchResult]
    assert config.system_instruction == "sys"
    assert config.automatic_function_calling.disable is True


def test_gemini_client_disables_sdk_retries(monkeypatch):
    models = fake_genai(monkeypatch, "[]")
    GeminiClient("test-key", "gemini-test")
    options = models.client_kwargs["http_options"]
    assert options.retry_options.attempts == 1


def test_real_sdk_client_makes_one_attempt():
    """The real google-genai client built with our options never retries (no network:
    constructing a client and its tenacity policy makes no request)."""
    client = GeminiClient("test-key", "gemini-test")
    retry = client._client._api_client._retry
    assert retry.stop.max_attempt_number == 1


def test_gemini_429_carries_retry_delay(monkeypatch):
    fake_genai(monkeypatch, api_error(429, "37s"))
    client = GeminiClient("test-key", "gemini-test")

    with pytest.raises(RateLimitError) as info:
        client.generate_json("sys", "user", list[BatchResult])
    assert info.value.retry_after == 37.0


def test_gemini_429_without_retry_delay(monkeypatch):
    fake_genai(monkeypatch, api_error(429))
    with pytest.raises(RateLimitError) as info:
        GeminiClient("k", "m").generate_json("sys", "user", list[BatchResult])
    assert info.value.retry_after is None


def test_gemini_503_is_service_unavailable(monkeypatch):
    fake_genai(monkeypatch, api_error(503))
    with pytest.raises(ServiceUnavailableError):
        GeminiClient("k", "m").generate_json("sys", "user", list[BatchResult])


def test_gemini_other_errors_are_llm_errors(monkeypatch):
    fake_genai(monkeypatch, api_error(500))
    with pytest.raises(LLMError) as info:
        GeminiClient("k", "m").generate_json("sys", "user", list[BatchResult])
    assert not isinstance(info.value, (RateLimitError, ServiceUnavailableError))


@pytest.mark.parametrize(
    "details, expected",
    [
        ({"error": {"details": [{"retryDelay": "12.5s"}]}}, 12.5),
        ({"error": {"details": [{"@type": "QuotaFailure"}]}}, None),
        ({"error": {}}, None),
        ("not a dict", None),
    ],
)
def test_parse_retry_delay(details, expected):
    assert parse_retry_delay(details) == expected


def test_get_llm_client_gemini(monkeypatch):
    fake_genai(monkeypatch, "[]")
    client = get_llm_client(Settings(llm_provider="gemini", gemini_api_key="k", gemini_model="m"))
    assert isinstance(client, GeminiClient)
    assert client.model == "gemini:m"


def test_fallback_model_read_from_env(monkeypatch):
    from config import get_settings

    monkeypatch.setenv("GEMINI_FALLBACK_MODEL", "backup-model")
    get_settings.cache_clear()
    try:
        assert get_settings().gemini_fallback_model == "backup-model"
    finally:
        get_settings.cache_clear()


def test_get_llm_client_unknown_provider():
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        get_llm_client(Settings(llm_provider="nope"))


def test_get_llm_client_missing_key():
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        get_llm_client(Settings(llm_provider="gemini", gemini_api_key=None))


# --- CLI ---------------------------------------------------------------------

class _NullSession:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.mark.parametrize("stopped", [0, 1])
def test_cli_prints_counts(monkeypatch, capsys, stopped):
    import db

    monkeypatch.setattr(db, "SessionLocal", lambda: _NullSession())
    monkeypatch.setattr(
        cli, "classify_pending",
        lambda session: {"prefiltered": 2, "classified": 1, "stopped_quota": stopped},
    )
    cli.main()
    out = capsys.readouterr().out
    assert "prefiltered: 2" in out and "classified: 1" in out
    assert ("quota exhausted" in out) == bool(stopped)


# --- eval script -------------------------------------------------------------

EVAL_CASES_SHOT = (
    "- title: Teen killed in shooting\n"
    "  expected: {is_crime: true, crime_type: homicide, in_stl: true, was_shooting: true}\n"
    "- title: Man stabbed and killed\n"
    "  expected: {is_crime: true, crime_type: homicide, in_stl: true, was_shooting: false}\n"
)

EVAL_CASES = (
    "- title: Man shot in Walnut Park\n"
    "  published_at: '2026-09-28T07:15:00-05:00'\n"
    "  expected: {is_crime: true, crime_type: shooting, in_stl: true}\n"
    "- title: Teen killed in Dutchtown\n"
    "  expected: {is_crime: true, crime_type: shooting, in_stl: true}\n"
    "- title: Cardinals win\n"
    "  expected: {is_crime: false, crime_type: null, in_stl: null}\n"
)


@pytest.fixture
def eval_env(monkeypatch, tmp_path):
    cases = tmp_path / "cases.yaml"
    cases.write_text(EVAL_CASES)
    cache = tmp_path / "cache.json"
    llms = []
    llms_settings = []

    def make(*responses):
        llm = FakeLLM(*responses)
        llms.append(llm)
        def entry(provider, model, settings=None):
            llms_settings.append((provider, model))
            return ChainEntry(llm, RateLimiter(0))

        monkeypatch.setattr(eval_script, "build_entry", entry)
        return llm

    return SimpleNamespace(cases=cases, cache=cache, make=make, settings=llms_settings)


def test_eval_fixture_has_26_labeled_cases():
    cases = eval_script.load_cases(eval_script.DEFAULT_PATH)
    assert len(cases) == 26
    for case in cases:
        expected = case["expected"]
        optional = {
            "was_shooting", "is_followup", "creates_incident", "occurred_at_null",
            "occurred_window",
        }
        assert set(expected) - optional == {"is_crime", "crime_type", "in_stl"}
        # was_shooting is labeled on exactly the crime cases.
        assert ("was_shooting" in expected) == expected["is_crime"]
    followups = [c["expected"] for c in cases if "is_followup" in c["expected"]]
    # 2 recent follow-ups (crimes), 2 about older crimes (not crimes), 1 detention,
    # 1 first report of a string of burglaries.
    assert [(e["is_crime"], e["is_followup"]) for e in followups] == [
        (True, True), (True, True), (False, False), (False, False), (True, True),
        (True, False),
    ]


def test_eval_fixture_has_detention_followup():
    cases = eval_script.load_cases(eval_script.DEFAULT_PATH)
    [case] = [c for c in cases if "person of interest" in c["title"]]
    assert case["expected"] == {
        "is_crime": True, "crime_type": "burglary", "in_stl": True, "was_shooting": False,
        "is_followup": True, "creates_incident": False, "occurred_at_null": True,
    }


def test_eval_fixture_has_string_of_burglaries():
    cases = eval_script.load_cases(eval_script.DEFAULT_PATH)
    [case] = [c for c in cases if "string of overnight burglaries" in c["title"]]
    expected = case["expected"]
    assert expected["creates_incident"] is True
    start, end = (datetime.fromisoformat(w) for w in expected["occurred_window"])
    published = datetime.fromisoformat(case["published_at"])
    # Early Sunday Oct 4, never the publish day.
    assert start.date() == end.date() == date(2026, 10, 4)
    assert end < published


@pytest.mark.parametrize(
    "occurred_at, ok",
    [
        ("2026-10-04T04:00:00-05:00", True),
        ("2026-10-04T09:00:00Z", True),  # 04:00 CDT, as UTC
        ("2026-10-05T00:00:00-05:00", False),  # the v5 answer
        (None, False),
    ],
)
def test_eval_scores_occurred_window(occurred_at, ok):
    pred = ClassifierOutput.model_validate(
        {**VALID, "crime_type": "burglary", "was_shooting": False, "occurred_at": occurred_at}
    )
    window = ["2026-10-04T00:00:00-05:00", "2026-10-04T05:00:00-05:00"]
    assert eval_script.score(pred, "occurred_window", window)[0] is ok


def test_eval_batches_and_scores(eval_env, capsys):
    llm = eval_env.make(echo())

    eval_script.main([str(eval_env.cases), "--cache", str(eval_env.cache)])

    assert len(llm.calls) == 1
    assert llm.batch_ids() == [0, 1]  # prefiltered "Cardinals win" not sent
    assert "Overall: 3/3 (100%)" in capsys.readouterr().out


def test_eval_cache_avoids_second_llm_call(eval_env):
    eval_env.make(echo())
    eval_script.main([str(eval_env.cases), "--cache", str(eval_env.cache)])
    assert len(json.loads(eval_env.cache.read_text())) == 2

    llm = eval_env.make()  # any request would fail: no scripted responses
    eval_script.main([str(eval_env.cases), "--cache", str(eval_env.cache)])
    assert llm.calls == []


def test_eval_no_cache_forces_llm_call(eval_env):
    eval_env.make(echo())
    eval_script.main([str(eval_env.cases), "--cache", str(eval_env.cache)])

    llm = eval_env.make(echo())
    eval_script.main([str(eval_env.cases), "--cache", str(eval_env.cache), "--no-cache"])
    assert len(llm.calls) == 1


def test_eval_quota_stop_reports_partial(eval_env, capsys):
    eval_env.make(RateLimitError("429", retry_after=None))

    eval_script.main([str(eval_env.cases), "--cache", str(eval_env.cache)])

    out = capsys.readouterr().out
    assert "quota exhausted" in out
    assert "2 case(s) not scored" in out


def test_eval_scores_was_shooting_when_labeled(eval_env, capsys):
    eval_env.cases.write_text(EVAL_CASES_SHOT)
    # Both predicted as fatal shootings: right for case 0, wrong for the stabbing.
    eval_env.make(echo({0: {"crime_type": "homicide", "was_shooting": True},
                        1: {"crime_type": "homicide", "was_shooting": True}}))

    eval_script.main([str(eval_env.cases), "--cache", str(eval_env.cache)])

    out = capsys.readouterr().out
    assert "was_shooting: want False, got True" in out
    assert "was_shooting: 1/2 (50%)" in out
    assert "Overall: 1/2 (50%)" in out


def test_eval_skips_was_shooting_when_unlabeled(eval_env, capsys):
    eval_env.make(echo())
    eval_script.main([str(eval_env.cases), "--cache", str(eval_env.cache)])
    assert "was_shooting:" not in capsys.readouterr().out


EVAL_CASES_DETAINED = (
    "- title: Police detain person of interest in burglaries\n"
    "  expected: {is_crime: true, crime_type: burglary, in_stl: true,"
    " is_followup: true, creates_incident: false, occurred_at_null: true}\n"
    "- title: Burglars hit Soulard bar\n"
    "  expected: {is_crime: true, crime_type: burglary, in_stl: true, creates_incident: true}\n"
)


def test_eval_scores_creates_incident_and_occurred_at_null(eval_env, capsys):
    eval_env.cases.write_text(EVAL_CASES_DETAINED)
    # Case 0 is a follow-up, but dated by the detention: it would not create an
    # incident, yet occurred_at is wrong. Case 1 is a first report.
    eval_env.make(echo({0: {"crime_type": "burglary", "is_followup": True},
                        1: {"crime_type": "burglary"}}))

    eval_script.main([str(eval_env.cases), "--cache", str(eval_env.cache)])

    out = capsys.readouterr().out
    assert "creates_incident: 2/2 (100%)" in out
    assert "occurred_at_null: want True, got False" in out
    assert "occurred_at_null: 0/1 (0%)" in out
    assert "Overall: 1/2 (50%)" in out


def test_eval_creates_incident_false_for_followup_or_outside_stl():
    base = {**VALID, "crime_type": "burglary"}
    assert eval_script.predicted(ClassifierOutput.model_validate(base), "creates_incident")
    for override in ({"is_followup": True}, {"in_stl": False}, {"is_crime": False}):
        out = ClassifierOutput.model_validate({**base, **override})
        assert not eval_script.predicted(out, "creates_incident")
    no_time = ClassifierOutput.model_validate({**base, "occurred_at": None})
    assert eval_script.predicted(no_time, "occurred_at_null")


def test_eval_model_flag_overrides_gemini_model(eval_env, capsys):
    llm = eval_env.make(echo())
    llm.model = "backup-model"
    eval_script.main([str(eval_env.cases), "--cache", str(eval_env.cache), "--model", "backup-model"])
    assert eval_env.settings[-1] == ("gemini", "backup-model")
    assert "Model: backup-model" in capsys.readouterr().out


def test_eval_defaults_to_gemini_model(eval_env, monkeypatch):
    from config import get_settings

    monkeypatch.setenv("GEMINI_MODEL", "primary-model")
    monkeypatch.setenv("LLM_CHAIN", "")  # blank: .env can't fill it in
    get_settings.cache_clear()
    try:
        eval_env.make(echo())
        eval_script.main([str(eval_env.cases), "--cache", str(eval_env.cache)])
        assert eval_env.settings[-1] == ("gemini", "primary-model")
    finally:
        get_settings.cache_clear()
