import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors
from pydantic import ValidationError
from sqlalchemy import select

import classifier.__main__ as cli
import classifier.eval as eval_script
from classifier.classify import classify_pending
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
from classifier.prompt import PROMPT_VERSION, build_user_prompt
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

    monkeypatch.setattr("classifier.classify.get_llm_client", boom)
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
    assert c.prompt_version == "v2"
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
    llm = FakeLLM(*[ServiceUnavailableError("503")] * 4, echo())

    counts = run(llm)

    assert sleeps == [5, 15, 45]
    assert all(i.status == "new" and i.retries == 0 for i in first)
    assert counts["deferred"] == 10
    assert second.status == "classified"  # next batch still runs


def test_503_then_success(make_item, run, sleeps):
    item = make_item()
    llm = FakeLLM(ServiceUnavailableError("503"), echo())

    run(llm)

    assert sleeps == [5]
    assert item.status == "classified"


def test_other_api_error_uses_same_backoff(make_item, run, sleeps):
    item = make_item()
    llm = FakeLLM(*[LLMError("400 bad request")] * 4)

    counts = run(llm)

    assert sleeps == [5, 15, 45]
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
    monkeypatch.setattr(
        "google.genai.Client",
        lambda api_key: SimpleNamespace(api_key=api_key, models=models),
    )
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
    assert client.model == "m"


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

    def make(*responses):
        llm = FakeLLM(*responses)
        llms.append(llm)
        monkeypatch.setattr(eval_script, "get_llm_client", lambda: llm)
        return llm

    monkeypatch.setattr(eval_script, "make_limiter", lambda sleep: RateLimiter(0))
    return SimpleNamespace(cases=cases, cache=cache, make=make)


def test_eval_fixture_has_20_labeled_cases():
    cases = eval_script.load_cases(eval_script.DEFAULT_PATH)
    assert len(cases) == 20
    for case in cases:
        assert set(case["expected"]) == {"is_crime", "crime_type", "in_stl"}


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
    assert "daily quota exhausted" in out
    assert "2 case(s) not scored" in out
