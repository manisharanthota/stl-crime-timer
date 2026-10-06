"""Provider chain (LLM_CHAIN), the OpenAI-compatible (Groq) client, token-aware rate
limiting and batch sizing. Every provider is mocked: FakeLLM or httpx.MockTransport."""

import json
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import select

import classifier.eval as eval_script
from classifier.classify import classify_pending, request_llm, take_batch
from classifier.llm import (
    GeminiClient,
    InvalidJSONResponse,
    LLMError,
    OpenAICompatibleClient,
    RateLimitError,
    RequestTooLargeError,
    ServiceUnavailableError,
    Usage,
    parse_duration,
    unwrap_results,
)
from classifier.prompt import SYSTEM_PROMPT, build_user_prompt
from classifier.providers import (
    ChainEntry,
    build_chain,
    chain_names,
    chain_specs,
    make_client,
    make_entry,
    parse_chain,
)
from classifier.ratelimit import RateLimiter, TokenRateLimiter
from classifier.tokens import count_items, output_tokens, request_tokens
from config import Settings
from models import Classification
from tests.test_classifier import VALID, FakeClock, FakeLLM, echo, make_item  # noqa: F401

QUOTA = RateLimitError("429 quota", retry_after=None)
OVERLOADED = ServiceUnavailableError("503")


def fake(model, *responses):
    llm = FakeLLM(*responses)
    llm.model = model
    return llm


def entries(*llms, max_batch_tokens=None):
    return [ChainEntry(llm, RateLimiter(0), max_batch_tokens) for llm in llms]


@pytest.fixture
def sleeps():
    return []


@pytest.fixture
def run_chain(session, sleeps):
    def _run(chain, **kwargs):
        return classify_pending(session, chain=chain, sleep=sleeps.append, **kwargs)

    return _run


def saved_models(session):
    return [c.model for c in session.scalars(select(Classification).order_by(Classification.id))]


# --- LLM_CHAIN parsing ---------------------------------------------------------

def test_parse_chain():
    assert parse_chain(" groq:openai/gpt-oss-120b , GEMINI:gemini-3.8-flash,gemini:lite ") == [
        ("groq", "openai/gpt-oss-120b"), ("gemini", "gemini-3.8-flash"), ("gemini", "lite"),
    ]


def test_parse_chain_keeps_colons_in_model_and_drops_duplicates():
    assert parse_chain("groq:a:b,groq:a:b,,") == [("groq", "a:b")]


@pytest.mark.parametrize(
    "text, match",
    [("groq", "provider:model"), ("groq:", "provider:model"), (":m", "provider:model"),
     ("openai:gpt", "unknown provider"), (" , ", "empty")],
)
def test_parse_chain_errors(text, match):
    with pytest.raises(ValueError, match=match):
        parse_chain(text)


def test_chain_specs_from_llm_chain():
    settings = Settings(llm_chain="groq:g,gemini:a", gemini_model="ignored")
    assert chain_specs(settings) == [("groq", "g"), ("gemini", "a")]
    assert chain_names(settings) == ["groq:g", "gemini:a"]


@pytest.mark.parametrize(
    "fallback, expected",
    [(None, [("gemini", "p")]), ("p", [("gemini", "p")]), ("f", [("gemini", "p"), ("gemini", "f")])],
)
def test_chain_specs_without_llm_chain_uses_gemini_settings(fallback, expected):
    assert chain_specs(Settings(gemini_model="p", gemini_fallback_model=fallback)) == expected


def test_chain_specs_unknown_legacy_provider():
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        chain_specs(Settings(llm_provider="nope"))
    assert chain_names(Settings(llm_provider="nope")) == []


# --- clients and entries from settings ----------------------------------------

def test_make_client_groq():
    client = make_client("groq", "llama-x", Settings(groq_api_key="gsk"))
    assert isinstance(client, OpenAICompatibleClient)
    assert client.model == "groq:llama-x"
    assert client.reasoning_effort is None


def test_make_client_groq_gpt_oss_defaults_to_low_reasoning():
    settings = Settings(groq_api_key="gsk")
    assert make_client("groq", "openai/gpt-oss-120b", settings).reasoning_effort == "low"
    settings = Settings(groq_api_key="gsk", groq_reasoning_effort="medium")
    assert make_client("groq", "openai/gpt-oss-120b", settings).reasoning_effort == "medium"


@pytest.mark.parametrize("provider, env", [("groq", "GROQ_API_KEY"), ("gemini", "GEMINI_API_KEY")])
def test_make_client_missing_key(provider, env):
    with pytest.raises(ValueError, match=env):
        make_client(provider, "m", Settings())


def test_groq_entry_has_token_limit_and_batch_budget():
    settings = Settings(groq_api_key="gsk", groq_tpm=8000, groq_rpm=30)
    entry = make_entry(make_client("groq", "m", settings), "groq", settings)
    assert isinstance(entry.limiter, TokenRateLimiter)
    assert entry.limiter.tpm == 8000
    assert entry.limiter.min_interval == 3  # 60/30 + 1
    assert entry.max_batch_tokens == 6400


def test_build_chain_gives_each_model_its_own_limiter():
    settings = Settings(
        llm_chain="groq:g,gemini:a,gemini:b", groq_api_key="gsk", gemini_api_key="k", gemini_rpm=5
    )
    chain = build_chain(settings)
    assert [e.client.model for e in chain] == ["groq:g", "gemini:a", "gemini:b"]
    assert isinstance(chain[1].client, GeminiClient)
    assert type(chain[1].limiter) is RateLimiter and chain[1].limiter.min_interval == 13
    assert chain[1].max_batch_tokens is None
    assert len({id(e.limiter) for e in chain}) == 3


# --- OpenAI-compatible client --------------------------------------------------

def results_body(user, overrides=None):
    """A chat completion whose content is {"results": [...]} for every item in `user`."""
    results = [{**VALID, "raw_item_id": e["raw_item_id"], **(overrides or {})}
               for e in json.loads(user)]
    return {
        "choices": [{"message": {"content": json.dumps({"results": results})}}],
        "usage": {"total_tokens": 1234},
    }


class Server:
    """MockTransport handler: records requests, answers with scripted responses
    (callables get the request's user prompt)."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request):
        body = json.loads(request.content)
        self.requests.append((request, body))
        response = self.responses.pop(0)
        if callable(response):
            return httpx.Response(200, json=response(body["messages"][1]["content"]))
        return response

    def client(self, model="openai/gpt-oss-120b", reasoning_effort=None):
        return OpenAICompatibleClient(
            "https://api.groq.test/openai/v1/", "gsk_test", model, "groq",
            reasoning_effort=reasoning_effort,
            client=httpx.Client(transport=httpx.MockTransport(self)),
        )


USER = json.dumps([{"raw_item_id": 1, "title": "Man shot"}, {"raw_item_id": 2, "title": "x"}])


def test_openai_client_request_shape_and_unwrap():
    server = Server(
        httpx.Response(
            200,
            json=results_body(USER),
            headers={"x-ratelimit-remaining-tokens": "6500", "x-ratelimit-reset-tokens": "7.5s"},
        )
    )
    client = server.client(reasoning_effort="low")

    text = client.generate_json("SYS", USER, None)

    [(request, body)] = server.requests
    assert str(request.url) == "https://api.groq.test/openai/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer gsk_test"
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["response_format"] == {"type": "json_object"}
    assert body["temperature"] == 0
    assert body["reasoning_effort"] == "low"
    assert body["max_tokens"] == output_tokens(2)
    assert body["messages"][0]["content"].startswith("SYS")
    assert '{"results": [...]}' in body["messages"][0]["content"]
    assert body["messages"][1] == {"role": "user", "content": USER}
    assert [r["raw_item_id"] for r in json.loads(text)] == [1, 2]
    assert client.last_usage == Usage(total_tokens=1234, remaining_tokens=6500, reset_seconds=7.5)


def test_openai_client_omits_reasoning_effort_when_unset():
    server = Server(results_body)
    server.client(model="llama-x").generate_json("SYS", USER, None)
    assert "reasoning_effort" not in server.requests[0][1]


@pytest.mark.parametrize(
    "response, error, retry_after",
    [
        (httpx.Response(429, headers={"retry-after": "12"}, json={"error": {"message": "TPM"}}),
         RateLimitError, 12.0),
        (httpx.Response(429, json={"error": {"message": "Please try again in 7.66s."}}),
         RateLimitError, 7.66),
        (httpx.Response(429, json={"error": {"message": "tokens per day (TPD): try again in 12m3s"}}),
         RateLimitError, 723.0),
        (httpx.Response(429, json={"error": {"message": "rate limited"}}), RateLimitError, None),
    ],
)
def test_openai_client_429_retry_after(response, error, retry_after):
    with pytest.raises(error) as info:
        Server(response).client().generate_json("SYS", USER, None)
    assert info.value.retry_after == retry_after


@pytest.mark.parametrize(
    "response, error",
    [
        (httpx.Response(503, text="over capacity"), ServiceUnavailableError),
        (httpx.Response(498, json={"error": {"message": "capacity"}}), ServiceUnavailableError),
        (httpx.Response(413, json={"error": {"message": "Request too large"}}), RequestTooLargeError),
        (httpx.Response(400, json={"error": {"code": "json_validate_failed", "message": "bad"}}),
         InvalidJSONResponse),
        (httpx.Response(500, text="boom"), LLMError),
        (httpx.Response(200, json={"choices": []}), LLMError),
        (httpx.Response(200, text="not json"), LLMError),
    ],
)
def test_openai_client_error_mapping(response, error):
    with pytest.raises(error) as info:
        Server(response).client().generate_json("SYS", USER, None)
    if error is LLMError:
        assert not isinstance(info.value, (RateLimitError, ServiceUnavailableError))


def test_openai_client_network_error_is_llm_error():
    def handler(request):
        raise httpx.ConnectError("refused")

    client = OpenAICompatibleClient(
        "https://x.test/v1", "k", "m", "groq", client=httpx.Client(transport=httpx.MockTransport(handler))
    )
    with pytest.raises(LLMError, match="ConnectError"):
        client.generate_json("SYS", USER, None)


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"results": [{"a": 1}]}', [{"a": 1}]),
        ('{"items": [{"a": 1}], "note": "x"}', [{"a": 1}]),
        ('[{"a": 1}]', [{"a": 1}]),
    ],
)
def test_unwrap_results(text, expected):
    assert json.loads(unwrap_results(text)) == expected


@pytest.mark.parametrize("text", ['{"a": [1], "b": [2]}', '{"a": 1}', "not json"])
def test_unwrap_results_leaves_other_shapes_alone(text):
    assert unwrap_results(text) == text


@pytest.mark.parametrize(
    "text, expected",
    [("12", 12.0), ("7.66s", 7.66), ("2m59.5s", 179.5), ("1h2m3s", 3723.0), ("120ms", 0.12),
     ("soon", None), ("", None), (None, None), ("5s junk", None)],
)
def test_parse_duration(text, expected):
    assert parse_duration(text) == pytest.approx(expected) if expected else parse_duration(text) is None


# --- tokens ----------------------------------------------------------------------

def test_token_estimates():
    assert count_items(USER) == 2
    assert count_items("nope") == 1
    assert output_tokens(2) > output_tokens(1)
    assert request_tokens("a" * 300, USER) > output_tokens(2) + 100


def test_token_limiter_waits_for_window():
    clock = FakeClock()
    limiter = TokenRateLimiter(0, 1000, clock=clock, sleep=clock.sleep)

    limiter.wait(600)
    clock.now += 10
    limiter.wait(300)  # 900 fits
    clock.now += 5
    limiter.wait(300)  # 1200 doesn't: wait until the first request leaves the window

    assert clock.sleeps == [pytest.approx(45)]


def test_token_limiter_record_replaces_estimate():
    clock = FakeClock()
    limiter = TokenRateLimiter(0, 1000, clock=clock, sleep=clock.sleep)
    limiter.wait(900)
    limiter.record(Usage(total_tokens=200))
    limiter.wait(700)  # 200 actual + 700 fits
    assert clock.sleeps == []
    assert limiter.used() == 900


def test_token_limiter_waits_for_server_reset():
    clock = FakeClock()
    limiter = TokenRateLimiter(0, 10_000, clock=clock, sleep=clock.sleep)
    limiter.wait(100)
    limiter.record(Usage(total_tokens=100, remaining_tokens=500, reset_seconds=8))
    limiter.wait(400)  # fits what the server says is left
    assert clock.sleeps == []
    limiter.record(Usage(total_tokens=400, remaining_tokens=100, reset_seconds=6))
    limiter.wait(400)
    assert clock.sleeps == [6]


def test_token_limiter_oversized_request_waits_for_empty_window():
    clock = FakeClock()
    limiter = TokenRateLimiter(0, 1000, clock=clock, sleep=clock.sleep)
    limiter.wait(100)
    limiter.wait(5000)
    assert clock.sleeps == [pytest.approx(60)]


def test_token_limiter_still_spaces_requests():
    clock = FakeClock()
    limiter = TokenRateLimiter(3, 1000, clock=clock, sleep=clock.sleep)
    limiter.wait(10)
    limiter.wait(10)
    assert clock.sleeps == [3]


def test_request_llm_records_usage():
    llm = fake("groq:m", echo())
    llm.last_usage = Usage(total_tokens=42)
    clock = FakeClock()
    limiter = TokenRateLimiter(0, 10_000, clock=clock, sleep=clock.sleep)
    request_llm(llm, json.dumps([{"raw_item_id": 1}]), limiter, clock.sleep)
    assert limiter.used() == 42


def test_request_llm_invalid_json_from_provider_is_an_invalid_response():
    llm = fake("groq:m", InvalidJSONResponse("400 json_validate_failed"))
    assert request_llm(llm, "[]", RateLimiter(0), lambda s: None) == ""


# --- batch sizing ------------------------------------------------------------------

def test_take_batch_respects_token_budget(make_item):
    items = [make_item(body="x" * 1500) for _ in range(10)]
    budget = request_tokens(SYSTEM_PROMPT, build_user_prompt(items[:3]))

    assert len(take_batch(items, 0, 10, None)) == 10
    assert len(take_batch(items, 0, 10, budget)) == 3
    assert [i.id for i in take_batch(items, 3, 10, budget)] == [i.id for i in items[3:6]]
    assert len(take_batch(items, 0, 10, 1)) == 1  # too big alone: sent alone


def test_batches_sized_per_provider(session, make_item, run_chain):
    items = [make_item(body="x" * 1500) for _ in range(13)]
    budget = request_tokens(SYSTEM_PROMPT, build_user_prompt(items[:3]))
    groq = fake("groq:m", echo(), QUOTA)
    gemini = fake("gemini:a", echo(), echo())
    chain = [ChainEntry(groq, RateLimiter(0), budget), ChainEntry(gemini, RateLimiter(0))]

    counts = run_chain(chain)

    assert len(groq.batch_ids(0)) == 3
    assert len(groq.batch_ids(1)) == 3  # second batch hit the quota...
    assert len(gemini.batch_ids(0)) == 3  # ...and moved on as it was
    assert len(gemini.calls) == 2
    assert len(gemini.batch_ids(1)) == 7  # then Gemini's own (unlimited) batch size
    assert counts["classified"] == 13
    assert counts["by_model"] == {"groq:m": 3, "gemini:a": 10}


# --- chain behavior ------------------------------------------------------------------

def test_three_provider_chain_falls_through(session, make_item, run_chain, sleeps):
    make_item()
    groq = fake("groq:g", QUOTA)
    flash = fake("gemini:flash", OVERLOADED)
    lite = fake("gemini:lite", echo())

    counts = run_chain(entries(groq, flash, lite))

    assert (len(groq.calls), len(flash.calls), len(lite.calls)) == (1, 1, 1)
    assert sleeps == []
    assert saved_models(session) == ["gemini:lite"]
    assert (counts["fallback"], counts["switched_quota"], counts["stopped_quota"]) == (1, 1, 0)
    assert counts["by_model"] == {"gemini:lite": 1}


def test_first_provider_stores_provider_model(session, make_item, run_chain):
    make_item()
    counts = run_chain(entries(fake("groq:openai/gpt-oss-120b", echo()), fake("gemini:a")))
    assert saved_models(session) == ["groq:openai/gpt-oss-120b"]
    assert counts["fallback"] == 0


def test_repeatedly_overloaded_provider_is_dropped_for_the_run(make_item, run_chain):
    for _ in range(25):  # three batches
        make_item()
    groq = fake("groq:g", OVERLOADED, OVERLOADED)
    gemini = fake("gemini:a", echo(), echo(), echo())

    counts = run_chain(entries(groq, gemini))

    assert len(groq.calls) == 2  # not asked for the third batch
    assert len(gemini.calls) == 3
    assert counts["classified"] == counts["fallback"] == 25
    assert counts["switched_quota"] == 0  # dropped for 503s, not quota


def test_overloaded_once_keeps_provider_first(make_item, run_chain):
    for _ in range(15):
        make_item()
    groq = fake("groq:g", OVERLOADED, echo())
    gemini = fake("gemini:a", echo())
    counts = run_chain(entries(groq, gemini))
    assert (len(groq.calls), len(gemini.calls)) == (2, 1)
    assert counts["by_model"] == {"gemini:a": 10, "groq:g": 5}


def test_last_provider_is_never_dropped_for_503s(make_item, run_chain, sleeps):
    for _ in range(25):
        make_item()
    groq = fake("groq:g", *[OVERLOADED] * 9)
    counts = run_chain(entries(groq))
    assert len(groq.calls) == 9  # every batch tried, with backoff
    assert sleeps == [5, 15] * 3
    assert (counts["deferred"], counts["stopped_quota"]) == (25, 0)


def test_too_large_batch_goes_to_next_provider(make_item, run_chain):
    for _ in range(15):
        make_item()
    groq = fake("groq:g", RequestTooLargeError("413"), echo())
    gemini = fake("gemini:a", echo())

    counts = run_chain(entries(groq, gemini))

    assert (len(groq.calls), len(gemini.calls)) == (2, 1)  # groq still used for batch 2
    assert counts["by_model"] == {"gemini:a": 10, "groq:g": 5}


def test_too_large_on_last_provider_defers(make_item, run_chain, sleeps):
    item = make_item()
    counts = run_chain(entries(fake("groq:g", RequestTooLargeError("413"))))
    assert counts["deferred"] == 1
    assert sleeps == []  # not retried
    assert (item.status, item.retries) == ("new", 0)


def test_all_providers_out_of_quota_stops_run(make_item, run_chain):
    items = [make_item() for _ in range(5)]
    counts = run_chain(entries(fake("groq:g", QUOTA), fake("gemini:a", QUOTA), fake("gemini:b", QUOTA)))
    assert (counts["stopped_quota"], counts["switched_quota"]) == (1, 1)
    assert all((i.status, i.retries) == ("new", 0) for i in items)


def test_short_rate_limit_waits_on_same_provider(make_item, run_chain, sleeps):
    make_item()
    groq = fake("groq:g", RateLimitError("429 TPM", retry_after=7.5), echo())
    gemini = fake("gemini:a")
    run_chain(entries(groq, gemini))
    assert sleeps == [7.5]
    assert gemini.calls == []


def test_rate_limit_exhaustion_moves_to_next_provider(make_item, run_chain, sleeps):
    make_item()
    groq = fake("groq:g", *[RateLimitError("429 TPM", retry_after=5)] * 4)
    gemini = fake("gemini:a", echo())
    counts = run_chain(entries(groq, gemini))
    assert sleeps == [5, 5, 5]
    assert counts["by_model"] == {"gemini:a": 1}


def test_provider_invalid_json_retried_then_back_to_new(make_item, run_chain):
    item = make_item()
    groq = fake("groq:g", InvalidJSONResponse("400"), InvalidJSONResponse("400"))
    counts = run_chain(entries(groq, fake("gemini:a")))
    assert len(groq.calls) == 2
    assert counts["retry_later"] == 1
    assert (item.status, item.retries) == ("new", 1)


def test_classify_pending_with_mocked_groq_end_to_end(session, make_item, run_chain):
    make_item()
    server = Server(results_body)
    entry = ChainEntry(server.client(), RateLimiter(0), 6400)

    counts = run_chain([entry])

    assert counts["classified"] == 1
    [c] = session.scalars(select(Classification)).all()
    assert (c.model, c.crime_type) == ("groq:openai/gpt-oss-120b", "shooting")


def test_settings_chain_used_by_default(session, make_item, monkeypatch, sleeps):
    make_item()
    llm = fake("groq:g", echo())
    monkeypatch.setattr(
        "classifier.classify.build_chain", lambda sleep: [ChainEntry(llm, RateLimiter(99))]
    )
    counts = classify_pending(session, sleep=sleeps.append)
    assert counts["by_model"] == {"groq:g": 1}


# --- eval --provider ---------------------------------------------------------------

EVAL_CASES = (
    "- title: Man shot in Walnut Park\n"
    "  expected: {is_crime: true, crime_type: shooting, in_stl: true}\n"
)


@pytest.fixture
def eval_env(monkeypatch, tmp_path):
    cases = tmp_path / "cases.yaml"
    cases.write_text(EVAL_CASES)
    built = []

    def entry(provider, model, settings=None):
        llm = fake(f"{provider}:{model}", echo())
        built.append(llm)
        return ChainEntry(llm, RateLimiter(0))

    monkeypatch.setattr(eval_script, "build_entry", entry)
    return SimpleNamespace(
        args=[str(cases), "--cache", str(tmp_path / "cache.json")], built=built
    )


def test_eval_provider_flag(eval_env, capsys):
    eval_script.main(eval_env.args + ["--provider", "groq:openai/gpt-oss-120b"])
    assert eval_env.built[-1].model == "groq:openai/gpt-oss-120b"
    assert len(eval_env.built[-1].calls) == 1
    out = capsys.readouterr().out
    assert "Model: groq:openai/gpt-oss-120b" in out
    assert "Overall: 1/1" in out


def test_eval_cache_is_per_provider(eval_env):
    eval_script.main(eval_env.args + ["--provider", "groq:m"])
    eval_script.main(eval_env.args + ["--provider", "groq:m"])
    assert eval_env.built[-1].calls == []  # cached
    eval_script.main(eval_env.args + ["--provider", "gemini:m"])
    assert len(eval_env.built[-1].calls) == 1  # same model name, other provider


@pytest.mark.parametrize("value", ["groq", "nope:m", "groq:a,gemini:b"])
def test_eval_bad_provider_flag(eval_env, value):
    with pytest.raises(SystemExit):
        eval_script.main(eval_env.args + ["--provider", value])


def test_eval_provider_and_model_are_exclusive(eval_env):
    with pytest.raises(SystemExit):
        eval_script.main(eval_env.args + ["--provider", "groq:m", "--model", "x"])


def test_eval_gemini_cache_keys_unchanged():
    case = {"title": "Man shot"}
    assert eval_script.cache_key(case, "gemini:flash") == eval_script.cache_key(case, "flash")
    assert eval_script.cache_key(case, "groq:flash") != eval_script.cache_key(case, "flash")


# --- per-minute vs per-day 429s, usage details, reasoning defaults ----------------

OTPM_MESSAGE = (
    "Request too large for model `qwen/qwen3.8-27b` in organization `org_x` service tier "
    "`on_demand` on output tokens per minute (OTPM): Limit 1000, Requested 1326."
)
TPD_MESSAGE = (
    "Rate limit reached for model `m` on tokens per day (TPD): Limit 200000, Used 199000, "
    "Requested 3000. Please try again in 7m12s."
)


@pytest.mark.parametrize(
    "message, expected",
    [
        (OTPM_MESSAGE, True),
        ("on tokens per minute (TPM): Limit 8000", True),
        ("on requests per minute (RPM)", True),
        (TPD_MESSAGE, False),
        ("on requests per day (RPD)", False),
        ("rate limited", None),
        (None, None),
    ],
)
def test_rate_limit_period(message, expected):
    from classifier.llm import rate_limit_period

    assert rate_limit_period(message) is expected


def test_openai_client_429_reports_period():
    response = httpx.Response(429, json={"error": {"message": OTPM_MESSAGE}})
    with pytest.raises(RateLimitError) as info:
        Server(response).client().generate_json("SYS", USER, None)
    assert info.value.per_minute is True
    assert info.value.retry_after is None


def test_per_minute_429_without_delay_waits_instead_of_stopping(make_item, run_chain, sleeps):
    make_item()
    groq = fake("groq:q", RateLimitError(OTPM_MESSAGE, None, per_minute=True), echo())
    counts = run_chain(entries(groq, fake("gemini:a")))
    assert sleeps == [61]
    assert counts["by_model"] == {"groq:q": 1}
    assert counts["switched_quota"] == 0


def test_per_minute_429_long_delay_is_capped(make_item, run_chain, sleeps):
    make_item()
    groq = fake("groq:q", RateLimitError("TPM", 300, per_minute=True), echo())
    run_chain(entries(groq))
    assert sleeps == [61]


def test_per_minute_429s_that_never_clear_move_to_next_provider(make_item, run_chain, sleeps):
    make_item()
    groq = fake("groq:q", *[RateLimitError("TPM", 20, per_minute=True)] * 4)
    gemini = fake("gemini:a", echo())
    counts = run_chain(entries(groq, gemini))
    assert sleeps == [20, 20, 20]
    assert counts["by_model"] == {"gemini:a": 1}


def test_per_day_429_stops_provider_even_with_short_delay(make_item, run_chain, sleeps):
    make_item()
    groq = fake("groq:q", RateLimitError(TPD_MESSAGE, 30, per_minute=False))
    gemini = fake("gemini:a", echo())
    counts = run_chain(entries(groq, gemini))
    assert sleeps == []
    assert counts["switched_quota"] == 1


def test_openai_client_reads_usage_details():
    body = {
        "choices": [{"message": {"content": '{"results": []}'}}],
        "usage": {"prompt_tokens": 2000, "completion_tokens": 900, "total_tokens": 2900,
                  "completion_tokens_details": {"reasoning_tokens": 300}},
    }
    client = Server(httpx.Response(200, json=body)).client()
    client.generate_json("SYS", USER, None)
    assert client.last_usage == Usage(
        total_tokens=2900, prompt_tokens=2000, completion_tokens=900, reasoning_tokens=300
    )


def test_qwen_reasoning_off_by_default():
    settings = Settings(groq_api_key="gsk")
    assert make_client("groq", "qwen/qwen3.8-27b", settings).reasoning_effort == "none"
    settings = Settings(groq_api_key="gsk", groq_reasoning_effort="default")
    assert make_client("groq", "qwen/qwen3.8-27b", settings).reasoning_effort == "default"


def test_eval_format_usage():
    usage = Usage(total_tokens=3731, prompt_tokens=2405, completion_tokens=1326, reasoning_tokens=200)
    assert eval_script.format_usage(10, usage) == (
        "Batch of 10: prompt 2405, completion 1326 (reasoning 200), total 3731"
    )
    assert "no token usage" in eval_script.format_usage(3, None)


def test_eval_prints_usage_per_batch(eval_env, capsys):
    eval_script.main(eval_env.args + ["--provider", "groq:m"])
    assert "Batch of 1: no token usage reported" in capsys.readouterr().out


# --- output tokens per minute (OTPM) ----------------------------------------------

def test_otpm_limit_parsed_from_429():
    response = httpx.Response(429, json={"error": {"message": OTPM_MESSAGE}})
    with pytest.raises(RateLimitError) as info:
        Server(response).client().generate_json("SYS", USER, None)
    assert info.value.output_limit == 1000
    tpm = httpx.Response(429, json={"error": {"message": "on tokens per minute (TPM): Limit 8000"}})
    with pytest.raises(RateLimitError) as info:
        Server(tpm).client().generate_json("SYS", USER, None)
    assert info.value.output_limit is None


def test_token_limiter_waits_for_output_window():
    clock = FakeClock()
    limiter = TokenRateLimiter(0, 100_000, clock=clock, sleep=clock.sleep, otpm=1000)

    limiter.wait(3000, 750)
    limiter.record(Usage(total_tokens=3000, completion_tokens=650))
    clock.now += 20
    limiter.wait(3000, 300)  # 650 + 300 fits
    limiter.record(Usage(total_tokens=3000, completion_tokens=300))
    clock.now += 5
    limiter.wait(3000, 600)  # 950 + 600 doesn't: wait for the first reply to age out

    assert clock.sleeps == [pytest.approx(35)]
    assert limiter.output_used() == 900


def test_token_limiter_without_otpm_ignores_output():
    clock = FakeClock()
    limiter = TokenRateLimiter(0, 100_000, clock=clock, sleep=clock.sleep)
    limiter.wait(10, 5000)
    limiter.wait(10, 5000)
    assert clock.sleeps == []


def test_token_limiter_learns_otpm_from_429():
    limiter = TokenRateLimiter(0, 8000)
    limiter.rate_limited(RateLimitError("x", 10, True, output_limit=1000))
    assert limiter.otpm == 1000
    limiter.rate_limited(RateLimitError("x", 10, True, output_limit=2000))
    assert limiter.otpm == 1000  # never raised
    RateLimiter(0).rate_limited(RateLimitError("x", output_limit=1000))  # no-op


@pytest.mark.parametrize("otpm, expected", [(None, 10), (1000, 5), (300, 1), (100, 1), (10_000, 10)])
def test_entry_batch_size_under_otpm(otpm, expected):
    entry = ChainEntry(fake("groq:q"), TokenRateLimiter(0, 8000, otpm=otpm), 6400)
    assert entry.batch_size(10) == expected


def test_groq_entry_gets_otpm_from_settings():
    settings = Settings(groq_api_key="gsk", groq_otpm=1000)
    entry = make_entry(make_client("groq", "qwen/qwen3.8-27b", settings), "groq", settings)
    assert entry.limiter.otpm == 1000
    assert entry.batch_size(10) == 5
    gemini = make_entry(fake("gemini:a"), "gemini", Settings(gemini_rpm=5))
    assert gemini.batch_size(10) == 10


def test_otpm_shrinks_batches_and_gemini_keeps_ten(session, make_item, run_chain):
    for _ in range(17):
        make_item()
    groq = fake("groq:q", echo(), QUOTA)
    gemini = fake("gemini:a", echo(), echo())
    clock = FakeClock()
    limiter = TokenRateLimiter(0, 100_000, clock=clock, sleep=clock.sleep, otpm=1000)
    chain = [ChainEntry(groq, limiter),
             ChainEntry(gemini, RateLimiter(0))]

    counts = run_chain(chain)

    assert len(groq.batch_ids(0)) == 5
    assert len(groq.batch_ids(1)) == 5  # quota: this batch moves to Gemini unchanged
    assert [len(gemini.batch_ids(i)) for i in range(2)] == [5, 7]
    assert counts["by_model"] == {"groq:q": 5, "gemini:a": 12}


def test_learned_otpm_shrinks_later_batches(session, make_item, run_chain, sleeps):
    for _ in range(20):
        make_item()
    otpm_429 = RateLimitError(OTPM_MESSAGE, None, per_minute=True, output_limit=1000)
    groq = fake("groq:q", otpm_429, echo(), echo(), echo())
    clock = FakeClock()
    chain = [ChainEntry(groq, TokenRateLimiter(0, 100_000, clock=clock, sleep=clock.sleep))]

    run_chain(chain)

    assert sleeps == [61]  # the 429 wait (the limiter's own waits use the fake clock)
    # First batch was sized before the limit was known; the next ones fit it.
    assert [len(groq.batch_ids(i)) for i in range(1, 4)] == [10, 5, 5]


def test_request_llm_reserves_expected_output():
    clock = FakeClock()
    limiter = TokenRateLimiter(0, 100_000, clock=clock, sleep=clock.sleep, otpm=1000)
    llm = fake("groq:q", echo())
    request_llm(llm, json.dumps([{"raw_item_id": i} for i in range(4)]), limiter, clock.sleep)
    assert limiter.output_used() == 600  # 4 x 150, no usage reported
