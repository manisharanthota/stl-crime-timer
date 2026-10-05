# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Rules

- Write tests for every module.
- Never call real external APIs in tests (Anthropic, crime data sources, etc.); mock them.
- Only work on the chunk the user asks for. The project is built one chunk at a time per `docs/design.md`; don't implement future chunks early.

## Commands

Use the virtualenv at `.venv` (`.venv\Scripts\activate` on Windows).

```bash
pip install -e ".[dev]"                  # install app + test deps
uvicorn api.main:app --reload            # run the API
pytest                                   # all tests
pytest tests/test_health.py::test_health # single test
alembic upgrade head                     # apply migrations
alembic revision --autogenerate -m "msg" # new migration
python seed.py                           # load sources.yaml into sources (skips known urls)
python -m fetchers                       # fetch all active sources once into raw_items
python -m classifier                     # classify all status=new raw_items once (needs GEMINI_API_KEY)
python -m classifier.eval                # score the live LLM on tests/fixtures/eval_headlines.yaml (manual, not pytest; cached, --no-cache to bypass)
```

## Architecture

- `config.py` — `get_settings()` loads `.env` (python-dotenv) into a cached Pydantic `Settings` (`database_url`, `anthropic_api_key`, `llm_provider`, `gemini_api_key`, `gemini_model`, `gemini_rpm`). Read config through this, not `os.getenv`.
- `db.py` — SQLAlchemy 2.0 `engine`, `SessionLocal`, declarative `Base`, and `UTCDateTime`. All models subclass `Base`; every datetime column uses `UTCDateTime`.
- `timeutil.py` — all datetimes are stored and compared in UTC. `to_utc()` treats naive input as America/Chicago (DST-aware); `to_local()` converts to St. Louis time and is for display only (and the LLM prompt). `UTCDateTime` applies `to_utc` on write (naive UTC on SQLite, timestamptz on Postgres) and returns aware UTC on read.
- `models.py` — `Source`, `RawItem`, `Classification`, `Incident`, `IncidentItem`, plus `hash_url()` (sha256 used for `raw_items.url_hash`). Enum-like columns are non-native `Enum`s with CHECK constraints (portable SQLite → Postgres). `db.py` enables SQLite foreign keys on every connection.
- `seed.py` — `seed_sources(session, path)` inserts sources from `sources.yaml` whose url isn't already stored.
- `alembic/env.py` — takes the DB URL from `config.py` (the value in `alembic.ini` is ignored) and uses `Base.metadata`; model modules must be imported there for autogenerate to see them.
- `api/main.py` — FastAPI `app`.
- `fetchers/` — `BaseFetcher` + `RSSFetcher` (httpx fetch, 15s timeout + User-Agent, feedparser parse; covers news and RSS.app feeds). `runner.run_fetchers(session, client=None)` loops active sources, each isolated: success sets `last_success_at`/resets `fail_count`, failure rolls back, increments `fail_count`, logs. Items dedup on `hash_url(normalize_url(url))` (query/fragment/trailing slash stripped). Tests inject an `httpx.MockTransport` client and use XML fixtures in `tests/fixtures/feeds/`.
- `classifier/` — built for the Gemini free tier (5 RPM, 20/day). `prefilter(item)` (word-boundary keyword regex) runs first; misses are saved as `is_crime=False`, `model="prefilter"` with no LLM call. The rest go up to 10 per request (`classify_batch`): the prompt is a JSON array keyed by `raw_item_id`, the response is `list[BatchResult]`, and each element is validated separately. A bad or missing element gives that item `retries += 1` and leaves it `new`; once `retries` reaches 3 it's marked `failed`. A response that isn't a JSON array is retried once. `RateLimiter` spaces requests `60/GEMINI_RPM + 1`s apart. `request_llm` policy: 429 with `retryDelay` < 2 min → wait and retry; longer/missing → `QuotaExhausted` stops the run (items stay `new`, retries untouched); 503/other errors → backoff 5/15/45s, then the batch stays `new`. `LLMClient` ABC with `GeminiClient` maps SDK errors to `RateLimitError`/`ServiceUnavailableError`/`LLMError`. `prompt.py` holds `SYSTEM_PROMPT` + `PROMPT_VERSION` (bump on any wording change). `eval.py` caches results in `.eval_cache.json` (`--no-cache` to bypass). Tests inject a fake `LLMClient`, `RateLimiter(0)`, and a recording `sleep`.
- `matcher/`, `jobs/` — top-level packages, empty until their chunks are built.
