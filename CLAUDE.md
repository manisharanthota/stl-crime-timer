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
python -m matcher                        # link all unlinked in-STL crime classifications into incidents once
python -m jobs once                      # run fetch -> classify -> match once (live network + LLM)
python -m jobs schedule                  # run it every PIPELINE_INTERVAL_MINUTES; Ctrl+C stops gracefully
python -m classifier.eval                # score the live LLM on tests/fixtures/eval_headlines.yaml (manual, not pytest; cached, --no-cache to bypass)
```

## Architecture

- `config.py` — `get_settings()` loads `.env` (python-dotenv) into a cached Pydantic `Settings` (`database_url`, `anthropic_api_key`, `llm_provider`, `gemini_api_key`, `gemini_model`, `gemini_rpm`, `match_location_threshold`, `pipeline_interval_minutes`, `admin_token`). Read config through this, not `os.getenv`.
- `db.py` — SQLAlchemy 2.0 `engine`, `SessionLocal`, declarative `Base`, and `UTCDateTime`. All models subclass `Base`; every datetime column uses `UTCDateTime`.
- `timeutil.py` — all datetimes are stored and compared in UTC. `to_utc()` treats naive input as America/Chicago (DST-aware); `to_local()` converts to St. Louis time and is for display only (and the LLM prompt). `UTCDateTime` applies `to_utc` on write (naive UTC on SQLite, timestamptz on Postgres) and returns aware UTC on read.
- `models.py` — `Source`, `RawItem`, `Classification`, `Incident`, `IncidentItem`, `PipelineRun`, `JobLock`, plus `hash_url()` (sha256 used for `raw_items.url_hash`). Enum-like columns are non-native `Enum`s with CHECK constraints (portable SQLite → Postgres). `db.py` enables SQLite foreign keys on every connection.
- `seed.py` — `seed_sources(session, path)` inserts sources from `sources.yaml` whose url isn't already stored.
- `alembic/env.py` — takes the DB URL from `config.py` (the value in `alembic.ini` is ignored) and uses `Base.metadata`; model modules must be imported there for autogenerate to see them.
- `api/` — FastAPI `app` in `main.py`; response models in `schemas.py` (datetimes serialize as UTC `...Z`); read queries in `queries.py`. Everything is computed per request from `status=confirmed` incidents (no caching), so confirm/reject shows up immediately. `/timer` = latest confirmed incident overall and per crime type (shooting = `crime_type=shooting OR was_shooting`, so a fatal shooting counts for both shooting and homicide; `/stats` uses the same rule) with `seconds_since` (clamped >= 0; `last: null` when none). `/incidents?limit=20` (1-100) = recent confirmed incidents with linked articles. `/stats` = longest gap between consecutive confirmed incidents overall and per type, counting the open gap up to now (`ongoing: true`). `/health` = latest `pipeline_runs` row. `/admin/*` needs header `X-Admin-Token` == `ADMIN_TOKEN` (401 missing/wrong, 503 if unset): `GET /admin/review`, `POST /admin/incidents/{id}/confirm|reject`. `/` serves `api/static/index.html` (single file, no build; times shown in America/Chicago, ticks every 1s, refetches every 60s). Dependencies `get_session`, `get_now`, `get_admin_token` are overridden in tests.
- `fetchers/` — `BaseFetcher` + `RSSFetcher` (httpx fetch, 15s timeout + User-Agent, feedparser parse; covers news and RSS.app feeds). `runner.run_fetchers(session, client=None)` loops active sources, each isolated: success sets `last_success_at`/resets `fail_count`, failure rolls back, increments `fail_count`, logs. Items dedup on `hash_url(normalize_url(url))` (query/fragment/trailing slash stripped). Tests inject an `httpx.MockTransport` client and use XML fixtures in `tests/fixtures/feeds/`.
- `classifier/` — built for the Gemini free tier (5 RPM, 20/day). `prefilter(item)` (word-boundary keyword regex) runs first; misses are saved as `is_crime=False`, `model="prefilter"` with no LLM call. The rest go up to 10 per request (`classify_batch`): the prompt is a JSON array keyed by `raw_item_id`, the response is `list[BatchResult]`, and each element is validated separately. A bad or missing element gives that item `retries += 1` and leaves it `new`; once `retries` reaches 3 it's marked `failed`. A response that isn't a JSON array is retried once. `RateLimiter` spaces requests `60/GEMINI_RPM + 1`s apart. `request_llm` policy: 429 with `retryDelay` < 2 min → wait and retry; longer/missing → `QuotaExhausted` stops the run (items stay `new`, retries untouched); 503/other errors → backoff 5/15/45s, then the batch stays `new`. `LLMClient` ABC with `GeminiClient` maps SDK errors to `RateLimitError`/`ServiceUnavailableError`/`LLMError`. The output has `was_shooting` (a fatal shooting is `crime_type=homicide, was_shooting=true`; the schema forces it true for shootings and false for burglaries/non-crime). `prompt.py` holds `SYSTEM_PROMPT` + `PROMPT_VERSION` (bump on any wording change). `eval.py` caches results in `.eval_cache.json` (`--no-cache` to bypass). Tests inject a fake `LLMClient`, `RateLimiter(0)`, and a recording `sleep`.
- `matcher/` — `match_pending(session, threshold=None)` takes the newest classification of each raw_item not in `incident_items` with `is_crime AND in_stl` (others ignored). Missing `occurred_at` falls back to `published_at` and sets `incidents.time_estimated`; items are processed oldest first. Merge needs compatible type (same, or shooting/homicide → upgraded to homicide), `occurred_at` within ±6h, and `location_similarity` (normalized, rapidfuzz `token_set_ratio`; null location never matches) ≥ `MATCH_LOCATION_THRESHOLD` (default 85). `incidents.was_shooting` is true if any linked item has `was_shooting` or `crime_type=shooting`, and stays true through the shooting → homicide upgrade. Merges keep the earliest time (a reported time beats an estimated one) and confirm on any item with confidence ≥ 0.8, never downgrade. Rejected incidents are never modified: an item matching one (and no live incident) stays unlinked. Idempotent because only unlinked items are selected.
- `jobs/` — `run_pipeline(session_factory=None, steps=None, stop_event=None)` runs `STEPS` (fetch → classify → match), each in its own session and try/except: a failing step is rolled back, its error recorded, and later steps still run. Every run that gets the lock writes a `pipeline_runs` row (`running` → `success`/`partial`/`failed`, per-step JSON counts and error text); a run that can't get the lock returns None and writes nothing. A classifier quota stop is not an error (it shows as `stopped_quota` in the counts). `lock.py` is a `job_locks` row with a 30-minute expiry, refreshed between steps; an expired lock is taken over with a conditional UPDATE, and `release` only deletes the caller's own lock. `scheduler.py` runs work off the main thread: APScheduler `BackgroundScheduler` (`max_instances=1`, `coalesce=True`, first run immediately). The first Ctrl+C sets `stop_event` (the current step finishes, the rest are recorded `skipped: shutdown`) and a second one force-exits. `logsetup.setup_logging()` (CLI only) logs to the console and `logs/pipeline.log` (rotating), with UTC timestamps. Tests pass fake steps and a `StaticPool` session factory (`session_factory` fixture).
