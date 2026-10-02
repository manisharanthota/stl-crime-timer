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
```

## Architecture

- `config.py` — `get_settings()` loads `.env` (python-dotenv) into a cached Pydantic `Settings` (`database_url`, `anthropic_api_key`). Read config through this, not `os.getenv`.
- `db.py` — SQLAlchemy 2.0 `engine`, `SessionLocal`, and declarative `Base`. All models subclass `Base`.
- `alembic/env.py` — takes the DB URL from `config.py` (the value in `alembic.ini` is ignored) and uses `Base.metadata`; model modules must be imported there for autogenerate to see them.
- `api/main.py` — FastAPI `app`.
- `fetchers/`, `classifier/`, `matcher/`, `jobs/` — top-level packages, empty until their chunks are built.
