# STL Crime Tracker

FastAPI service for tracking St. Louis crime data. See `docs/design.md` for the design.

## Setup

Requires Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -e ".[dev]"
copy .env.example .env          # Windows (cp on macOS/Linux), then fill in values
```

`DATABASE_URL` defaults to a local SQLite file (`sqlite:///./stl_crime.db`).

## Run

```bash
uvicorn api.main:app --reload
```

Health check: http://127.0.0.1:8000/health

## Test

```bash
pytest                                   # all tests
pytest tests/test_health.py::test_health # single test
```

## Migrations

```bash
alembic upgrade head                          # apply migrations
alembic revision --autogenerate -m "message"  # create a migration from model changes
```
