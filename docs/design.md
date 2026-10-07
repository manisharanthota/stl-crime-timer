# STL Crime Tracker — System Design

Goal: track how long St. Louis goes without a shooting, burglary, or killing.

## Key Decision
Don't reset a timer. Store incidents and compute `now - max(occurred_at)` of confirmed incidents on read.

## Pipeline
Scheduler (every 10 min) → Fetchers (RSS / RSS.app for FB) → raw_items (dedup by url_hash) → keyword prefilter → LLM classifier (JSON) → incident matcher → incidents → API /timer → web page

## Sources
All RSS, listed in `sources.yaml` (loaded by `seed.py`); items older than 7 days are skipped (`MAX_ITEM_AGE_DAYS`).
- KSDK 5 (news)
- Fox2 (news)
- KMOV 4 (news)
- St. Louis Post-Dispatch, crime & courts section (news)
- St. Louis Public Radio (news)
- SLMPD, St. Louis Metropolitan Police Department news releases (police)

## Stack
Python, FastAPI, SQLAlchemy, Alembic, SQLite (Postgres later), Claude API, APScheduler

## Data Model
- sources: id, name, url, type, active, last_success_at, fail_count
- raw_items: id, source_id, url, url_hash (unique), title, body, published_at, status (new/classified/failed), retries
- classifications: raw_item_id, is_crime, crime_type (shooting/burglary/homicide), in_stl, occurred_at, location, confidence, model, prompt_version
- incidents: id, crime_type, occurred_at, location, status (confirmed/review/rejected)
- incident_items: incident_id, raw_item_id

## Failure Handling
1. Source down → isolate per source, track fail_count, alert after 5 failures
2. Duplicate articles → unique url_hash + title similarity
3. Same crime, many sources → merge by type + ±6h + location
4. LLM errors → retry with backoff, item stays new
5. Bad LLM JSON → Pydantic validation, retry once, else failed
6. False positives (old crimes, court news, county) → extract occurred_at and in_stl; confidence < 0.8 → review
7. Only confirmed incidents affect the timer
8. Scheduler overlap → lock
9. Cost → keyword prefilter before LLM
10. Silent failure → heartbeat, alert if no run in 30 min

## Chunks
1. Skeleton
2. DB schema + seed sources
3. Fetchers
4. Prefilter + classifier
5. Incident matcher
6. Job runner
7. API + page
8. Alerts
9. Deploy
