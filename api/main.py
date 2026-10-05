from collections.abc import Iterator

from fastapi import Depends, FastAPI
from sqlalchemy import select
from sqlalchemy.orm import Session

from models import PipelineRun

app = FastAPI(title="STL Crime Tracker")


def get_session() -> Iterator[Session]:
    from db import SessionLocal

    with SessionLocal() as session:
        yield session


def _iso(value) -> str | None:
    return value.isoformat().replace("+00:00", "Z") if value else None


@app.get("/health")
def health(session: Session = Depends(get_session)) -> dict:
    run = session.scalars(
        select(PipelineRun).order_by(PipelineRun.started_at.desc(), PipelineRun.id.desc())
    ).first()
    last_run = None
    if run is not None:
        last_run = {
            "started_at": _iso(run.started_at),
            "finished_at": _iso(run.finished_at),
            "status": run.status,
        }
    return {"status": "ok", "last_run": last_run}
