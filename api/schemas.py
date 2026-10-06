"""Pydantic response models. Datetimes are aware UTC and serialize with a trailing Z."""

from datetime import datetime, timezone
from typing import Annotated

from pydantic import BaseModel, PlainSerializer


def _iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


UTC = Annotated[datetime, PlainSerializer(_iso_z, return_type=str)]


class LastRun(BaseModel):
    started_at: UTC
    finished_at: UTC | None
    status: str


class HealthResponse(BaseModel):
    status: str  # "ok", or "stale" (served with HTTP 503)
    last_run: LastRun | None
    last_success_at: UTC | None


class LastIncident(BaseModel):
    incident_id: int
    crime_type: str
    occurred_at: UTC
    seconds_since: int
    location: str | None
    time_estimated: bool
    was_shooting: bool


class TimerEntry(BaseModel):
    crime_type: str | None  # None = overall (any type)
    last: LastIncident | None  # None = no confirmed incident yet


class TimerResponse(BaseModel):
    now: UTC
    overall: TimerEntry
    by_type: list[TimerEntry]


class ArticleOut(BaseModel):
    title: str
    url: str
    source_name: str
    published_at: UTC | None


class IncidentOut(BaseModel):
    id: int
    crime_type: str
    occurred_at: UTC
    time_estimated: bool
    was_shooting: bool
    location: str | None
    neighborhood: str | None
    status: str
    articles: list[ArticleOut]


class Gap(BaseModel):
    seconds: int
    start_incident_id: int
    end_incident_id: int | None  # None when ongoing
    start: UTC
    end: UTC  # "now" when ongoing
    ongoing: bool


class GapEntry(BaseModel):
    crime_type: str | None  # None = overall
    longest: Gap | None  # None = no confirmed incident yet


class StatsResponse(BaseModel):
    now: UTC
    overall: GapEntry
    by_type: list[GapEntry]
