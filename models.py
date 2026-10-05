"""SQLAlchemy models for sources, raw items, classifications, incidents, and jobs."""

import hashlib
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from db import Base, UTCDateTime

RAW_ITEM_STATUSES = ("new", "classified", "failed")
CRIME_TYPES = ("shooting", "burglary", "homicide")
# merged: folded into another incident (merged_into_id); never counted or matched.
INCIDENT_STATUSES = ("confirmed", "review", "rejected", "merged")
PIPELINE_RUN_STATUSES = ("running", "success", "partial", "failed")

# Stored as VARCHAR + CHECK so the schema is portable between SQLite and Postgres.
_enum_opts = {"native_enum": False, "create_constraint": True}
RawItemStatus = Enum(*RAW_ITEM_STATUSES, name="raw_item_status", **_enum_opts)
CrimeType = Enum(*CRIME_TYPES, name="crime_type", **_enum_opts)
IncidentStatus = Enum(*INCIDENT_STATUSES, name="incident_status", **_enum_opts)
PipelineRunStatus = Enum(*PIPELINE_RUN_STATUSES, name="pipeline_run_status", **_enum_opts)


def hash_url(url: str) -> str:
    """Stable sha256 hex digest of a URL, used for raw_items dedup."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


class Source(Base):
    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    url: Mapped[str] = mapped_column(String(2048), unique=True)
    type: Mapped[str] = mapped_column(String(50))
    active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")
    last_success_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    fail_count: Mapped[int] = mapped_column(Integer, default=0, server_default="0")

    raw_items: Mapped[list["RawItem"]] = relationship(back_populates="source")


class RawItem(Base):
    __tablename__ = "raw_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    source_id: Mapped[int] = mapped_column(ForeignKey("sources.id"))
    url: Mapped[str] = mapped_column(String(2048))
    url_hash: Mapped[str] = mapped_column(String(64), unique=True)
    title: Mapped[str] = mapped_column(String(1000))
    body: Mapped[str | None] = mapped_column(Text)
    published_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    status: Mapped[str] = mapped_column(
        RawItemStatus, default="new", server_default="new", index=True
    )
    retries: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, server_default=func.now()
    )

    source: Mapped[Source] = relationship(back_populates="raw_items")


class Classification(Base):
    __tablename__ = "classifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    raw_item_id: Mapped[int] = mapped_column(ForeignKey("raw_items.id"))
    is_crime: Mapped[bool] = mapped_column(Boolean)
    crime_type: Mapped[str | None] = mapped_column(CrimeType)
    # Someone was shot (always true for crime_type=shooting; may be true for homicide).
    was_shooting: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0"
    )
    in_stl: Mapped[bool] = mapped_column(Boolean)
    occurred_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    location: Mapped[str | None] = mapped_column(String(500))
    neighborhood: Mapped[str | None] = mapped_column(String(100))
    is_followup: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0"
    )
    confidence: Mapped[float] = mapped_column(Float)
    model: Mapped[str] = mapped_column(String(100))
    prompt_version: Mapped[str] = mapped_column(String(50))
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, server_default=func.now()
    )


class Incident(Base):
    __tablename__ = "incidents"

    id: Mapped[int] = mapped_column(primary_key=True)
    crime_type: Mapped[str] = mapped_column(CrimeType)
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    location: Mapped[str | None] = mapped_column(String(500))
    neighborhood: Mapped[str | None] = mapped_column(String(100))
    # True while every linked item lacked occurred_at (time is a published_at fallback).
    time_estimated: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0"
    )
    # True if any linked item was a shooting; survives a shooting -> homicide upgrade.
    was_shooting: Mapped[bool] = mapped_column(
        Boolean, default=False, server_default="0"
    )
    status: Mapped[str] = mapped_column(
        IncidentStatus, default="review", server_default="review"
    )
    # Set when status=merged: the incident this one was folded into.
    merged_into_id: Mapped[int | None] = mapped_column(ForeignKey("incidents.id"))
    created_at: Mapped[datetime] = mapped_column(
        UTCDateTime, server_default=func.now()
    )

    raw_items: Mapped[list[RawItem]] = relationship(secondary="incident_items")


class IncidentItem(Base):
    __tablename__ = "incident_items"

    incident_id: Mapped[int] = mapped_column(
        ForeignKey("incidents.id"), primary_key=True
    )
    raw_item_id: Mapped[int] = mapped_column(
        ForeignKey("raw_items.id"), primary_key=True
    )


class PipelineRun(Base):
    """One fetch -> classify -> match run; the pipeline heartbeat."""

    __tablename__ = "pipeline_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime)
    status: Mapped[str] = mapped_column(
        PipelineRunStatus, default="running", server_default="running"
    )
    fetch_counts: Mapped[dict | None] = mapped_column(JSON)
    classify_counts: Mapped[dict | None] = mapped_column(JSON)
    match_counts: Mapped[dict | None] = mapped_column(JSON)
    fetch_error: Mapped[str | None] = mapped_column(Text)
    classify_error: Mapped[str | None] = mapped_column(Text)
    match_error: Mapped[str | None] = mapped_column(Text)


class JobLock(Base):
    """A named lock row; stale once expires_at has passed."""

    __tablename__ = "job_locks"

    name: Mapped[str] = mapped_column(String(100), primary_key=True)
    owner: Mapped[str] = mapped_column(String(64))
    acquired_at: Mapped[datetime] = mapped_column(UTCDateTime)
    expires_at: Mapped[datetime] = mapped_column(UTCDateTime)
