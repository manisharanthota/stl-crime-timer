"""Pydantic model for the classifier's JSON response."""

from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, field_validator, model_validator

LOCAL_TZ = ZoneInfo("America/Chicago")


class ClassifierOutput(BaseModel):
    is_crime: bool
    crime_type: Literal["shooting", "burglary", "homicide"] | None = None
    in_stl: bool
    occurred_at: datetime | None = None
    location: str | None = None
    confidence: float = Field(ge=0, le=1)

    @field_validator("occurred_at")
    @classmethod
    def _assume_local_time(cls, value: datetime | None) -> datetime | None:
        # Times without an offset are St. Louis local time.
        if value is not None and value.tzinfo is None:
            return value.replace(tzinfo=LOCAL_TZ)
        return value

    @model_validator(mode="after")
    def _crime_type_matches_is_crime(self) -> "ClassifierOutput":
        if not self.is_crime:
            self.crime_type = None
        elif self.crime_type is None:
            raise ValueError("crime_type is required when is_crime is true")
        return self


class BatchResult(ClassifierOutput):
    """One element of a batch response; raw_item_id ties it back to the input item."""

    raw_item_id: int
