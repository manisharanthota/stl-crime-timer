"""Pydantic model for the classifier's JSON response."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from timeutil import to_utc


class ClassifierOutput(BaseModel):
    is_crime: bool
    crime_type: Literal["shooting", "burglary", "homicide"] | None = None
    # Someone was shot; lets a fatal shooting (crime_type=homicide) count as a shooting too.
    was_shooting: bool = False
    in_stl: bool
    occurred_at: datetime | None = None
    location: str | None = None
    confidence: float = Field(ge=0, le=1)

    @field_validator("occurred_at")
    @classmethod
    def _to_utc(cls, value: datetime | None) -> datetime | None:
        # Times without an offset are St. Louis local time.
        return to_utc(value) if value is not None else None

    @model_validator(mode="after")
    def _crime_type_matches_is_crime(self) -> "ClassifierOutput":
        if not self.is_crime:
            self.crime_type = None
        elif self.crime_type is None:
            raise ValueError("crime_type is required when is_crime is true")
        if self.crime_type == "shooting":
            self.was_shooting = True
        elif self.crime_type != "homicide":
            self.was_shooting = False
        return self


class BatchResult(ClassifierOutput):
    """One element of a batch response; raw_item_id ties it back to the input item."""

    raw_item_id: int
