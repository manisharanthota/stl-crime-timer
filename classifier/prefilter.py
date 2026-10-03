"""Cheap keyword gate so only plausibly crime-related items reach the LLM."""

import re

from models import RawItem

# Word boundaries keep "dead" from matching "deadline" and "shot" from "screenshot".
CRIME_KEYWORDS = re.compile(
    r"\b(?:"
    r"shots?|shootings?|shooters?|gunfire"
    r"|burglary|burglaries|burglars?|break-ins?"
    r"|killed|homicides?|murder(?:s|ed)?|stabbed|dead"
    r")\b",
    re.IGNORECASE,
)


def prefilter(item: RawItem) -> bool:
    """True if the item's title or body mentions a crime keyword."""
    return bool(CRIME_KEYWORDS.search(f"{item.title or ''} {item.body or ''}"))
