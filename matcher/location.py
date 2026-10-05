"""Location normalization and fuzzy comparison for incident matching."""

import re

from rapidfuzz import fuzz

_CITY_NAMES = re.compile(r"\b(?:city of\s+)?(?:st\.?|saint)\s+louis\b")
_PUNCTUATION = re.compile(r"[^\w\s]")
_WHITESPACE = re.compile(r"\s+")


def normalize_location(location: str | None) -> str:
    """Lowercase, drop "St. Louis" variants and punctuation, collapse whitespace."""
    if not location:
        return ""
    text = _CITY_NAMES.sub(" ", location.lower())
    text = _PUNCTUATION.sub(" ", text)
    return _WHITESPACE.sub(" ", text).strip()


def location_similarity(a: str | None, b: str | None) -> float:
    """0-100 similarity of two locations; 0 if either is empty after normalizing."""
    a, b = normalize_location(a), normalize_location(b)
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b)
