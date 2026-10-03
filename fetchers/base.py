"""Fetcher interface and shared types."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import httpx

from models import Source

REQUEST_TIMEOUT = 15.0
USER_AGENT = "stl-crime-tracker/0.1 (+RSS fetcher)"


class FetchError(Exception):
    """A source responded but its content couldn't be used."""


@dataclass
class FetchedItem:
    url: str
    title: str
    body: str | None = None
    published_at: datetime | None = None


class BaseFetcher(ABC):
    def __init__(self, client: httpx.Client):
        self.client = client

    @abstractmethod
    def fetch(self, source: Source) -> list[FetchedItem]:
        """Return the items currently published by `source`. Raises on failure."""
