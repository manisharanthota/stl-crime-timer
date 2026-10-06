import pytest

from config import get_settings, normalize_database_url


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("postgresql://u:p@db.example.com:5432/postgres",
         "postgresql+psycopg://u:p@db.example.com:5432/postgres"),
        ("postgres://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        ("postgresql+psycopg://u:p@host/db", "postgresql+psycopg://u:p@host/db"),
        ("sqlite:///./stl_crime.db", "sqlite:///./stl_crime.db"),
    ],
)
def test_normalize_database_url(url, expected):
    assert normalize_database_url(url) == expected


@pytest.fixture
def env(monkeypatch):
    """Settings from the process environment only (no .env)."""
    monkeypatch.setattr("config.load_dotenv", lambda: None)
    get_settings.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()


def test_settings_normalize_database_url(env):
    env.setenv("DATABASE_URL", "postgresql://u:p@host:6543/postgres")
    assert get_settings().database_url == "postgresql+psycopg://u:p@host:6543/postgres"


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, True), ("", True), ("true", True), ("1", True),
     ("false", False), ("0", False), ("no", False), ("off", False)],
)
def test_log_to_file(env, value, expected):
    if value is None:
        env.delenv("LOG_TO_FILE", raising=False)
    else:
        env.setenv("LOG_TO_FILE", value)
    assert get_settings().log_to_file is expected
