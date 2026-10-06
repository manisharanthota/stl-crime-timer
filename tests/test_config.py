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


PAD = " \t{}\r\n"  # what a value pasted into a secret / dashboard can look like

# env var -> (value, Settings attribute, expected)
STRIPPED = {
    "DATABASE_URL": ("postgresql://u:p@host:5432/postgres", "database_url",
                     "postgresql+psycopg://u:p@host:5432/postgres"),
    "ANTHROPIC_API_KEY": ("sk-ant-x", "anthropic_api_key", "sk-ant-x"),
    "LLM_PROVIDER": ("Gemini", "llm_provider", "gemini"),
    "GEMINI_API_KEY": ("AIza-key", "gemini_api_key", "AIza-key"),
    "GEMINI_MODEL": ("gemini-x", "gemini_model", "gemini-x"),
    "GEMINI_FALLBACK_MODEL": ("gemini-y", "gemini_fallback_model", "gemini-y"),
    "GEMINI_RPM": ("7", "gemini_rpm", 7.0),
    "MATCH_LOCATION_THRESHOLD": ("90", "match_location_threshold", 90.0),
    "PIPELINE_INTERVAL_MINUTES": ("15", "pipeline_interval_minutes", 15.0),
    "ADMIN_TOKEN": ("tok3n", "admin_token", "tok3n"),
    "ALERT_WEBHOOK_URL": ("https://discord.com/api/webhooks/1/abc",
                          "alert_webhook_url", "https://discord.com/api/webhooks/1/abc"),
    "ALERT_ON_NEW_INCIDENT": ("1", "alert_on_new_incident", True),
    "ALERT_COOLDOWN_HOURS": ("3", "alert_cooldown_hours", 3.0),
    "LOG_TO_FILE": ("false", "log_to_file", False),
}


def test_stripped_settings_cover_every_field():
    from config import Settings

    assert {attr for _, attr, _ in STRIPPED.values()} == set(Settings.model_fields)


@pytest.mark.parametrize("name", STRIPPED)
def test_settings_strip_whitespace_and_newlines(env, name):
    value, attr, expected = STRIPPED[name]
    env.setenv(name, PAD.format(value))
    assert getattr(get_settings(), attr) == expected


@pytest.mark.parametrize("name", STRIPPED)
def test_whitespace_only_setting_counts_as_unset(env, name):
    from config import Settings

    _, attr, _ = STRIPPED[name]
    env.setenv(name, " \r\n\t ")
    assert getattr(get_settings(), attr) == Settings.model_fields[attr].default
