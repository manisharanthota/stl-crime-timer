"""The deployment files say what docs/deploy.md promises."""

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load(path: str) -> dict:
    # PyYAML reads the workflow key `on:` as True.
    data = yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))
    if True in data:
        data["on"] = data.pop(True)
    return data


def steps_run(job: dict) -> list[str]:
    return [s["run"] for s in job["steps"] if "run" in s]


def test_pipeline_workflow():
    wf = load(".github/workflows/pipeline.yml")
    # Only cron-job.org (workflow_dispatch) starts runs; no GitHub schedule.
    assert wf["on"] == {"workflow_dispatch": {}}
    assert wf["concurrency"] == {"group": "pipeline", "cancel-in-progress": False}

    job = wf["jobs"]["run"]
    uses = [s.get("uses", "") for s in job["steps"]]
    assert any(u.startswith("actions/checkout@") for u in uses)
    setup = next(s for s in job["steps"] if s.get("uses", "").startswith("actions/setup-python@"))
    assert setup["with"]["cache"] == "pip"

    runs = steps_run(job)
    order = [next(i for i, r in enumerate(runs) if cmd in r)
             for cmd in ("pip install", "alembic upgrade head", "python -m jobs once")]
    assert order == sorted(order)

    # Every value comes from secrets, except the logging switch.
    env = job["env"]
    assert env.pop("LOG_TO_FILE") == "false"
    for key, value in env.items():
        assert value == f"${{{{ secrets.{key} }}}}", key
    assert {"DATABASE_URL", "GEMINI_API_KEY"} <= set(env)


def test_tests_workflow_runs_on_every_push():
    wf = load(".github/workflows/tests.yml")
    assert "push" in wf["on"] and wf["on"]["push"] is None  # no branch filter
    job = wf["jobs"]["pytest"]
    assert any(r.strip().startswith("pytest") for r in steps_run(job))
    assert "DATABASE_URL_TEST" in job["env"]
    assert "secrets." not in (ROOT / ".github/workflows/tests.yml").read_text()


def test_render_blueprint():
    (svc,) = load("render.yaml")["services"]
    assert svc["type"] == "web" and svc["plan"] == "free"
    assert re.search(r"uvicorn api\.main:app .*--port \$PORT", svc["startCommand"])
    assert "--host 0.0.0.0" in svc["startCommand"]
    assert "python -m pip install ." in svc["buildCommand"]
    assert "poetry" not in svc["buildCommand"].lower()
    assert svc["healthCheckPath"] != "/health"  # /health is 503 when the pipeline is stale
    for var in svc["envVars"]:
        assert var == {"key": var["key"], "sync": False}  # values only in the dashboard
    assert {v["key"] for v in svc["envVars"]} >= {"DATABASE_URL", "ADMIN_TOKEN"}


def test_python_version_pinned_once():
    assert (ROOT / ".python-version").read_text().strip() == "3.13"


def test_no_poetry_config():
    """Render switches to Poetry when it finds Poetry config; keep the repo pip-only."""
    assert not (ROOT / "poetry.lock").exists()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[tool.poetry" not in pyproject
    assert 'build-backend = "setuptools.build_meta"' in pyproject
