"""scripts/copy_sqlite_to_postgres.py, SQLite -> SQLite (the Postgres target is
covered in test_postgres.py)."""

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

from models import Incident, JobLock, Source
from scripts.copy_sqlite_to_postgres import CopyError, copy_database, main
from tests.dbhelpers import add_sample_data, run_alembic, snapshot


@pytest.fixture
def dbs(tmp_path, monkeypatch):
    """(source engine, target engine): both migrated to head, source with data."""
    urls = {}
    for name in ("source", "target"):
        urls[name] = f"sqlite:///{(tmp_path / f'{name}.db').as_posix()}"
        run_alembic(monkeypatch, urls[name])
    source, target = create_engine(urls["source"]), create_engine(urls["target"])
    with Session(source) as s:
        add_sample_data(s)
    yield source, target
    source.dispose()
    target.dispose()


def test_copies_every_table_with_ids(dbs):
    source, target = dbs
    counts = copy_database(source, target)

    assert counts == {
        "incidents": 2, "sources": 1, "raw_items": 3, "classifications": 3,
        "incident_items": 3, "pipeline_runs": 2, "alerts_sent": 1,
    }
    assert snapshot(target) == snapshot(source)


def test_merged_incident_pointing_at_later_id(dbs):
    source, target = dbs
    copy_database(source, target)
    with Session(target) as s:
        merged = s.scalars(select(Incident).where(Incident.status == "merged")).one()
        assert merged.merged_into_id > merged.id
        assert s.get(Incident, merged.merged_into_id).status == "confirmed"


def test_job_locks_not_copied(dbs):
    source, target = dbs
    copy_database(source, target)
    with Session(target) as s:
        assert s.scalar(select(func.count()).select_from(JobLock)) == 0


def test_new_rows_get_new_ids_after_copy(dbs):
    source, target = dbs
    copy_database(source, target)
    with Session(target) as s:
        s.add(Source(name="New", url="https://example.com/new", type="rss"))
        s.commit()
        assert s.scalar(select(func.max(Source.id))) == 2


def test_refuses_non_empty_target(dbs):
    source, target = dbs
    with Session(target) as s:
        s.add(Source(name="Already here", url="https://example.com/x", type="rss"))
        s.commit()
    before = snapshot(target)

    with pytest.raises(CopyError, match="already has rows in sources"):
        copy_database(source, target)
    assert snapshot(target) == before


def test_refuses_unmigrated_target(dbs, tmp_path):
    source, _ = dbs
    blank = create_engine(f"sqlite:///{(tmp_path / 'blank.db').as_posix()}")
    with pytest.raises(CopyError, match="target database is at revision none"):
        copy_database(source, blank)
    blank.dispose()


def test_refuses_source_behind_head(tmp_path, monkeypatch, dbs):
    _, target = dbs
    old_url = f"sqlite:///{(tmp_path / 'old.db').as_posix()}"
    run_alembic(monkeypatch, old_url, revision="c9a4e2d7f813")
    old = create_engine(old_url)
    with pytest.raises(CopyError, match="source database is at revision c9a4e2d7f813"):
        copy_database(old, target)
    old.dispose()


def test_rolls_back_on_count_mismatch(dbs, monkeypatch):
    source, target = dbs
    import scripts.copy_sqlite_to_postgres as script

    real = script._count

    def lying_count(conn, table):
        # The source claims one more raw_item than it really copied.
        n = real(conn, table)
        return n + 1 if conn.engine is source and table.name == "raw_items" else n

    monkeypatch.setattr(script, "_count", lying_count)
    with pytest.raises(CopyError, match="raw_items: source has 4 rows, target 3"):
        copy_database(source, target)
    monkeypatch.setattr(script, "_count", real)
    assert all(rows == [] for rows in snapshot(target).values())


def test_cli(dbs, capsys):
    source, target = dbs
    code = main(["--from", source.url.database, "--to", str(target.url)])
    out = capsys.readouterr().out
    assert code == 0
    assert "raw_items" in out and "Done: 15 rows copied" in out
    assert snapshot(target) == snapshot(source)


def test_cli_missing_source(tmp_path, capsys):
    assert main(["--from", str(tmp_path / "nope.db"), "--to", "sqlite://"]) == 1
    assert "not found" in capsys.readouterr().err


def test_cli_same_database(dbs, capsys):
    source, _ = dbs
    assert main(["--from", source.url.database, "--to", str(source.url)]) == 1
    assert "same database" in capsys.readouterr().err


def test_cli_reports_refusal(dbs, capsys):
    source, target = dbs
    main(["--from", source.url.database, "--to", str(target.url)])
    assert main(["--from", source.url.database, "--to", str(target.url)]) == 1
    assert "Nothing copied: target already has rows" in capsys.readouterr().err
