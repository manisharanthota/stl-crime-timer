"""One-time copy of the local SQLite database into an empty Postgres database.

    python scripts/copy_sqlite_to_postgres.py                     # stl_crime.db -> DATABASE_URL
    python scripts/copy_sqlite_to_postgres.py --from other.db --to postgresql://...

Both databases must be at the same (latest) alembic revision, and the target's tables
must be empty: run `alembic upgrade head` against the target first and don't start the
scheduled pipeline until the copy is done. Everything is copied in one transaction
(ids kept, Postgres id sequences moved past the copied rows), so a failure leaves the
target empty. job_locks is not copied: a lock only matters while a run is going.
"""

import argparse
import sys
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import Connection, Engine, Table, func, inspect, make_url, select, text, update

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # run as a file: make the app modules importable
    sys.path.insert(0, str(ROOT))

import models  # noqa: E402,F401  (registers tables on Base.metadata)
from config import get_settings, normalize_database_url  # noqa: E402
from db import Base, make_engine  # noqa: E402

SKIP_TABLES = {"job_locks"}
BATCH_SIZE = 500


class CopyError(Exception):
    """The copy can't (or didn't) go through; nothing was written to the target."""


def head_revision() -> str:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "alembic"))
    return ScriptDirectory.from_config(cfg).get_current_head()


def _revision(conn: Connection) -> str | None:
    if not inspect(conn).has_table("alembic_version"):
        return None
    return conn.execute(text("SELECT version_num FROM alembic_version")).scalar()


def tables() -> list[Table]:
    """App tables in foreign-key order (parents first)."""
    return [t for t in Base.metadata.sorted_tables if t.name not in SKIP_TABLES]


def _self_refs(table: Table) -> list[str]:
    """Columns referencing their own table (incidents.merged_into_id). They're filled
    in after all rows exist, since a row can point at a later id."""
    return [
        fk.parent.name for fk in table.foreign_keys if fk.column.table is table
    ]


def _count(conn: Connection, table: Table) -> int:
    return conn.execute(select(func.count()).select_from(table)).scalar_one()


def _check_ready(src: Connection, dst: Connection, head: str) -> None:
    for name, conn in (("source", src), ("target", dst)):
        rev = _revision(conn)
        if rev != head:
            raise CopyError(
                f"{name} database is at revision {rev or 'none'}, expected {head}: "
                f"run `alembic upgrade head` against it first"
            )
    non_empty = [t.name for t in tables() if _count(dst, t)]
    if non_empty:
        raise CopyError(
            f"target already has rows in {', '.join(non_empty)}; "
            "this script only copies into an empty database"
        )


def _copy_table(src: Connection, dst: Connection, table: Table) -> int:
    self_refs = _self_refs(table)
    pk = list(table.primary_key.columns)
    rows = src.execute(select(table).order_by(*pk)).mappings()
    deferred: list[dict] = []
    copied = 0
    batch: list[dict] = []
    for row in rows:
        values = dict(row)
        if self_refs and any(values[c] is not None for c in self_refs):
            deferred.append({c.name: values[c.name] for c in pk} | {
                c: values[c] for c in self_refs
            })
            values.update({c: None for c in self_refs})
        batch.append(values)
        if len(batch) >= BATCH_SIZE:
            dst.execute(table.insert(), batch)
            copied += len(batch)
            batch = []
    if batch:
        dst.execute(table.insert(), batch)
        copied += len(batch)
    for values in deferred:
        where = [c == values[c.name] for c in pk]
        dst.execute(
            update(table).where(*where).values({c: values[c] for c in self_refs})
        )
    return copied


def _reset_sequences(dst: Connection) -> None:
    """Rows were inserted with explicit ids, so Postgres' id sequences still start at
    1; move each one past the highest copied id."""
    if dst.dialect.name != "postgresql":
        return
    for table in tables():
        if "id" not in table.c or not table.c.id.primary_key:
            continue
        dst.execute(
            text(
                "SELECT setval(pg_get_serial_sequence(:t, 'id'), "
                f"COALESCE(MAX(id), 1), MAX(id) IS NOT NULL) FROM {table.name}"
            ),
            {"t": table.name},
        )


def copy_database(source: Engine, target: Engine) -> dict[str, int]:
    """Copy every app table from source to target; returns rows copied per table.
    Raises CopyError (target untouched) if the databases aren't ready or the row
    counts don't match afterwards."""
    head = head_revision()
    counts: dict[str, int] = {}
    with source.connect() as src, target.begin() as dst:
        _check_ready(src, dst, head)
        for table in tables():
            counts[table.name] = _copy_table(src, dst, table)
        _reset_sequences(dst)
        for table in tables():
            expected, got = _count(src, table), _count(dst, table)
            if expected != got:
                raise CopyError(f"{table.name}: source has {expected} rows, target {got}")
    return counts


def _as_url(value: str) -> str:
    """Accept a plain file path for SQLite."""
    if "://" in value:
        return normalize_database_url(value)
    return f"sqlite:///{Path(value).resolve().as_posix()}"


def _sqlite_path(url: str) -> Path | None:
    parsed = make_url(url)
    if parsed.get_backend_name() != "sqlite" or not parsed.database:
        return None
    return Path(parsed.database).resolve()


def _same_database(a: str, b: str) -> bool:
    path_a, path_b = _sqlite_path(a), _sqlite_path(b)
    if path_a or path_b:
        return path_a == path_b
    return make_url(a) == make_url(b)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--from", dest="source", default="stl_crime.db",
        help="SQLite file or URL to copy from (default: stl_crime.db)",
    )
    parser.add_argument(
        "--to", dest="target", default=None,
        help="database URL to copy into (default: DATABASE_URL)",
    )
    args = parser.parse_args(argv)
    source_url = _as_url(args.source)
    target_url = _as_url(args.target) if args.target else get_settings().database_url
    source_path = _sqlite_path(source_url)
    if source_path is not None and not source_path.exists():
        print(f"Source database not found: {args.source}", file=sys.stderr)
        return 1
    if _same_database(source_url, target_url):
        print("Source and target are the same database.", file=sys.stderr)
        return 1

    source, target = make_engine(source_url), make_engine(target_url)
    try:
        print(f"Copying {source.url} -> {target.url} ...")  # passwords print as ***
        counts = copy_database(source, target)
    except CopyError as exc:
        print(f"Nothing copied: {exc}", file=sys.stderr)
        return 1
    finally:
        source.dispose()
        target.dispose()
    width = max(map(len, counts))
    for name, n in counts.items():
        print(f"  {name:<{width}}  {n:>7}")
    print(f"Done: {sum(counts.values())} rows copied, counts verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
