"""Guard against drift between the Alembic chain and the runtime DDL.

``DatabaseManager._init_schema`` creates the schema from ``storage/schema.py``
so the app can boot without running migrations, while Alembic owns the
migration history. Two creators of the same tables will drift unless something
fails when they do — this is that something.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config

from agent_bargain.config import PROJECT_ROOT
from agent_bargain.models.negotiation import NegotiationRecord
from agent_bargain.storage.database import DatabaseManager


def _schema_of(db_path: Path) -> dict[str, set[tuple]]:
    """Return ``{table: {(name, type, notnull, default), ...}}`` for a database.

    Order-insensitive on purpose: Alembic reaches the final shape through
    ``ALTER TABLE ... ADD COLUMN`` (which appends), while the runtime DDL
    declares columns in their logical order. Only the resulting column set
    should have to match.
    """
    conn = sqlite3.connect(db_path)
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%' AND name != 'alembic_version'"
            )
        ]
        # PRAGMA table_info -> (cid, name, type, notnull, dflt_value, pk);
        # ``pk`` is dropped because column position is not part of the contract.
        return {
            table: {row[1:5] for row in conn.execute(f"PRAGMA table_info({table})")}
            for table in tables
        }
    finally:
        conn.close()


def _build_with_alembic(db_path: Path) -> None:
    """Create a database by running the full migration chain against it."""
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "alembic"))
    command.upgrade(cfg, "head")
    assert db_path.exists(), "alembic upgrade head did not create the database"


def _indexes_of(db_path: Path) -> dict[str, str]:
    """Return ``{index_name: sql}`` for explicitly created indexes.

    SQLite's implicit ``sqlite_autoindex_*`` entries back PRIMARY KEY and UNIQUE
    constraints; they are derived from the table DDL (already compared above)
    and their names are not stable, so they are excluded.
    """
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT name, sql FROM sqlite_master WHERE type='index' "
            "AND name NOT LIKE 'sqlite_autoindex_%'"
        )
        return {name: " ".join((sql or "").split()) for name, sql in rows}
    finally:
        conn.close()


async def test_alembic_and_runtime_schemas_match(isolated_settings, tmp_path):
    """A migrated database and an app-initialized database must be identical."""
    # ``alembic/env.py`` derives its URL from settings, which the fixture
    # already points at tmp_path/test.db.
    alembic_db = Path(isolated_settings.sqlite_db_full_path)
    _build_with_alembic(alembic_db)

    runtime_db = tmp_path / "runtime.db"
    db = DatabaseManager(db_path=runtime_db)
    await db.connect()
    await db.close()

    migrated = _schema_of(alembic_db)
    runtime = _schema_of(runtime_db)

    assert migrated.keys() == runtime.keys(), (
        f"table sets differ: alembic={sorted(migrated)} runtime={sorted(runtime)}"
    )

    diffs = []
    for table in sorted(migrated):
        only_migrated = migrated[table] - runtime[table]
        only_runtime = runtime[table] - migrated[table]
        if only_migrated or only_runtime:
            diffs.append(
                f"{table}: alembic-only={sorted(only_migrated)} runtime-only={sorted(only_runtime)}"
            )

    assert not diffs, "Schema drift between Alembic and storage/schema.py:\n" + "\n".join(diffs)

    # Indexes are part of the schema contract too: they are what keeps the hot
    # queries off a full table scan, and only one of the two paths creating them
    # would silently leave half of all deployments unindexed.
    migrated_indexes = _indexes_of(alembic_db)
    runtime_indexes = _indexes_of(runtime_db)

    assert migrated_indexes == runtime_indexes, (
        "Index drift between Alembic and storage/schema.py:\n"
        f"  alembic-only: {sorted(set(migrated_indexes) - set(runtime_indexes))}\n"
        f"  runtime-only: {sorted(set(runtime_indexes) - set(migrated_indexes))}"
    )
    assert migrated_indexes, "expected the hot-path indexes to exist at all"


def test_max_rounds_column_is_persisted(isolated_settings):
    """``negotiations.max_rounds`` must exist so analytics need not guess the limit."""
    db_path = Path(isolated_settings.sqlite_db_full_path)
    _build_with_alembic(db_path)
    columns = {
        row[1] for row in sqlite3.connect(db_path).execute("PRAGMA table_info(negotiations)")
    }
    assert "max_rounds" in columns


async def test_foreign_keys_are_enforced(tmp_path):
    """A row referencing a missing parent must be rejected.

    SQLite ships with ``PRAGMA foreign_keys=OFF``, so without an explicit
    enable in ``connect()`` every declared FOREIGN KEY is decorative.
    """
    db = DatabaseManager(db_path=tmp_path / "fk.db")
    await db.connect()
    try:
        cursor = await db.conn.execute("PRAGMA foreign_keys")
        assert (await cursor.fetchone())[0] == 1

        with pytest.raises(sqlite3.IntegrityError):
            await db.save_negotiation(
                NegotiationRecord(
                    negotiation_id="neg_fk",
                    buyer_id="ghost_buyer",
                    seller_id="ghost_seller",
                    service_id="ghost_service",
                    status="active",
                    rounds=0,
                )
            )
    finally:
        await db.close()


async def test_stale_schema_fails_with_an_actionable_error(tmp_path):
    """An unmigrated database must not start up with a cryptic column error.

    ``CREATE TABLE IF NOT EXISTS`` leaves an existing table alone, so a database
    from before a migration is silently missing columns until some query hits
    one. The guard converts that into a single message naming the fix.
    """
    stale = tmp_path / "stale.db"
    conn = sqlite3.connect(stale)
    conn.executescript(
        """
        CREATE TABLE services (
            service_id TEXT PRIMARY KEY,
            seller_id  TEXT NOT NULL,
            title      TEXT NOT NULL
            -- deliberately missing description, category_*, base_price,
            -- floor_price and sla_terms
        );
        INSERT INTO services VALUES ('legacy', 's1', 'Old Listing');
        """
    )
    conn.commit()
    conn.close()

    db = DatabaseManager(db_path=stale)
    with pytest.raises(RuntimeError) as excinfo:
        await db.connect()

    message = str(excinfo.value)
    assert "alembic upgrade head" in message, message
    assert "services.floor_price" in message, message


async def test_schema_guard_accepts_both_creation_paths(isolated_settings, tmp_path):
    """The guard must not false-positive on a healthy database."""
    migrated = Path(isolated_settings.sqlite_db_full_path)
    _build_with_alembic(migrated)

    for path in (migrated, tmp_path / "runtime.db"):
        db = DatabaseManager(db_path=path)
        await db.connect()  # must not raise
        await db.close()
