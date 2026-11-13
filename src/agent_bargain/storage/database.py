"""Async SQLite database abstraction using aiosqlite."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import aiosqlite

from agent_bargain.config import settings
from agent_bargain.engine.reputation import NEUTRAL_REPUTATION_SCORE
from agent_bargain.models.negotiation import Agreement, NegotiationMessage, NegotiationRecord
from agent_bargain.models.reputation import FeedbackScore
from agent_bargain.models.service import ServiceCategory, ServiceListing
from agent_bargain.storage.schema import SCHEMA_SQL

logger = logging.getLogger(__name__)


def _expected_schema() -> dict[str, set[str]]:
    """``{table: {column, ...}}`` as declared by :data:`SCHEMA_SQL`.

    Derived by building a throwaway in-memory database rather than hardcoding a
    list, so it cannot drift from the DDL it is meant to check.
    """
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(SCHEMA_SQL)
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        return {
            table: {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            for table in tables
        }
    finally:
        conn.close()


def select_representative_feedbacks(feedbacks: list[FeedbackScore]) -> list[FeedbackScore]:
    """Pick up to 4 unique feedbacks that summarise an agent's history.

    Selection strategy:
    1. Highest scored feedback
    2. Median scored feedback (closest to the middle)
    3. Lowest scored feedback
    4. Most recent feedback

    Duplicates are removed, so the result is 0-4 unique items.

    Pure so the negotiation pipeline can reuse the list it already fetched for
    scoring instead of triggering a second full table scan per agent.
    """
    if not feedbacks:
        return []

    seen_ids: set[str] = set()
    representatives: list[FeedbackScore] = []

    def _add(fb: FeedbackScore) -> None:
        if fb.feedback_id not in seen_ids:
            seen_ids.add(fb.feedback_id)
            representatives.append(fb)

    # Sort by score for best/median/worst selection
    by_score = sorted(feedbacks, key=lambda f: f.score)

    _add(by_score[-1])  # best
    _add(by_score[len(by_score) // 2])  # median
    _add(by_score[0])  # worst
    _add(feedbacks[0])  # most recent (the query orders by timestamp DESC)

    return representatives


async def _write_reputation(
    conn: aiosqlite.Connection, agent_id: str, reputation_score: float
) -> None:
    await conn.execute(
        """
        INSERT INTO agents (agent_id, budget, capacity, reputation_score)
        VALUES (?, 0.0, 1, ?)
        ON CONFLICT(agent_id) DO UPDATE SET
            reputation_score = excluded.reputation_score
        """,
        (agent_id, reputation_score),
    )


async def _write_negotiation(conn: aiosqlite.Connection, negotiation: NegotiationRecord) -> None:
    await conn.execute(
        """
        INSERT INTO negotiations (negotiation_id, buyer_id, seller_id, service_id,
                                  max_budget, min_price, initial_price,
                                  status, rounds, max_rounds, final_price, final_sla,
                                  created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(negotiation_id) DO UPDATE SET
            max_budget = excluded.max_budget,
            min_price = excluded.min_price,
            initial_price = excluded.initial_price,
            status = excluded.status,
            rounds = excluded.rounds,
            max_rounds = excluded.max_rounds,
            final_price = excluded.final_price,
            final_sla = excluded.final_sla,
            updated_at = excluded.updated_at
        """,
        (
            negotiation.negotiation_id,
            negotiation.buyer_id,
            negotiation.seller_id,
            negotiation.service_id,
            negotiation.max_budget,
            negotiation.min_price,
            negotiation.initial_price,
            str(negotiation.status),
            negotiation.rounds,
            negotiation.max_rounds,
            negotiation.final_price,
            negotiation.final_sla,
            negotiation.created_at.isoformat(),
            negotiation.updated_at.isoformat(),
        ),
    )


async def _write_agreement(conn: aiosqlite.Connection, agreement: Agreement) -> None:
    await conn.execute(
        """
        INSERT INTO agreements (agreement_id, buyer_id, seller_id, service_id,
                                final_price, final_sla, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            agreement.agreement_id,
            agreement.buyer_id,
            agreement.seller_id,
            agreement.service_id,
            agreement.final_price,
            agreement.final_sla,
            agreement.timestamp.isoformat(),
        ),
    )


async def _write_feedback(conn: aiosqlite.Connection, feedback: FeedbackScore) -> None:
    await conn.execute(
        """
        INSERT INTO feedback (feedback_id, agreement_id, evaluator_id,
                              target_id, score, comments, timestamp, evaluator_model)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            feedback.feedback_id,
            feedback.agreement_id,
            feedback.evaluator_id,
            feedback.target_id,
            feedback.score,
            feedback.comments,
            feedback.timestamp.isoformat(),
            feedback.evaluator_model,
        ),
    )


async def _write_messages(conn: aiosqlite.Connection, messages: list[NegotiationMessage]) -> None:
    if not messages:
        return
    await conn.executemany(
        """
        INSERT OR IGNORE INTO negotiation_messages
        (message_id, negotiation_id, role, content, raw_data, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                msg.message_id,
                msg.negotiation_id,
                msg.role,
                msg.content,
                msg.raw_data,
                msg.timestamp.isoformat(),
            )
            for msg in messages
        ],
    )


async def _feedbacks_for(conn: aiosqlite.Connection, agent_id: str) -> list[FeedbackScore]:
    cursor = await conn.execute(
        "SELECT * FROM feedback WHERE target_id = ? ORDER BY timestamp DESC",
        (agent_id,),
    )
    return [
        FeedbackScore(
            feedback_id=row["feedback_id"],
            agreement_id=row["agreement_id"],
            evaluator_id=row["evaluator_id"],
            target_id=row["target_id"],
            score=row["score"],
            comments=row["comments"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            evaluator_model=row["evaluator_model"],
        )
        for row in await cursor.fetchall()
    ]


class DatabaseManager:
    """Manages a persistent async SQLite connection and CRUD operations."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        # ``settings.sqlite_db_full_path`` is a str so that the ``:memory:``
        # token can be passed through unwrapped.
        self.db_path: Path | str = db_path or settings.sqlite_db_full_path
        self._conn: aiosqlite.Connection | None = None
        # One connection is shared by every coroutine, and SQLite transactions
        # are per connection: without serialising writers, one coroutine's
        # commit would commit another's half-finished transaction (and a
        # rollback would discard someone else's writes).
        self._write_lock = asyncio.Lock()

    # -- Lifecycle -----------------------------------------------------

    async def connect(self, *, read_only: bool = False) -> None:
        """Open a persistent connection and initialize the schema.

        ``read_only=True`` opens the file with SQLite's ``mode=ro``: no DDL, no
        journal-mode switch, and any write raises. The schema is still verified.
        For processes that run beside the API server against the same file (the
        MCP server); the file must already exist and be migrated.
        """
        if read_only:
            if str(self.db_path) == ":memory:":
                raise ValueError("A read-only connection needs a database file, not :memory:.")
            if not Path(self.db_path).is_file():
                raise RuntimeError(
                    f"Database {self.db_path} does not exist. Start the API server "
                    "once (or run `uv run alembic upgrade head`) to create it."
                )
            self._conn = await aiosqlite.connect(
                f"{Path(self.db_path).resolve().as_uri()}?mode=ro", uri=True
            )
        else:
            self._conn = await aiosqlite.connect(self.db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA busy_timeout=5000")
        if read_only:
            await self._verify_schema()
        else:
            # WAL lets readers (including a read-only MCP process) work while
            # this connection writes.
            await self._conn.execute("PRAGMA journal_mode=WAL")
            # SQLite does NOT enforce FOREIGN KEY constraints unless this is set
            # on every connection; without it the declared keys are decorative.
            await self._conn.execute("PRAGMA foreign_keys=ON")
            await self._init_schema()
        logger.info("Database connected%s: %s", " (read-only)" if read_only else "", self.db_path)

    async def _init_schema(self) -> None:
        """Create tables if they do not already exist.

        The DDL lives in :mod:`agent_bargain.storage.schema` and is shared with
        the Alembic chain; ``tests/test_schema_drift.py`` keeps them aligned.

        ``CREATE TABLE IF NOT EXISTS`` will not add columns to a table that
        already exists, so an existing database still needs
        ``alembic upgrade head``. ``_verify_schema`` turns the resulting cryptic
        "no such column" failures into one actionable startup error.
        """
        await self.conn.executescript(SCHEMA_SQL)
        await self._verify_schema()

    async def _verify_schema(self) -> None:
        """Fail fast with a clear message if the database predates a migration."""
        expected = _expected_schema()

        cursor = await self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )
        actual_tables = {row[0] for row in await cursor.fetchall()}

        missing: list[str] = []
        for table, columns in sorted(expected.items()):
            if table not in actual_tables:
                missing.append(f"{table} (whole table)")
                continue
            cursor = await self.conn.execute(f"PRAGMA table_info({table})")
            present = {row[1] for row in await cursor.fetchall()}
            missing.extend(f"{table}.{col}" for col in sorted(columns - present))

        if missing:
            raise RuntimeError(
                f"Database at {self.db_path} is missing schema element(s): "
                f"{', '.join(missing)}. Run `uv run alembic upgrade head` "
                "(the Docker image does this automatically on start)."
            )

    async def close(self) -> None:
        """Close the persistent connection."""
        if self._conn:
            await self._conn.close()
            self._conn = None
            logger.info("Database connection closed.")

    @property
    def conn(self) -> aiosqlite.Connection:
        """Return the active connection or raise if not connected."""
        if self._conn is None:
            raise RuntimeError("DatabaseManager is not connected. Call connect() first.")
        return self._conn

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[aiosqlite.Connection]:
        """Serialised write transaction: commit on success, roll back on error."""
        async with self._write_lock:
            conn = self.conn
            try:
                yield conn
            except BaseException:
                await conn.rollback()
                raise
            await conn.commit()

    # -- Agents --------------------------------------------------------

    async def ensure_agent(self, agent_id: str) -> None:
        """Register an agent with defaults only if it does not already exist.

        Use this for auto-registration at negotiation start so that a pre-existing
        agent's ``budget`` and ``capacity`` are never overwritten by placeholder zeros.
        """
        async with self._transaction() as conn:
            # Neutral, not 0.0: a new agent has no history, and 0.0 ranked it
            # as the worst-rated agent on the leaderboard.
            await conn.execute(
                """
                INSERT OR IGNORE INTO agents (agent_id, budget, capacity, reputation_score)
                VALUES (?, 0.0, 1, ?)
                """,
                (agent_id, NEUTRAL_REPUTATION_SCORE),
            )

    async def update_agent_reputation(self, agent_id: str, reputation_score: float) -> None:
        """Update only the reputation_score for an existing agent.

        Leaves ``budget`` and ``capacity`` untouched.
        If the agent row does not exist it is created with placeholder defaults.
        """
        async with self._transaction() as conn:
            await _write_reputation(conn, agent_id, reputation_score)

    # -- Agreements and Negotiations -----------------------------------

    async def save_negotiation(self, negotiation: NegotiationRecord) -> None:
        """Save a negotiation record (active, accepted, rejected, failed)."""
        async with self._transaction() as conn:
            await _write_negotiation(conn, negotiation)

    async def save_agreement(self, agreement: Agreement) -> None:
        """Save a finalized agreement."""
        async with self._transaction() as conn:
            await _write_agreement(conn, agreement)

    async def upsert_service(self, service: ServiceListing) -> bool:
        """Insert a listing, or update it if ``service.seller_id`` already owns it.

        Returns ``False`` when the ``service_id`` belongs to a different seller
        and nothing was written. The ownership check is part of the upsert
        itself, so two sellers racing for one id cannot both "win" the way a
        separate read-then-write let them.
        """
        async with self._transaction() as conn:
            await conn.execute(
                """
                INSERT INTO services (service_id, seller_id, title, description,
                                      category_id, category_name, base_price,
                                      floor_price, sla_terms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(service_id) DO UPDATE SET
                    title = excluded.title,
                    description = excluded.description,
                    category_id = excluded.category_id,
                    category_name = excluded.category_name,
                    base_price = excluded.base_price,
                    floor_price = excluded.floor_price,
                    sla_terms = excluded.sla_terms
                WHERE services.seller_id = excluded.seller_id
                """,
                (
                    service.service_id,
                    service.seller_id,
                    service.title,
                    service.description,
                    service.category.id,
                    service.category.name,
                    service.base_price,
                    service.floor_price,
                    service.sla_terms,
                ),
            )
            # ``cursor.rowcount`` reads -1 on this connection setup, so ask
            # SQLite directly; ``changes()`` is 0 when the WHERE refused the update.
            cursor = await conn.execute("SELECT changes()")
            row = await cursor.fetchone()
            return bool(row and row[0])

    async def get_service(self, service_id: str) -> ServiceListing | None:
        """Retrieve a service listing by ID, or None if it is not registered."""
        cursor = await self.conn.execute(
            "SELECT * FROM services WHERE service_id = ?", (service_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return ServiceListing(
            service_id=row["service_id"],
            seller_id=row["seller_id"],
            title=row["title"],
            description=row["description"],
            category=ServiceCategory(id=row["category_id"], name=row["category_name"]),
            base_price=row["base_price"],
            floor_price=row["floor_price"],
            sla_terms=row["sla_terms"],
        )

    async def save_negotiation_messages(self, messages: list[NegotiationMessage]) -> None:
        """Save a batch of negotiation messages to the database."""
        if not messages:
            return
        async with self._transaction() as conn:
            await _write_messages(conn, messages)

    # -- Feedback ------------------------------------------------------

    async def save_feedback(self, feedback: FeedbackScore) -> None:
        """Save a new feedback score."""
        async with self._transaction() as conn:
            await _write_feedback(conn, feedback)

    async def get_feedbacks_for_agent(self, agent_id: str) -> list[FeedbackScore]:
        """Retrieve all feedback scores targeting a specific agent, newest first."""
        return await _feedbacks_for(self.conn, agent_id)

    # -- Atomic multi-row writes ---------------------------------------

    async def record_outcome(
        self,
        negotiation: NegotiationRecord,
        messages: list[NegotiationMessage],
        agreement: Agreement | None,
    ) -> None:
        """Persist a finished negotiation's row, transcript and agreement atomically.

        Either all of it lands or none of it: written one by one, a failing
        transcript insert left an ``accepted`` row without its agreement.
        """
        async with self._transaction() as conn:
            await _write_negotiation(conn, negotiation)
            await _write_messages(conn, messages)
            if agreement is not None:
                await _write_agreement(conn, agreement)

    async def record_evaluation(
        self,
        feedbacks: list[FeedbackScore],
        score: Callable[[list[FeedbackScore]], float],
    ) -> None:
        """Insert feedback and refresh each target's stored reputation atomically.

        ``score`` maps an agent's full feedback list (newest first, including
        the rows just inserted) to its reputation.
        """
        async with self._transaction() as conn:
            for feedback in feedbacks:
                await _write_feedback(conn, feedback)
            for target_id in dict.fromkeys(fb.target_id for fb in feedbacks):
                await _write_reputation(
                    conn, target_id, score(await _feedbacks_for(conn, target_id))
                )

    # -- Negotiation queries -------------------------------------------

    async def fail_orphaned_negotiations(self) -> int:
        """Mark leftover ``active`` negotiations as ``failed``; return how many.

        A negotiation is written as ``active`` before its background task runs.
        If the process dies mid-negotiation — a crash, a restart, a redeploy —
        nothing ever reaches Phase 3, so the row stays ``active`` forever:
        indistinguishable from one that is genuinely in flight, and permanently
        skewing the dashboard's accept-rate and aggregate statistics.

        Safe to call at startup because no request has been served yet, so any
        ``active`` row necessarily belongs to a dead process.

        .. caution::
           With multiple Uvicorn workers a later worker starting up would mark
           an earlier worker's *live* negotiations as failed. The default entry
           point is single-process; multi-worker deployments need a real
           liveness signal (heartbeat timestamp) instead of this sweep.
        """
        async with self._transaction() as conn:
            cursor = await conn.execute(
                "UPDATE negotiations SET status = 'failed' WHERE status = 'active'"
            )
            return cursor.rowcount or 0

    async def get_negotiation(self, negotiation_id: str) -> dict | None:
        """Retrieve a single negotiation record by ID."""
        cursor = await self.conn.execute(
            "SELECT * FROM negotiations WHERE negotiation_id = ?",
            (negotiation_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_negotiations(
        self, limit: int = 20, offset: int = 0, *, status: str | None = None
    ) -> list[dict]:
        """List negotiation records, most recent first, optionally of one status."""
        if status is None:
            cursor = await self.conn.execute(
                "SELECT * FROM negotiations ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        else:
            cursor = await self.conn.execute(
                "SELECT * FROM negotiations WHERE status = ? "
                "ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (status, limit, offset),
            )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def negotiation_stats(self) -> dict:
        """Marketplace-wide counts and averages, computed over *every* row.

        Aggregating a fetched page instead silently capped "total" at the page
        size once the marketplace outgrew it.
        """
        cursor = await self.conn.execute(
            """
            SELECT COUNT(*)                                             AS total,
                   COALESCE(SUM(status = 'accepted'), 0)                AS accepted,
                   COALESCE(SUM(status = 'rejected'), 0)                AS rejected,
                   COALESCE(SUM(status = 'timeout'), 0)                 AS timeout,
                   COALESCE(AVG(rounds), 0.0)                           AS avg_rounds,
                   COALESCE(AVG(CASE WHEN status = 'accepted'
                                     THEN final_price END), 0.0)        AS avg_final_price
            FROM negotiations
            """
        )
        row = await cursor.fetchone()
        return dict(row) if row else {}

    async def get_messages_for_negotiation(self, negotiation_id: str) -> list[dict]:
        """Retrieve all messages for a negotiation in chronological order.

        ``rowid`` breaks timestamp ties: the strategy engine can emit two turns
        within the clock's resolution, and a transcript is inserted in turn
        order, so insertion order is the true order.
        """
        cursor = await self.conn.execute(
            """SELECT * FROM negotiation_messages
               WHERE negotiation_id = ?
               ORDER BY timestamp ASC, rowid ASC""",
            (negotiation_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # -- Service & Agent queries ---------------------------------------

    async def list_services(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """List all registered service listings."""
        cursor = await self.conn.execute(
            "SELECT * FROM services ORDER BY title ASC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def list_all_services(self) -> list[dict]:
        """Return every service listing (used to seed the discovery index)."""
        cursor = await self.conn.execute("SELECT * FROM services ORDER BY title ASC")
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def list_agents(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """List all registered agents ordered by reputation score."""
        cursor = await self.conn.execute(
            "SELECT * FROM agents ORDER BY reputation_score DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]
