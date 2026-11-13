"""Tests for the storage layer helpers that are pure enough to test directly."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from agent_bargain.models.negotiation import Agreement, NegotiationRecord, NegotiationStatus
from agent_bargain.models.reputation import FeedbackScore
from agent_bargain.models.service import ServiceListing
from agent_bargain.storage.database import DatabaseManager, select_representative_feedbacks


def _fb(feedback_id: str, score: float, *, minutes_ago: int = 0) -> FeedbackScore:
    return FeedbackScore(
        feedback_id=feedback_id,
        agreement_id="agr_1",
        evaluator_id="system_evaluator",
        target_id="seller_1",
        score=score,
        timestamp=datetime.now(UTC) - timedelta(minutes=minutes_ago),
    )


async def _seed_agreement(db: DatabaseManager) -> None:
    """Create the rows ``feedback.agreement_id`` points at.

    Foreign keys are enforced, so feedback cannot be inserted against a
    non-existent agreement.
    """
    await db.ensure_agent("buyer_1")
    await db.ensure_agent("seller_1")
    await db.upsert_service(
        ServiceListing(
            service_id="svc_1",
            seller_id="seller_1",
            title="Test Service",
            base_price=100.0,
            sla_terms="Standard",
        )
    )
    await db.save_agreement(
        Agreement(
            agreement_id="agr_1",
            buyer_id="buyer_1",
            seller_id="seller_1",
            service_id="svc_1",
            final_price=100.0,
            final_sla="Standard",
        )
    )


# -- representative feedback selection -------------------------------------


def test_select_representative_empty_history():
    assert select_representative_feedbacks([]) == []


def test_select_representative_single_feedback_is_not_duplicated():
    """One row can be best, median, worst and most recent at once."""
    only = [_fb("fb_1", 4.0)]

    assert select_representative_feedbacks(only) == only


def test_select_representative_picks_extremes_median_and_latest():
    # Ordered newest-first, as get_feedbacks_for_agent returns them.
    feedbacks = [
        _fb("newest", 3.0, minutes_ago=0),
        _fb("mid_old", 1.0, minutes_ago=30),
        _fb("mid", 5.0, minutes_ago=60),
        _fb("oldest", 4.0, minutes_ago=90),
    ]

    selected = select_representative_feedbacks(feedbacks)
    ids = [f.feedback_id for f in selected]

    assert len(ids) == len(set(ids)), f"duplicates: {ids}"
    assert len(ids) <= 4
    assert "mid" in ids, "highest score missing"
    assert "mid_old" in ids, "lowest score missing"
    assert "newest" in ids, "most recent missing"


def test_select_representative_caps_at_four():
    """Best, median, worst and newest — never more than those four."""
    # Newest is deliberately a mid score so it cannot coincide with best/worst,
    # and the spread guarantees all four picks are distinct rows.
    feedbacks = [_fb("newest", 3.0, minutes_ago=0)]
    feedbacks += [_fb(f"fb_{i}", float(i % 5 + 1), minutes_ago=i + 1) for i in range(50)]

    selected = select_representative_feedbacks(feedbacks)

    assert len(selected) == 4
    assert len({f.feedback_id for f in selected}) == 4, "picks were not deduplicated"
    assert "newest" in [f.feedback_id for f in selected]


def test_select_representative_dedupes_when_categories_overlap():
    """With little history the same row is best, median, worst and newest."""
    two = [_fb("recent_mid", 3.0, minutes_ago=0), _fb("old_best", 5.0, minutes_ago=60)]

    selected = select_representative_feedbacks(two)

    assert {f.feedback_id for f in selected} == {"recent_mid", "old_best"}
    assert len(selected) == 2


def test_select_representative_is_pure():
    """Callers reuse an already-fetched list, so it must not mutate it."""
    feedbacks = [_fb(f"fb_{i}", float(i % 5 + 1), minutes_ago=i) for i in range(6)]
    before = list(feedbacks)

    select_representative_feedbacks(feedbacks)
    select_representative_feedbacks(feedbacks)

    assert feedbacks == before


async def test_get_feedbacks_returns_newest_first(tmp_path):
    """The selection strategy relies on this ordering for 'most recent'."""
    db = DatabaseManager(db_path=tmp_path / "order.db")
    await db.connect()
    try:
        await _seed_agreement(db)
        await db.save_feedback(_fb("old", 2.0, minutes_ago=100))
        await db.save_feedback(_fb("new", 4.0, minutes_ago=1))

        fetched = await db.get_feedbacks_for_agent("seller_1")
    finally:
        await db.close()

    assert [f.feedback_id for f in fetched][:1] == ["new"]


async def test_feedback_is_isolated_per_agent(tmp_path):
    """The indexed target_id lookup must not leak another agent's history."""
    db = DatabaseManager(db_path=tmp_path / "isolate.db")
    await db.connect()
    try:
        await _seed_agreement(db)
        await db.ensure_agent("seller_2")
        await db.save_feedback(_fb("fb_a", 5.0))
        await db.save_feedback(_fb("fb_b", 1.0).model_copy(update={"target_id": "seller_2"}))

        mine = await db.get_feedbacks_for_agent("seller_1")
        theirs = await db.get_feedbacks_for_agent("seller_2")
    finally:
        await db.close()

    assert [f.feedback_id for f in mine] == ["fb_a"]
    assert [f.feedback_id for f in theirs] == ["fb_b"]


async def test_feedback_requires_an_existing_agreement(tmp_path):
    """FK enforcement reaches the reputation ledger, not just negotiations."""
    db = DatabaseManager(db_path=tmp_path / "nofk.db")
    await db.connect()
    try:
        await db.ensure_agent("seller_1")
        with pytest.raises(sqlite3.IntegrityError):
            await db.save_feedback(_fb("orphan", 5.0))
    finally:
        await db.close()


# -- crash recovery -------------------------------------------------


async def _seed_negotiation(db, negotiation_id: str, status: str) -> None:
    await db.ensure_agent("buyer_1")
    await db.ensure_agent("seller_1")
    await db.upsert_service(
        ServiceListing(
            service_id="svc_1",
            seller_id="seller_1",
            title="Test Service",
            base_price=100.0,
            sla_terms="Standard",
        )
    )
    await db.save_negotiation(
        NegotiationRecord(
            negotiation_id=negotiation_id,
            buyer_id="buyer_1",
            seller_id="seller_1",
            service_id="svc_1",
            status=NegotiationStatus(status),
            rounds=0,
        )
    )


async def test_orphaned_active_negotiations_are_failed(tmp_path):
    """A process that dies mid-negotiation must not leave an immortal 'active' row.

    The row is written before the background task runs, so a crash leaves it
    indistinguishable from a live negotiation — and it skews the dashboard's
    accept rate forever.
    """
    db = DatabaseManager(db_path=tmp_path / "orphan.db")
    await db.connect()
    try:
        await _seed_negotiation(db, "neg_orphan", "active")
        await _seed_negotiation(db, "neg_done", "accepted")
        await _seed_negotiation(db, "neg_gone", "rejected")

        swept = await db.fail_orphaned_negotiations()

        assert swept == 1
        assert (await db.get_negotiation("neg_orphan"))["status"] == "failed"
        # Terminal states must be left alone — especially a real agreement.
        assert (await db.get_negotiation("neg_done"))["status"] == "accepted"
        assert (await db.get_negotiation("neg_gone"))["status"] == "rejected"
    finally:
        await db.close()


async def test_orphan_sweep_is_idempotent(tmp_path):
    """Running the sweep twice must not keep re-writing rows."""
    db = DatabaseManager(db_path=tmp_path / "orphan2.db")
    await db.connect()
    try:
        await _seed_negotiation(db, "neg_a", "active")

        assert await db.fail_orphaned_negotiations() == 1
        assert await db.fail_orphaned_negotiations() == 0
    finally:
        await db.close()


async def test_read_only_connection_reads_but_refuses_writes(tmp_path):
    """The MCP server opens the API server's database with mode=ro."""
    path = tmp_path / "shared.db"
    writer = DatabaseManager(path)
    await writer.connect()
    await writer.ensure_agent("seller_1")

    reader = DatabaseManager(path)
    await reader.connect(read_only=True)
    try:
        assert [a["agent_id"] for a in await reader.list_agents()] == ["seller_1"]
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            await reader.ensure_agent("intruder")
    finally:
        await reader.close()
        await writer.close()


async def test_read_only_connection_requires_an_existing_file(tmp_path):
    db = DatabaseManager(tmp_path / "missing.db")
    with pytest.raises(RuntimeError, match="does not exist"):
        await db.connect(read_only=True)
    assert not (tmp_path / "missing.db").exists()
