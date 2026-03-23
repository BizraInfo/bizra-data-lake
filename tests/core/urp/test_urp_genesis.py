"""Tests for URP Genesis — the constitutional membrane lifecycle."""

from __future__ import annotations

import pytest

from core.urp.constitution import Constitution
from core.urp.genesis import get_urp, mint_urp_genesis, reset_urp
from core.urp.membrane import ConstitutionalMembrane
from core.urp.resource_pool import ResourcePool
from core.urp.service import URPService


@pytest.fixture(autouse=True)
def _clean_urp():
    """Reset URP singleton between tests."""
    reset_urp()
    yield
    reset_urp()


class TestConstitution:
    def test_constitution_is_immutable(self) -> None:
        c = Constitution()
        with pytest.raises(AttributeError):
            c.ihsan_floor = 0.50  # type: ignore[misc]

    def test_hash_is_deterministic(self) -> None:
        c1 = Constitution()
        c2 = Constitution()
        assert c1.hash() == c2.hash()
        assert len(c1.hash()) == 64

    def test_check_receipt_passes_above_threshold(self) -> None:
        c = Constitution()
        ok, reason = c.check_receipt({"ihsan_score": 0.97, "signed": True})
        assert ok
        assert reason == "constitutional"

    def test_check_receipt_rejects_below_threshold(self) -> None:
        c = Constitution()
        ok, reason = c.check_receipt({"ihsan_score": 0.80, "signed": True})
        assert not ok
        assert "ihsan" in reason

    def test_check_receipt_rejects_unsigned(self) -> None:
        c = Constitution()
        ok, reason = c.check_receipt({"ihsan_score": 0.97, "signed": False})
        assert not ok
        assert "ZANN_ZERO" in reason

    def test_gini_check(self) -> None:
        c = Constitution()
        ok, _ = c.check_gini(0.30)
        assert ok
        ok, reason = c.check_gini(0.40)
        assert not ok
        assert "gini" in reason


class TestMembrane:
    def test_admit_valid_receipt(self) -> None:
        m = ConstitutionalMembrane(Constitution())
        admitted, reason, record = m.filter_inbound(
            "node-0", "receipt", {"ihsan_score": 0.97, "signed": True}
        )
        assert admitted
        assert reason == "admitted"
        assert record.chain_hash != "0" * 64

    def test_reject_missing_identity(self) -> None:
        m = ConstitutionalMembrane(Constitution())
        admitted, reason, _ = m.filter_inbound("", "receipt", {})
        assert not admitted
        assert reason == "missing_node_identity"

    def test_reject_below_ihsan(self) -> None:
        m = ConstitutionalMembrane(Constitution())
        admitted, reason, _ = m.filter_inbound(
            "node-0", "receipt", {"ihsan_score": 0.80, "signed": True}
        )
        assert not admitted
        assert "ihsan" in reason

    def test_chain_integrity(self) -> None:
        m = ConstitutionalMembrane(Constitution())
        for i in range(5):
            m.filter_inbound(f"node-{i}", "event", {"data": i})
        ok, errors = m.verify_chain()
        assert ok
        assert errors == []

    def test_stats_track_admitted_and_rejected(self) -> None:
        m = ConstitutionalMembrane(Constitution())
        m.filter_inbound("node-0", "event", {})
        m.filter_inbound("", "event", {})  # rejected
        stats = m.stats()
        assert stats["admitted"] == 1
        assert stats["rejected"] == 1
        assert stats["total_crossings"] == 2


class TestResourcePool:
    def test_genesis_seed(self) -> None:
        pool = ResourcePool()
        receipt = pool.mint_genesis_seed("node-0", 100_000.0, 2_500.0)
        assert pool.seed_treasury == 100_000.0
        assert pool.zakat_pool == 2_500.0
        assert receipt["type"] == "genesis_mint"

    def test_contribute_knowledge_accepted(self) -> None:
        pool = ResourcePool()
        ok = pool.contribute_knowledge(
            "BIZRA is a sovereign AI OS",
            "node-0",
            ihsan_score=0.97,
            receipt_id="r-001",
        )
        assert ok
        assert len(pool.knowledge) == 1

    def test_contribute_knowledge_rejected_low_ihsan(self) -> None:
        pool = ResourcePool()
        ok = pool.contribute_knowledge(
            "low quality", "node-0", ihsan_score=0.80, receipt_id="r-002"
        )
        assert not ok
        assert len(pool.knowledge) == 0

    def test_duplicate_knowledge_rejected(self) -> None:
        pool = ResourcePool()
        pool.contribute_knowledge("same content", "node-0", 0.97, "r-001")
        ok = pool.contribute_knowledge("same content", "node-1", 0.97, "r-002")
        assert not ok
        assert len(pool.knowledge) == 1

    def test_draw_excludes_own_contributions(self) -> None:
        pool = ResourcePool()
        pool.contribute_knowledge("from node-0", "node-0", 0.97, "r-001")
        pool.contribute_knowledge("from node-1", "node-1", 0.97, "r-002")
        results = pool.draw_knowledge("node-0")
        assert len(results) == 1
        assert results[0].contributor_node == "node-1"

    def test_shared_reflex_adoption(self) -> None:
        pool = ResourcePool()
        pool.contribute_reflex("hash-1", "fast auth", "node-0", 0.90)
        pool.contribute_reflex("hash-1", "fast auth", "node-1", 0.92)
        reflex = pool.shared_reflexes["hash-1"]
        assert len(reflex.adopters) == 2
        assert reflex.confidence > 0.90


class TestURPService:
    def test_genesis_creates_complete_urp(self) -> None:
        urp = URPService()
        receipt = urp.mint_genesis("node-0", "pubkey-0")
        assert urp.genesis_complete
        assert len(urp.sat_agents) == 5
        assert "S2-Oracle" in urp.sat_agents
        assert urp.sat_agents["S2-Oracle"].frozen
        assert "node-0" in urp.connected_nodes
        assert receipt.sat_count == 5

    def test_genesis_only_once(self) -> None:
        urp = URPService()
        urp.mint_genesis("node-0", "pk-0")
        with pytest.raises(RuntimeError, match="one-time only"):
            urp.mint_genesis("node-0", "pk-0")

    def test_register_node(self) -> None:
        urp = URPService()
        urp.mint_genesis("node-0", "pk-0")
        ok, reason = urp.register_node("node-1", "pk-1")
        assert ok
        assert reason == "registered"
        assert "node-1" in urp.connected_nodes

    def test_submit_receipt_accepted(self) -> None:
        urp = URPService()
        urp.mint_genesis("node-0", "pk-0")
        ok, reason = urp.submit_receipt(
            "node-0", {"ihsan_score": 0.97, "signed": True, "id": "r-001"}
        )
        assert ok
        assert reason == "accepted"

    def test_submit_receipt_rejected_unregistered(self) -> None:
        urp = URPService()
        urp.mint_genesis("node-0", "pk-0")
        ok, reason = urp.submit_receipt("node-99", {"ihsan_score": 0.97})
        assert not ok
        assert reason == "not_registered"

    def test_contribute_and_query_knowledge(self) -> None:
        urp = URPService()
        urp.mint_genesis("node-0", "pk-0")
        urp.register_node("node-1", "pk-1")
        urp.contribute_knowledge("node-0", "BIZRA uses BLAKE3", 0.97, "r-001")
        ok, results = urp.query_knowledge("node-1", "BLAKE3")
        assert ok
        assert len(results) == 1

    def test_status(self) -> None:
        urp = URPService()
        urp.mint_genesis("node-0", "pk-0")
        status = urp.status()
        assert status["genesis_complete"]
        assert status["connected_nodes"] == 1
        assert status["sat_agents"] == 5
        assert status["sat_frozen"] == 1


class TestURPGenesis:
    def test_mint_urp_genesis(self) -> None:
        urp, receipt = mint_urp_genesis("node-0", "pk-0")
        assert urp.genesis_complete
        assert receipt.founder_node == "node-0"

    def test_idempotent(self) -> None:
        urp1, r1 = mint_urp_genesis("node-0", "pk-0")
        urp2, r2 = mint_urp_genesis("node-0", "pk-0")
        assert urp1 is urp2

    def test_get_urp(self) -> None:
        assert get_urp() is None
        mint_urp_genesis("node-0", "pk-0")
        assert get_urp() is not None
        assert get_urp().genesis_complete
