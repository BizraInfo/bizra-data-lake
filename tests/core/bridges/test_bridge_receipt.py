"""
Tests for BridgeReceiptEngine (core/bridges/bridge_receipt.py).

Covers:
- Receipt creation (accepted + rejected)
- Signature validity
- Disk persistence + atomic writes
- Cache retrieval + disk fallback
- Receipt not found
- Receipt directory auto-creation
- Signer loading from env / fallback
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from core.bridges.bridge_receipt import BridgeReceiptEngine, load_signer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_receipt_dir(tmp_path: Path) -> Path:
    d = tmp_path / "receipts"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def _receipt_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BIZRA_RECEIPT_PRIVATE_KEY_HEX", "11" * 32)


@pytest.fixture
def engine(tmp_receipt_dir: Path) -> BridgeReceiptEngine:
    return BridgeReceiptEngine(receipt_dir=tmp_receipt_dir)


# ---------------------------------------------------------------------------
# Signer loading
# ---------------------------------------------------------------------------


class TestLoadSigner:
    def test_load_signer_from_env(self) -> None:
        key = "aa" * 32
        with patch.dict(os.environ, {"BIZRA_RECEIPT_PRIVATE_KEY_HEX": key}):
            signer = load_signer()
            assert signer.secret == bytes.fromhex(key)

    def test_load_signer_requires_key(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("BIZRA_RECEIPT_PRIVATE_KEY_HEX", None)
            with pytest.raises(RuntimeError, match="BIZRA_RECEIPT_PRIVATE_KEY_HEX"):
                load_signer()


# ---------------------------------------------------------------------------
# Receipt creation
# ---------------------------------------------------------------------------


class TestReceiptCreation:
    def test_create_accepted_receipt(self, engine: BridgeReceiptEngine) -> None:
        receipt = engine.create_receipt(
            method="ping",
            query_data={"method": "ping"},
            result_data={"status": "alive"},
            fate_score=0.98,
            snr_score=0.95,
            gate_passed="FATE",
            status="accepted",
            duration_ms=1.5,
        )
        assert receipt["status"] == "accepted"
        assert receipt["method"] == "ping"
        assert receipt["receipt_id"].startswith("br-")
        assert receipt["fate_score"] == 0.98
        assert receipt["snr_score"] == 0.95
        assert receipt["gate_passed"] == "FATE"
        assert receipt["duration_ms"] == 1.5
        assert receipt["reason"] is None
        assert receipt["origin"]["designation"] == "ephemeral_node"
        assert len(receipt["origin_digest"]) == 64
        assert "signature" in receipt
        assert "receipt_digest" in receipt
        assert len(receipt["signature"]) > 0
        assert len(receipt["receipt_digest"]) > 0

    def test_create_rejected_receipt(self, engine: BridgeReceiptEngine) -> None:
        receipt = engine.create_receipt(
            method="sovereign_query",
            query_data={"query": "blocked"},
            result_data={"error": "FATE gate blocked"},
            fate_score=0.4,
            snr_score=0.3,
            gate_passed="FATE",
            status="rejected",
            reason="Below ihsan threshold",
        )
        assert receipt["status"] == "rejected"
        assert receipt["reason"] == "Below ihsan threshold"
        assert receipt["fate_score"] == 0.4

    def test_receipt_has_valid_signature(self, engine: BridgeReceiptEngine) -> None:
        receipt = engine.create_receipt(
            method="status",
            query_data={},
            result_data={"node": "node0"},
            fate_score=1.0,
            snr_score=0.95,
            gate_passed="all",
            status="accepted",
        )
        assert engine.verify_receipt(receipt)

    def test_receipt_tamper_on_origin_fails_signature(
        self, engine: BridgeReceiptEngine
    ) -> None:
        receipt = engine.create_receipt(
            method="status",
            query_data={},
            result_data={"node": "node0"},
            fate_score=1.0,
            snr_score=0.95,
            gate_passed="all",
            status="accepted",
            origin={
                "designation": "node0",
                "genesis_node": True,
                "genesis_block": True,
                "block_id": "block0",
                "home_base_device": True,
                "node_id": "node0_fixture",
                "node_name": "Node0 Fixture",
                "authority_source": "genesis_files",
                "hash_validated": True,
            },
        )
        tampered = dict(receipt)
        tampered["origin"] = dict(tampered["origin"])
        tampered["origin"]["hash_validated"] = False
        assert engine.verify_receipt(tampered) is False

    def test_receipt_id_is_unique(self, engine: BridgeReceiptEngine) -> None:
        r1 = engine.create_receipt(
            method="ping", query_data={}, result_data={},
            fate_score=1.0, snr_score=0.95, gate_passed="all", status="accepted",
        )
        r2 = engine.create_receipt(
            method="ping", query_data={}, result_data={},
            fate_score=1.0, snr_score=0.95, gate_passed="all", status="accepted",
        )
        assert r1["receipt_id"] != r2["receipt_id"]

    def test_receipt_digests_are_deterministic(
        self, tmp_receipt_dir: Path
    ) -> None:
        """Same input -> same digests (excluding timestamp/id)."""
        e1 = BridgeReceiptEngine(receipt_dir=tmp_receipt_dir)
        e2 = BridgeReceiptEngine(receipt_dir=tmp_receipt_dir)

        kwargs = dict(
            method="ping",
            query_data={"a": 1},
            result_data={"b": 2},
            fate_score=1.0,
            snr_score=0.95,
            gate_passed="all",
            status="accepted",
        )
        r1 = e1.create_receipt(**kwargs)
        r2 = e2.create_receipt(**kwargs)

        # Digests of same content are identical
        assert r1["query_digest"] == r2["query_digest"]
        assert r1["payload_digest"] == r2["payload_digest"]
        assert r1["policy_digest"] == r2["policy_digest"]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_receipt_persisted_to_disk(
        self, engine: BridgeReceiptEngine, tmp_receipt_dir: Path
    ) -> None:
        receipt = engine.create_receipt(
            method="ping", query_data={}, result_data={},
            fate_score=1.0, snr_score=0.95, gate_passed="all", status="accepted",
        )
        path = tmp_receipt_dir / f"{receipt['receipt_id']}.json"
        assert path.exists()

        loaded = json.loads(path.read_text())
        assert loaded["receipt_id"] == receipt["receipt_id"]
        assert loaded["signature"] == receipt["signature"]

    def test_atomic_write_no_tmp_leftover(
        self, engine: BridgeReceiptEngine, tmp_receipt_dir: Path
    ) -> None:
        engine.create_receipt(
            method="ping", query_data={}, result_data={},
            fate_score=1.0, snr_score=0.95, gate_passed="all", status="accepted",
        )
        tmp_files = list(tmp_receipt_dir.glob("*.tmp"))
        assert len(tmp_files) == 0, "Temp files should not remain after atomic write"

    def test_receipt_dir_auto_created(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "auto" / "created" / "receipts"
        assert not new_dir.exists()
        engine = BridgeReceiptEngine(receipt_dir=new_dir)
        assert new_dir.exists()
        # Still functional
        receipt = engine.create_receipt(
            method="test", query_data={}, result_data={},
            fate_score=1.0, snr_score=0.95, gate_passed="all", status="accepted",
        )
        assert (new_dir / f"{receipt['receipt_id']}.json").exists()


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


class TestRetrieval:
    def test_get_receipt_from_cache(self, engine: BridgeReceiptEngine) -> None:
        receipt = engine.create_receipt(
            method="ping", query_data={}, result_data={},
            fate_score=1.0, snr_score=0.95, gate_passed="all", status="accepted",
        )
        retrieved = engine.get_receipt(receipt["receipt_id"])
        assert retrieved is not None
        assert retrieved["receipt_id"] == receipt["receipt_id"]

    def test_get_receipt_from_disk(
        self, tmp_receipt_dir: Path
    ) -> None:
        # Create with one engine
        e1 = BridgeReceiptEngine(receipt_dir=tmp_receipt_dir)
        receipt = e1.create_receipt(
            method="ping", query_data={}, result_data={},
            fate_score=1.0, snr_score=0.95, gate_passed="all", status="accepted",
        )

        # Retrieve with a fresh engine (empty cache)
        e2 = BridgeReceiptEngine(receipt_dir=tmp_receipt_dir)
        retrieved = e2.get_receipt(receipt["receipt_id"])
        assert retrieved is not None
        assert retrieved["receipt_id"] == receipt["receipt_id"]

    def test_get_receipt_not_found(self, engine: BridgeReceiptEngine) -> None:
        assert engine.get_receipt("br-nonexistent-0001-ping") is None

    def test_cache_eviction(self, tmp_receipt_dir: Path) -> None:
        """Cache evicts oldest when over _MAX_CACHE."""
        from core.bridges.bridge_receipt import _MAX_CACHE

        engine = BridgeReceiptEngine(receipt_dir=tmp_receipt_dir)
        ids = []
        for i in range(_MAX_CACHE + 5):
            r = engine.create_receipt(
                method="ping", query_data={"i": i}, result_data={},
                fate_score=1.0, snr_score=0.95, gate_passed="all",
                status="accepted",
            )
            ids.append(r["receipt_id"])

        # First few should be evicted from cache
        assert ids[0] not in engine._cache
        # But still retrievable from disk
        assert engine.get_receipt(ids[0]) is not None
        # Latest should be in cache
        assert ids[-1] in engine._cache
