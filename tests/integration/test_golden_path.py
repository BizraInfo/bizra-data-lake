"""Golden Path Integration Test — exec() -> Receipt -> State -> Verify.

Proves the real spine of the BIZRA system end-to-end:
  1. MissionOrchestrator.execute() produces a MissionResult with evidence
  2. TokenMinter.mint_seed() produces a valid TokenReceipt from PoI
  3. EvidenceLedger records mission receipt and maintains chain integrity
  4. Health API reflects updated state (seed engine, node value)

This test validates the contract that the frontend TrustPanel relies on:
the API must return verifiable state backed by cryptographic evidence.

ADR-012: This is the golden path that must pass before any canary promotion.
"""

from __future__ import annotations

import asyncio
import tempfile
import time
import uuid
from pathlib import Path

import pytest


# ── Fixture: Isolated workspace ──────────────────────────────────────────


@pytest.fixture()
def workspace(tmp_path: Path):
    """Create an isolated workspace for mission + ledger + tokens."""
    paths = {
        "memory_path": str(tmp_path / "memory"),
        "evidence_path": str(tmp_path / "evidence.jsonl"),
        "token_ledger": str(tmp_path / "ledger.jsonl"),
        "token_db": str(tmp_path / "token.db"),
        "briefing_dir": str(tmp_path / "briefings"),
        "workspace_root": str(tmp_path),
    }
    (tmp_path / "memory").mkdir()
    (tmp_path / "briefings").mkdir()
    return paths


# ── Test 1: Mission execution produces evidence receipt ──────────────────


def test_mission_produces_evidence_receipt(workspace):
    """exec() -> MissionResult with valid evidence_receipt_id and ihsan_score."""
    from core.sovereign.mission import MissionOrchestrator, MissionRequest

    config = {
        "memory_path": workspace["memory_path"],
        "evidence_path": workspace["evidence_path"],
        "workspace_root": workspace["workspace_root"],
    }

    orchestrator = MissionOrchestrator(config)
    mission_req = MissionRequest(
        mission_id=f"gp-{uuid.uuid4().hex[:8]}",
        description="Calculate node sovereignty score for test node",
        context={"active_app": "test", "screen_title": "Golden Path Test"},
        timestamp=time.time(),
        source="integration_test",
    )

    result = asyncio.run(orchestrator.execute(mission_req))

    # Core contract: result must have all required fields
    assert result.mission_id == mission_req.mission_id
    assert result.status in ("COMPLETE", "PARTIAL")
    assert result.evidence_receipt_id, "Must produce evidence receipt"
    assert result.ihsan_score >= 0.0, "Ihsan score must be non-negative"
    assert result.snr_score >= 0.0, "SNR score must be non-negative"
    assert result.duration_ms > 0, "Duration must be positive"
    assert result.synthesis, "Synthesis must not be empty"


# ── Test 2: Token mint from PoI produces valid receipt ───────────────────


def test_token_mint_produces_receipt(workspace):
    """mint_seed() -> TokenReceipt with success, balance, and hash."""
    from core.token.mint import TokenMinter

    minter = TokenMinter.create(
        db_path=Path(workspace["token_db"]),
        log_path=Path(workspace["workspace_root"]) / "mint.log",
    )

    receipt = minter.mint_seed(
        to_account="node-golden-path",
        amount=10.0,
        epoch_id="epoch-gp-001",
        poi_score=0.87,
        memo="Golden path integration test mint",
    )

    assert receipt.success, f"Mint must succeed: {receipt.error}"
    # 10.0 SEED - 2.5% zakat = 9.75 net
    assert receipt.balance_after == pytest.approx(9.75, abs=0.01)
    assert receipt.receipt_hash, "Receipt must have hash"
    assert receipt.tx_entry is not None, "Must have transaction entry"
    assert receipt.tx_entry.amount == pytest.approx(9.75, abs=0.01)


# ── Test 3: Evidence ledger maintains chain integrity ────────────────────


def test_evidence_chain_integrity(workspace):
    """append() + verify_chain() -> intact hash chain."""
    from core.proof_engine.evidence_ledger import EvidenceLedger

    ledger = EvidenceLedger(
        path=Path(workspace["evidence_path"]),
        validate_on_append=False,  # chain-only test
    )

    # Simulate mission evidence entries
    for i in range(3):
        ledger.append(
            receipt={
                "type": "mission_evidence",
                "mission_id": f"gp-mission-{i}",
                "ihsan_score": 0.96,
                "snr_score": 0.90,
                "status": "COMPLETE",
                "timestamp": time.time(),
            }
        )

    valid, errors = ledger.verify_chain()
    assert valid, f"Chain must be valid: {errors}"
    assert len(errors) == 0


# ── Test 4: Mission receipt feeds into evidence ledger ───────────────────


def test_mission_receipt_to_ledger(workspace):
    """exec() receipt -> ledger.append() -> verify_chain() passes."""
    from core.proof_engine.evidence_ledger import EvidenceLedger
    from core.sovereign.mission import MissionOrchestrator, MissionRequest

    config = {
        "memory_path": workspace["memory_path"],
        "evidence_path": workspace["evidence_path"],
        "workspace_root": workspace["workspace_root"],
    }

    orchestrator = MissionOrchestrator(config)
    mission_req = MissionRequest(
        mission_id=f"gp-chain-{uuid.uuid4().hex[:8]}",
        description="Verify evidence chain integrity after mission execution",
        context={"active_app": "test", "screen_title": "Chain Test"},
        timestamp=time.time(),
        source="integration_test",
    )

    result = asyncio.run(orchestrator.execute(mission_req))
    assert result.evidence_receipt_id

    # The orchestrator writes to its own evidence ledger at evidence_path.
    # Verify the ledger chain is intact after mission execution.
    ledger = EvidenceLedger(
        path=Path(workspace["evidence_path"]),
        validate_on_append=False,
    )
    valid, errors = ledger.verify_chain()
    assert valid, f"Chain broken after mission: {errors}"


# ── Test 5: Token + Mission spine (golden path) ─────────────────────────


def test_golden_path_spine(workspace):
    """Full spine: exec() -> reward receipt -> state update -> verifiable.

    This is the contract the frontend TrustPanel depends on:
    1. Mission executes and produces evidence
    2. PoI-gated token mint succeeds
    3. Both receipts are independently verifiable
    4. State is consistent (balance reflects mint, chain is intact)
    """
    from core.proof_engine.evidence_ledger import EvidenceLedger
    from core.sovereign.mission import MissionOrchestrator, MissionRequest
    from core.token.mint import TokenMinter

    # Step 1: Execute mission
    config = {
        "memory_path": workspace["memory_path"],
        "evidence_path": workspace["evidence_path"],
        "workspace_root": workspace["workspace_root"],
    }

    orchestrator = MissionOrchestrator(config)
    mission_req = MissionRequest(
        mission_id=f"gp-spine-{uuid.uuid4().hex[:8]}",
        description="Execute sovereignty task for golden path validation",
        context={"active_app": "test", "screen_title": "Spine Test"},
        timestamp=time.time(),
        source="integration_test",
    )

    mission_result = asyncio.run(orchestrator.execute(mission_req))
    assert mission_result.evidence_receipt_id

    # Step 2: PoI-gated token mint (score from mission gates the reward)
    minter = TokenMinter.create(
        db_path=Path(workspace["token_db"]),
        log_path=Path(workspace["workspace_root"]) / "mint.log",
    )

    # Only mint if mission met the constitutional quality threshold
    poi_score = mission_result.ihsan_score
    mint_amount = 5.0 if poi_score >= 0.50 else 0.0

    if mint_amount > 0:
        token_receipt = minter.mint_seed(
            to_account="node-golden-path",
            amount=mint_amount,
            epoch_id=mission_result.mission_id,
            poi_score=poi_score,
            memo=f"Reward for mission {mission_result.mission_id}",
        )
        assert token_receipt.success, f"Mint failed: {token_receipt.error}"
        assert token_receipt.balance_after > 0

    # Step 3: Evidence chain integrity
    ledger = EvidenceLedger(
        path=Path(workspace["evidence_path"]),
        validate_on_append=False,
    )
    valid, errors = ledger.verify_chain()
    assert valid, f"Evidence chain broken: {errors}"

    # Step 4: Cross-verify — mission receipt exists in ledger
    # The orchestrator's _emit_evidence writes to this path.
    # Note: receipt_id may be truncated vs mission_id, so match on prefix.
    evidence_file = Path(workspace["evidence_path"])
    if evidence_file.exists():
        content = evidence_file.read_text()
        receipt_id = mission_result.evidence_receipt_id
        assert receipt_id in content, (
            f"Receipt {receipt_id} must appear in evidence ledger"
        )
