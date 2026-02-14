"""
7-Layer Consciousness Stack Integration Smoke Test
====================================================
Exercises all 7 layers of the BIZRA DDAGI OS stack end-to-end:

  L2 SIC  : Sovereign Identity Core (Ed25519 unified signer)
  L3 PCE  : Proof-Carrying Execution (SovereignRuntime, GoT, InferenceGateway)
  L4 CPM  : Consensus & P2P Mesh (Gossip, ConsensusEngine, Federation)
  L5 TFSC : Task/Feedback/Swarm Coordination (A2A, Orchestrator)
  L6 WAN  : World-Aware Neocortex (KnowledgeIntegrator, SNRMaximizer)
  L7 MCG  : Meta-Cognitive Governance (IhsanGate, IhsanFloorWatchdog)

Cross-layer: Ed25519 signer flows from L2 into L3/L7; IhsanFloorWatchdog
enforces IHSAN_FLOOR invariant; GoT artifacts feed evidence ledger.

Standing on Giants:
- Bernstein (Ed25519, 2011): Sovereign identity
- Besta (GoT, 2024): Graph reasoning artifacts
- Shannon (1948): SNR scoring
- Lamport (fail-closed): IHSAN_FLOOR watchdog
- Castro & Liskov (PBFT, 1999): Consensus engine
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest


def _test_policy():
    """Create a minimal CanonPolicy for testing."""
    from core.proof_engine.canonical import CanonPolicy

    return CanonPolicy(
        policy_id="test-policy-v1",
        version="1.0.0",
        rules={"ihsan_gate": True, "snr_gate": True},
        thresholds={"snr": 0.85, "ihsan": 0.95},
    )


def _hex_id(tag: str, length: int = 32) -> str:
    """Generate a valid hex ID from a tag string."""
    return hashlib.sha256(tag.encode()).hexdigest()[:length]


# ===========================================================================
# LAYER 2: Sovereign Identity Core (SIC)
# ===========================================================================


class TestLayer2SIC:
    """L2: Ed25519 identity, unified signer, key generation."""

    def test_ed25519_signer_creation(self):
        """Ed25519Signer generates a valid keypair on construction."""
        from core.proof_engine.receipt import Ed25519Signer

        signer = Ed25519Signer.generate()
        assert signer.public_key_hex is not None
        assert len(signer.public_key_hex) == 64  # 32 bytes = 64 hex chars

    def test_ed25519_signer_sign_verify(self):
        """Ed25519Signer can sign and verify messages."""
        from core.proof_engine.receipt import Ed25519Signer

        signer = Ed25519Signer.generate()
        msg = b"sovereignty integrity check"
        sig = signer.sign(msg)
        assert len(sig) > 0
        assert signer.verify(msg, sig) is True

    def test_ed25519_signer_tamper_detection(self):
        """Ed25519Signer detects tampered messages."""
        from core.proof_engine.receipt import Ed25519Signer

        signer = Ed25519Signer.generate()
        msg = b"original message"
        sig = signer.sign(msg)
        assert signer.verify(b"tampered message", sig) is False

    def test_ed25519_signer_public_key_bytes(self):
        """Ed25519Signer public_key_bytes matches hex representation."""
        from core.proof_engine.receipt import Ed25519Signer

        signer = Ed25519Signer.generate()
        pk_bytes = signer.public_key_bytes()
        assert pk_bytes == bytes.fromhex(signer.public_key_hex)

    def test_ed25519_signer_deterministic_with_key(self):
        """Ed25519Signer with explicit keys is deterministic."""
        from core.pci.crypto import generate_keypair
        from core.proof_engine.receipt import Ed25519Signer

        priv, pub = generate_keypair()
        s1 = Ed25519Signer(private_key_hex=priv, public_key_hex=pub)
        s2 = Ed25519Signer(private_key_hex=priv, public_key_hex=pub)
        assert s1.public_key_hex == s2.public_key_hex

        msg = b"test"
        sig1 = s1.sign(msg)
        assert s2.verify(msg, sig1) is True

    def test_receipt_sign_with_ed25519(self):
        """Receipt can be signed and verified with Ed25519Signer."""
        from core.proof_engine.canonical import CanonQuery
        from core.proof_engine.receipt import Ed25519Signer, ReceiptBuilder

        signer = Ed25519Signer.generate()
        builder = ReceiptBuilder(signer)

        query = CanonQuery(user_id="test-user", user_state="active", intent="test query")
        policy = _test_policy()

        receipt = builder.accepted(
            query=query,
            policy=policy,
            payload=b"test payload",
            snr=0.92,
            ihsan_score=0.96,
        )
        assert receipt.verify_signature(signer) is True

    def test_receipt_cross_signer_fails(self):
        """Receipt signed with one signer fails with another."""
        from core.proof_engine.canonical import CanonQuery
        from core.proof_engine.receipt import Ed25519Signer, ReceiptBuilder

        signer1 = Ed25519Signer.generate()
        signer2 = Ed25519Signer.generate()

        builder = ReceiptBuilder(signer1)
        query = CanonQuery(user_id="test-user", user_state="active", intent="test")
        policy = _test_policy()

        receipt = builder.accepted(
            query=query, policy=policy, payload=b"data",
            snr=0.90, ihsan_score=0.95,
        )
        # Verify with different signer should fail
        assert receipt.verify_signature(signer2) is False


# ===========================================================================
# LAYER 3: Proof-Carrying Execution (PCE)
# ===========================================================================


class TestLayer3PCE:
    """L3: SovereignRuntime, GoT, Gate Chain."""

    def test_graph_of_thoughts_creation(self):
        """GoT can be created and produces thoughts."""
        from core.sovereign.graph_core import GraphOfThoughts
        from core.sovereign.graph_types import ThoughtType

        got = GraphOfThoughts()
        t = got.add_thought("What is sovereignty?", ThoughtType.QUESTION)
        assert t is not None
        assert t.content == "What is sovereignty?"

    def test_graph_hash_determinism(self):
        """GoT graph hash is deterministic."""
        from core.sovereign.graph_core import GraphOfThoughts
        from core.sovereign.graph_types import EdgeType, ThoughtType

        got = GraphOfThoughts()
        q = got.add_thought("Query", ThoughtType.QUESTION)
        h = got.add_thought("Hypothesis", ThoughtType.HYPOTHESIS, parent_id=q.id)
        e = got.add_thought("Evidence", ThoughtType.EVIDENCE, parent_id=h.id)
        got.add_edge(e.id, h.id, EdgeType.SUPPORTS)

        h1 = got.compute_graph_hash()
        h2 = got.compute_graph_hash()
        assert h1 == h2
        assert len(h1) == 64

    def test_graph_artifact_schema(self):
        """GoT artifact has the required schema fields."""
        from core.sovereign.graph_core import GraphOfThoughts
        from core.sovereign.graph_types import ThoughtType

        got = GraphOfThoughts()
        got.add_thought("Test", ThoughtType.QUESTION)
        artifact = got.to_artifact(build_id="l3-test")

        assert "graph_hash" in artifact
        assert "nodes" in artifact
        assert "edges" in artifact
        assert "build_id" in artifact

    def test_graph_signing_with_ed25519(self):
        """GoT graph can be signed with Ed25519."""
        from core.pci.crypto import generate_keypair
        from core.sovereign.graph_core import GraphOfThoughts
        from core.sovereign.graph_types import ThoughtType

        got = GraphOfThoughts()
        got.add_thought("Signed thought", ThoughtType.EVIDENCE)
        priv, pub = generate_keypair()
        sig = got.sign_graph(priv)
        assert sig is not None
        assert len(sig) == 128  # 64 bytes = 128 hex

    def test_gate_chain_schema_to_commit(self):
        """6-gate chain can be loaded and has the right gate count."""
        from core.proof_engine.gates import (
            CommitGate,
            ConstraintGate,
            ProvenanceGate,
            SafetyGate,
            SchemaGate,
            SNRGate,
        )

        gates = [SchemaGate, ProvenanceGate, SNRGate, ConstraintGate, SafetyGate, CommitGate]
        assert len(gates) == 6
        for gate_cls in gates:
            assert hasattr(gate_cls, "evaluate")

    def test_snr_maximizer_analysis(self):
        """SNRMaximizer produces valid analysis."""
        from core.sovereign.snr_maximizer import SNRMaximizer

        snr = SNRMaximizer()
        analysis = snr.analyze("This is a well-grounded factual statement with evidence.")
        assert analysis.snr_linear >= 0.0
        assert isinstance(analysis.ihsan_achieved, bool)


# ===========================================================================
# LAYER 4: Consensus & P2P Mesh (CPM)
# ===========================================================================


class TestLayer4CPM:
    """L4: Gossip, Consensus, Federation."""

    def test_consensus_engine_creation(self):
        """ConsensusEngine can be instantiated with Ed25519 keys."""
        from core.federation.consensus import ConsensusEngine
        from core.pci.crypto import generate_keypair

        priv, pub = generate_keypair()
        engine = ConsensusEngine(node_id="test-node-L4", private_key=priv, public_key=pub)
        assert engine is not None
        assert engine.node_id == "test-node-L4"

    def test_consensus_phases_defined(self):
        """ConsensusPhase enum has expected phases."""
        from core.federation.consensus import ConsensusPhase

        phases = [p.name for p in ConsensusPhase]
        assert "PRE_PREPARE" in phases or "IDLE" in phases

    def test_propagation_engine_exists(self):
        """PropagationEngine subsystem exists."""
        from core.federation.propagation import PropagationEngine

        assert PropagationEngine is not None


# ===========================================================================
# LAYER 5: Task/Feedback/Swarm Coordination (TFSC)
# ===========================================================================


class TestLayer5TFSC:
    """L5: A2A protocol, AgentCard, SovereignOrchestrator."""

    def test_agent_card_schema(self):
        """AgentCard has required fields."""
        from core.a2a.schema import AgentCard

        card = AgentCard(
            agent_id="test-agent",
            name="Test Agent",
            description="A test agent for L5 integration",
        )
        assert card.agent_id == "test-agent"
        assert card.description == "A test agent for L5 integration"

    def test_a2a_engine_creation(self):
        """A2A Engine can be instantiated with agent card and key."""
        from core.a2a.engine import A2AEngine
        from core.a2a.schema import AgentCard
        from core.pci.crypto import generate_keypair

        priv, pub = generate_keypair()
        card = AgentCard(
            agent_id="l5-test",
            name="L5 Test",
            description="Layer 5 integration test",
            public_key=pub,
        )
        engine = A2AEngine(agent_card=card, private_key=priv)
        assert engine is not None

    def test_sovereign_orchestrator_exists(self):
        """SovereignOrchestrator is importable."""
        from core.sovereign.orchestrator import SovereignOrchestrator

        assert SovereignOrchestrator is not None


# ===========================================================================
# LAYER 6: World-Aware Neocortex (WAN)
# ===========================================================================


class TestLayer6WAN:
    """L6: KnowledgeIntegrator, SNR protocol facade."""

    def test_knowledge_integrator_exists(self):
        """KnowledgeIntegrator is importable."""
        from core.bridges.knowledge_integrator import KnowledgeIntegrator

        assert KnowledgeIntegrator is not None

    def test_snr_facade_normalized_output(self):
        """SNR facade produces [0,1]-clamped output."""
        from core.snr_protocol import SNRFacade

        facade = SNRFacade()
        result = facade.calculate(text="A verified claim with supporting evidence.")
        assert 0.0 <= result.score <= 1.0


# ===========================================================================
# LAYER 7: Meta-Cognitive Governance (MCG)
# ===========================================================================


class TestLayer7MCG:
    """L7: IhsanGate, IhsanFloorWatchdog, constitutional validation."""

    def test_ihsan_gate_approve(self):
        """IhsanGate approves high-quality components."""
        from core.proof_engine.ihsan_gate import IhsanComponents, IhsanGate

        gate = IhsanGate(threshold=0.90)
        components = IhsanComponents(
            correctness=0.95, safety=0.98, efficiency=0.92, user_benefit=0.94,
        )
        result = gate.evaluate(components)
        assert result.decision == "APPROVED"
        assert result.score >= 0.90

    def test_ihsan_gate_reject(self):
        """IhsanGate rejects low-quality components."""
        from core.proof_engine.ihsan_gate import IhsanComponents, IhsanGate

        gate = IhsanGate(threshold=0.95)
        components = IhsanComponents(
            correctness=0.40, safety=0.50, efficiency=0.30, user_benefit=0.40,
        )
        result = gate.evaluate(components)
        assert result.decision == "REJECTED"
        assert "IHSAN_BELOW_THRESHOLD" in result.reason_codes

    def test_ihsan_watchdog_healthy_path(self):
        """Watchdog stays healthy when scores are above floor."""
        from core.proof_engine.ihsan_gate import IhsanFloorWatchdog

        wd = IhsanFloorWatchdog(max_consecutive_failures=3, floor=0.90)
        assert wd.is_degraded is False

        # Record good scores
        assert wd.record(0.95) is True
        assert wd.record(0.92) is True
        assert wd.record(0.91) is True
        assert wd.is_degraded is False

    def test_ihsan_watchdog_degraded_path(self):
        """Watchdog enters DEGRADED after max consecutive failures."""
        from core.proof_engine.ihsan_gate import IhsanFloorWatchdog

        wd = IhsanFloorWatchdog(max_consecutive_failures=3, floor=0.90)

        # 3 consecutive failures
        wd.record(0.80)
        wd.record(0.70)
        result = wd.record(0.60)
        assert result is False
        assert wd.is_degraded is True

    def test_ihsan_watchdog_recovery(self):
        """Watchdog recovers after reset."""
        from core.proof_engine.ihsan_gate import IhsanFloorWatchdog

        wd = IhsanFloorWatchdog(max_consecutive_failures=2, floor=0.90)
        wd.record(0.50)
        wd.record(0.50)
        assert wd.is_degraded is True

        wd.reset()
        assert wd.is_degraded is False
        assert wd.consecutive_failures == 0

    def test_ihsan_watchdog_reset_by_good_score(self):
        """A good score resets consecutive failure count (but not degraded state)."""
        from core.proof_engine.ihsan_gate import IhsanFloorWatchdog

        wd = IhsanFloorWatchdog(max_consecutive_failures=3, floor=0.90)
        wd.record(0.80)  # fail
        wd.record(0.70)  # fail
        assert wd.consecutive_failures == 2

        wd.record(0.95)  # good — resets consecutive counter
        assert wd.consecutive_failures == 0

    def test_ihsan_watchdog_status_dict(self):
        """Watchdog status returns expected schema."""
        from core.proof_engine.ihsan_gate import IhsanFloorWatchdog

        wd = IhsanFloorWatchdog(max_consecutive_failures=3, floor=0.90)
        wd.record(0.95)
        wd.record(0.80)
        status = wd.status()

        assert status["degraded"] is False
        assert status["consecutive_failures"] == 1
        assert status["total_evaluations"] == 2
        assert status["total_failures"] == 1
        assert status["floor"] == 0.90

    def test_ihsan_score_receipt_shape(self):
        """ihsan_score() returns receipt-compatible dict."""
        from core.proof_engine.ihsan_gate import IhsanComponents, IhsanGate

        gate = IhsanGate(threshold=0.95)
        components = IhsanComponents(
            correctness=0.96, safety=0.98, efficiency=0.94, user_benefit=0.95,
        )
        result = gate.ihsan_score(components)

        assert "score" in result
        assert "threshold" in result
        assert "decision" in result
        assert "passed" in result
        assert "reason_codes" in result
        assert "components" in result
        assert "version" in result


# ===========================================================================
# CROSS-LAYER: L2→L3 (Ed25519 flows into Gate Chain + Receipt)
# ===========================================================================


class TestCrossLayerL2L3:
    """Cross-layer: Ed25519 signer flows from identity into execution."""

    def test_ed25519_receipt_sign_verify_chain(self):
        """Full chain: generate key → build receipt → sign → verify."""
        from core.proof_engine.canonical import CanonQuery
        from core.proof_engine.receipt import Ed25519Signer, ReceiptBuilder, ReceiptVerifier

        signer = Ed25519Signer.generate()
        builder = ReceiptBuilder(signer)
        verifier = ReceiptVerifier(signer)

        query = CanonQuery(user_id="test-user", user_state="active", intent="cross-layer test")
        policy = _test_policy()

        receipt = builder.accepted(
            query=query, policy=policy, payload=b"verified output",
            snr=0.93, ihsan_score=0.97,
        )
        valid, error = verifier.verify(receipt)
        assert valid is True
        assert error is None

    def test_rejected_receipt_with_ed25519(self):
        """Rejection receipts are also properly signed."""
        from core.proof_engine.canonical import CanonQuery
        from core.proof_engine.receipt import Ed25519Signer, ReceiptBuilder, ReceiptVerifier

        signer = Ed25519Signer.generate()
        builder = ReceiptBuilder(signer)
        verifier = ReceiptVerifier(signer)

        query = CanonQuery(user_id="test-user", user_state="active", intent="rejected query")
        policy = _test_policy()

        receipt = builder.rejected(
            query=query, policy=policy,
            snr=0.40, ihsan_score=0.50,
            gate_failed="snr", reason="SNR_BELOW_THRESHOLD",
        )
        valid, error = verifier.verify(receipt)
        assert valid is True
        assert receipt.status.value == "rejected"


# ===========================================================================
# CROSS-LAYER: L3→L7 (GoT + Evidence → IhsanGate → Watchdog)
# ===========================================================================


class TestCrossLayerL3L7:
    """Cross-layer: GoT results feed IhsanGate, watchdog monitors."""

    def test_got_evidence_ihsan_pipeline(self):
        """GoT → evidence hash → Ihsan evaluation → watchdog record."""
        from core.proof_engine.ihsan_gate import (
            IhsanComponents,
            IhsanFloorWatchdog,
            IhsanGate,
        )
        from core.sovereign.graph_core import GraphOfThoughts
        from core.sovereign.graph_types import EdgeType, ThoughtType

        # L3: Build GoT
        got = GraphOfThoughts()
        q = got.add_thought("What is truth?", ThoughtType.QUESTION)
        h = got.add_thought("Truth is verifiable.", ThoughtType.HYPOTHESIS, parent_id=q.id)
        e = got.add_thought("Cryptographic proof.", ThoughtType.EVIDENCE, parent_id=h.id)
        got.add_edge(e.id, h.id, EdgeType.SUPPORTS)
        graph_hash = got.compute_graph_hash()
        assert len(graph_hash) == 64

        # L7: Evaluate with IhsanGate
        gate = IhsanGate(threshold=0.90)
        components = IhsanComponents(
            correctness=0.95, safety=0.97, efficiency=0.91, user_benefit=0.93,
        )
        result = gate.evaluate(components)
        assert result.decision == "APPROVED"

        # L7: Record in watchdog
        watchdog = IhsanFloorWatchdog(max_consecutive_failures=3, floor=0.90)
        healthy = watchdog.record(result.score)
        assert healthy is True
        assert watchdog.is_degraded is False

    def test_watchdog_degrades_on_repeated_failure(self):
        """Repeated low-quality GoT outputs trigger watchdog degradation."""
        from core.proof_engine.ihsan_gate import (
            IhsanComponents,
            IhsanFloorWatchdog,
            IhsanGate,
        )

        gate = IhsanGate(threshold=0.95)
        watchdog = IhsanFloorWatchdog(max_consecutive_failures=3, floor=0.90)

        # Simulate 3 consecutive low-quality outputs
        low_components = IhsanComponents(
            correctness=0.30, safety=0.40, efficiency=0.20, user_benefit=0.30,
        )
        for _ in range(3):
            result = gate.evaluate(low_components)
            watchdog.record(result.score)

        assert watchdog.is_degraded is True
        assert watchdog.consecutive_failures == 3


# ===========================================================================
# CROSS-LAYER: Full Stack Smoke (L2→L3→L7 with Evidence)
# ===========================================================================


class TestFullStackSmoke:
    """Full stack: Identity → Execution → Governance → Evidence chain."""

    def test_full_stack_approved_flow(self, tmp_path):
        """Full approved flow: keygen → GoT → Ihsan → Receipt → Evidence."""
        from core.proof_engine.canonical import CanonQuery
        from core.proof_engine.evidence_ledger import EvidenceLedger, emit_receipt
        from core.proof_engine.ihsan_gate import (
            IhsanComponents,
            IhsanFloorWatchdog,
            IhsanGate,
        )
        from core.proof_engine.receipt import Ed25519Signer, ReceiptBuilder
        from core.sovereign.graph_core import GraphOfThoughts
        from core.sovereign.graph_types import EdgeType, ThoughtType

        # L2: Generate identity
        signer = Ed25519Signer.generate()
        node_pubkey = signer.public_key_hex
        assert len(node_pubkey) == 64

        # L3: Build GoT reasoning graph
        got = GraphOfThoughts()
        q = got.add_thought("What is sovereignty?", ThoughtType.QUESTION)
        h = got.add_thought(
            "Self-determination in the digital age.", ThoughtType.HYPOTHESIS,
            parent_id=q.id,
        )
        e = got.add_thought(
            "Ed25519 cryptographic identity.", ThoughtType.EVIDENCE,
            parent_id=h.id,
        )
        got.add_edge(e.id, h.id, EdgeType.SUPPORTS)
        graph_hash = got.compute_graph_hash()

        # L7: Evaluate Ihsan
        gate = IhsanGate(threshold=0.90)
        components = IhsanComponents(
            correctness=0.96, safety=0.98, efficiency=0.93, user_benefit=0.95,
        )
        ihsan_result = gate.evaluate(components)
        assert ihsan_result.decision == "APPROVED"

        # L7: Record in watchdog
        watchdog = IhsanFloorWatchdog(max_consecutive_failures=3, floor=0.90)
        healthy = watchdog.record(ihsan_result.score)
        assert healthy is True

        # L3: Build signed receipt
        builder = ReceiptBuilder(signer)
        query = CanonQuery(user_id="node0", user_state="active", intent="What is sovereignty?")
        policy = _test_policy()
        receipt = builder.accepted(
            query=query, policy=policy, payload=b"Self-determination answer",
            snr=0.93, ihsan_score=ihsan_result.score,
        )
        assert receipt.verify_signature(signer) is True

        # Evidence chain: append to ledger
        ledger = EvidenceLedger(tmp_path / "evidence.jsonl")
        entry = emit_receipt(
            ledger,
            receipt_id=_hex_id("l2l3l7-approved"),
            node_id="BIZRA-NODE0-L2L3L7",
            policy_version="1.0.0",
            status="accepted",
            decision="APPROVED",
            snr_score=0.93,
            ihsan_score=ihsan_result.score,
            ihsan_threshold=0.90,
            seal_digest=receipt.hex_digest(),
            graph_hash=graph_hash,
        )
        assert entry.sequence == 1

        # Verify chain integrity
        is_valid, errors = ledger.verify_chain()
        assert is_valid is True
        assert errors == []

    def test_full_stack_rejected_flow(self, tmp_path):
        """Full rejected flow: low scores → reject receipt → evidence trail."""
        from core.proof_engine.canonical import CanonQuery
        from core.proof_engine.evidence_ledger import EvidenceLedger, emit_receipt
        from core.proof_engine.ihsan_gate import (
            IhsanComponents,
            IhsanFloorWatchdog,
            IhsanGate,
        )
        from core.proof_engine.receipt import Ed25519Signer, ReceiptBuilder

        # L2: Identity
        signer = Ed25519Signer.generate()

        # L7: Evaluate — should fail
        gate = IhsanGate(threshold=0.95)
        components = IhsanComponents(
            correctness=0.40, safety=0.50, efficiency=0.30, user_benefit=0.40,
        )
        ihsan_result = gate.evaluate(components)
        assert ihsan_result.decision == "REJECTED"

        # L7: Watchdog records failure
        watchdog = IhsanFloorWatchdog(max_consecutive_failures=3, floor=0.90)
        watchdog.record(ihsan_result.score)
        assert watchdog.consecutive_failures == 1

        # L3: Build rejection receipt
        builder = ReceiptBuilder(signer)
        query = CanonQuery(user_id="node0", user_state="active", intent="Low quality query")
        policy = _test_policy()
        receipt = builder.rejected(
            query=query, policy=policy,
            snr=0.30, ihsan_score=ihsan_result.score,
            gate_failed="ihsan", reason="IHSAN_BELOW_THRESHOLD",
        )
        assert receipt.verify_signature(signer) is True
        assert receipt.status.value == "rejected"

        # Evidence: rejection still gets recorded
        ledger = EvidenceLedger(tmp_path / "evidence.jsonl")
        entry = emit_receipt(
            ledger,
            receipt_id=_hex_id("rejected-flow"),
            node_id="BIZRA-NODE0-REJECTED",
            status="rejected",
            decision="REJECTED",
            reason_codes=["IHSAN_BELOW_THRESHOLD"],
            snr_score=0.30,
            ihsan_score=ihsan_result.score,
            ihsan_threshold=0.95,
            seal_digest=receipt.hex_digest(),
        )
        assert entry.sequence == 1

        is_valid, errors = ledger.verify_chain()
        assert is_valid is True

    def test_full_stack_multi_query_chain(self, tmp_path):
        """Multiple queries form a hash-chained evidence trail."""
        from core.proof_engine.canonical import CanonQuery
        from core.proof_engine.evidence_ledger import EvidenceLedger, emit_receipt
        from core.proof_engine.ihsan_gate import (
            IhsanComponents,
            IhsanFloorWatchdog,
            IhsanGate,
        )
        from core.proof_engine.receipt import Ed25519Signer, ReceiptBuilder

        signer = Ed25519Signer.generate()
        builder = ReceiptBuilder(signer)
        gate = IhsanGate(threshold=0.90)
        watchdog = IhsanFloorWatchdog(max_consecutive_failures=3, floor=0.90)
        ledger = EvidenceLedger(tmp_path / "evidence.jsonl")

        entries = []
        for i in range(5):
            query = CanonQuery(user_id="node0", user_state="active", intent=f"Query {i}")
            policy = _test_policy()

            components = IhsanComponents(
                correctness=0.95, safety=0.97,
                efficiency=0.92, user_benefit=0.94,
            )
            ihsan_result = gate.evaluate(components)
            watchdog.record(ihsan_result.score)

            receipt = builder.accepted(
                query=query, policy=policy, payload=f"Answer {i}".encode(),
                snr=0.92, ihsan_score=ihsan_result.score,
            )
            entry = emit_receipt(
                ledger,
                receipt_id=_hex_id(f"chain-query-{i}"),
                node_id="BIZRA-NODE0-CHAIN",
                seal_digest=receipt.hex_digest(),
                snr_score=0.92,
                ihsan_score=ihsan_result.score,
            )
            entries.append(entry)

        # Verify chain
        assert len(entries) == 5
        assert entries[-1].sequence == 5

        # Each entry links to previous
        for i in range(1, len(entries)):
            assert entries[i].prev_hash == entries[i - 1].entry_hash

        is_valid, errors = ledger.verify_chain()
        assert is_valid is True
        assert errors == []

        # Watchdog stayed healthy
        assert watchdog.is_degraded is False
        assert watchdog.status()["total_evaluations"] == 5

    def test_full_stack_watchdog_degradation(self, tmp_path):
        """Consecutive failures across queries trigger watchdog degradation."""
        from core.proof_engine.ihsan_gate import (
            IhsanComponents,
            IhsanFloorWatchdog,
            IhsanGate,
        )

        gate = IhsanGate(threshold=0.95)
        watchdog = IhsanFloorWatchdog(max_consecutive_failures=3, floor=0.90)

        # 3 consecutive bad queries
        low = IhsanComponents(
            correctness=0.30, safety=0.40, efficiency=0.20, user_benefit=0.25,
        )
        for _ in range(3):
            result = gate.evaluate(low)
            watchdog.record(result.score)

        assert watchdog.is_degraded is True

        # Good score doesn't auto-recover (needs explicit reset)
        high = IhsanComponents(
            correctness=0.98, safety=0.99, efficiency=0.96, user_benefit=0.97,
        )
        result = gate.evaluate(high)
        watchdog.record(result.score)
        assert watchdog.is_degraded is True  # Still degraded until reset

        # Explicit human reset
        watchdog.reset()
        assert watchdog.is_degraded is False
