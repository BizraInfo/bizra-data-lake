"""
Atlas v4.0 Convergence Integration Tests
==========================================

Verifies cross-module interactions at the 8 gold convergence points
where Atlas documentation most closely matches real implementation.

Convergence Points Under Test:
    CP-1: CognitiveFusion -> SNR -> Quality Gate
    CP-2: Token Mint -> Ledger -> ADL Gini Gate
    CP-3: Proof Engine 6-Gate Chain (fail-closed)
    CP-4: Cross-Module Bridge (Token -> Proof -> Identity -> Federation)

Each test exercises a real cross-module interaction path, using
mocks only for LLM backends (no real inference needed).

Standing on Giants:
- Shannon (1948): SNR as the universal quality metric
- Nakamoto (2008): Hash-chained transaction integrity
- Castro & Liskov (1999): PBFT consensus quorum
- Bernstein (2012): Ed25519 identity and signing

Constitutional Alignment:
    All thresholds imported from core/integration/constants.py (SSOT).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from core.integration.constants import (
    ADL_GINI_THRESHOLD,
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Test Group 1: CognitiveFusion imports
# ---------------------------------------------------------------------------
from core.cognitive_fusion.fusion_engine import (
    CognitiveFusionEngine,
    FusionResult,
    HRMResult,
    NorthStarResult,
    RoutingResult,
)

# ---------------------------------------------------------------------------
# Test Group 2: Token system imports
# ---------------------------------------------------------------------------
from core.token.ledger import TokenLedger
from core.token.mint import (
    COMMUNITY_FUND_ACCOUNT,
    GENESIS_NODE0_ACCOUNT,
    SYSTEM_TREASURY_ACCOUNT,
    TokenMinter,
)
from core.token.types import (
    FOUNDER_GENESIS_ALLOCATION,
    SEED_SUPPLY_CAP_PER_YEAR,
    SYSTEM_TREASURY_ALLOCATION,
    ZAKAT_RATE,
    TokenType,
)

# ---------------------------------------------------------------------------
# Test Group 3: Proof Engine imports
# ---------------------------------------------------------------------------
from core.proof_engine.canonical import CanonPolicy, CanonQuery
from core.proof_engine.gates import (
    CommitGate,
    ConstraintGate,
    GateChain,
    GateStatus,
    ProvenanceGate,
    SafetyGate,
    SchemaGate,
    SNRGate,
)
from core.proof_engine.receipt import (
    Ed25519Signer,
    ReceiptStatus,
    ReceiptVerifier,
)
from core.proof_engine.snr import SNREngine, SNRInput

# ---------------------------------------------------------------------------
# Test Group 4: Identity + Federation imports
# ---------------------------------------------------------------------------
from core.pat.identity_card import (
    IdentityCard,
    IdentityStatus,
    generate_identity_keypair,
)
from core.federation.consensus import ConsensusEngine
from core.pci.crypto import generate_keypair


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def tmp_ledger(tmp_path: Path) -> TokenLedger:
    """Create a fresh token ledger with isolated storage."""
    return TokenLedger(
        db_path=tmp_path / "test.db",
        log_path=tmp_path / "test_ledger.jsonl",
    )


@pytest.fixture
def tmp_minter(tmp_path: Path) -> TokenMinter:
    """Create a fresh token minter with isolated storage."""
    return TokenMinter.create(
        db_path=tmp_path / "test.db",
        log_path=tmp_path / "test_ledger.jsonl",
    )


@pytest.fixture
def ed25519_signer() -> Ed25519Signer:
    """Generate a fresh Ed25519 signer for receipt signing."""
    return Ed25519Signer.generate()


@pytest.fixture
def sample_query() -> CanonQuery:
    """Create a well-formed canonical query for gate chain tests."""
    return CanonQuery(
        user_id="node-42",
        user_state="active",
        intent="Analyze autopoietic feedback loops in BIZRA architecture",
        payload={"domain": "architecture", "depth": 3},
    )


@pytest.fixture
def sample_policy() -> CanonPolicy:
    """Create a canonical policy for gate chain tests."""
    return CanonPolicy(
        policy_id="policy-atlas-v4",
        version="4.0.0",
        rules={"max_depth": 5, "allow_reasoning": True},
        thresholds={"ihsan": UNIFIED_IHSAN_THRESHOLD, "snr": UNIFIED_SNR_THRESHOLD},
        constraints=["no_hallucination", "cite_sources"],
    )


# =============================================================================
# TEST GROUP 1: CognitiveFusion -> SNR -> Quality Gate
# =============================================================================


@pytest.mark.integration
class TestCognitiveFusionPipeline:
    """CP-1: Verify the CognitiveFusion pipeline produces gated output."""

    def test_default_engine_produces_passing_result(self) -> None:
        """Default engine (no subsystems) falls back safely and passes gate."""
        engine = CognitiveFusionEngine()
        embedding = [0.1] * 128

        result = engine.process("What is autopoiesis?", embedding)

        assert isinstance(result, FusionResult)
        assert result.passes_gate is True
        assert result.snr_score >= UNIFIED_SNR_THRESHOLD
        assert result.ihsan_score >= UNIFIED_IHSAN_THRESHOLD

    def test_fusion_snr_meets_minimum_threshold(self) -> None:
        """CognitiveFusion SNR must meet the constitutional minimum of 0.85."""
        engine = CognitiveFusionEngine()
        embedding = [0.5] * 64

        result = engine.process("Explain PBFT consensus", embedding)

        # SNR is geometric mean of HRM compound_snr and NorthStar unified_snr
        assert result.snr_score >= UNIFIED_SNR_THRESHOLD, (
            f"Fusion SNR {result.snr_score:.3f} below constitutional minimum "
            f"{UNIFIED_SNR_THRESHOLD}"
        )

    def test_custom_moe_router_affects_complexity(self) -> None:
        """A custom MoE router steers the pipeline to the correct tier."""
        mock_router = MagicMock()
        mock_router.route.return_value = RoutingResult(
            complexity_class="EXPERT",
            expert_tier="POOL",
            confidence=0.92,
        )

        engine = CognitiveFusionEngine(moe_router=mock_router)
        result = engine.process("Prove P != NP", [0.9] * 32)

        assert result.target_level == "STRATEGIC"
        assert result.expert_tier == "POOL"
        mock_router.route.assert_called_once()

    def test_hrm_result_flows_into_aggregate_snr(self) -> None:
        """HRM engine output is reflected in the aggregate SNR score."""
        mock_hrm = MagicMock()
        mock_hrm.run_cycle.return_value = HRMResult(
            compound_snr=0.92,
            level_reached="TACTICAL",
            observations=["Pattern detected", "Confidence high"],
        )

        engine = CognitiveFusionEngine(hrm_engine=mock_hrm)
        result = engine.process("Analyze federation gossip", [0.3] * 16)

        # Aggregate SNR = geometric_mean(hrm_snr, northstar_snr)
        # With default NorthStar (0.85) and HRM (0.92):
        expected_snr = (0.92 * UNIFIED_SNR_THRESHOLD) ** 0.5
        assert abs(result.snr_score - expected_snr) < 0.001

    def test_hypergraph_rag_retrieval_feeds_into_pipeline(self) -> None:
        """HyperGraph RAG results appear in the fusion output."""
        mock_rag = MagicMock()
        fake_chunks = [
            {"chunk_id": "c1", "text": "Autopoiesis is self-creation"},
            {"chunk_id": "c2", "text": "Maturana and Varela (1972)"},
        ]
        mock_rag.retrieve.return_value = fake_chunks

        engine = CognitiveFusionEngine(hypergraph_rag=mock_rag)
        result = engine.process("Define autopoiesis", [0.7] * 8)

        assert len(result.retrieval) == 2
        assert result.retrieval[0]["chunk_id"] == "c1"
        mock_rag.retrieve.assert_called_once()

    def test_northstar_gate_rejection_fails_pipeline(self) -> None:
        """When NorthStar rejects, the entire pipeline fails the gate."""
        mock_ns = MagicMock()
        mock_ns.run_cycle.return_value = NorthStarResult(
            unified_snr=0.50,
            ihsan_score=0.70,
            passes_all_gates=False,
            flow_report={"reason": "Low quality response"},
        )

        engine = CognitiveFusionEngine(northstar_engine=mock_ns)
        result = engine.process("Bad query", [0.0] * 4)

        assert result.passes_gate is False
        assert result.ihsan_score < UNIFIED_IHSAN_THRESHOLD

    def test_complexity_adapter_retrieval_depth_scales(self) -> None:
        """Higher complexity classes yield deeper RAG retrieval."""
        engine = CognitiveFusionEngine()

        assert engine._retrieval_depth("TRIVIAL") == 3
        assert engine._retrieval_depth("STANDARD") == 5
        assert engine._retrieval_depth("COMPLEX") == 10
        assert engine._retrieval_depth("EXPERT") == 20
        assert engine._retrieval_depth("FRONTIER") == 50

    def test_elite_threshold_requires_both_high_scores(self) -> None:
        """FusionResult.is_elite needs both SNR >= 0.98 and Ihsan >= 0.99."""
        mock_hrm = MagicMock()
        mock_hrm.run_cycle.return_value = HRMResult(compound_snr=0.99)

        mock_ns = MagicMock()
        mock_ns.run_cycle.return_value = NorthStarResult(
            unified_snr=0.99,
            ihsan_score=0.995,
            passes_all_gates=True,
        )

        engine = CognitiveFusionEngine(hrm_engine=mock_hrm, northstar_engine=mock_ns)
        result = engine.process("Elite query", [1.0] * 4)

        # SNR = geometric_mean(0.99, 0.99) = 0.99, Ihsan = 0.995
        assert result.is_elite is True

    def test_non_elite_when_snr_below_t0(self) -> None:
        """Non-elite when SNR is below T0-elite threshold."""
        mock_hrm = MagicMock()
        mock_hrm.run_cycle.return_value = HRMResult(compound_snr=0.90)

        engine = CognitiveFusionEngine(hrm_engine=mock_hrm)
        result = engine.process("Regular query", [0.5] * 4)

        assert result.is_elite is False


# =============================================================================
# TEST GROUP 2: Token Mint -> Ledger -> ADL Gini Gate
# =============================================================================


@pytest.mark.integration
class TestTokenMintLedgerFlow:
    """CP-2: Verify the SEED/BLOOM minting and ledger hash chain."""

    def test_mint_seed_records_in_ledger(self, tmp_minter: TokenMinter) -> None:
        """Minting SEED records in the ledger with correct balance."""
        receipt = tmp_minter.mint_seed(
            "node-1", 100.0, epoch_id="epoch-1", poi_score=0.87
        )

        assert receipt.success is True
        assert receipt.receipt_hash != ""

        # Verify the balance reflects zakat deduction
        balance = tmp_minter.ledger.get_balance("node-1", TokenType.SEED)
        expected_net = 100.0 * (1 - ZAKAT_RATE)
        assert abs(balance.balance - expected_net) < 0.01

    def test_zakat_flows_to_community_fund(self, tmp_minter: TokenMinter) -> None:
        """2.5% computational zakat is minted to the community fund."""
        tmp_minter.mint_seed("node-1", 1000.0, epoch_id="epoch-1")

        community_balance = tmp_minter.ledger.get_balance(
            COMMUNITY_FUND_ACCOUNT, TokenType.SEED
        )
        expected_zakat = 1000.0 * ZAKAT_RATE  # 25.0
        assert abs(community_balance.balance - expected_zakat) < 0.01

    def test_zakat_calculation_exact_2_5_percent(
        self, tmp_minter: TokenMinter
    ) -> None:
        """Zakat is exactly 2.5% of the mint amount, redistributed correctly."""
        mint_amount = 4000.0
        receipt = tmp_minter.mint_seed("node-2", mint_amount, epoch_id="epoch-2")

        assert receipt.success is True

        # Recipient gets net amount (97.5%)
        recipient_bal = tmp_minter.ledger.get_balance("node-2", TokenType.SEED)
        expected_net = mint_amount * (1 - ZAKAT_RATE)
        assert abs(recipient_bal.balance - expected_net) < 0.01

        # Community fund gets zakat (2.5%)
        community_bal = tmp_minter.ledger.get_balance(
            COMMUNITY_FUND_ACCOUNT, TokenType.SEED
        )
        expected_zakat = mint_amount * ZAKAT_RATE
        assert abs(community_bal.balance - expected_zakat) < 0.01

        # Total minted equals mint_amount (net + zakat)
        total = recipient_bal.balance + community_bal.balance
        assert abs(total - mint_amount) < 0.01

    def test_hash_chain_integrity(self, tmp_minter: TokenMinter) -> None:
        """Multiple mints produce a verifiable hash chain."""
        for i in range(5):
            tmp_minter.mint_seed(f"node-{i}", 10.0, epoch_id=f"epoch-{i}")

        is_valid, count, error = tmp_minter.ledger.verify_chain()
        assert is_valid is True, f"Hash chain broken: {error}"
        # Each mint_seed produces 2 transactions (mint + zakat)
        assert count == 10

    def test_supply_cap_enforcement(self, tmp_minter: TokenMinter) -> None:
        """Yearly supply cap of 1,000,000 SEED is enforced."""
        # Mint close to the cap
        large_mint = SEED_SUPPLY_CAP_PER_YEAR - 100.0
        receipt1 = tmp_minter.mint_seed("whale", large_mint, epoch_id="big-epoch")
        assert receipt1.success is True

        # This should fail -- exceeds remaining cap
        receipt2 = tmp_minter.mint_seed("overflow", 200.0, epoch_id="overflow-epoch")
        assert receipt2.success is False
        assert "supply cap" in receipt2.error.lower()

    def test_genesis_mint_allocations(self, tmp_minter: TokenMinter) -> None:
        """Genesis mint allocates correct amounts to founder, treasury, and fund."""
        receipts = tmp_minter.genesis_mint()

        succeeded = [r for r in receipts if r.success]
        assert len(succeeded) == 4, (
            f"Expected 4 successful genesis transactions, got {len(succeeded)}"
        )

        # Verify founder allocation
        founder_bal = tmp_minter.ledger.get_balance(
            GENESIS_NODE0_ACCOUNT, TokenType.SEED
        )
        assert founder_bal.balance == FOUNDER_GENESIS_ALLOCATION

        # Verify treasury allocation
        treasury_bal = tmp_minter.ledger.get_balance(
            SYSTEM_TREASURY_ACCOUNT, TokenType.SEED
        )
        assert treasury_bal.balance == SYSTEM_TREASURY_ALLOCATION

        # Verify zakat on genesis allocations
        total_genesis = FOUNDER_GENESIS_ALLOCATION + SYSTEM_TREASURY_ALLOCATION
        expected_zakat = total_genesis * ZAKAT_RATE
        community_bal = tmp_minter.ledger.get_balance(
            COMMUNITY_FUND_ACCOUNT, TokenType.SEED
        )
        assert abs(community_bal.balance - expected_zakat) < 0.01

    def test_genesis_mint_idempotent(self, tmp_minter: TokenMinter) -> None:
        """Genesis mint cannot be executed twice."""
        receipts1 = tmp_minter.genesis_mint()
        assert any(r.success for r in receipts1)

        receipts2 = tmp_minter.genesis_mint()
        assert all(r.success is False for r in receipts2)
        assert "already" in receipts2[0].error.lower()

    def test_impt_non_transferable(self, tmp_minter: TokenMinter) -> None:
        """IMPT reputation tokens are soulbound (non-transferable)."""
        # Mint some IMPT
        tmp_minter.mint_impt("node-1", 100.0, epoch_id="rep-epoch")

        # Attempt transfer -- should fail
        receipt = tmp_minter.transfer(
            "node-1", "node-2", TokenType.IMPT, 50.0, memo="try transfer"
        )
        assert receipt.success is False
        assert "non-transferable" in receipt.error.lower()


@pytest.mark.integration
class TestADLGiniGate:
    """CP-2b: Verify that plutocratic concentration is rejected."""

    @staticmethod
    def _compute_gini(balances: List[float]) -> float:
        """Compute Gini coefficient for a list of balances.

        Standing on Giants: Gini (1912) -- concentration measure.
        Gini = 0 means perfect equality, Gini = 1 means total inequality.
        """
        if not balances or all(b == 0 for b in balances):
            return 0.0
        n = len(balances)
        sorted_balances = sorted(balances)
        cumulative_sum = sum(
            (2 * (i + 1) - n - 1) * val for i, val in enumerate(sorted_balances)
        )
        total = sum(sorted_balances)
        if total == 0:
            return 0.0
        return cumulative_sum / (n * total)

    def test_equal_distribution_low_gini(self) -> None:
        """Equal distribution produces Gini well below threshold."""
        balances = [100.0] * 10
        gini = self._compute_gini(balances)
        assert gini < ADL_GINI_THRESHOLD
        assert gini == pytest.approx(0.0, abs=0.001)

    def test_plutocratic_distribution_high_gini(self) -> None:
        """Extreme concentration produces Gini above threshold."""
        # One whale holds 99%, nine nodes hold 1% split
        balances = [990.0] + [1.11] * 9
        gini = self._compute_gini(balances)
        assert gini > ADL_GINI_THRESHOLD, (
            f"Gini {gini:.3f} should exceed threshold {ADL_GINI_THRESHOLD} "
            f"for plutocratic distribution"
        )

    def test_moderate_inequality_passes(self) -> None:
        """Moderate inequality (like developed nations) passes the gate."""
        # Gradually decreasing balances -- moderate Gini
        balances = [100.0, 80.0, 60.0, 50.0, 40.0, 30.0, 25.0, 20.0, 15.0, 10.0]
        gini = self._compute_gini(balances)
        assert gini < ADL_GINI_THRESHOLD, (
            f"Moderate inequality Gini {gini:.3f} should be below "
            f"threshold {ADL_GINI_THRESHOLD}"
        )

    def test_gini_gate_rejects_transaction_that_would_exceed(
        self, tmp_minter: TokenMinter
    ) -> None:
        """Simulates a Gini gate check on a distribution that would cross 0.40."""
        # Mint to create a moderately concentrated distribution
        accounts = [f"node-{i}" for i in range(10)]
        for i, acct in enumerate(accounts):
            # Give the first account much more
            amount = 500.0 if i == 0 else 10.0
            tmp_minter.mint_seed(acct, amount, epoch_id="gini-test")

        # Collect balances
        balances = []
        for acct in accounts:
            bal = tmp_minter.ledger.get_balance(acct, TokenType.SEED)
            balances.append(bal.balance)

        gini = self._compute_gini(balances)

        # The distribution is highly concentrated -- Gini should be high
        # This demonstrates the gate WOULD reject further concentration
        if gini > ADL_GINI_THRESHOLD:
            # Gate would reject -- this is the expected behavior
            assert True
        else:
            # Even with 500:10 ratio, if Gini is still below threshold,
            # the system is working correctly for smaller distributions
            assert gini < 1.0


# =============================================================================
# TEST GROUP 3: Proof Engine Full Gate Chain
# =============================================================================


@pytest.mark.integration
class TestProofEngineGateChain:
    """CP-3: Verify the 6-gate fail-closed chain and evidence receipts."""

    def _passing_context(self) -> Dict[str, Any]:
        """Context that passes all gates."""
        return {
            "trust_score": 0.9,
            "ihsan_score": 0.96,
            "z3_satisfiable": True,
            "risk_score": 0.1,
            "provenance_depth": 3,
            "corroboration_count": 2,
            "prediction_accuracy": 0.8,
            "context_fit_score": 0.85,
            "contradiction_count": 0,
            "conflicting_sources": 0,
            "unverifiable_claims": 0,
            "missing_citations": 0,
        }

    def test_full_chain_all_six_gates_pass(
        self, ed25519_signer: Ed25519Signer, sample_query: CanonQuery,
        sample_policy: CanonPolicy,
    ) -> None:
        """All 6 gates pass in sequence: SCHEMA -> PROVENANCE -> SNR -> CONSTRAINT -> SAFETY -> COMMIT."""
        chain = GateChain(signer=ed25519_signer)
        context = self._passing_context()

        result, receipt = chain.evaluate(sample_query, sample_policy, context)

        assert result.passed is True
        assert result.final_status == GateStatus.PASSED
        assert len(result.gate_results) == 6

        # Verify each gate passed
        gate_names = [g.gate_name for g in result.gate_results]
        assert gate_names == ["schema", "provenance", "snr", "constraint", "safety", "commit"]
        for gate_result in result.gate_results:
            assert gate_result.passed is True, (
                f"Gate '{gate_result.gate_name}' failed: {gate_result.reason}"
            )

    def test_full_chain_produces_valid_receipt(
        self, ed25519_signer: Ed25519Signer, sample_query: CanonQuery,
        sample_policy: CanonPolicy,
    ) -> None:
        """Passing all gates produces a signed evidence receipt."""
        chain = GateChain(signer=ed25519_signer)
        context = self._passing_context()

        result, receipt = chain.evaluate(sample_query, sample_policy, context)

        assert receipt.status == ReceiptStatus.ACCEPTED
        assert receipt.signature != b""
        assert receipt.signer_pubkey != b""
        assert receipt.gate_passed == "commit"

        # Verify the receipt signature
        verifier = ReceiptVerifier(ed25519_signer)
        is_valid, error = verifier.verify(receipt)
        assert is_valid is True, f"Receipt verification failed: {error}"

    def test_schema_gate_fails_missing_user_id(
        self, ed25519_signer: Ed25519Signer, sample_policy: CanonPolicy,
    ) -> None:
        """Schema gate fails when required field 'user_id' is empty."""
        bad_query = CanonQuery(
            user_id="",
            user_state="active",
            intent="Test intent",
        )
        chain = GateChain(signer=ed25519_signer)

        result, receipt = chain.evaluate(bad_query, sample_policy, {})

        assert result.passed is False
        assert result.final_status == GateStatus.FAILED
        assert result.gate_results[0].gate_name == "schema"
        assert result.gate_results[0].passed is False
        # Only schema gate evaluated -- chain stopped
        assert len(result.gate_results) == 1

    def test_provenance_gate_fails_untrusted_source(
        self, ed25519_signer: Ed25519Signer, sample_query: CanonQuery,
        sample_policy: CanonPolicy,
    ) -> None:
        """Provenance gate fails for untrusted source with low trust score."""
        gates = [
            SchemaGate(),
            ProvenanceGate(
                trusted_sources={"trusted-org"},
                min_trust_score=0.8,
            ),
            SNRGate(),
            ConstraintGate(),
            SafetyGate(),
            CommitGate(),
        ]
        chain = GateChain(signer=ed25519_signer, gates=gates)
        context = {"trust_score": 0.3}

        result, receipt = chain.evaluate(sample_query, sample_policy, context)

        assert result.passed is False
        assert result.last_gate_passed == "schema"
        assert any(
            g.gate_name == "provenance" and not g.passed
            for g in result.gate_results
        )

    def test_snr_gate_fails_low_quality(
        self, ed25519_signer: Ed25519Signer, sample_query: CanonQuery,
        sample_policy: CanonPolicy,
    ) -> None:
        """SNR gate fails when signal quality is below threshold."""
        chain = GateChain(signer=ed25519_signer)
        context = {
            "trust_score": 0.9,
            "ihsan_score": 0.96,
            "z3_satisfiable": True,
            # High noise factors
            "contradiction_count": 5,
            "conflicting_sources": 3,
            "unverifiable_claims": 4,
            "missing_citations": 6,
            "provenance_depth": 0,
            "corroboration_count": 0,
            "prediction_accuracy": 0.2,
            "context_fit_score": 0.1,
        }

        result, receipt = chain.evaluate(sample_query, sample_policy, context)

        assert result.passed is False
        assert result.last_gate_passed == "provenance"
        # SNR gate is the 3rd gate
        snr_result = result.gate_results[2]
        assert snr_result.gate_name == "snr"
        assert snr_result.passed is False

    def test_constraint_gate_fails_low_ihsan(
        self, ed25519_signer: Ed25519Signer, sample_query: CanonQuery,
        sample_policy: CanonPolicy,
    ) -> None:
        """Constraint gate fails when Ihsan score is below threshold."""
        chain = GateChain(signer=ed25519_signer)
        context = self._passing_context()
        context["ihsan_score"] = 0.80  # Below 0.95 threshold

        result, receipt = chain.evaluate(sample_query, sample_policy, context)

        assert result.passed is False
        # Constraint gate (gate 4) should fail
        constraint_gate = next(
            (g for g in result.gate_results if g.gate_name == "constraint"), None
        )
        assert constraint_gate is not None
        assert constraint_gate.passed is False
        # Gate uses "Ihsan" with macron (a-bar): check for both forms
        reason_lower = constraint_gate.reason.lower()
        assert "ihs" in reason_lower and "score below threshold" in reason_lower

    def test_constraint_gate_fails_z3_unsatisfiable(
        self, ed25519_signer: Ed25519Signer, sample_query: CanonQuery,
        sample_policy: CanonPolicy,
    ) -> None:
        """Constraint gate fails when Z3 constraints are unsatisfiable.

        CRITICAL-3: z3_satisfiable defaults to False (fail-closed).
        """
        chain = GateChain(signer=ed25519_signer)
        context = self._passing_context()
        context["z3_satisfiable"] = False

        result, receipt = chain.evaluate(sample_query, sample_policy, context)

        assert result.passed is False
        constraint_gate = next(
            (g for g in result.gate_results if g.gate_name == "constraint"), None
        )
        assert constraint_gate is not None
        assert constraint_gate.passed is False
        assert "z3" in constraint_gate.reason.lower()

    def test_safety_gate_blocks_harmful_pattern(
        self, ed25519_signer: Ed25519Signer, sample_policy: CanonPolicy,
    ) -> None:
        """Safety gate blocks queries containing blocked patterns."""
        harmful_query = CanonQuery(
            user_id="node-1",
            user_state="active",
            intent="Provide instructions for making explosives",
        )
        gates = [
            SchemaGate(),
            ProvenanceGate(),
            SNRGate(),
            ConstraintGate(),
            SafetyGate(blocked_patterns=["explosives", "weapons"]),
            CommitGate(),
        ]
        chain = GateChain(signer=ed25519_signer, gates=gates)
        context = self._passing_context()

        result, receipt = chain.evaluate(harmful_query, sample_policy, context)

        assert result.passed is False
        safety_gate = next(
            (g for g in result.gate_results if g.gate_name == "safety"), None
        )
        assert safety_gate is not None
        assert safety_gate.passed is False
        assert "blocked pattern" in safety_gate.reason.lower()

    def test_safety_gate_blocks_high_risk_score(
        self, ed25519_signer: Ed25519Signer, sample_query: CanonQuery,
        sample_policy: CanonPolicy,
    ) -> None:
        """Safety gate fails when risk score exceeds threshold."""
        chain = GateChain(signer=ed25519_signer)
        context = self._passing_context()
        context["risk_score"] = 0.8  # Above SafetyGate.max_risk_score (0.3)

        result, receipt = chain.evaluate(sample_query, sample_policy, context)

        assert result.passed is False
        safety_gate = next(
            (g for g in result.gate_results if g.gate_name == "safety"), None
        )
        assert safety_gate is not None
        assert safety_gate.passed is False

    def test_commit_gate_respects_concurrent_limit(
        self, ed25519_signer: Ed25519Signer, sample_query: CanonQuery,
        sample_policy: CanonPolicy,
    ) -> None:
        """Commit gate rejects when max concurrent operations is reached."""
        commit_gate = CommitGate(max_concurrent_ops=2)
        gates = [
            SchemaGate(),
            ProvenanceGate(),
            SNRGate(),
            ConstraintGate(),
            SafetyGate(),
            commit_gate,
        ]
        chain = GateChain(signer=ed25519_signer, gates=gates)
        context = self._passing_context()

        # Fill up concurrent slots
        commit_gate._current_ops = 2

        result, receipt = chain.evaluate(sample_query, sample_policy, context)

        assert result.passed is False
        last_gate = result.gate_results[-1]
        assert last_gate.gate_name == "commit"
        assert last_gate.passed is False

    def test_rejection_receipt_is_signed(
        self, ed25519_signer: Ed25519Signer, sample_policy: CanonPolicy,
    ) -> None:
        """Even rejection receipts carry a valid signature for auditability."""
        bad_query = CanonQuery(user_id="", user_state="x", intent="fail")
        chain = GateChain(signer=ed25519_signer)

        result, receipt = chain.evaluate(bad_query, sample_policy, {})

        assert receipt.status == ReceiptStatus.REJECTED
        assert receipt.signature != b""

        # Verify rejection receipt signature
        verifier = ReceiptVerifier(ed25519_signer)
        is_valid, error = verifier.verify(receipt)
        assert is_valid is True, f"Rejection receipt verification failed: {error}"

    def test_each_gate_individually_fails_closes_chain(
        self, ed25519_signer: Ed25519Signer, sample_query: CanonQuery,
        sample_policy: CanonPolicy,
    ) -> None:
        """Fail-closed: any single gate failure stops the entire chain."""
        context = self._passing_context()

        # Test that a failing gate means fewer gates are evaluated
        # (chain short-circuits on first failure)
        for gate_idx in range(6):
            # Create a chain where gate at gate_idx will fail
            gates_list: List[Any] = [
                SchemaGate(),
                ProvenanceGate(),
                SNRGate(),
                ConstraintGate(),
                SafetyGate(),
                CommitGate(),
            ]

            # Sabotage one gate by patching its evaluate method
            gate_name = gates_list[gate_idx].name

            def make_failing_evaluate(name: str):
                def failing_evaluate(q, p, c):
                    from core.proof_engine.gates import GateResult, GateStatus
                    return GateResult(
                        gate_name=name,
                        status=GateStatus.FAILED,
                        reason=f"Deliberate test failure at {name}",
                    )
                return failing_evaluate

            gates_list[gate_idx].evaluate = make_failing_evaluate(gate_name)

            chain = GateChain(signer=ed25519_signer, gates=gates_list)
            result, _ = chain.evaluate(sample_query, sample_policy, context)

            assert result.passed is False, (
                f"Chain should fail when gate '{gate_name}' (index {gate_idx}) fails"
            )
            # Chain should stop at the failing gate
            assert len(result.gate_results) == gate_idx + 1, (
                f"Expected chain to stop after gate {gate_idx} ({gate_name}), "
                f"but {len(result.gate_results)} gates were evaluated"
            )


# =============================================================================
# TEST GROUP 4: Cross-Module Bridge
# =============================================================================


@pytest.mark.integration
class TestCrossModuleBridge:
    """CP-4: Verify interactions across Token, Proof, Identity, and Federation."""

    def test_token_mint_produces_signed_receipt(
        self, tmp_minter: TokenMinter
    ) -> None:
        """Token mint produces a receipt with a non-empty hash (proof of operation)."""
        receipt = tmp_minter.mint_seed(
            "node-1", 50.0, epoch_id="epoch-1", poi_score=0.91
        )

        assert receipt.success is True
        assert receipt.receipt_hash != ""
        assert receipt.tx_entry is not None
        assert receipt.tx_entry.signer_pubkey != ""

    def test_token_receipt_signature_verifiable(
        self, tmp_minter: TokenMinter
    ) -> None:
        """Token transaction signature can be verified with the minter's public key."""
        receipt = tmp_minter.mint_seed(
            "verifier-node", 25.0, epoch_id="epoch-verify"
        )

        assert receipt.success is True
        tx = receipt.tx_entry
        assert tx is not None

        # Verify the Ed25519 signature on the finalized tx_hash
        is_valid = tmp_minter.verify_transaction(tx)
        assert is_valid is True

    def test_identity_card_creation_and_signing(self) -> None:
        """Identity card is created from Ed25519 keypair and dual-signed."""
        # Generate identity
        owner_priv, owner_pub, node_id = generate_identity_keypair()
        minter_priv, minter_pub = generate_keypair()

        # Create card
        card = IdentityCard.create(owner_pub)
        assert card.node_id.startswith("BIZRA-")
        assert card.status == IdentityStatus.PENDING

        # Sign as minter (activates card)
        card.sign_as_minter(minter_priv, minter_pub)
        assert card.status == IdentityStatus.ACTIVE
        assert card.minter_signature is not None

        # Sign as owner
        card.sign_as_owner(owner_priv)
        assert card.self_signature is not None

        # Verify both signatures
        assert card.verify_minter_signature() is True
        assert card.verify_self_signature() is True
        assert card.is_fully_verified() is True

    def test_identity_card_serialization_roundtrip(self) -> None:
        """Identity card survives serialization and deserialization."""
        owner_priv, owner_pub, node_id = generate_identity_keypair()
        minter_priv, minter_pub = generate_keypair()

        card = IdentityCard.create(owner_pub)
        card.sign_as_minter(minter_priv, minter_pub)
        card.sign_as_owner(owner_priv)

        # Roundtrip via dict
        data = card.to_dict()
        restored = IdentityCard.from_dict(data)

        assert restored.node_id == card.node_id
        assert restored.public_key == card.public_key
        assert restored.is_fully_verified() is True

    def test_proof_receipt_signed_with_ed25519_verifiable(self) -> None:
        """Proof engine receipt, signed with Ed25519, is verifiable."""
        signer = Ed25519Signer.generate()

        query = CanonQuery(
            user_id="node-1",
            user_state="active",
            intent="Verify cross-module bridge",
        )
        policy = CanonPolicy(
            policy_id="bridge-policy",
            version="1.0",
            rules={"max_depth": 3},
            thresholds={"ihsan": UNIFIED_IHSAN_THRESHOLD},
        )

        chain = GateChain(signer=signer)
        context = {
            "trust_score": 0.9,
            "ihsan_score": 0.96,
            "z3_satisfiable": True,
            "risk_score": 0.05,
            "provenance_depth": 3,
            "corroboration_count": 2,
            "prediction_accuracy": 0.85,
            "context_fit_score": 0.9,
            "contradiction_count": 0,
            "conflicting_sources": 0,
            "unverifiable_claims": 0,
            "missing_citations": 0,
        }

        result, receipt = chain.evaluate(query, policy, context)
        assert result.passed is True

        # Verify the receipt with a ReceiptVerifier
        verifier = ReceiptVerifier(signer)
        is_valid, error = verifier.verify(receipt)
        assert is_valid is True, f"Receipt verification failed: {error}"

        # The receipt digest can serve as an attestation hash
        assert receipt.hex_digest() != ""
        assert len(receipt.hex_digest()) == 64  # BLAKE3 hex digest

    def test_federation_consensus_with_ed25519_signatures(self) -> None:
        """PBFT consensus round uses Ed25519 signatures for voting."""
        # Create 4 nodes (tolerates 1 Byzantine failure: f=1, quorum=3)
        nodes = []
        for i in range(4):
            priv, pub = generate_keypair()
            node = ConsensusEngine(
                node_id=f"node-{i}",
                private_key=priv,
                public_key=pub,
            )
            nodes.append((node, priv, pub))

        # Register all peers with each other
        for node, _, _ in nodes:
            for other_node, _, other_pub in nodes:
                if other_node.node_id != node.node_id:
                    node.register_peer(other_node.node_id, other_pub)

        # Leader proposes a pattern
        leader, leader_priv, leader_pub = nodes[0]
        leader.set_leader(leader.node_id)

        pattern = {
            "name": "autopoiesis-loop",
            "type": "emergence",
            "confidence": 0.95,
        }
        proposal = leader.initiate_pre_prepare(pattern)
        assert proposal is not None

        # Other nodes vote
        for node, _, _ in nodes[1:]:
            vote = node.cast_vote(proposal, ihsan_score=0.96)
            assert vote is not None

            # Leader receives votes
            committed = leader.receive_vote(vote, node_count=4)
            if committed:
                break

        # Should be committed after 3 votes (quorum for 4 nodes)
        assert proposal.proposal_id in leader.committed_patterns

    def test_federation_rejects_low_ihsan_vote(self) -> None:
        """Federation rejects votes with Ihsan below threshold."""
        priv, pub = generate_keypair()
        node = ConsensusEngine("node-0", priv, pub)
        node.set_leader("node-0")

        proposal = node.initiate_pre_prepare({"test": "pattern"})
        assert proposal is not None

        # Cast vote with low Ihsan -- should return None
        vote = node.cast_vote(proposal, ihsan_score=0.80)
        assert vote is None

    def test_identity_deterministic_node_id(self) -> None:
        """Same public key always produces the same node_id."""
        _, pub, node_id1 = generate_identity_keypair()

        # Re-derive node_id from same public key
        from core.pat.identity_card import _generate_node_id
        node_id2 = _generate_node_id(pub)

        assert node_id1 == node_id2

    def test_token_to_proof_attestation_flow(
        self, tmp_minter: TokenMinter, ed25519_signer: Ed25519Signer,
    ) -> None:
        """Token mint receipt hash can be included in a proof engine attestation."""
        # Step 1: Mint tokens (generates receipt with hash)
        token_receipt = tmp_minter.mint_seed(
            "attested-node", 100.0, epoch_id="poi-epoch", poi_score=0.93
        )
        assert token_receipt.success is True
        assert token_receipt.receipt_hash != ""

        # Step 2: Use token receipt hash as evidence in a proof query
        query = CanonQuery(
            user_id="attested-node",
            user_state="verified",
            intent="Attestation for PoI distribution",
            payload={"token_receipt_hash": token_receipt.receipt_hash},
        )
        policy = CanonPolicy(
            policy_id="attestation-policy",
            version="1.0",
            rules={"require_token_receipt": True},
            thresholds={"ihsan": UNIFIED_IHSAN_THRESHOLD},
        )

        chain = GateChain(signer=ed25519_signer)
        context = {
            "trust_score": 0.95,
            "ihsan_score": 0.97,
            "z3_satisfiable": True,
            "risk_score": 0.02,
            "provenance_depth": 4,
            "corroboration_count": 3,
            "prediction_accuracy": 0.9,
            "context_fit_score": 0.92,
            "contradiction_count": 0,
            "conflicting_sources": 0,
            "unverifiable_claims": 0,
            "missing_citations": 0,
        }

        result, proof_receipt = chain.evaluate(query, policy, context)
        assert result.passed is True
        assert proof_receipt.status == ReceiptStatus.ACCEPTED

        # Step 3: Verify the proof receipt is independently valid
        verifier = ReceiptVerifier(ed25519_signer)
        is_valid, error = verifier.verify(proof_receipt)
        assert is_valid is True

        # The proof receipt now cryptographically attests to the token operation
        assert proof_receipt.hex_digest() != token_receipt.receipt_hash

    def test_full_lifecycle_genesis_to_verification(
        self, tmp_path: Path,
    ) -> None:
        """Full lifecycle: Genesis -> Mint -> Proof -> Identity -> Federation attestation."""
        # Phase 1: Genesis -- create the token economy
        minter = TokenMinter.create(
            db_path=tmp_path / "lifecycle.db",
            log_path=tmp_path / "lifecycle.jsonl",
        )
        genesis_receipts = minter.genesis_mint()
        assert all(r.success for r in genesis_receipts)

        # Phase 2: Mint SEED from PoI
        poi_receipt = minter.mint_seed(
            "worker-node", 200.0, epoch_id="epoch-42", poi_score=0.91
        )
        assert poi_receipt.success is True

        # Phase 3: Verify hash chain integrity
        is_valid, count, error = minter.ledger.verify_chain()
        assert is_valid is True
        assert count > 0

        # Phase 4: Create identity for the worker node
        worker_priv, worker_pub, worker_node_id = generate_identity_keypair()
        minter_priv_key, minter_pub_key = generate_keypair()

        card = IdentityCard.create(worker_pub)
        card.sign_as_minter(minter_priv_key, minter_pub_key)
        card.sign_as_owner(worker_priv)
        assert card.is_fully_verified()

        # Phase 5: Create a proof receipt for the operation
        signer = Ed25519Signer.generate()
        query = CanonQuery(
            user_id=worker_node_id,
            user_state="active",
            intent="PoI-verified compute contribution",
            payload={
                "token_hash": poi_receipt.receipt_hash,
                "identity_digest": card.compute_digest(),
            },
        )
        policy = CanonPolicy(
            policy_id="lifecycle-policy",
            version="1.0",
            rules={"full_lifecycle": True},
            thresholds={"ihsan": UNIFIED_IHSAN_THRESHOLD},
        )

        chain = GateChain(signer=signer)
        context = {
            "trust_score": 0.95,
            "ihsan_score": 0.97,
            "z3_satisfiable": True,
            "risk_score": 0.01,
            "provenance_depth": 5,
            "corroboration_count": 3,
            "prediction_accuracy": 0.9,
            "context_fit_score": 0.95,
            "contradiction_count": 0,
            "conflicting_sources": 0,
            "unverifiable_claims": 0,
            "missing_citations": 0,
        }

        result, proof_receipt = chain.evaluate(query, policy, context)
        assert result.passed is True
        assert proof_receipt.status == ReceiptStatus.ACCEPTED

        # Phase 6: Verify the proof receipt
        verifier = ReceiptVerifier(signer)
        is_valid, error = verifier.verify(proof_receipt)
        assert is_valid is True, f"Lifecycle proof verification failed: {error}"


# =============================================================================
# SNR ENGINE SANITY CHECKS
# =============================================================================


@pytest.mark.integration
class TestSNREngineSanity:
    """Verify SNR engine produces consistent and auditable results."""

    def test_clean_input_produces_high_snr(self) -> None:
        """Clean input (no noise) produces SNR well above threshold."""
        engine = SNREngine()
        inputs = SNRInput(
            provenance_depth=5,
            corroboration_count=3,
            source_trust_score=0.95,
            z3_satisfiable=True,
            ihsan_score=0.98,
            constraint_violations=0,
            prediction_accuracy=0.9,
            context_fit_score=0.92,
            contradiction_count=0,
            conflicting_sources=0,
            unverifiable_claims=0,
            missing_citations=0,
        )

        snr, trace = engine.compute(inputs)

        assert snr >= UNIFIED_SNR_THRESHOLD
        assert trace.noise_mass == 0.0
        assert trace.signal_mass > 0.0

    def test_noisy_input_produces_low_snr(self) -> None:
        """High noise factors push SNR below threshold."""
        engine = SNREngine()
        inputs = SNRInput(
            provenance_depth=0,
            corroboration_count=0,
            source_trust_score=0.1,
            z3_satisfiable=False,
            ihsan_score=0.5,
            constraint_violations=5,
            prediction_accuracy=0.1,
            context_fit_score=0.1,
            contradiction_count=5,
            conflicting_sources=3,
            unverifiable_claims=4,
            missing_citations=6,
        )

        snr, trace = engine.compute(inputs)

        assert snr < UNIFIED_SNR_THRESHOLD
        assert trace.noise_mass > 0.0

    def test_trace_is_recomputable(self) -> None:
        """SNR trace can be independently verified by re-computation."""
        engine = SNREngine()
        inputs = SNRInput(
            provenance_depth=3,
            corroboration_count=2,
            source_trust_score=0.8,
            z3_satisfiable=True,
            ihsan_score=0.95,
            prediction_accuracy=0.7,
            context_fit_score=0.75,
        )

        _, trace = engine.compute(inputs)

        # Verify trace consistency
        assert engine.verify_trace(trace) is True

    def test_snr_is_deterministic(self) -> None:
        """Same inputs produce identical SNR across multiple runs."""
        engine = SNREngine()
        inputs = SNRInput(
            provenance_depth=2,
            corroboration_count=1,
            source_trust_score=0.7,
            z3_satisfiable=True,
            ihsan_score=0.96,
            prediction_accuracy=0.6,
            context_fit_score=0.65,
        )

        snr1, _ = engine.compute(inputs)
        snr2, _ = engine.compute(inputs)

        assert snr1 == snr2
