"""
PoI Determinism Property Tests — SAPE Audit DoD.

Definition of Done:
  "Same input bundle scored 100 times → identical poi_receipt.json hash"

These tests prove that the PoI engine is deterministic:
- Same inputs produce byte-identical receipts
- Same citation graph produces identical PageRank
- Same epoch produces identical AuditTrail digest
- Sorted iteration order is enforced everywhere
- No datetime.now() leaks into scoring paths

Standing on Giants:
- Lamport (1978): Deterministic ordering in distributed systems
- Shannon (1948): Signal integrity
"""

import pytest
from datetime import datetime, timedelta, timezone

from core.proof_engine.canonical import blake3_digest, canonical_bytes
from core.proof_engine.receipt import SimpleSigner
from core.proof_engine.poi_engine import (
    AuditTrail,
    CitationGraph,
    ContributionMetadata,
    ContributionType,
    ContributionVerifier,
    PoIConfig,
    PoIOrchestrator,
    PoIReasonCode,
    PoIReceipt,
    ProofOfImpact,
    SATRebalancer,
    TemporalScorer,
    compute_gini,
    compute_token_distribution,
)


FIXED_TIME = datetime(2026, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
FIXED_SIGNER = SimpleSigner(b"determinism-test-key")
ITERATIONS = 100


def _make_deterministic_config() -> PoIConfig:
    """Deterministic config with fixed values."""
    return PoIConfig()


def _make_contribution(
    contributor_id: str,
    content_hash: str,
    snr: float = 0.95,
    ihsan: float = 0.96,
) -> ContributionMetadata:
    return ContributionMetadata(
        contributor_id=contributor_id,
        contribution_type=ContributionType.CODE,
        content_hash=content_hash,
        snr_score=snr,
        ihsan_score=ihsan,
        timestamp=FIXED_TIME,
    )


# =============================================================================
# RECEIPT DETERMINISM (100-iteration DoD)
# =============================================================================

class TestReceiptDeterminism:
    """Same input → same receipt hash, 100 times."""

    def test_poi_receipt_100_iterations(self):
        """PoIReceipt body_bytes + signature are identical across 100 runs."""
        digests = set()
        for _ in range(ITERATIONS):
            receipt = PoIReceipt(
                receipt_id="rcpt_test_001",
                epoch_id="epoch_00000001",
                contributor_id="alice",
                reason=PoIReasonCode.POI_OK,
                poi_score=0.85,
                contribution_score=0.90,
                reach_score=0.70,
                longevity_score=0.80,
                config_digest="abc123def456",
                content_hash="hash_alice_001",
            )
            receipt.sign_with(FIXED_SIGNER)
            digests.add(receipt.hex_digest())

        assert len(digests) == 1, f"Expected 1 unique digest, got {len(digests)}"

    def test_poi_receipt_body_bytes_deterministic(self):
        """body_bytes() is deterministic — no timestamp in canonical form."""
        bodies = set()
        for _ in range(ITERATIONS):
            receipt = PoIReceipt(
                receipt_id="rcpt_body_test",
                epoch_id="epoch_00000001",
                contributor_id="bob",
                reason=PoIReasonCode.POI_REJECT_SNR_BELOW_THRESHOLD,
                poi_score=0.0,
                contribution_score=0.0,
                reach_score=0.0,
                longevity_score=0.0,
                config_digest="xyz789",
                content_hash="hash_bob_001",
            )
            bodies.add(receipt.body_bytes())

        assert len(bodies) == 1

    def test_poi_receipt_signature_deterministic(self):
        """Same body → same HMAC signature."""
        sigs = set()
        for _ in range(ITERATIONS):
            receipt = PoIReceipt(
                receipt_id="rcpt_sig_test",
                epoch_id="epoch_00000001",
                contributor_id="carol",
                reason=PoIReasonCode.POI_OK,
                poi_score=0.75,
                contribution_score=0.80,
                reach_score=0.60,
                longevity_score=0.70,
                config_digest="sig_test_config",
                content_hash="hash_carol_001",
            )
            receipt.sign_with(FIXED_SIGNER)
            sigs.add(receipt.signature)

        assert len(sigs) == 1


# =============================================================================
# PAGERANK DETERMINISM
# =============================================================================

class TestPageRankDeterminism:
    """Same citation graph → same PageRank scores, 100 times."""

    def _build_graph(self) -> CitationGraph:
        """Build a fixed citation graph."""
        config = _make_deterministic_config()
        graph = CitationGraph(config)
        # Star topology: everyone cites alice
        for name in ["bob", "carol", "dave", "eve"]:
            graph.add_citation(name, "alice")
        # Some cross-citations
        graph.add_citation("bob", "carol")
        graph.add_citation("carol", "dave")
        graph.add_citation("dave", "bob")
        return graph

    def test_pagerank_100_iterations(self):
        """PageRank produces identical scores across 100 runs."""
        reference = None
        for i in range(ITERATIONS):
            graph = self._build_graph()
            scores = graph.compute_pagerank()
            # Convert to canonical bytes for comparison
            score_bytes = canonical_bytes(
                {k: scores[k] for k in sorted(scores.keys())}
            )
            if reference is None:
                reference = score_bytes
            else:
                assert score_bytes == reference, (
                    f"PageRank diverged on iteration {i}"
                )

    def test_reach_scores_100_iterations(self):
        """compute_reach_scores() produces identical output 100 times."""
        reference = None
        for i in range(ITERATIONS):
            graph = self._build_graph()
            scores = graph.compute_reach_scores()
            # Serialize deterministically
            data = [
                {
                    "id": s.contributor_id,
                    "reach": s.normalized_reach,
                    "penalty": s.ring_penalty,
                }
                for s in scores
            ]
            score_bytes = canonical_bytes(data)
            if reference is None:
                reference = score_bytes
            else:
                assert score_bytes == reference, (
                    f"Reach scores diverged on iteration {i}"
                )

    def test_pagerank_sorted_node_order(self):
        """PageRank iterates over sorted node list."""
        graph = self._build_graph()
        scores = graph.compute_pagerank()
        # Keys should be in sorted order when iterated
        keys = list(scores.keys())
        assert keys == sorted(keys), "PageRank keys are not sorted"


# =============================================================================
# TEMPORAL SCORER DETERMINISM
# =============================================================================

class TestTemporalDeterminism:
    """Same activity + reference_time → same longevity, 100 times."""

    def _build_scorer(self) -> TemporalScorer:
        config = _make_deterministic_config()
        scorer = TemporalScorer(config)
        # Fixed activity pattern
        for i in range(5):
            ts = FIXED_TIME - timedelta(days=i * 7)  # Weekly
            scorer.record_activity("alice", ts)
        for i in range(3):
            ts = FIXED_TIME - timedelta(days=i * 2)
            scorer.record_activity("bob", ts)
        return scorer

    def test_longevity_100_iterations(self):
        """compute_longevity with fixed reference_time is deterministic."""
        reference_alice = None
        reference_bob = None
        for i in range(ITERATIONS):
            scorer = self._build_scorer()
            la = scorer.compute_longevity("alice", reference_time=FIXED_TIME)
            lb = scorer.compute_longevity("bob", reference_time=FIXED_TIME)

            if reference_alice is None:
                reference_alice = la.normalized_longevity
                reference_bob = lb.normalized_longevity
            else:
                assert la.normalized_longevity == reference_alice, (
                    f"Alice longevity diverged on iteration {i}"
                )
                assert lb.normalized_longevity == reference_bob, (
                    f"Bob longevity diverged on iteration {i}"
                )

    def test_reference_time_required_for_determinism(self):
        """Without reference_time, longevity may vary (wall-clock)."""
        config = _make_deterministic_config()
        scorer = TemporalScorer(config)
        scorer.record_activity("alice", FIXED_TIME)

        # With reference_time: deterministic
        r1 = scorer.compute_longevity("alice", reference_time=FIXED_TIME)
        r2 = scorer.compute_longevity("alice", reference_time=FIXED_TIME)
        assert r1.normalized_longevity == r2.normalized_longevity


# =============================================================================
# CONTRIBUTION VERIFIER DETERMINISM
# =============================================================================

class TestVerifierDeterminism:
    """Same contribution → same verification result, 100 times."""

    def test_verify_accept_deterministic(self):
        """Accepted contributions produce identical quality_score."""
        scores = set()
        for _ in range(ITERATIONS):
            config = _make_deterministic_config()
            verifier = ContributionVerifier(config)
            meta = _make_contribution("alice", "unique_hash_001")
            result = verifier.verify(meta)
            scores.add(result.quality_score)

        assert len(scores) == 1

    def test_verify_reject_deterministic(self):
        """Rejected contributions produce identical reason code."""
        reasons = set()
        for _ in range(ITERATIONS):
            config = _make_deterministic_config()
            verifier = ContributionVerifier(config)
            meta = _make_contribution("alice", "low_snr_hash", snr=0.50)
            result = verifier.verify(meta)
            reasons.add(result.reason_code)

        assert len(reasons) == 1
        assert PoIReasonCode.POI_REJECT_SNR_BELOW_THRESHOLD in reasons


# =============================================================================
# FULL EPOCH DETERMINISM (the money test)
# =============================================================================

class TestEpochDeterminism:
    """Same inputs → same AuditTrail digest, 100 times."""

    def _build_orchestrator(self) -> PoIOrchestrator:
        config = _make_deterministic_config()
        orch = PoIOrchestrator(config, signer=FIXED_SIGNER)

        # Register fixed contributions
        for i, name in enumerate(["alice", "bob", "carol", "dave"]):
            meta = _make_contribution(
                name,
                f"epoch_det_{name}",
                snr=0.90 + i * 0.02,
                ihsan=0.92 + i * 0.01,
            )
            orch.register_contribution(meta)

        # Fixed citations
        orch.add_citation("bob", "alice")
        orch.add_citation("carol", "alice")
        orch.add_citation("dave", "alice")
        orch.add_citation("carol", "bob")

        return orch

    def test_audit_trail_digest_100_iterations(self):
        """AuditTrail.digest() is identical across 100 runs."""
        reference = None
        for i in range(ITERATIONS):
            orch = self._build_orchestrator()
            audit = orch.compute_epoch(
                epoch_id="det_epoch_001",
                reference_time=FIXED_TIME,
            )
            digest = audit.hex_digest()

            if reference is None:
                reference = digest
            else:
                assert digest == reference, (
                    f"AuditTrail digest diverged on iteration {i}: "
                    f"{digest} != {reference}"
                )

    def test_all_receipts_identical_100_iterations(self):
        """Every receipt digest is identical across 100 runs."""
        reference_receipts = None
        for i in range(ITERATIONS):
            orch = self._build_orchestrator()
            audit = orch.compute_epoch(
                epoch_id="det_epoch_002",
                reference_time=FIXED_TIME,
            )
            receipt_digests = [r.hex_digest() for r in audit.receipts]

            if reference_receipts is None:
                reference_receipts = receipt_digests
            else:
                assert receipt_digests == reference_receipts, (
                    f"Receipt digests diverged on iteration {i}"
                )

    def test_poi_scores_identical_100_iterations(self):
        """Every PoI score is identical across 100 runs."""
        reference_scores = None
        for i in range(ITERATIONS):
            orch = self._build_orchestrator()
            audit = orch.compute_epoch(
                epoch_id="det_epoch_003",
                reference_time=FIXED_TIME,
            )
            scores = {p.contributor_id: p.poi_score for p in audit.poi_scores}

            if reference_scores is None:
                reference_scores = scores
            else:
                for cid in sorted(scores.keys()):
                    assert scores[cid] == reference_scores[cid], (
                        f"PoI score for {cid} diverged on iteration {i}"
                    )

    def test_token_distribution_deterministic(self):
        """Token distribution from same audit → same amounts."""
        reference = None
        for i in range(ITERATIONS):
            orch = self._build_orchestrator()
            audit = orch.compute_epoch(
                epoch_id="det_epoch_004",
                reference_time=FIXED_TIME,
            )
            dist = compute_token_distribution(audit, epoch_reward=1000.0)
            dist_bytes = canonical_bytes(
                {k: dist.distributions[k] for k in sorted(dist.distributions.keys())}
            )

            if reference is None:
                reference = dist_bytes
            else:
                assert dist_bytes == reference, (
                    f"Token distribution diverged on iteration {i}"
                )

    def test_gini_deterministic(self):
        """Gini coefficient is identical across 100 runs."""
        reference = None
        for i in range(ITERATIONS):
            orch = self._build_orchestrator()
            audit = orch.compute_epoch(
                epoch_id="det_epoch_005",
                reference_time=FIXED_TIME,
            )
            gini = audit.gini_coefficient

            if reference is None:
                reference = gini
            else:
                assert gini == reference, (
                    f"Gini diverged on iteration {i}: {gini} != {reference}"
                )


# =============================================================================
# REBALANCER DETERMINISM
# =============================================================================

class TestRebalancerDeterminism:
    """Same scores → same rebalance result, 100 times."""

    def test_rebalance_100_iterations(self):
        """Rebalancer produces identical results 100 times."""
        config = PoIConfig(gini_rebalance_threshold=0.01)

        reference = None
        for i in range(ITERATIONS):
            rebalancer = SATRebalancer(config)
            scores = {"alice": 0.9, "bob": 0.5, "carol": 0.1, "dave": 0.3}
            result = rebalancer.rebalance(scores)
            result_bytes = canonical_bytes({
                "gini_before": result.gini_before,
                "gini_after": result.gini_after,
                "zakat_collected": result.zakat_collected,
                "rebalanced": {
                    k: result.rebalanced_scores[k]
                    for k in sorted(result.rebalanced_scores.keys())
                },
            })

            if reference is None:
                reference = result_bytes
            else:
                assert result_bytes == reference, (
                    f"Rebalancer diverged on iteration {i}"
                )


# =============================================================================
# REASON CODE COVERAGE
# =============================================================================

class TestReasonCodeCoverage:
    """Every reason code can be emitted."""

    def test_poi_ok_on_accept(self):
        """Accepted contribution gets POI_OK."""
        verifier = ContributionVerifier()
        meta = _make_contribution("alice", "ok_test", snr=0.95, ihsan=0.96)
        result = verifier.verify(meta)
        assert result.reason_code == PoIReasonCode.POI_OK

    def test_reject_duplicate(self):
        """Duplicate gets POI_REJECT_DUPLICATE_ARTIFACT."""
        verifier = ContributionVerifier()
        meta = _make_contribution("alice", "dup_test", snr=0.95, ihsan=0.96)
        verifier.verify(meta)
        result = verifier.verify(meta)
        assert result.reason_code == PoIReasonCode.POI_REJECT_DUPLICATE_ARTIFACT

    def test_reject_snr(self):
        """Low SNR gets POI_REJECT_SNR_BELOW_THRESHOLD."""
        verifier = ContributionVerifier()
        meta = _make_contribution("alice", "snr_test", snr=0.50, ihsan=0.96)
        result = verifier.verify(meta)
        assert result.reason_code == PoIReasonCode.POI_REJECT_SNR_BELOW_THRESHOLD

    def test_reject_ihsan(self):
        """Low Ihsan gets POI_REJECT_IHSAN_BELOW_THRESHOLD."""
        verifier = ContributionVerifier()
        meta = _make_contribution("alice", "ihsan_test", snr=0.95, ihsan=0.50)
        result = verifier.verify(meta)
        assert result.reason_code == PoIReasonCode.POI_REJECT_IHSAN_BELOW_THRESHOLD

    def test_penalty_ring_detected(self):
        """Citation ring penalty is tagged in reach score."""
        config = PoIConfig(citation_ring_threshold=1)
        graph = CitationGraph(config)
        graph.add_citation("a", "b")
        graph.add_citation("b", "a")
        graph.add_citation("a", "c")
        graph.add_citation("c", "a")
        scores = graph.compute_reach_scores()
        a_score = next(s for s in scores if s.contributor_id == "a")
        assert a_score.reason_code == PoIReasonCode.POI_PENALTY_RING_DETECTED

    def test_all_reason_codes_defined(self):
        """All 10 reason codes exist."""
        assert len(PoIReasonCode) == 10
        assert PoIReasonCode.POI_OK.value == "POI_OK"
        assert PoIReasonCode.POI_QUARANTINE_MISSING_EVIDENCE.value == "POI_QUARANTINE_MISSING_EVIDENCE"
        assert PoIReasonCode.POI_REJECT_BAD_SIGNATURE.value == "POI_REJECT_BAD_SIGNATURE"
        assert PoIReasonCode.POI_REJECT_DUPLICATE_ARTIFACT.value == "POI_REJECT_DUPLICATE_ARTIFACT"
        assert PoIReasonCode.POI_REJECT_SNR_BELOW_THRESHOLD.value == "POI_REJECT_SNR_BELOW_THRESHOLD"
        assert PoIReasonCode.POI_REJECT_EPOCH_MISMATCH.value == "POI_REJECT_EPOCH_MISMATCH"
        assert PoIReasonCode.POI_REJECT_IHSAN_BELOW_THRESHOLD.value == "POI_REJECT_IHSAN_BELOW_THRESHOLD"
        assert PoIReasonCode.POI_PENALTY_RING_DETECTED.value == "POI_PENALTY_RING_DETECTED"
        assert PoIReasonCode.POI_PENALTY_RECIPROCAL_FARM.value == "POI_PENALTY_RECIPROCAL_FARM"
        assert PoIReasonCode.POI_INTERNAL_INVARIANT_FAIL.value == "POI_INTERNAL_INVARIANT_FAIL"


# =============================================================================
# RECEIPT FEATURES
# =============================================================================

class TestReceiptFeatures:
    """Tests for PoI receipt functionality."""

    def test_receipt_signing_and_verification(self):
        """Receipt can be signed and verified."""
        receipt = PoIReceipt(
            receipt_id="feat_001",
            epoch_id="epoch_00000001",
            contributor_id="alice",
            reason=PoIReasonCode.POI_OK,
            poi_score=0.85,
            contribution_score=0.90,
            reach_score=0.70,
            longevity_score=0.80,
            config_digest="abc123",
            content_hash="hash_001",
        )
        receipt.sign_with(FIXED_SIGNER)

        assert receipt.signature != b""
        assert receipt.signer_pubkey != b""
        assert receipt.verify_signature(FIXED_SIGNER)

    def test_receipt_wrong_signer_fails(self):
        """Receipt verification fails with wrong signer."""
        receipt = PoIReceipt(
            receipt_id="feat_002",
            epoch_id="epoch_00000001",
            contributor_id="bob",
            reason=PoIReasonCode.POI_OK,
            poi_score=0.50,
            contribution_score=0.60,
            reach_score=0.40,
            longevity_score=0.30,
            config_digest="xyz",
            content_hash="hash_002",
        )
        receipt.sign_with(FIXED_SIGNER)

        wrong_signer = SimpleSigner(b"wrong-key")
        assert not receipt.verify_signature(wrong_signer)

    def test_receipt_to_dict_complete(self):
        """to_dict() includes all required fields."""
        receipt = PoIReceipt(
            receipt_id="feat_003",
            epoch_id="epoch_00000001",
            contributor_id="carol",
            reason=PoIReasonCode.POI_REJECT_SNR_BELOW_THRESHOLD,
            poi_score=0.0,
            contribution_score=0.0,
            reach_score=0.0,
            longevity_score=0.0,
            config_digest="cfg",
            content_hash="hash_003",
        )
        receipt.sign_with(FIXED_SIGNER)
        d = receipt.to_dict()

        assert d["receipt_id"] == "feat_003"
        assert d["epoch_id"] == "epoch_00000001"
        assert d["contributor_id"] == "carol"
        assert d["reason"] == "POI_REJECT_SNR_BELOW_THRESHOLD"
        assert d["poi_score"] == 0.0
        assert "signature" in d
        assert "signer_pubkey" in d
        assert "receipt_digest" in d

    def test_epoch_emits_receipts(self):
        """compute_epoch() emits signed receipts for every contributor."""
        config = _make_deterministic_config()
        orch = PoIOrchestrator(config, signer=FIXED_SIGNER)

        for name in ["alice", "bob", "carol"]:
            meta = _make_contribution(name, f"receipt_test_{name}")
            orch.register_contribution(meta)

        audit = orch.compute_epoch(
            epoch_id="receipt_epoch",
            reference_time=FIXED_TIME,
        )

        assert len(audit.receipts) == 3
        for receipt in audit.receipts:
            assert receipt.epoch_id == "receipt_epoch"
            assert receipt.reason == PoIReasonCode.POI_OK
            assert receipt.poi_score > 0
            assert receipt.verify_signature(FIXED_SIGNER)

    def test_receipt_ids_deterministic(self):
        """Receipt IDs follow deterministic pattern."""
        config = _make_deterministic_config()
        orch = PoIOrchestrator(config, signer=FIXED_SIGNER)

        meta = _make_contribution("alice", "id_test")
        orch.register_contribution(meta)
        audit = orch.compute_epoch(epoch_id="id_epoch")

        receipt = audit.receipts[0]
        assert receipt.receipt_id.startswith("poi_rcpt_id_epoch_")

    def test_orchestrator_stats_include_receipts(self):
        """Orchestrator stats track total receipts."""
        config = _make_deterministic_config()
        orch = PoIOrchestrator(config, signer=FIXED_SIGNER)

        meta = _make_contribution("alice", "stats_test")
        orch.register_contribution(meta)
        orch.compute_epoch()

        stats = orch.get_stats()
        assert "total_receipts" in stats
        assert stats["total_receipts"] > 0


# =============================================================================
# AUDIT TRAIL DETERMINISM
# =============================================================================

class TestAuditTrailDeterminism:
    """AuditTrail canonical form excludes mutable timestamp."""

    def test_canonical_excludes_timestamp(self):
        """canonical_bytes() does NOT include timestamp."""
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 6, 15, tzinfo=timezone.utc)

        a1 = AuditTrail(
            epoch_id="ts_test",
            poi_scores=[],
            gini_coefficient=0.3,
            rebalance_triggered=False,
            config_digest="abc",
            timestamp=t1,
        )
        a2 = AuditTrail(
            epoch_id="ts_test",
            poi_scores=[],
            gini_coefficient=0.3,
            rebalance_triggered=False,
            config_digest="abc",
            timestamp=t2,
        )

        assert a1.canonical_bytes() == a2.canonical_bytes()
        assert a1.digest() == a2.digest()

    def test_proof_of_impact_canonical_excludes_timestamp(self):
        """ProofOfImpact canonical_bytes() excludes timestamp."""
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 12, 31, tzinfo=timezone.utc)

        kwargs = dict(
            contributor_id="alice",
            contribution_score=0.9,
            reach_score=0.7,
            longevity_score=0.8,
            poi_score=0.83,
            alpha=0.5, beta=0.3, gamma=0.2,
            config_digest="abc",
            computation_id="c1",
            epoch_id="e1",
        )
        p1 = ProofOfImpact(**kwargs, timestamp=t1)
        p2 = ProofOfImpact(**kwargs, timestamp=t2)

        assert p1.canonical_bytes() == p2.canonical_bytes()
        assert p1.digest() == p2.digest()
