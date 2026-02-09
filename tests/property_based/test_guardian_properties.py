"""
Property-Based Tests — Guardian Council Vote Integrity
=======================================================

Standing on Giants:
- Lamport et al. (1982) — Byzantine Fault Tolerance
- Shapley & Shubik (1954) — Weighted Voting Power Index
- Hypothesis (MacIver, 2016) — Property-based testing

Invariants verified:
1. Every Guardian-produced vote is Ed25519-signed and verifiable
2. Vote numeric_value is bounded by [-1.0, 1.0]
3. IhsanVector.score() is bounded by [0.0, 1.0]
4. Tampered votes are always rejected
5. Council verdict consistency: unanimous ⟹ all approve
6. Ihsān gate: if any dimension < 0.7, passes_gate(0.95) == False
"""

import asyncio

import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

from core.sovereign.guardian_council import (
    ConsensusMode,
    CouncilVerdict,
    Guardian,
    GuardianCouncil,
    GuardianRole,
    GuardianVote,
    IhsanVector,
    Proposal,
    VoteType,
)


# ── Strategies ──────────────────────────────────────────────────────────

ihsan_float = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
confidence_float = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
guardian_roles = st.sampled_from(list(GuardianRole))
vote_types = st.sampled_from(list(VoteType))

ihsan_vectors = st.builds(
    IhsanVector,
    correctness=ihsan_float,
    safety=ihsan_float,
    beneficence=ihsan_float,
    transparency=ihsan_float,
    sustainability=ihsan_float,
)


# ── IhsanVector Properties ─────────────────────────────────────────────

class TestIhsanVectorProperties:
    """Algebraic properties of the Ihsān quality vector."""

    @given(vec=ihsan_vectors)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_score_bounded(self, vec: IhsanVector):
        """∀ v ∈ IhsanVector: 0.0 ≤ score(v) ≤ 1.0"""
        s = vec.score()
        assert 0.0 <= s <= 1.0 + 1e-9, f"Score {s} out of bounds"

    @given(vec=ihsan_vectors)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
    def test_gate_requires_min_dimensions(self, vec: IhsanVector):
        """If any dimension < 0.7, passes_gate(0.95) must be False."""
        min_dim = min(
            vec.correctness, vec.safety, vec.beneficence,
            vec.transparency, vec.sustainability,
        )
        if min_dim < 0.7:
            assert not vec.passes_gate(0.95), (
                f"Gate should fail when min dimension = {min_dim}"
            )

    def test_perfect_vector_passes(self):
        """A perfect vector (all 1.0) must always pass."""
        perfect = IhsanVector(1.0, 1.0, 1.0, 1.0, 1.0)
        assert perfect.passes_gate(0.95)
        assert perfect.score() == pytest.approx(1.0, abs=1e-9)

    def test_zero_vector_fails(self):
        """A zero vector must always fail."""
        zero = IhsanVector(0.0, 0.0, 0.0, 0.0, 0.0)
        assert not zero.passes_gate(0.95)
        assert zero.score() == pytest.approx(0.0, abs=1e-9)


# ── GuardianVote Properties ────────────────────────────────────────────

class TestGuardianVoteProperties:
    """Invariants of Guardian votes."""

    @given(role=guardian_roles, vtype=vote_types, conf=confidence_float)
    @settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
    def test_vote_numeric_bounded(self, role, vtype, conf):
        """∀ vote: -1.0 ≤ numeric_value ≤ 1.0"""
        vote = GuardianVote(
            guardian=role,
            vote_type=vtype,
            confidence=conf,
            reasoning="test",
            ihsan_assessment=IhsanVector(0.8, 0.8, 0.8, 0.8, 0.8),
        )
        assert -1.0 <= vote.numeric_value <= 1.0 + 1e-9

    @given(role=guardian_roles, vtype=vote_types, conf=confidence_float)
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_vote_hmac_fallback_verifies(self, role, vtype, conf):
        """Unsigned votes (no Ed25519) use HMAC fallback and verify."""
        vote = GuardianVote(
            guardian=role,
            vote_type=vtype,
            confidence=conf,
            reasoning="test",
            ihsan_assessment=IhsanVector(0.8, 0.8, 0.8, 0.8, 0.8),
        )
        # No .sign() called → HMAC fallback
        assert vote.signer_public_key == ""
        assert vote.verify(), "HMAC fallback verification must succeed"


# ── Guardian Ed25519 Properties ─────────────────────────────────────────

class TestGuardianEd25519Properties:
    """Every Guardian-produced vote is Ed25519-signed and verifiable."""

    @given(role=guardian_roles)
    @settings(max_examples=8, suppress_health_check=[HealthCheck.too_slow])
    def test_guardian_vote_signed(self, role):
        """∀ guardian role: evaluate() produces Ed25519-signed vote."""
        g = Guardian(role)
        p = Proposal(
            id="prop-test", title="Test", content="test",
            proposer="hypothesis", required_mode=ConsensusMode.MAJORITY,
        )
        vote = asyncio.run(g.evaluate(p))
        assert vote.signer_public_key == g.public_key
        assert vote.verify(), "Ed25519 vote verification must succeed"

    @given(role=guardian_roles)
    @settings(max_examples=8, suppress_health_check=[HealthCheck.too_slow])
    def test_guardian_tamper_detection(self, role):
        """∀ guardian: tampering with vote signature → verify fails."""
        g = Guardian(role)
        p = Proposal(
            id="prop-tamper", title="Tamper Test", content="test",
            proposer="hypothesis", required_mode=ConsensusMode.MAJORITY,
        )
        vote = asyncio.run(g.evaluate(p))

        # Tamper with the signature
        sig_bytes = bytearray(bytes.fromhex(vote.signature))
        sig_bytes[0] ^= 0xFF  # Flip all bits in first byte
        vote.signature = sig_bytes.hex()
        assert not vote.verify(), "Tampered signature must be rejected"

    @given(role=guardian_roles)
    @settings(max_examples=8, suppress_health_check=[HealthCheck.too_slow])
    def test_guardian_cross_key_rejection(self, role):
        """∀ guardian: vote signed by one key cannot verify with another."""
        g1 = Guardian(role)
        g2 = Guardian(role)
        assume(g1.public_key != g2.public_key)

        p = Proposal(
            id="prop-xkey", title="Cross-Key Test", content="test",
            proposer="hypothesis", required_mode=ConsensusMode.MAJORITY,
        )
        vote = asyncio.run(g1.evaluate(p))

        # Replace public key with g2's key
        vote.signer_public_key = g2.public_key
        assert not vote.verify(), "Cross-key verification must fail"


# ── Council Verdict Properties ──────────────────────────────────────────

class TestCouncilVerdictProperties:
    """Invariants of the council deliberation."""

    def test_council_produces_verdict(self):
        """A standard council deliberation always produces a verdict."""
        council = GuardianCouncil(ihsan_threshold=0.95)
        p = Proposal(
            id="prop-council", title="Council Test", content="test",
            proposer="hypothesis", required_mode=ConsensusMode.MAJORITY,
        )
        verdict = asyncio.run(council.deliberate(p))
        assert isinstance(verdict, CouncilVerdict)
        assert len(verdict.votes) > 0
        assert verdict.deliberation_time_ms >= 0

    def test_unanimous_implies_all_approve(self):
        """If verdict.unanimous == True, all votes are approve-type."""
        council = GuardianCouncil(ihsan_threshold=0.50)  # Low threshold
        p = Proposal(
            id="prop-unan", title="Unanimity Test", content="test",
            proposer="hypothesis", required_mode=ConsensusMode.UNANIMOUS,
        )
        verdict = asyncio.run(council.deliberate(p))
        if verdict.unanimous:
            for vote in verdict.votes:
                assert vote.vote_type in [
                    VoteType.APPROVE, VoteType.APPROVE_WITH_CONCERNS
                ], f"Unanimous but {vote.guardian.name} voted {vote.vote_type.name}"

    def test_all_votes_verified(self):
        """Every vote in a verdict must pass Ed25519 verification."""
        council = GuardianCouncil(ihsan_threshold=0.95)
        p = Proposal(
            id="prop-verify", title="Verify All", content="test",
            proposer="hypothesis", required_mode=ConsensusMode.MAJORITY,
        )
        verdict = asyncio.run(council.deliberate(p))
        for vote in verdict.votes:
            assert vote.verify(), (
                f"Vote from {vote.guardian.name} failed Ed25519 verification"
            )
