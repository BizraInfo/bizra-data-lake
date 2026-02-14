"""
Guardian Council — Comprehensive Test Suite
=============================================================================

Tests for core.sovereign.guardian_council covering:
- Enums: GuardianRole, VoteType, ConsensusMode
- Data classes: IhsanVector, GuardianVote, Proposal, CouncilVerdict
- Guardian agent: evaluate, default heuristic, domain weights, Ed25519 signing
- GuardianCouncil: deliberate, consensus modes, Ihsan gate, veto logic
- Convenience: create_council, validate() monkey-patch

Created: 2026-02-11
"""

import asyncio
import hashlib
import hmac as hmac_mod
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.pci.crypto import (
    domain_separated_digest,
    generate_keypair,
    sign_message,
    verify_signature,
)
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
    create_council,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def high_ihsan() -> IhsanVector:
    """IhsanVector that passes the default 0.95 gate."""
    return IhsanVector(
        correctness=0.98,
        safety=0.97,
        beneficence=0.96,
        transparency=0.95,
        sustainability=0.94,
    )


@pytest.fixture
def low_ihsan() -> IhsanVector:
    """IhsanVector that fails the gate on overall score."""
    return IhsanVector(
        correctness=0.6,
        safety=0.6,
        beneficence=0.5,
        transparency=0.5,
        sustainability=0.5,
    )


@pytest.fixture
def one_weak_dimension_ihsan() -> IhsanVector:
    """IhsanVector with high overall score but one dimension below 0.7."""
    return IhsanVector(
        correctness=0.99,
        safety=0.99,
        beneficence=0.99,
        transparency=0.65,  # Below the 0.7 minimum
        sustainability=0.99,
    )


@pytest.fixture
def sample_proposal() -> Proposal:
    """A standard proposal for testing."""
    return Proposal(
        id="test-001",
        title="Test Proposal",
        content="This is a test proposal with evidence and reasoning. "
        "Because of verified sources, therefore we conclude this is valid.",
        proposer="tester",
        context={"snr_score": 0.90, "ihsan_score": 0.92},
        required_mode=ConsensusMode.MAJORITY,
        urgency=0.5,
    )


@pytest.fixture
def minimal_proposal() -> Proposal:
    """A minimal proposal with very little content."""
    return Proposal(
        id="min-001",
        title="Minimal",
        content="x",
        proposer="tester",
    )


@pytest.fixture
def unsafe_proposal() -> Proposal:
    """A proposal containing safety concern keywords."""
    return Proposal(
        id="unsafe-001",
        title="Unsafe Proposal",
        content="This could be dangerous and exploit a vulnerability causing harm.",
        proposer="tester",
    )


@pytest.fixture
def council() -> GuardianCouncil:
    """A default-initialized Guardian Council."""
    return GuardianCouncil()


@pytest.fixture
def keypair():
    """A fresh Ed25519 keypair."""
    return generate_keypair()


# =============================================================================
# 1. TestGuardianRole
# =============================================================================


class TestGuardianRole:
    """Tests for the GuardianRole enum."""

    def test_has_eight_roles(self):
        """GuardianRole must define exactly 8 roles."""
        assert len(GuardianRole) == 8

    def test_role_names(self):
        """All expected role names must be present."""
        expected = {
            "ARCHITECT",
            "SECURITY",
            "ETHICS",
            "REASONING",
            "KNOWLEDGE",
            "CREATIVE",
            "INTEGRATION",
            "NUCLEUS",
        }
        actual = {role.name for role in GuardianRole}
        assert actual == expected

    def test_roles_are_unique(self):
        """Each role must have a unique auto() value."""
        values = [role.value for role in GuardianRole]
        assert len(values) == len(set(values))


# =============================================================================
# 2. TestVoteType
# =============================================================================


class TestVoteType:
    """Tests for the VoteType enum."""

    def test_has_five_types(self):
        """VoteType must define exactly 5 types."""
        assert len(VoteType) == 5

    def test_type_names(self):
        """All expected vote type names must be present."""
        expected = {
            "APPROVE",
            "APPROVE_WITH_CONCERNS",
            "ABSTAIN",
            "REJECT_SOFT",
            "REJECT_HARD",
        }
        actual = {vt.name for vt in VoteType}
        assert actual == expected

    def test_types_are_unique(self):
        """Each type must have a unique auto() value."""
        values = [vt.value for vt in VoteType]
        assert len(values) == len(set(values))


# =============================================================================
# 3. TestConsensusMode
# =============================================================================


class TestConsensusMode:
    """Tests for the ConsensusMode enum."""

    def test_has_five_modes(self):
        """ConsensusMode must define exactly 5 modes."""
        assert len(ConsensusMode) == 5

    def test_mode_names(self):
        """All expected mode names must be present."""
        expected = {
            "UNANIMOUS",
            "SUPERMAJORITY",
            "MAJORITY",
            "WEIGHTED",
            "NUCLEUS_OVERRIDE",
        }
        actual = {m.name for m in ConsensusMode}
        assert actual == expected

    def test_modes_are_unique(self):
        """Each mode must have a unique auto() value."""
        values = [m.value for m in ConsensusMode]
        assert len(values) == len(set(values))


# =============================================================================
# 4. TestIhsanVector
# =============================================================================


class TestIhsanVector:
    """Tests for the IhsanVector dataclass."""

    def test_default_values_are_zero(self):
        """All dimensions default to 0.0."""
        vec = IhsanVector()
        assert vec.correctness == 0.0
        assert vec.safety == 0.0
        assert vec.beneficence == 0.0
        assert vec.transparency == 0.0
        assert vec.sustainability == 0.0

    def test_score_with_default_weights(self, high_ihsan):
        """Score with default weights must sum correctly."""
        # Default weights: correctness=0.25, safety=0.25, beneficence=0.20,
        #                  transparency=0.15, sustainability=0.15
        expected = (
            0.98 * 0.25
            + 0.97 * 0.25
            + 0.96 * 0.20
            + 0.95 * 0.15
            + 0.94 * 0.15
        )
        assert abs(high_ihsan.score() - expected) < 1e-9

    def test_score_with_custom_weights(self):
        """Score with custom weights applies those weights."""
        vec = IhsanVector(
            correctness=1.0,
            safety=0.0,
            beneficence=0.0,
            transparency=0.0,
            sustainability=0.0,
        )
        custom_weights = {
            "correctness": 1.0,
            "safety": 0.0,
            "beneficence": 0.0,
            "transparency": 0.0,
            "sustainability": 0.0,
        }
        assert vec.score(custom_weights) == 1.0

    def test_score_uniform_weights(self):
        """When all dimensions are equal, score equals that value regardless of weights."""
        vec = IhsanVector(
            correctness=0.8,
            safety=0.8,
            beneficence=0.8,
            transparency=0.8,
            sustainability=0.8,
        )
        assert abs(vec.score() - 0.8) < 1e-9

    def test_score_zero_vector(self):
        """All-zero vector produces score 0.0."""
        vec = IhsanVector()
        assert vec.score() == 0.0

    def test_passes_gate_high_quality(self, high_ihsan):
        """A high-quality vector passes the default 0.95 gate."""
        assert high_ihsan.passes_gate(0.95) is True

    def test_passes_gate_fails_on_low_score(self, low_ihsan):
        """A low-score vector fails the gate."""
        assert low_ihsan.passes_gate(0.95) is False

    def test_passes_gate_fails_on_weak_dimension(self, one_weak_dimension_ihsan):
        """Even if overall score is high, one dimension below 0.7 fails the gate."""
        # The min dimension is 0.65 (< 0.7)
        assert one_weak_dimension_ihsan.passes_gate(0.95) is False

    def test_passes_gate_custom_threshold(self):
        """A vector that fails at 0.95 can pass at a lower threshold."""
        vec = IhsanVector(
            correctness=0.85,
            safety=0.85,
            beneficence=0.85,
            transparency=0.85,
            sustainability=0.85,
        )
        assert vec.passes_gate(0.80) is True
        assert vec.passes_gate(0.95) is False

    def test_passes_gate_boundary_min_dimension(self):
        """A dimension at exactly 0.7 should still pass the minimum check."""
        vec = IhsanVector(
            correctness=1.0,
            safety=1.0,
            beneficence=1.0,
            transparency=0.70,
            sustainability=1.0,
        )
        # Score = 1.0*0.25 + 1.0*0.25 + 1.0*0.20 + 0.70*0.15 + 1.0*0.15 = 0.955
        assert vec.passes_gate(0.95) is True


# =============================================================================
# 5. TestGuardianVote
# =============================================================================


class TestGuardianVote:
    """Tests for the GuardianVote dataclass."""

    def _make_vote(self, vote_type=VoteType.APPROVE, confidence=0.9):
        """Helper to create a GuardianVote with HMAC fallback."""
        return GuardianVote(
            guardian=GuardianRole.ARCHITECT,
            vote_type=vote_type,
            confidence=confidence,
            reasoning="Test reasoning",
            ihsan_assessment=IhsanVector(0.9, 0.9, 0.9, 0.9, 0.9),
        )

    def test_hmac_signature_generated_on_creation(self):
        """A new vote without explicit signature gets an HMAC fallback."""
        vote = self._make_vote()
        assert vote.signature != ""
        assert len(vote.signature) == 64  # SHA-256 hex digest

    def test_hmac_signature_is_deterministic(self):
        """Two votes with the same data and timestamp produce the same HMAC."""
        ts = datetime(2026, 1, 1, 12, 0, 0)
        ihsan = IhsanVector(0.9, 0.9, 0.9, 0.9, 0.9)
        vote_a = GuardianVote(
            guardian=GuardianRole.SECURITY,
            vote_type=VoteType.APPROVE,
            confidence=0.95,
            reasoning="Test",
            ihsan_assessment=ihsan,
            timestamp=ts,
        )
        vote_b = GuardianVote(
            guardian=GuardianRole.SECURITY,
            vote_type=VoteType.APPROVE,
            confidence=0.95,
            reasoning="Test",
            ihsan_assessment=ihsan,
            timestamp=ts,
        )
        assert vote_a.signature == vote_b.signature

    def test_verify_hmac_valid(self):
        """Verify returns True for a freshly created HMAC-signed vote."""
        vote = self._make_vote()
        assert vote.signer_public_key == ""  # HMAC fallback
        assert vote.verify() is True

    def test_verify_hmac_tampered(self):
        """Verify returns False when HMAC signature is tampered."""
        vote = self._make_vote()
        vote.signature = "a" * 64  # Tampered
        assert vote.verify() is False

    def test_ed25519_sign_and_verify(self, keypair):
        """Ed25519 sign replaces HMAC, and verify succeeds."""
        priv, pub = keypair
        vote = self._make_vote()
        original_sig = vote.signature

        vote.sign(priv, pub)

        assert vote.signature != original_sig
        assert vote.signer_public_key == pub
        assert vote.verify() is True

    def test_ed25519_verify_fails_with_wrong_key(self, keypair):
        """Ed25519 verify fails when a different public key is used."""
        priv, pub = keypair
        _, wrong_pub = generate_keypair()

        vote = self._make_vote()
        vote.sign(priv, pub)

        # Replace public key with a different one
        vote.signer_public_key = wrong_pub
        assert vote.verify() is False

    def test_ed25519_verify_fails_with_tampered_signature(self, keypair):
        """Ed25519 verify fails with a tampered signature."""
        priv, pub = keypair
        vote = self._make_vote()
        vote.sign(priv, pub)

        # Tamper with the signature (flip last byte)
        sig_bytes = bytes.fromhex(vote.signature)
        tampered = sig_bytes[:-1] + bytes([(sig_bytes[-1] ^ 0xFF)])
        vote.signature = tampered.hex()

        assert vote.verify() is False

    def test_numeric_value_approve(self):
        """APPROVE: 1.0 * confidence."""
        vote = self._make_vote(VoteType.APPROVE, 0.9)
        assert abs(vote.numeric_value - 0.9) < 1e-9

    def test_numeric_value_approve_with_concerns(self):
        """APPROVE_WITH_CONCERNS: 0.7 * confidence."""
        vote = self._make_vote(VoteType.APPROVE_WITH_CONCERNS, 1.0)
        assert abs(vote.numeric_value - 0.7) < 1e-9

    def test_numeric_value_abstain(self):
        """ABSTAIN: 0.0 regardless of confidence."""
        vote = self._make_vote(VoteType.ABSTAIN, 0.99)
        assert vote.numeric_value == 0.0

    def test_numeric_value_reject_soft(self):
        """REJECT_SOFT: -0.5 * confidence."""
        vote = self._make_vote(VoteType.REJECT_SOFT, 0.8)
        assert abs(vote.numeric_value - (-0.4)) < 1e-9

    def test_numeric_value_reject_hard(self):
        """REJECT_HARD: -1.0 * confidence."""
        vote = self._make_vote(VoteType.REJECT_HARD, 1.0)
        assert abs(vote.numeric_value - (-1.0)) < 1e-9

    def test_canonical_data_str(self):
        """_canonical_data_str includes guardian name, vote type, confidence, and timestamp ISO."""
        ts = datetime(2026, 6, 15, 10, 30, 0)
        vote = GuardianVote(
            guardian=GuardianRole.NUCLEUS,
            vote_type=VoteType.REJECT_HARD,
            confidence=0.75,
            reasoning="Reason",
            ihsan_assessment=IhsanVector(),
            timestamp=ts,
        )
        canonical = vote._canonical_data_str()
        assert "NUCLEUS" in canonical
        assert "REJECT_HARD" in canonical
        assert "0.75" in canonical
        assert ts.isoformat() in canonical


# =============================================================================
# 6. TestProposal
# =============================================================================


class TestProposal:
    """Tests for the Proposal dataclass."""

    def test_creation_with_all_fields(self):
        """Proposal stores all provided fields."""
        p = Proposal(
            id="p-1",
            title="Title",
            content="Content",
            proposer="alice",
            context={"key": "val"},
            required_mode=ConsensusMode.UNANIMOUS,
            urgency=0.9,
        )
        assert p.id == "p-1"
        assert p.title == "Title"
        assert p.content == "Content"
        assert p.proposer == "alice"
        assert p.context == {"key": "val"}
        assert p.required_mode == ConsensusMode.UNANIMOUS
        assert p.urgency == 0.9

    def test_default_values(self):
        """Proposal defaults: empty context, MAJORITY mode, 0.5 urgency."""
        p = Proposal(id="d-1", title="T", content="C", proposer="bob")
        assert p.context == {}
        assert p.required_mode == ConsensusMode.MAJORITY
        assert p.urgency == 0.5
        assert isinstance(p.created_at, datetime)

    def test_content_can_be_any_type(self):
        """Content field accepts Any type."""
        p = Proposal(id="a-1", title="T", content={"nested": [1, 2]}, proposer="x")
        assert p.content == {"nested": [1, 2]}


# =============================================================================
# 7. TestCouncilVerdict
# =============================================================================


class TestCouncilVerdict:
    """Tests for the CouncilVerdict dataclass."""

    def _make_vote(self, role, vote_type, confidence=0.9):
        """Helper to create a vote for verdict testing."""
        return GuardianVote(
            guardian=role,
            vote_type=vote_type,
            confidence=confidence,
            reasoning=f"{role.name} reasoning",
            ihsan_assessment=IhsanVector(0.9, 0.9, 0.9, 0.9, 0.9),
        )

    def test_vote_summary(self):
        """vote_summary counts votes by type correctly."""
        votes = [
            self._make_vote(GuardianRole.ARCHITECT, VoteType.APPROVE),
            self._make_vote(GuardianRole.SECURITY, VoteType.APPROVE),
            self._make_vote(GuardianRole.ETHICS, VoteType.APPROVE_WITH_CONCERNS),
            self._make_vote(GuardianRole.REASONING, VoteType.ABSTAIN),
            self._make_vote(GuardianRole.KNOWLEDGE, VoteType.REJECT_SOFT),
        ]
        verdict = CouncilVerdict(
            proposal_id="v-1",
            approved=True,
            consensus_mode=ConsensusMode.MAJORITY,
            votes=votes,
            aggregate_score=0.5,
            ihsan_passed=True,
            dissenting_opinions=[],
            recommendations=[],
        )
        summary = verdict.vote_summary
        assert summary[VoteType.APPROVE] == 2
        assert summary[VoteType.APPROVE_WITH_CONCERNS] == 1
        assert summary[VoteType.ABSTAIN] == 1
        assert summary[VoteType.REJECT_SOFT] == 1
        assert summary[VoteType.REJECT_HARD] == 0

    def test_unanimous_true(self):
        """unanimous is True when all votes are APPROVE or APPROVE_WITH_CONCERNS."""
        votes = [
            self._make_vote(GuardianRole.ARCHITECT, VoteType.APPROVE),
            self._make_vote(GuardianRole.SECURITY, VoteType.APPROVE_WITH_CONCERNS),
            self._make_vote(GuardianRole.ETHICS, VoteType.APPROVE),
        ]
        verdict = CouncilVerdict(
            proposal_id="u-1",
            approved=True,
            consensus_mode=ConsensusMode.UNANIMOUS,
            votes=votes,
            aggregate_score=0.8,
            ihsan_passed=True,
            dissenting_opinions=[],
            recommendations=[],
        )
        assert verdict.unanimous is True

    def test_unanimous_false(self):
        """unanimous is False when any vote is not an approval."""
        votes = [
            self._make_vote(GuardianRole.ARCHITECT, VoteType.APPROVE),
            self._make_vote(GuardianRole.SECURITY, VoteType.REJECT_SOFT),
        ]
        verdict = CouncilVerdict(
            proposal_id="u-2",
            approved=False,
            consensus_mode=ConsensusMode.MAJORITY,
            votes=votes,
            aggregate_score=0.3,
            ihsan_passed=True,
            dissenting_opinions=[],
            recommendations=[],
        )
        assert verdict.unanimous is False

    def test_empty_votes(self):
        """Verdict with no votes has empty summary and unanimous=True (vacuously)."""
        verdict = CouncilVerdict(
            proposal_id="e-1",
            approved=False,
            consensus_mode=ConsensusMode.MAJORITY,
            votes=[],
            aggregate_score=0.0,
            ihsan_passed=False,
            dissenting_opinions=[],
            recommendations=[],
        )
        for vt in VoteType:
            assert verdict.vote_summary[vt] == 0
        # 0 == 0 is True (vacuous truth)
        assert verdict.unanimous is True


# =============================================================================
# 8. TestGuardian
# =============================================================================


class TestGuardian:
    """Tests for the Guardian agent class."""

    def test_guardian_has_keypair(self):
        """Each Guardian gets a unique Ed25519 keypair at init."""
        g = Guardian(GuardianRole.ARCHITECT)
        assert len(g.private_key) == 64  # 32 bytes hex
        assert len(g.public_key) == 64  # 32 bytes hex
        assert g.private_key != g.public_key

    def test_guardian_unique_keypairs(self):
        """Different Guardian instances have different keypairs."""
        g1 = Guardian(GuardianRole.ARCHITECT)
        g2 = Guardian(GuardianRole.ARCHITECT)
        assert g1.public_key != g2.public_key

    def test_domain_weights_for_all_roles(self):
        """DOMAIN_WEIGHTS must cover every GuardianRole."""
        for role in GuardianRole:
            assert role in Guardian.DOMAIN_WEIGHTS
            weights = Guardian.DOMAIN_WEIGHTS[role]
            # Weights should sum to ~1.0
            total = sum(weights.values())
            assert abs(total - 1.0) < 1e-9, f"{role.name} weights sum to {total}"

    def test_domain_weights_all_dimensions(self):
        """Each role's weights include all five Ihsan dimensions."""
        dims = {"correctness", "safety", "beneficence", "transparency", "sustainability"}
        for role in GuardianRole:
            assert set(Guardian.DOMAIN_WEIGHTS[role].keys()) == dims

    @pytest.mark.asyncio
    async def test_evaluate_returns_guardian_vote(self, sample_proposal):
        """evaluate() returns a GuardianVote with valid Ed25519 signature."""
        g = Guardian(GuardianRole.SECURITY)
        vote = await g.evaluate(sample_proposal)

        assert isinstance(vote, GuardianVote)
        assert vote.guardian == GuardianRole.SECURITY
        assert vote.signer_public_key == g.public_key
        assert vote.verify() is True

    @pytest.mark.asyncio
    async def test_evaluate_appends_to_history(self, sample_proposal):
        """evaluate() appends the vote to vote_history."""
        g = Guardian(GuardianRole.REASONING)
        assert len(g.vote_history) == 0
        vote = await g.evaluate(sample_proposal)
        assert len(g.vote_history) == 1
        assert g.vote_history[0] is vote

    @pytest.mark.asyncio
    async def test_default_evaluate_with_evidence(self, sample_proposal):
        """Default evaluate assigns higher correctness when evidence keywords present."""
        g = Guardian(GuardianRole.KNOWLEDGE)
        vote = await g.evaluate(sample_proposal)
        # "because", "verified", "therefore", "conclude" are in sample_proposal
        assert vote.ihsan_assessment.correctness > 0.75
        assert vote.ihsan_assessment.transparency > 0.78

    @pytest.mark.asyncio
    async def test_default_evaluate_safety_concern(self, unsafe_proposal):
        """Default evaluate reduces safety score when danger keywords present."""
        g = Guardian(GuardianRole.SECURITY)
        vote = await g.evaluate(unsafe_proposal)
        assert vote.ihsan_assessment.safety == 0.70

    @pytest.mark.asyncio
    async def test_default_evaluate_minimal_content(self, minimal_proposal):
        """Default evaluate still produces valid scores for minimal content."""
        g = Guardian(GuardianRole.CREATIVE)
        vote = await g.evaluate(minimal_proposal)
        assert 0.0 <= vote.ihsan_assessment.correctness <= 1.0
        assert 0.0 <= vote.ihsan_assessment.safety <= 1.0

    @pytest.mark.asyncio
    async def test_custom_evaluate_fn(self):
        """Guardian can use a custom evaluate_fn."""

        def custom_eval(proposal: Proposal) -> IhsanVector:
            return IhsanVector(1.0, 1.0, 1.0, 1.0, 1.0)

        g = Guardian(GuardianRole.NUCLEUS, evaluate_fn=custom_eval)
        proposal = Proposal(id="c-1", title="T", content="C", proposer="x")
        vote = await g.evaluate(proposal)
        assert vote.ihsan_assessment.correctness == 1.0
        assert vote.vote_type == VoteType.APPROVE

    @pytest.mark.asyncio
    async def test_vote_type_thresholds(self):
        """Test vote type determination at different score levels."""
        cases = [
            (IhsanVector(1.0, 1.0, 1.0, 1.0, 1.0), VoteType.APPROVE),
            (IhsanVector(0.5, 0.5, 0.5, 0.5, 0.5), VoteType.REJECT_SOFT),  # score=0.5, >= 0.50
            (IhsanVector(0.3, 0.3, 0.3, 0.3, 0.3), VoteType.REJECT_HARD),  # score=0.3, < 0.50
        ]
        for ihsan, expected_type in cases:
            g = Guardian(
                GuardianRole.ARCHITECT,
                evaluate_fn=lambda p, iv=ihsan: iv,
            )
            proposal = Proposal(id="t-1", title="T", content="C", proposer="x")
            vote = await g.evaluate(proposal)
            assert vote.vote_type == expected_type, (
                f"Score {ihsan.score(Guardian.DOMAIN_WEIGHTS[GuardianRole.ARCHITECT]):.3f} "
                f"expected {expected_type.name}, got {vote.vote_type.name}"
            )

    def test_generate_reasoning_includes_weakest(self):
        """_generate_reasoning mentions the weakest dimension."""
        g = Guardian(GuardianRole.ETHICS)
        ihsan = IhsanVector(0.9, 0.9, 0.9, 0.5, 0.9)  # transparency is weakest
        proposal = Proposal(id="r-1", title="T", content="C", proposer="x")
        reasoning = g._generate_reasoning(proposal, ihsan, 0.85)
        assert "ETHICS" in reasoning
        assert "transparency" in reasoning
        assert "0.50" in reasoning


# =============================================================================
# 9. TestGuardianCouncil
# =============================================================================


class TestGuardianCouncil:
    """Tests for the GuardianCouncil orchestrator."""

    def test_initialization_has_all_guardians(self, council):
        """Council initializes with all 8 guardians."""
        assert len(council.guardians) == 8
        for role in GuardianRole:
            assert role in council.guardians
            assert isinstance(council.guardians[role], Guardian)

    def test_default_ihsan_threshold(self, council):
        """Default Ihsan threshold is 0.95."""
        assert council.ihsan_threshold == 0.95

    def test_default_veto_enabled(self, council):
        """Veto is enabled by default."""
        assert council.enable_veto is True

    def test_custom_init(self):
        """Council can be initialized with custom threshold and veto setting."""
        c = GuardianCouncil(ihsan_threshold=0.80, enable_veto=False)
        assert c.ihsan_threshold == 0.80
        assert c.enable_veto is False

    def test_voting_power_defined(self):
        """VOTING_POWER has entries for all 8 roles."""
        for role in GuardianRole:
            assert role in GuardianCouncil.VOTING_POWER

    def test_nucleus_has_highest_weight(self):
        """NUCLEUS has the highest voting power."""
        max_role = max(GuardianCouncil.VOTING_POWER, key=GuardianCouncil.VOTING_POWER.get)
        assert max_role == GuardianRole.NUCLEUS

    def test_set_guardian_evaluator(self, council):
        """set_guardian_evaluator replaces a guardian's evaluate_fn."""
        custom_fn = lambda p: IhsanVector(1.0, 1.0, 1.0, 1.0, 1.0)
        council.set_guardian_evaluator(GuardianRole.ARCHITECT, custom_fn)
        assert council.guardians[GuardianRole.ARCHITECT].evaluate_fn is custom_fn

    def test_set_inference_gateway(self, council):
        """set_inference_gateway stores the gateway object."""
        mock_gateway = MagicMock()
        council.set_inference_gateway(mock_gateway)
        assert council._inference_gateway is mock_gateway

    @pytest.mark.asyncio
    async def test_deliberate_returns_verdict(self, council, sample_proposal):
        """deliberate() returns a CouncilVerdict."""
        verdict = await council.deliberate(sample_proposal)
        assert isinstance(verdict, CouncilVerdict)
        assert verdict.proposal_id == "test-001"
        assert verdict.consensus_mode == ConsensusMode.MAJORITY

    @pytest.mark.asyncio
    async def test_deliberate_records_verdict(self, council, sample_proposal):
        """deliberate() appends the verdict to council.verdicts."""
        assert len(council.verdicts) == 0
        await council.deliberate(sample_proposal)
        assert len(council.verdicts) == 1

    @pytest.mark.asyncio
    async def test_deliberate_has_deliberation_time(self, council, sample_proposal):
        """Verdict includes a positive deliberation_time_ms."""
        verdict = await council.deliberate(sample_proposal)
        assert verdict.deliberation_time_ms >= 0.0

    @pytest.mark.asyncio
    async def test_deliberate_all_votes_verified(self, council, sample_proposal):
        """All votes in the verdict have valid Ed25519 signatures."""
        verdict = await council.deliberate(sample_proposal)
        for vote in verdict.votes:
            assert vote.verify() is True

    @pytest.mark.asyncio
    async def test_deliberate_majority_approval(self):
        """When all guardians return high scores, MAJORITY proposal is approved."""
        council = GuardianCouncil(ihsan_threshold=0.90)

        # Make all guardians approve with high Ihsan
        def perfect_eval(p: Proposal) -> IhsanVector:
            return IhsanVector(0.99, 0.99, 0.99, 0.99, 0.99)

        for role in GuardianRole:
            council.set_guardian_evaluator(role, perfect_eval)

        proposal = Proposal(
            id="maj-1",
            title="Good Proposal",
            content="Excellent content",
            proposer="tester",
            required_mode=ConsensusMode.MAJORITY,
        )
        verdict = await council.deliberate(proposal)
        assert verdict.approved is True
        assert verdict.ihsan_passed is True
        assert len(verdict.votes) == 8

    @pytest.mark.asyncio
    async def test_deliberate_hard_veto_blocks(self):
        """A single REJECT_HARD vote blocks the proposal when veto is enabled."""
        council = GuardianCouncil(enable_veto=True, ihsan_threshold=0.80)

        def low_eval(p: Proposal) -> IhsanVector:
            return IhsanVector(0.3, 0.3, 0.3, 0.3, 0.3)

        def high_eval(p: Proposal) -> IhsanVector:
            return IhsanVector(0.99, 0.99, 0.99, 0.99, 0.99)

        # All guardians approve except SECURITY which rejects hard
        for role in GuardianRole:
            council.set_guardian_evaluator(role, high_eval)
        council.set_guardian_evaluator(GuardianRole.SECURITY, low_eval)

        proposal = Proposal(
            id="veto-1",
            title="Vetoed",
            content="Content",
            proposer="tester",
            required_mode=ConsensusMode.MAJORITY,
        )
        verdict = await council.deliberate(proposal)
        assert verdict.approved is False
        assert verdict.aggregate_score == -1.0
        assert len(verdict.recommendations) > 0

    @pytest.mark.asyncio
    async def test_deliberate_hard_veto_disabled(self):
        """When veto is disabled, REJECT_HARD does not block by itself."""
        council = GuardianCouncil(enable_veto=False, ihsan_threshold=0.80)

        def low_eval(p: Proposal) -> IhsanVector:
            return IhsanVector(0.3, 0.3, 0.3, 0.3, 0.3)

        def high_eval(p: Proposal) -> IhsanVector:
            return IhsanVector(0.99, 0.99, 0.99, 0.99, 0.99)

        for role in GuardianRole:
            council.set_guardian_evaluator(role, high_eval)
        council.set_guardian_evaluator(GuardianRole.SECURITY, low_eval)

        proposal = Proposal(
            id="noveto-1",
            title="No Veto",
            content="Content",
            proposer="tester",
            required_mode=ConsensusMode.MAJORITY,
        )
        verdict = await council.deliberate(proposal)
        # Aggregate score is still > 0 because 7 of 8 approve
        # But ihsan gate may still override
        assert verdict.aggregate_score != -1.0

    @pytest.mark.asyncio
    async def test_deliberate_unanimous_mode(self):
        """UNANIMOUS mode fails if any guardian does not approve."""
        council = GuardianCouncil(ihsan_threshold=0.80)

        def approve_eval(p: Proposal) -> IhsanVector:
            return IhsanVector(0.99, 0.99, 0.99, 0.99, 0.99)

        def abstain_eval(p: Proposal) -> IhsanVector:
            # Score ~0.75 -> ABSTAIN
            return IhsanVector(0.75, 0.75, 0.75, 0.75, 0.75)

        for role in GuardianRole:
            council.set_guardian_evaluator(role, approve_eval)
        council.set_guardian_evaluator(GuardianRole.CREATIVE, abstain_eval)

        proposal = Proposal(
            id="unan-1",
            title="Unanimity Test",
            content="Content",
            proposer="tester",
            required_mode=ConsensusMode.UNANIMOUS,
        )
        verdict = await council.deliberate(proposal)
        assert verdict.approved is False
        assert verdict.consensus_mode == ConsensusMode.UNANIMOUS

    @pytest.mark.asyncio
    async def test_deliberate_unanimous_all_approve(self):
        """UNANIMOUS mode succeeds when all guardians approve."""
        council = GuardianCouncil(ihsan_threshold=0.80)

        def perfect_eval(p: Proposal) -> IhsanVector:
            return IhsanVector(0.99, 0.99, 0.99, 0.99, 0.99)

        for role in GuardianRole:
            council.set_guardian_evaluator(role, perfect_eval)

        proposal = Proposal(
            id="unan-2",
            title="Unanimity Pass",
            content="Content",
            proposer="tester",
            required_mode=ConsensusMode.UNANIMOUS,
        )
        verdict = await council.deliberate(proposal)
        assert verdict.approved is True

    @pytest.mark.asyncio
    async def test_ihsan_gate_override(self):
        """
        Even when all guardians approve, if combined Ihsan fails,
        the verdict is REJECTED.
        """
        council = GuardianCouncil(ihsan_threshold=0.95)

        # Ihsan dimensions are just below passing but high enough to approve individually
        def borderline_eval(p: Proposal) -> IhsanVector:
            return IhsanVector(0.88, 0.88, 0.88, 0.88, 0.88)

        for role in GuardianRole:
            council.set_guardian_evaluator(role, borderline_eval)

        proposal = Proposal(
            id="gate-1",
            title="Ihsan Gate Test",
            content="Content",
            proposer="tester",
            required_mode=ConsensusMode.MAJORITY,
        )
        verdict = await council.deliberate(proposal)
        # All vote APPROVE_WITH_CONCERNS (score ~0.88), aggregate > 0
        # But Ihsan gate at 0.95 should fail (combined ~0.88 < 0.95)
        assert verdict.ihsan_passed is False
        assert verdict.approved is False

    @pytest.mark.asyncio
    async def test_supermajority_mode(self):
        """SUPERMAJORITY requires 2/3 approvals."""
        council = GuardianCouncil(ihsan_threshold=0.80)

        def perfect_eval(p: Proposal) -> IhsanVector:
            return IhsanVector(0.99, 0.99, 0.99, 0.99, 0.99)

        for role in GuardianRole:
            council.set_guardian_evaluator(role, perfect_eval)

        proposal = Proposal(
            id="super-1",
            title="Supermajority Test",
            content="Content",
            proposer="tester",
            required_mode=ConsensusMode.SUPERMAJORITY,
        )
        verdict = await council.deliberate(proposal)
        assert verdict.approved is True

    @pytest.mark.asyncio
    async def test_nucleus_override_mode(self):
        """NUCLEUS_OVERRIDE: approval depends only on Nucleus vote."""
        council = GuardianCouncil(ihsan_threshold=0.80)

        def perfect_eval(p: Proposal) -> IhsanVector:
            return IhsanVector(0.99, 0.99, 0.99, 0.99, 0.99)

        def reject_eval(p: Proposal) -> IhsanVector:
            return IhsanVector(0.3, 0.3, 0.3, 0.3, 0.3)

        # All reject except Nucleus approves
        for role in GuardianRole:
            council.set_guardian_evaluator(role, reject_eval)
        council.set_guardian_evaluator(GuardianRole.NUCLEUS, perfect_eval)

        proposal = Proposal(
            id="nuc-1",
            title="Nucleus Override",
            content="Content",
            proposer="tester",
            required_mode=ConsensusMode.NUCLEUS_OVERRIDE,
        )
        verdict = await council.deliberate(proposal)
        # Ihsan gate will still check, but nucleus_override checks nucleus only
        # With mostly low scores, combined ihsan may fail
        assert verdict.consensus_mode == ConsensusMode.NUCLEUS_OVERRIDE

    @pytest.mark.asyncio
    async def test_weighted_mode(self):
        """WEIGHTED mode requires aggregate_score >= 0.5."""
        council = GuardianCouncil(ihsan_threshold=0.80)

        def high_eval(p: Proposal) -> IhsanVector:
            return IhsanVector(0.99, 0.99, 0.99, 0.99, 0.99)

        for role in GuardianRole:
            council.set_guardian_evaluator(role, high_eval)

        proposal = Proposal(
            id="wt-1",
            title="Weighted Test",
            content="Content",
            proposer="tester",
            required_mode=ConsensusMode.WEIGHTED,
        )
        verdict = await council.deliberate(proposal)
        assert verdict.approved is True
        assert verdict.aggregate_score >= 0.5

    @pytest.mark.asyncio
    async def test_dissenting_opinions_collected(self):
        """Dissenting opinions include reasoning from REJECT votes."""
        council = GuardianCouncil(enable_veto=False, ihsan_threshold=0.80)

        def reject_eval(p: Proposal) -> IhsanVector:
            return IhsanVector(0.4, 0.4, 0.4, 0.4, 0.4)

        for role in GuardianRole:
            council.set_guardian_evaluator(role, reject_eval)

        proposal = Proposal(
            id="diss-1",
            title="Dissent Test",
            content="C",
            proposer="tester",
        )
        verdict = await council.deliberate(proposal)
        assert len(verdict.dissenting_opinions) > 0

    @pytest.mark.asyncio
    async def test_recommendations_on_weak_dimensions(self):
        """When dimensions are weak, recommendations suggest improvements."""
        council = GuardianCouncil(ihsan_threshold=0.80)

        def weak_eval(p: Proposal) -> IhsanVector:
            return IhsanVector(0.9, 0.9, 0.5, 0.5, 0.9)  # beneficence & transparency weak

        for role in GuardianRole:
            council.set_guardian_evaluator(role, weak_eval)

        proposal = Proposal(
            id="rec-1",
            title="Rec Test",
            content="C",
            proposer="tester",
        )
        verdict = await council.deliberate(proposal)
        recommendation_text = " ".join(verdict.recommendations).lower()
        assert "beneficence" in recommendation_text or "transparency" in recommendation_text

    def test_combine_ihsan_vectors_empty(self, council):
        """Combining empty vote list returns zero vector."""
        combined = council._combine_ihsan_vectors([])
        assert combined.score() == 0.0

    def test_combine_ihsan_vectors_single(self, council):
        """Combining a single vote returns that vote's ihsan (weighted by confidence and power)."""
        ihsan = IhsanVector(0.9, 0.8, 0.7, 0.6, 0.5)
        vote = GuardianVote(
            guardian=GuardianRole.ARCHITECT,
            vote_type=VoteType.APPROVE,
            confidence=1.0,
            reasoning="R",
            ihsan_assessment=ihsan,
        )
        combined = council._combine_ihsan_vectors([vote])
        # With weight 1.0 (power) * 1.0 (confidence), values should be unchanged
        assert abs(combined.correctness - 0.9) < 1e-9
        assert abs(combined.safety - 0.8) < 1e-9

    def test_create_veto_verdict(self, council):
        """_create_veto_verdict produces a rejected verdict with -1.0 score."""
        veto_vote = GuardianVote(
            guardian=GuardianRole.SECURITY,
            vote_type=VoteType.REJECT_HARD,
            confidence=1.0,
            reasoning="Security flaw detected",
            ihsan_assessment=IhsanVector(),
        )
        proposal = Proposal(id="v-1", title="T", content="C", proposer="x")
        verdict = council._create_veto_verdict(proposal, [veto_vote], [veto_vote])
        assert verdict.approved is False
        assert verdict.aggregate_score == -1.0
        assert "SECURITY" in verdict.recommendations[0]

    def test_generate_recommendations_no_weak(self, council):
        """When all dimensions are strong, no recommendations are generated."""
        vote = GuardianVote(
            guardian=GuardianRole.ARCHITECT,
            vote_type=VoteType.APPROVE,
            confidence=1.0,
            reasoning="R",
            ihsan_assessment=IhsanVector(0.95, 0.95, 0.95, 0.95, 0.95),
        )
        recs = council._generate_recommendations([vote], 0.95)
        assert len(recs) == 0

    def test_generate_recommendations_weak_dims(self, council):
        """Dimensions below 0.8 average generate recommendations."""
        vote = GuardianVote(
            guardian=GuardianRole.ARCHITECT,
            vote_type=VoteType.APPROVE,
            confidence=1.0,
            reasoning="R",
            ihsan_assessment=IhsanVector(0.5, 0.95, 0.95, 0.95, 0.95),
        )
        recs = council._generate_recommendations([vote], 0.85)
        assert any("correctness" in r for r in recs)


# =============================================================================
# 10. TestCreateCouncil
# =============================================================================


class TestCreateCouncil:
    """Tests for the create_council() convenience function."""

    def test_returns_guardian_council(self):
        """create_council() returns a GuardianCouncil."""
        c = create_council()
        assert isinstance(c, GuardianCouncil)

    def test_default_params(self):
        """create_council() uses default ihsan_threshold=0.95, enable_veto=True."""
        c = create_council()
        assert c.ihsan_threshold == 0.95
        assert c.enable_veto is True

    def test_custom_params(self):
        """create_council() passes custom params through."""
        c = create_council(ihsan_threshold=0.80, enable_veto=False)
        assert c.ihsan_threshold == 0.80
        assert c.enable_veto is False

    def test_has_all_guardians(self):
        """create_council() initializes all 8 guardians."""
        c = create_council()
        assert len(c.guardians) == 8


# =============================================================================
# 11. TestGuardianCouncilValidate
# =============================================================================


class TestGuardianCouncilValidate:
    """Tests for the monkey-patched .validate() method."""

    def test_validate_method_exists(self, council):
        """GuardianCouncil has a .validate() method via monkey-patch."""
        assert hasattr(council, "validate")
        assert callable(council.validate)

    @pytest.mark.asyncio
    async def test_validate_returns_dict(self, council):
        """validate() returns a dict with expected keys."""
        result = await council.validate("Test content for validation", {"key": "val"})
        assert isinstance(result, dict)
        expected_keys = {
            "approved",
            "votes",
            "consensus_score",
            "verdict",
            "ihsan_score",
            "recommendations",
            "dissenting_opinions",
            "deliberation_time_ms",
        }
        assert set(result.keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_validate_approved_verdict_string(self):
        """When approved, verdict string is 'APPROVED'."""
        council = GuardianCouncil(ihsan_threshold=0.80)

        def perfect_eval(p: Proposal) -> IhsanVector:
            return IhsanVector(0.99, 0.99, 0.99, 0.99, 0.99)

        for role in GuardianRole:
            council.set_guardian_evaluator(role, perfect_eval)

        result = await council.validate("Good content", {})
        assert result["verdict"] == "APPROVED"
        assert result["approved"] is True
        assert result["ihsan_score"] == 1.0

    @pytest.mark.asyncio
    async def test_validate_rejected_verdict_string(self):
        """When rejected (not vetoed), verdict string is 'REJECTED'."""
        council = GuardianCouncil(ihsan_threshold=0.99, enable_veto=False)

        def borderline_eval(p: Proposal) -> IhsanVector:
            return IhsanVector(0.85, 0.85, 0.85, 0.85, 0.85)

        for role in GuardianRole:
            council.set_guardian_evaluator(role, borderline_eval)

        result = await council.validate("Borderline content", {})
        assert result["verdict"] == "REJECTED"
        assert result["approved"] is False

    @pytest.mark.asyncio
    async def test_validate_vetoed_verdict_string(self):
        """When vetoed, verdict string is 'VETOED' and aggregate_score <= -1.0."""
        council = GuardianCouncil(enable_veto=True, ihsan_threshold=0.80)

        def reject_eval(p: Proposal) -> IhsanVector:
            return IhsanVector(0.2, 0.2, 0.2, 0.2, 0.2)

        def high_eval(p: Proposal) -> IhsanVector:
            return IhsanVector(0.99, 0.99, 0.99, 0.99, 0.99)

        for role in GuardianRole:
            council.set_guardian_evaluator(role, high_eval)
        council.set_guardian_evaluator(GuardianRole.SECURITY, reject_eval)

        result = await council.validate("Vetoed content", {})
        assert result["verdict"] == "VETOED"
        assert result["approved"] is False

    @pytest.mark.asyncio
    async def test_validate_vote_counts(self):
        """validate() returns correct vote counts in approve/reject/abstain."""
        council = GuardianCouncil(ihsan_threshold=0.80)

        def perfect_eval(p: Proposal) -> IhsanVector:
            return IhsanVector(0.99, 0.99, 0.99, 0.99, 0.99)

        for role in GuardianRole:
            council.set_guardian_evaluator(role, perfect_eval)

        result = await council.validate("Content", {})
        votes = result["votes"]
        assert "approve" in votes
        assert "reject" in votes
        assert "abstain" in votes
        assert votes["approve"] + votes["reject"] + votes["abstain"] == 8

    @pytest.mark.asyncio
    async def test_validate_consensus_score_non_negative(self):
        """consensus_score is normalized to >= 0."""
        council = GuardianCouncil(ihsan_threshold=0.80)
        result = await council.validate("Some content here", {})
        assert result["consensus_score"] >= 0.0

    @pytest.mark.asyncio
    async def test_validate_proposal_id_is_hash(self):
        """validate() creates a proposal with ID derived from content hash."""
        council = GuardianCouncil(ihsan_threshold=0.80)

        def perfect_eval(p: Proposal) -> IhsanVector:
            return IhsanVector(0.99, 0.99, 0.99, 0.99, 0.99)

        for role in GuardianRole:
            council.set_guardian_evaluator(role, perfect_eval)

        content = "Test content"
        await council.validate(content, {})

        # The verdict was appended to council.verdicts
        assert len(council.verdicts) == 1
        expected_id = hashlib.sha256(content.encode()).hexdigest()[:12]
        assert council.verdicts[0].proposal_id == expected_id

    @pytest.mark.asyncio
    async def test_validate_deliberation_time_positive(self):
        """validate() reports a positive deliberation time."""
        council = GuardianCouncil(ihsan_threshold=0.80)
        result = await council.validate("Content", {})
        assert result["deliberation_time_ms"] >= 0.0
