"""
Crown Verdict System -- Comprehensive Tests
=============================================

Tests the three-tier invariant verification watchdog (H0/H1/H2)
and the Crown Verdict adjudication logic.

Covers:
- H0: Ethical + Shariah invariants (gharar, riba, Gini, Ihsan)
- H1: Performance invariants (SLA, throughput, cost, SNR)
- H2: Safety invariants (reversibility, blast, human override, harm)
- Crown adjudication logic (ACCEPT, REJECT, REVISE)
- Ed25519 signature verification
- Gini coefficient computation
- Edge cases and boundary conditions
"""

import pytest

from core.governance.crown_verdict import (
    ActionScope,
    CrownVerdict,
    CrownVerdictResult,
    H0Result,
    H1Result,
    H2Result,
    SovereignAction,
    TierStatus,
    Verdict,
    _compute_gini,
    create_crown_verdict,
)
from core.integration.constants import (
    ADL_GINI_THRESHOLD,
    IHSAN_THRESHOLD,
    SNR_THRESHOLD,
)

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def crown() -> CrownVerdict:
    """Create a CrownVerdict engine with default constitutional thresholds."""
    return create_crown_verdict()


@pytest.fixture
def passing_action() -> SovereignAction:
    """An action that passes all three tiers."""
    return SovereignAction(
        action_id="test-pass-001",
        action_type="query",
        description="Read-only data retrieval",
        agent_id="agent-alpha",
        # H0: All ethical checks pass
        has_audit_trail=True,
        involves_interest=False,
        resource_distribution=[0.25, 0.25, 0.25, 0.25],
        ihsan_score=0.97,
        # H1: All performance checks pass
        sla_deadline_ms=1000.0,
        estimated_duration_ms=500.0,
        throughput_rps=100.0,
        resource_cost=0.3,
        resource_cost_ceiling=1.0,
        snr_score=0.92,
        # H2: All safety checks pass
        reversible=True,
        blast_radius=ActionScope.SELF,
        max_allowed_scope=ActionScope.LOCAL,
        human_override_available=True,
        harm_assessment=0.05,
    )


@pytest.fixture
def failing_action() -> SovereignAction:
    """An action that fails multiple tiers."""
    return SovereignAction(
        action_id="test-fail-001",
        action_type="transfer",
        description="Large irreversible transfer with no audit",
        agent_id="agent-rogue",
        # H0: Fails gharar (no audit trail) and riba
        has_audit_trail=False,
        involves_interest=True,
        resource_distribution=[0.01, 0.01, 0.01, 0.97],
        ihsan_score=0.50,
        # H1: Fails SLA and SNR
        sla_deadline_ms=100.0,
        estimated_duration_ms=500.0,
        throughput_rps=0.1,
        resource_cost=5.0,
        resource_cost_ceiling=1.0,
        snr_score=0.40,
        # H2: Fails blast, reversibility, harm
        reversible=False,
        blast_radius=ActionScope.FEDERATION,
        max_allowed_scope=ActionScope.LOCAL,
        human_override_available=False,
        harm_assessment=0.90,
    )


# =============================================================================
# GINI COEFFICIENT TESTS
# =============================================================================


class TestGiniCoefficient:
    """Tests for the Gini coefficient computation."""

    def test_perfect_equality(self):
        """Equal distribution should yield Gini = 0."""
        assert _compute_gini([1.0, 1.0, 1.0, 1.0]) == pytest.approx(0.0, abs=1e-10)

    def test_perfect_inequality(self):
        """One person has everything, rest have nothing -> Gini near 1.0."""
        # With [0, 0, 0, 100], Gini = (n-1)/n = 0.75 for n=4
        gini = _compute_gini([0.0, 0.0, 0.0, 100.0])
        assert gini > 0.5

    def test_moderate_inequality(self):
        """Moderate inequality should be between 0 and 1."""
        gini = _compute_gini([1.0, 2.0, 3.0, 4.0])
        assert 0.0 < gini < 1.0

    def test_empty_distribution(self):
        """Empty distribution should return 0."""
        assert _compute_gini([]) == 0.0

    def test_single_element(self):
        """Single element should return 0."""
        assert _compute_gini([42.0]) == 0.0

    def test_all_zeros(self):
        """All zeros should return 0 (no resources to distribute)."""
        assert _compute_gini([0.0, 0.0, 0.0]) == 0.0

    def test_below_threshold(self):
        """Equal-ish distribution should be below ADL threshold."""
        gini = _compute_gini([0.2, 0.25, 0.25, 0.3])
        assert gini <= ADL_GINI_THRESHOLD

    def test_above_threshold(self):
        """Highly concentrated distribution should exceed ADL threshold."""
        gini = _compute_gini([0.01, 0.01, 0.01, 0.97])
        assert gini > ADL_GINI_THRESHOLD


# =============================================================================
# H0: ETHICAL + SHARIAH INVARIANT TESTS
# =============================================================================


class TestH0EthicalInvariants:
    """Tests for the H0 tier: ethical and Shariah invariants."""

    def test_all_pass(self, crown: CrownVerdict, passing_action: SovereignAction):
        """Action with full ethical compliance should pass H0."""
        result = crown.verify_h0(passing_action)
        assert result.passed
        assert result.status == TierStatus.PASSED
        assert not result.gharar_detected
        assert not result.riba_detected
        assert result.gini_passed
        assert result.ihsan_passed
        assert len(result.violations) == 0

    def test_gharar_detection(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Missing audit trail triggers gharar detection."""
        passing_action.has_audit_trail = False
        result = crown.verify_h0(passing_action)
        assert not result.passed
        assert result.gharar_detected
        assert any("GHARAR" in v for v in result.violations)

    def test_riba_detection(self, crown: CrownVerdict, passing_action: SovereignAction):
        """Interest-bearing pattern triggers riba detection."""
        passing_action.involves_interest = True
        result = crown.verify_h0(passing_action)
        assert not result.passed
        assert result.riba_detected
        assert any("RIBA" in v for v in result.violations)

    def test_gini_violation(self, crown: CrownVerdict, passing_action: SovereignAction):
        """Concentrated resource distribution violates Gini threshold."""
        passing_action.resource_distribution = [0.01, 0.01, 0.01, 0.97]
        result = crown.verify_h0(passing_action)
        assert not result.passed
        assert not result.gini_passed
        assert result.gini_coefficient > ADL_GINI_THRESHOLD
        assert any("ADL" in v for v in result.violations)

    def test_ihsan_below_threshold(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Ihsan score below threshold fails H0."""
        passing_action.ihsan_score = 0.80
        result = crown.verify_h0(passing_action)
        assert not result.passed
        assert not result.ihsan_passed
        assert any("IHSAN" in v for v in result.violations)

    def test_ihsan_at_threshold(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Ihsan score exactly at threshold should pass."""
        passing_action.ihsan_score = IHSAN_THRESHOLD
        result = crown.verify_h0(passing_action)
        assert result.ihsan_passed

    def test_gini_none_distribution(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """None distribution should not trigger Gini check."""
        passing_action.resource_distribution = None
        result = crown.verify_h0(passing_action)
        assert result.gini_passed
        assert result.gini_coefficient == 0.0

    def test_evidence_populated(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """H0 result should contain structured evidence."""
        result = crown.verify_h0(passing_action)
        assert "has_audit_trail" in result.evidence
        assert "ihsan_score" in result.evidence
        assert "gini_threshold" in result.evidence

    def test_to_dict(self, crown: CrownVerdict, passing_action: SovereignAction):
        """H0 result should serialize to dict."""
        result = crown.verify_h0(passing_action)
        d = result.to_dict()
        assert d["tier"] == "H0"
        assert d["tier_name"] == "Ethical + Shariah Invariants"
        assert d["status"] == "passed"


# =============================================================================
# H1: PERFORMANCE INVARIANT TESTS
# =============================================================================


class TestH1PerformanceInvariants:
    """Tests for the H1 tier: performance invariants."""

    def test_all_pass(self, crown: CrownVerdict, passing_action: SovereignAction):
        """Action within all performance bounds should pass H1."""
        result = crown.verify_h1(passing_action)
        assert result.passed
        assert result.sla_met
        assert result.throughput_adequate
        assert result.cost_within_ceiling
        assert result.snr_passed
        assert len(result.violations) == 0

    def test_sla_violation(self, crown: CrownVerdict, passing_action: SovereignAction):
        """Estimated duration exceeding SLA deadline fails H1."""
        passing_action.sla_deadline_ms = 100.0
        passing_action.estimated_duration_ms = 500.0
        result = crown.verify_h1(passing_action)
        assert not result.passed
        assert not result.sla_met
        assert any("SLA" in v for v in result.violations)

    def test_sla_with_margin(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """SLA check applies safety margin (default 0.9)."""
        # Deadline 100ms * 0.9 margin = 90ms effective
        passing_action.sla_deadline_ms = 100.0
        passing_action.estimated_duration_ms = 95.0
        result = crown.verify_h1(passing_action)
        assert not result.sla_met  # 95 > 90

    def test_sla_none_deadline(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """No SLA deadline means SLA check is skipped."""
        passing_action.sla_deadline_ms = None
        result = crown.verify_h1(passing_action)
        assert result.sla_met

    def test_cost_ceiling_violation(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Resource cost exceeding ceiling fails H1."""
        passing_action.resource_cost = 2.0
        passing_action.resource_cost_ceiling = 1.0
        result = crown.verify_h1(passing_action)
        assert not result.passed
        assert not result.cost_within_ceiling
        assert any("COST" in v for v in result.violations)

    def test_snr_below_minimum(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """SNR below minimum threshold fails H1."""
        passing_action.snr_score = 0.50
        result = crown.verify_h1(passing_action)
        assert not result.passed
        assert not result.snr_passed
        assert any("SNR" in v for v in result.violations)

    def test_snr_at_minimum(self, crown: CrownVerdict, passing_action: SovereignAction):
        """SNR exactly at minimum should pass."""
        passing_action.snr_score = SNR_THRESHOLD
        result = crown.verify_h1(passing_action)
        assert result.snr_passed

    def test_throughput_check_disabled_by_default(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Default min_throughput_rps=0 means throughput check is skipped."""
        passing_action.throughput_rps = 0.001
        result = crown.verify_h1(passing_action)
        assert result.throughput_adequate

    def test_throughput_check_with_floor(self, passing_action: SovereignAction):
        """Explicit throughput floor enforces minimum."""
        crown = CrownVerdict(min_throughput_rps=10.0)
        passing_action.throughput_rps = 5.0
        result = crown.verify_h1(passing_action)
        assert not result.throughput_adequate
        assert any("THROUGHPUT" in v for v in result.violations)

    def test_to_dict(self, crown: CrownVerdict, passing_action: SovereignAction):
        """H1 result should serialize to dict."""
        result = crown.verify_h1(passing_action)
        d = result.to_dict()
        assert d["tier"] == "H1"
        assert d["tier_name"] == "Performance Invariants"


# =============================================================================
# H2: SAFETY INVARIANT TESTS
# =============================================================================


class TestH2SafetyInvariants:
    """Tests for the H2 tier: safety invariants."""

    def test_all_pass(self, crown: CrownVerdict, passing_action: SovereignAction):
        """Safe action should pass H2."""
        result = crown.verify_h2(passing_action)
        assert result.passed
        assert result.status == TierStatus.PASSED
        assert result.reversible
        assert result.blast_contained
        assert result.human_override_available
        assert result.no_harm_verified
        assert len(result.violations) == 0

    def test_irreversible_degrades(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Irreversible action degrades to DEGRADED (not hard failure)."""
        passing_action.reversible = False
        result = crown.verify_h2(passing_action)
        assert result.status == TierStatus.DEGRADED
        assert not result.reversible
        assert any("REVERSIBILITY" in v for v in result.violations)

    def test_no_human_override_degrades(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Missing human override degrades to DEGRADED."""
        passing_action.human_override_available = False
        result = crown.verify_h2(passing_action)
        assert result.status == TierStatus.DEGRADED
        assert not result.human_override_available
        assert any("HUMAN" in v for v in result.violations)

    def test_blast_radius_breach(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Blast radius exceeding allowed scope hard fails."""
        passing_action.blast_radius = ActionScope.FEDERATION
        passing_action.max_allowed_scope = ActionScope.LOCAL
        result = crown.verify_h2(passing_action)
        assert result.status == TierStatus.FAILED
        assert not result.blast_contained
        assert any("BLAST" in v for v in result.violations)

    def test_blast_within_scope(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Blast radius within allowed scope passes."""
        passing_action.blast_radius = ActionScope.SELF
        passing_action.max_allowed_scope = ActionScope.CLUSTER
        result = crown.verify_h2(passing_action)
        assert result.blast_contained

    def test_harm_hard_reject(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Harm above hard threshold causes FAILED status."""
        passing_action.harm_assessment = 0.85
        result = crown.verify_h2(passing_action)
        assert result.status == TierStatus.FAILED
        assert not result.no_harm_verified
        assert any("HARM" in v for v in result.violations)

    def test_harm_below_soft_threshold(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Low harm should pass cleanly."""
        passing_action.harm_assessment = 0.1
        result = crown.verify_h2(passing_action)
        assert result.no_harm_verified

    def test_to_dict(self, crown: CrownVerdict, passing_action: SovereignAction):
        """H2 result should serialize to dict."""
        result = crown.verify_h2(passing_action)
        d = result.to_dict()
        assert d["tier"] == "H2"
        assert d["tier_name"] == "Safety Invariants"


# =============================================================================
# CROWN VERDICT ADJUDICATION TESTS
# =============================================================================


class TestCrownAdjudication:
    """Tests for the full Crown Verdict adjudication logic."""

    def test_all_pass_yields_accept(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Action passing all tiers should receive ACCEPT verdict."""
        result = crown.adjudicate(passing_action)
        assert result.verdict == Verdict.ACCEPT
        assert result.accepted
        assert result.all_tiers_passed
        assert len(result.total_violations) == 0

    def test_h0_failure_yields_reject(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Gharar violation (H0) should yield REJECT."""
        passing_action.has_audit_trail = False
        result = crown.adjudicate(passing_action)
        assert result.verdict == Verdict.REJECT
        assert not result.accepted

    def test_riba_yields_hard_reject(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Riba violation is a hard rejection -- no REVISE possible."""
        passing_action.involves_interest = True
        result = crown.adjudicate(passing_action)
        assert result.verdict == Verdict.REJECT

    def test_gini_yields_hard_reject(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Gini violation is a hard rejection."""
        passing_action.resource_distribution = [0.01, 0.01, 0.01, 0.97]
        result = crown.adjudicate(passing_action)
        assert result.verdict == Verdict.REJECT

    def test_ihsan_only_failure_yields_revise(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Ihsan-only failure (no gharar/riba/Gini) should yield REVISE with remediation."""
        passing_action.ihsan_score = 0.90
        result = crown.adjudicate(passing_action)
        assert result.verdict == Verdict.REVISE
        assert len(result.remediations) > 0
        assert any("Ihsan" in r for r in result.remediations)

    def test_snr_only_failure_yields_revise(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """SNR-only failure should yield REVISE with remediation."""
        passing_action.snr_score = 0.70
        result = crown.adjudicate(passing_action)
        assert result.verdict == Verdict.REVISE
        assert any("SNR" in r for r in result.remediations)

    def test_sla_only_failure_yields_revise(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """SLA-only failure should yield REVISE with remediation."""
        passing_action.sla_deadline_ms = 100.0
        passing_action.estimated_duration_ms = 200.0
        result = crown.adjudicate(passing_action)
        assert result.verdict == Verdict.REVISE
        assert any("SLA" in r for r in result.remediations)

    def test_irreversible_yields_revise(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Irreversible (but safe) action should yield REVISE."""
        passing_action.reversible = False
        result = crown.adjudicate(passing_action)
        assert result.verdict == Verdict.REVISE
        assert any("REVERSIBILITY" in r for r in result.remediations)

    def test_blast_breach_yields_reject(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Blast radius breach is always REJECT (never REVISE)."""
        passing_action.blast_radius = ActionScope.FEDERATION
        passing_action.max_allowed_scope = ActionScope.LOCAL
        result = crown.adjudicate(passing_action)
        assert result.verdict == Verdict.REJECT

    def test_harm_hard_reject(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """High harm is always REJECT."""
        passing_action.harm_assessment = 0.9
        result = crown.adjudicate(passing_action)
        assert result.verdict == Verdict.REJECT

    def test_total_failure_yields_reject(
        self, crown: CrownVerdict, failing_action: SovereignAction
    ):
        """Action failing everything should definitely REJECT."""
        result = crown.adjudicate(failing_action)
        assert result.verdict == Verdict.REJECT
        assert not result.accepted
        assert not result.all_tiers_passed
        assert len(result.total_violations) > 0


# =============================================================================
# SIGNATURE VERIFICATION TESTS
# =============================================================================


class TestSignatureVerification:
    """Tests for Ed25519 signature on verdicts."""

    def test_verdict_is_signed(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Adjudicated verdict should carry a valid signature."""
        result = crown.adjudicate(passing_action)
        assert result.signature != ""
        assert result.signer_public_key != ""
        assert result.signer_public_key == crown.public_key

    def test_signature_verifies(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Valid signature should verify successfully."""
        result = crown.adjudicate(passing_action)
        assert result.verify()

    def test_tampered_verdict_fails_verification(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Modifying the verdict after signing should break verification."""
        result = crown.adjudicate(passing_action)
        # Tamper with the action_id
        result.action_id = "tampered-id"
        assert not result.verify()

    def test_unsigned_verdict_fails_verification(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Verdict with empty signature should fail verification."""
        result = crown.adjudicate(passing_action)
        result.signature = ""
        assert not result.verify()

    def test_missing_public_key_fails(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Verdict with empty public key should fail verification."""
        result = crown.adjudicate(passing_action)
        result.signer_public_key = ""
        assert not result.verify()


# =============================================================================
# SERIALIZATION TESTS
# =============================================================================


class TestSerialization:
    """Tests for CrownVerdictResult serialization."""

    def test_to_dict_complete(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Full verdict should serialize with all fields."""
        result = crown.adjudicate(passing_action)
        d = result.to_dict()

        assert "action_id" in d
        assert "verdict" in d
        assert "h0" in d
        assert "h1" in d
        assert "h2" in d
        assert "remediations" in d
        assert "total_violations" in d
        assert "signature" in d
        assert "signer_public_key" in d
        assert "timestamp" in d

    def test_verdict_value_in_dict(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Verdict enum should serialize as string."""
        result = crown.adjudicate(passing_action)
        d = result.to_dict()
        assert d["verdict"] == "ACCEPT"

    def test_tier_dicts_nested(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Tier results should be nested dicts with proper structure."""
        result = crown.adjudicate(passing_action)
        d = result.to_dict()

        for tier_key in ("h0", "h1", "h2"):
            tier = d[tier_key]
            assert "tier" in tier
            assert "status" in tier
            assert "violations" in tier
            assert "evidence" in tier


# =============================================================================
# AUDIT & STATISTICS TESTS
# =============================================================================


class TestAuditStatistics:
    """Tests for verdict history and statistics."""

    def test_verdict_recorded(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Each adjudication should be recorded in history."""
        crown.adjudicate(passing_action)
        history = crown.get_verdict_history()
        assert len(history) == 1

    def test_multiple_verdicts_recorded(
        self,
        crown: CrownVerdict,
        passing_action: SovereignAction,
        failing_action: SovereignAction,
    ):
        """Multiple adjudications should accumulate."""
        crown.adjudicate(passing_action)
        crown.adjudicate(failing_action)
        history = crown.get_verdict_history()
        assert len(history) == 2

    def test_stats_empty(self, crown: CrownVerdict):
        """Stats with no verdicts should show zeros."""
        stats = crown.get_stats()
        assert stats["total_verdicts"] == 0

    def test_stats_populated(
        self,
        crown: CrownVerdict,
        passing_action: SovereignAction,
        failing_action: SovereignAction,
    ):
        """Stats should reflect verdict counts."""
        crown.adjudicate(passing_action)
        crown.adjudicate(failing_action)
        stats = crown.get_stats()
        assert stats["total_verdicts"] == 2
        assert stats["accept_count"] == 1
        assert stats["reject_count"] == 1
        assert "tier_failures" in stats

    def test_history_limit(self, crown: CrownVerdict, passing_action: SovereignAction):
        """History limit should cap returned results."""
        for _ in range(5):
            crown.adjudicate(passing_action)
        history = crown.get_verdict_history(limit=3)
        assert len(history) == 3


# =============================================================================
# CONSTRUCTOR & FACTORY TESTS
# =============================================================================


class TestConstructor:
    """Tests for CrownVerdict construction and configuration."""

    def test_default_thresholds(self):
        """Default thresholds should match constitutional constants."""
        crown = CrownVerdict()
        assert crown.ihsan_threshold == IHSAN_THRESHOLD
        assert crown.snr_minimum == SNR_THRESHOLD
        assert crown.gini_threshold == ADL_GINI_THRESHOLD

    def test_custom_thresholds(self):
        """Custom thresholds should be respected."""
        crown = CrownVerdict(
            ihsan_threshold=0.99,
            snr_minimum=0.95,
            gini_threshold=0.30,
        )
        assert crown.ihsan_threshold == 0.99
        assert crown.snr_minimum == 0.95
        assert crown.gini_threshold == 0.30

    def test_factory_function(self):
        """Factory function should create valid engine."""
        crown = create_crown_verdict()
        assert isinstance(crown, CrownVerdict)
        assert crown.ihsan_threshold == IHSAN_THRESHOLD

    def test_public_key_exposed(self):
        """Public key should be accessible for verification."""
        crown = CrownVerdict()
        assert len(crown.public_key) == 64  # 32 bytes = 64 hex chars


# =============================================================================
# EDGE CASES & BOUNDARY CONDITIONS
# =============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_action_scope_ordering(self):
        """ActionScope enum values should be ordered by blast radius."""
        assert ActionScope.SELF.value < ActionScope.LOCAL.value
        assert ActionScope.LOCAL.value < ActionScope.CLUSTER.value
        assert ActionScope.CLUSTER.value < ActionScope.FEDERATION.value
        assert ActionScope.FEDERATION.value < ActionScope.EXTERNAL.value

    def test_cost_at_ceiling(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Cost exactly at ceiling should pass."""
        passing_action.resource_cost = 1.0
        passing_action.resource_cost_ceiling = 1.0
        result = crown.verify_h1(passing_action)
        assert result.cost_within_ceiling

    def test_harm_at_zero(self, crown: CrownVerdict, passing_action: SovereignAction):
        """Zero harm should pass H2 cleanly."""
        passing_action.harm_assessment = 0.0
        result = crown.verify_h2(passing_action)
        assert result.no_harm_verified

    def test_verdict_enum_values(self):
        """Verdict enum should have exactly three values."""
        assert len(Verdict) == 3
        assert Verdict.ACCEPT.value == "ACCEPT"
        assert Verdict.REJECT.value == "REJECT"
        assert Verdict.REVISE.value == "REVISE"

    def test_tier_status_enum_values(self):
        """TierStatus should have three values."""
        assert len(TierStatus) == 3
        assert TierStatus.PASSED.value == "passed"
        assert TierStatus.FAILED.value == "failed"
        assert TierStatus.DEGRADED.value == "degraded"

    def test_duration_tracked(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Each tier and total should track duration."""
        result = crown.adjudicate(passing_action)
        assert result.total_duration_us >= 0
        assert result.h0.duration_us >= 0
        assert result.h1.duration_us >= 0
        assert result.h2.duration_us >= 0

    def test_multiple_h1_failures(
        self, crown: CrownVerdict, passing_action: SovereignAction
    ):
        """Multiple H1 failures should all be recorded as violations."""
        passing_action.snr_score = 0.5
        passing_action.resource_cost = 10.0
        passing_action.sla_deadline_ms = 10.0
        passing_action.estimated_duration_ms = 500.0
        result = crown.verify_h1(passing_action)
        assert len(result.violations) >= 3  # SLA + cost + SNR

    def test_import_from_governance_package(self):
        """Crown Verdict types should be importable from core.governance."""
        from core.governance import (
            CrownVerdict,
            CrownVerdictResult,
            H0Result,
            H1Result,
            H2Result,
            SovereignAction,
            Verdict,
            create_crown_verdict,
        )

        assert CrownVerdict is not None
        assert CrownVerdictResult is not None
        assert Verdict is not None
