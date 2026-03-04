"""
BIZRA Constitutional Conformance Tests — DO NOT EDIT MANUALLY
═════════════════════════════════════════════════════════════
Generated from constitution.toml v5.0.0-GENESIS
SHA-256: 8b020123e2b04e8ade720eb285339ee3967a321f142a7a9d253d7d47cd562422
Generated: 2026-03-03T19:49:37.136982+00:00

These tests verify that the running system conforms to the constitution.
To modify: edit constitution.toml, then re-run generate_from_constitution.py
"""

import pytest
from pathlib import Path


# ── Fixture: Load constitution ──

@pytest.fixture(scope="session")
def constitution():
    from bizra_constitution import load_constitution
    return load_constitution()


# ═══════════════════════════════════════════════════════════
# §1 — Constitutional Self-Consistency
# ═══════════════════════════════════════════════════════════

class TestConstitutionalInvariants:
    """The constitution must be internally consistent."""

    def test_ihsan_weights_sum_to_one(self, constitution):
        s = constitution.ihsan.canonical_weights.sum()
        assert abs(s - 1.0) < 0.001, f"Ihsan weights sum to {s}, expected 1.0"

    def test_gate_weights_sum_to_one(self, constitution):
        s = constitution.gates.total_weight()
        assert abs(s - 1.0) < 0.001, f"Gate weights sum to {s}, expected 1.0"

    def test_pat_agent_count_matches(self, constitution):
        assert len(constitution.pat.agents) == 7

    def test_trust_stages_are_unique(self, constitution):
        stages = [a.trust_stage for a in constitution.pat.agents]
        assert len(set(stages)) == len(stages), "Trust stages must be unique"

    def test_identity_rights_minimum(self, constitution):
        assert len(constitution.identity.rights.rights) >= 7

    def test_fail_modes_are_closed(self, constitution):
        assert constitution.ihsan.fail_mode == "closed"
        assert constitution.gates.fail_mode == "closed"

    def test_zakat_is_positive(self, constitution):
        assert constitution.economics.zakat_rate > 0, "Zakat is constitutional"

    def test_bloom_threshold_above_gate(self, constitution):
        assert (
            constitution.economics.bloom_ihsan_threshold
            >= constitution.ihsan.thresholds.gate_minimum
        )

    def test_constitution_hash_is_populated(self, constitution):
        assert len(constitution.raw_hash) == 64, "SHA-256 hash must be 64 hex chars"

    def test_seven_eliminated_attacks(self, constitution):
        assert len(constitution.interaction_laws.eliminated_attacks) == 7

    def test_no_riba_no_gharar(self, constitution):
        assert constitution.economics.no_riba is True
        assert constitution.economics.no_gharar is True


# ═══════════════════════════════════════════════════════════
# §2 — Ihsan Tensor Verification
# ═══════════════════════════════════════════════════════════

class TestIhsanTensor:
    """The Ihsan tensor must be correctly configured."""

    def test_canonical_is_8_dimensional(self, constitution):
        assert constitution.ihsan.dimensions == 8

    def test_operational_is_6_dimensional(self, constitution):
        assert len(constitution.ihsan.operational_dimensions) == 6

    def test_operational_is_subset_of_canonical(self, constitution):
        canonical = set(constitution.ihsan.canonical_weights.as_dict().keys())
        operational = set(constitution.ihsan.operational_dimensions)
        assert operational.issubset(canonical)

    def test_operational_projection_renormalizes(self, constitution):
        op = constitution.ihsan.operational_weights()
        total = sum(op.values())
        assert abs(total - 1.0) < 0.001

    def test_gate_minimum_threshold(self, constitution):
        assert constitution.ihsan.thresholds.gate_minimum == 0.85

    def test_bloom_eligibility_threshold(self, constitution):
        assert constitution.ihsan.thresholds.bloom_eligibility == 0.9

    def test_ihsan_excellence_standard(self, constitution):
        assert constitution.ihsan.thresholds.ihsan_excellence == 0.95


# ═══════════════════════════════════════════════════════════
# §3 — Gate Configuration
# ═══════════════════════════════════════════════════════════

class TestGates:
    """Constitutional gates must be correctly weighted."""

    def test_five_gates_defined(self, constitution):
        assert constitution.gates.count == 5

    def test_gate_overhead_budget(self, constitution):
        assert constitution.gates.total_overhead_budget_ms == 50

    @pytest.mark.parametrize("gate_name,expected_weight", [
        ("alpha_4", 0.15),
        ("alpha_7", 0.25),
        ("alpha_8", 0.2),
        ("alpha_9", 0.25),
        ("alpha_10", 0.15),
    ])
    def test_individual_gate_weight(self, constitution, gate_name, expected_weight):
        gate = getattr(constitution.gates, gate_name)
        assert gate.weight == expected_weight


# ═══════════════════════════════════════════════════════════
# §4 — Security Domain Separation
# ═══════════════════════════════════════════════════════════

class TestSecurity:
    """Domain separation must be enforced for all signing contexts."""

    @pytest.mark.parametrize("context_name,expected_value", [
        ("evidence_receipt", "bizra-evidence-v1"),
        ("urp_lease", "bizra-urp-lease-v1"),
        ("poi_attestation", "bizra-poi-v1"),
        ("identity_genesis", "bizra-identity-genesis-v1"),
        ("telescript_publish", "bizra-telescript-v1"),
        ("bloom_mint", "bizra-bloom-mint-v1"),
    ])
    def test_domain_context_defined(self, constitution, context_name, expected_value):
        actual = getattr(constitution.security.domain_separation, context_name)
        assert actual == expected_value

    def test_equivocation_impossible(self, constitution):
        assert constitution.security.equivocation_possible is False

    def test_default_privacy_is_local_only(self, constitution):
        assert constitution.security.default_privacy == "LOCAL_ONLY"


# ═══════════════════════════════════════════════════════════
# §5 — Daughter Test (Liveness Property)
# ═══════════════════════════════════════════════════════════

class TestDaughterTest:
    """The Daughter Test must be enforced as a CI test, not philosophy."""

    def test_daughter_test_is_liveness(self, constitution):
        assert constitution.daughter_test.type == "liveness_property"

    def test_daughter_test_enforced_in_ci(self, constitution):
        assert constitution.daughter_test.enforcement == "ci_test"

    def test_rejection_test_defined(self, constitution):
        assert constitution.daughter_test.test_safe_rejection != ""

    def test_acceptance_test_defined(self, constitution):
        assert constitution.daughter_test.test_safe_acceptance != ""


# ═══════════════════════════════════════════════════════════
# §6 — Economics (Thermodynamic Invariants)
# ═══════════════════════════════════════════════════════════

class TestEconomics:
    """Economic parameters must maintain thermodynamic stability."""

    def test_zakat_rate(self, constitution):
        assert constitution.economics.zakat_rate == 0.025

    def test_zakat_is_constitutional(self, constitution):
        assert constitution.economics.zakat_constitutional is True

    def test_gini_threshold(self, constitution):
        assert constitution.economics.gini_threshold == 0.45

    def test_local_model_advantage(self, constitution):
        assert constitution.economics.local_cost_per_mission < constitution.economics.cloud_cost_per_mission

    def test_bloom_not_purchasable(self, constitution):
        # BLOOM threshold requires real Ihsan, cannot be bought
        assert constitution.economics.bloom_ihsan_threshold >= 0.90


# ═══════════════════════════════════════════════════════════
# §7 — Cross-Language Consistency Marker
# ═══════════════════════════════════════════════════════════

class TestCrossLanguage:
    """Marker tests for cross-language verification."""

    def test_zero_tolerance_on_cross_language(self, constitution):
        assert constitution.conformance.cross_language_tolerance == 0.0

    def test_constitution_hash_deterministic(self, constitution):
        """Re-loading must produce the same hash."""
        from bizra_constitution import load_constitution
        c2 = load_constitution()
        assert c2.raw_hash == constitution.raw_hash
