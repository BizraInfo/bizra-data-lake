"""Tests for constitution.toml v5.0.0-GENESIS — Phase 60 Step 1.

Standing on Giants: Dijkstra (correctness by construction) · Al-Ghazali (Ihsan as obligation)

Validates:
- TOML parses successfully with all 13 required sections
- Ihsan canonical weights (8-dim) sum to 1.0
- Operational dimensions are subset of canonical
- 5 constitutional gates with weights summing to 1.0
- Identity rights meet minimum_rights_count
- All thresholds bounded [0, 1]
- Zakat rate valid
- PAT + SAT agent counts
- Interaction laws and security properties
- Error handling for missing/malformed constitutions
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from core.integration.constitution_parser import (
    ConstitutionError,
    canonical_ihsan_weights,
    load_constitution,
    operational_ihsan_weights,
    validate_constitution,
)

# ═══════════════════════════════════════════════════════════════════════════════
# §1: Constitution Loads and Has Required Sections
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstitutionLoading:
    """Verify constitution.toml loads and has all 13 required sections."""

    def test_constitution_parses_successfully(self):
        data = load_constitution()
        assert isinstance(data, dict)

    def test_has_all_required_sections(self):
        data = load_constitution()
        required = [
            "meta",
            "identity",
            "interaction_laws",
            "ihsan_tensor",
            "pat",
            "sat",
            "gates",
            "hhmm",
            "economics",
            "reflex",
            "conformance",
            "security",
            "daughter_test",
        ]
        for section in required:
            assert section in data, f"Missing required section: [{section}]"

    def test_meta_has_version(self):
        data = load_constitution()
        assert data["meta"]["version"] == "5.0.0-GENESIS"

    def test_loading_is_deterministic(self):
        d1 = load_constitution()
        d2 = load_constitution()
        assert d1 == d2


# ═══════════════════════════════════════════════════════════════════════════════
# §2: Ihsan Tensor (8-Dimensional Canonical Weights)
# ═══════════════════════════════════════════════════════════════════════════════


class TestIhsanTensor:
    """Verify 8-dimensional canonical Ihsan tensor."""

    def test_canonical_weights_sum_to_one(self):
        data = load_constitution()
        weights = data["ihsan_tensor"]["canonical_weights"]
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6

    def test_eight_canonical_dimensions(self):
        data = load_constitution()
        weights = data["ihsan_tensor"]["canonical_weights"]
        assert len(weights) == 8

    def test_canonical_dimension_names(self):
        data = load_constitution()
        weights = data["ihsan_tensor"]["canonical_weights"]
        expected = {
            "moral_clarity",
            "epistemic_humility",
            "structural_integrity",
            "verifiability",
            "contextual_relevance",
            "intent_alignment",
            "resilience",
            "efficiency",
        }
        assert set(weights.keys()) == expected

    def test_all_weights_positive(self):
        data = load_constitution()
        for name, weight in data["ihsan_tensor"]["canonical_weights"].items():
            assert weight > 0, f"Non-positive weight for {name}"

    def test_fail_mode_is_closed(self):
        data = load_constitution()
        assert data["ihsan_tensor"]["fail_mode"] == "closed"

    def test_operational_dimensions_subset_of_canonical(self):
        data = load_constitution()
        canonical = set(data["ihsan_tensor"]["canonical_weights"].keys())
        operational = set(data["ihsan_tensor"]["operational_dimensions"]["dimensions"])
        assert operational.issubset(canonical)

    def test_six_operational_dimensions(self):
        data = load_constitution()
        op_dims = data["ihsan_tensor"]["operational_dimensions"]["dimensions"]
        assert len(op_dims) == 6

    def test_canonical_weights_helper(self):
        data = load_constitution()
        weights = canonical_ihsan_weights(data)
        assert isinstance(weights, dict)
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_operational_weights_helper(self):
        data = load_constitution()
        weights = operational_ihsan_weights(data)
        assert isinstance(weights, dict)
        assert len(weights) == 6
        assert abs(sum(weights.values()) - 1.0) < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# §3: Ihsan Thresholds
# ═══════════════════════════════════════════════════════════════════════════════


class TestIhsanThresholds:
    """Verify all Ihsan thresholds are in valid range."""

    def test_thresholds_in_range(self):
        data = load_constitution()
        thresholds = data["ihsan_tensor"]["thresholds"]
        for field in [
            "gate_minimum",
            "poi_consensus",
            "bloom_eligibility",
            "ihsan_excellence",
            "conformance_join",
        ]:
            value = thresholds[field]
            assert 0.0 <= value <= 1.0, f"{field}={value} out of range"

    def test_gate_minimum_is_085(self):
        data = load_constitution()
        assert data["ihsan_tensor"]["thresholds"]["gate_minimum"] == 0.85

    def test_bloom_eligibility_is_090(self):
        data = load_constitution()
        assert data["ihsan_tensor"]["thresholds"]["bloom_eligibility"] == 0.90

    def test_ihsan_excellence_is_095(self):
        data = load_constitution()
        assert data["ihsan_tensor"]["thresholds"]["ihsan_excellence"] == 0.95

    def test_threshold_ordering(self):
        """gate_minimum <= poi_consensus <= bloom <= excellence <= join."""
        data = load_constitution()
        t = data["ihsan_tensor"]["thresholds"]
        assert t["gate_minimum"] <= t["poi_consensus"]
        assert t["poi_consensus"] <= t["bloom_eligibility"]
        assert t["bloom_eligibility"] <= t["ihsan_excellence"]
        assert t["ihsan_excellence"] <= t["conformance_join"]


# ═══════════════════════════════════════════════════════════════════════════════
# §4: Constitutional Gates (5 alpha gates)
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstitutionalGates:
    """Verify 5 alpha gates with weights summing to 1.0."""

    def test_five_gates(self):
        data = load_constitution()
        assert data["gates"]["count"] == 5

    def test_gate_weights_sum_to_one(self):
        data = load_constitution()
        gates = data["gates"]
        gate_keys = ["alpha_4", "alpha_7", "alpha_8", "alpha_9", "alpha_10"]
        total = sum(gates[k]["weight"] for k in gate_keys)
        assert abs(total - 1.0) < 1e-6

    def test_all_gates_have_name_and_description(self):
        data = load_constitution()
        gates = data["gates"]
        for key in ["alpha_4", "alpha_7", "alpha_8", "alpha_9", "alpha_10"]:
            assert "name" in gates[key], f"{key} missing name"
            assert "description" in gates[key], f"{key} missing description"
            assert "weight" in gates[key], f"{key} missing weight"

    def test_fail_mode_closed(self):
        data = load_constitution()
        assert data["gates"]["fail_mode"] == "closed"

    def test_overhead_budget(self):
        data = load_constitution()
        assert data["gates"]["total_overhead_budget_ms"] == 50


# ═══════════════════════════════════════════════════════════════════════════════
# §5: Identity and Rights
# ═══════════════════════════════════════════════════════════════════════════════


class TestIdentity:
    """Verify identity section and rights."""

    def test_identity_agents_per_node(self):
        data = load_constitution()
        assert data["identity"]["agents_per_node"] == 12

    def test_key_algorithm_ed25519(self):
        data = load_constitution()
        assert data["identity"]["key_algorithm"] == "Ed25519"

    def test_seven_rights(self):
        data = load_constitution()
        rights = data["identity"]["rights"]["rights"]
        assert len(rights) == 7

    def test_rights_meet_minimum(self):
        data = load_constitution()
        rights = data["identity"]["rights"]["rights"]
        minimum = data["identity"]["rights"]["minimum_rights_count"]
        assert len(rights) >= minimum

    def test_right_to_leave(self):
        """Nodes can exit at any time."""
        data = load_constitution()
        rights = data["identity"]["rights"]["rights"]
        assert "Leave" in rights


# ═══════════════════════════════════════════════════════════════════════════════
# §6: PAT + SAT Agent Teams
# ═══════════════════════════════════════════════════════════════════════════════


class TestAgentTeams:
    """Verify PAT and SAT agent configuration."""

    def test_pat_seven_agents(self):
        data = load_constitution()
        assert data["pat"]["agent_count"] == 7
        assert len(data["pat"]["agents"]) == 7

    def test_sat_five_agents_per_node(self):
        data = load_constitution()
        assert data["sat"]["agents_per_node"] == 5

    def test_pat_plus_sat_equals_twelve(self):
        data = load_constitution()
        total = data["pat"]["agent_count"] + data["sat"]["agents_per_node"]
        assert total == data["identity"]["agents_per_node"]

    def test_pat_trust_monotonicity(self):
        data = load_constitution()
        assert data["pat"]["trust_monotonicity"] is True

    def test_sat_bootstrap_roles(self):
        data = load_constitution()
        roles = data["sat"]["bootstrap_roles"]["roles"]
        assert len(roles) == 5

    def test_sat_infra_minimum(self):
        data = load_constitution()
        assert data["sat"]["dynamic_roles"]["minimum_infrastructure_pct"] == 20


# ═══════════════════════════════════════════════════════════════════════════════
# §7: Interaction Laws
# ═══════════════════════════════════════════════════════════════════════════════


class TestInteractionLaws:
    """Verify interaction boundary axioms."""

    def test_three_laws(self):
        data = load_constitution()
        assert "law_1" in data["interaction_laws"]
        assert "law_2" in data["interaction_laws"]
        assert "law_3" in data["interaction_laws"]

    def test_seven_eliminated_attacks(self):
        data = load_constitution()
        attacks = data["interaction_laws"]["eliminated_attacks"]
        assert len(attacks) == 7

    def test_sybil_remains(self):
        data = load_constitution()
        remaining = data["interaction_laws"]["remaining_attacks"]
        assert "Sybil" in remaining


# ═══════════════════════════════════════════════════════════════════════════════
# §8: Economics
# ═══════════════════════════════════════════════════════════════════════════════


class TestEconomics:
    """Verify dual-token economy and zakat."""

    def test_dual_token(self):
        data = load_constitution()
        assert data["economics"]["dual_token"] is True

    def test_zakat_rate(self):
        data = load_constitution()
        assert data["economics"]["zakat"]["rate"] == 0.025

    def test_zakat_constitutional(self):
        data = load_constitution()
        assert data["economics"]["zakat"]["constitutional"] is True

    def test_no_riba(self):
        data = load_constitution()
        assert data["economics"]["seed"]["no_riba"] is True

    def test_bloom_not_transferable(self):
        data = load_constitution()
        assert data["economics"]["bloom"]["transferable"] is False

    def test_gini_threshold(self):
        data = load_constitution()
        assert data["economics"]["gini"]["threshold"] == 0.45

    def test_local_model_advantage(self):
        data = load_constitution()
        local = data["economics"]["local_model_advantage"]
        assert local["local_cost_per_mission"] == 0.0
        assert local["cloud_cost_per_mission"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# §9: Security
# ═══════════════════════════════════════════════════════════════════════════════


class TestSecurity:
    """Verify security configuration."""

    def test_signature_scheme(self):
        data = load_constitution()
        assert data["security"]["signature_scheme"] == "Ed25519"

    def test_domain_separation_contexts(self):
        data = load_constitution()
        ds = data["security"]["domain_separation"]
        expected_keys = [
            "evidence_receipt",
            "urp_lease",
            "poi_attestation",
            "identity_genesis",
            "telescript_publish",
            "bloom_mint",
        ]
        for key in expected_keys:
            assert key in ds, f"Missing domain separation: {key}"

    def test_byzantine_no_equivocation(self):
        data = load_constitution()
        assert data["security"]["byzantine"]["equivocation_possible"] is False

    def test_privacy_default_local_only(self):
        data = load_constitution()
        assert data["security"]["privacy_classes"]["default"] == "LOCAL_ONLY"


# ═══════════════════════════════════════════════════════════════════════════════
# §10: HHMM + Action Bus
# ═══════════════════════════════════════════════════════════════════════════════


class TestHHMM:
    """Verify HHMM and Action Bus configuration."""

    def test_observation_window(self):
        data = load_constitution()
        assert data["hhmm"]["observation_window"] == 50

    def test_complexity_tiers(self):
        data = load_constitution()
        tiers = data["hhmm"]["complexity_tiers"]
        assert "trivial" in tiers
        assert "simple" in tiers
        assert "complex" in tiers
        assert "sovereign" in tiers

    def test_action_bus_gcd(self):
        data = load_constitution()
        bus = data["hhmm"]["action_bus"]
        assert bus["gcd_tick_ms"] == 100


# ═══════════════════════════════════════════════════════════════════════════════
# §11: Reflex Cache
# ═══════════════════════════════════════════════════════════════════════════════


class TestReflexCache:
    """Verify reflex cache configuration."""

    def test_store_type_hashmap(self):
        data = load_constitution()
        assert data["reflex"]["store_type"] == "HashMap"

    def test_precipitation_ihsan_minimum(self):
        data = load_constitution()
        assert data["reflex"]["precipitation"]["ihsan_minimum"] == 0.90

    def test_invalidation_staleness(self):
        data = load_constitution()
        assert data["reflex"]["invalidation"]["staleness_max_days"] == 30


# ═══════════════════════════════════════════════════════════════════════════════
# §12: Daughter Test
# ═══════════════════════════════════════════════════════════════════════════════


class TestDaughterTest:
    """Verify daughter test liveness property."""

    def test_daughter_test_exists(self):
        data = load_constitution()
        assert "description" in data["daughter_test"]

    def test_is_liveness_property(self):
        data = load_constitution()
        assert data["daughter_test"]["type"] == "liveness_property"

    def test_has_enforcement(self):
        data = load_constitution()
        assert data["daughter_test"]["enforcement"] == "ci_test"


# ═══════════════════════════════════════════════════════════════════════════════
# §13: Error Handling
# ═══════════════════════════════════════════════════════════════════════════════


class TestErrorHandling:
    """Verify proper error handling for edge cases."""

    def test_missing_constitution_raises_error(self):
        with pytest.raises(ConstitutionError):
            load_constitution(Path("/nonexistent/constitution.toml"))

    def test_malformed_weights_detected(self):
        """Weights that don't sum to 1.0 are rejected."""
        data = load_constitution(validate=False)
        bad_data = copy.deepcopy(data)
        bad_data["ihsan_tensor"]["canonical_weights"]["moral_clarity"] = 0.99
        with pytest.raises(ConstitutionError, match="weights sum"):
            validate_constitution(bad_data)

    def test_missing_section_detected(self):
        """Missing a required section raises error."""
        data = load_constitution(validate=False)
        bad_data = copy.deepcopy(data)
        del bad_data["gates"]
        with pytest.raises(ConstitutionError, match="missing required section"):
            validate_constitution(bad_data)

    def test_gate_weight_mismatch_detected(self):
        """Gate weights not summing to 1.0 are rejected."""
        data = load_constitution(validate=False)
        bad_data = copy.deepcopy(data)
        bad_data["gates"]["alpha_4"]["weight"] = 0.99
        with pytest.raises(ConstitutionError, match="gate weights sum"):
            validate_constitution(bad_data)

    def test_rights_count_violation_detected(self):
        """Fewer rights than minimum_rights_count raises error."""
        data = load_constitution(validate=False)
        bad_data = copy.deepcopy(data)
        bad_data["identity"]["rights"]["rights"] = ["Exist"]
        bad_data["identity"]["rights"]["minimum_rights_count"] = 7
        with pytest.raises(ConstitutionError, match="rights count"):
            validate_constitution(bad_data)


# ═══════════════════════════════════════════════════════════════════════════════
# §14: Deployment and Conformance
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeploymentConformance:
    """Verify deployment milestones and conformance suite."""

    def test_conformance_cross_language(self):
        data = load_constitution()
        cl = data["conformance"]["cross_language"]
        assert "python" in cl["languages"]
        assert "rust" in cl["languages"]

    def test_psi_targets_monotonic(self):
        data = load_constitution()
        psi = data["psi_targets"]
        values = [
            psi["nodes_1"],
            psi["nodes_10"],
            psi["nodes_100"],
            psi["nodes_1000"],
            psi["nodes_5000"],
            psi["nodes_1B"],
            psi["nodes_8B"],
        ]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1], f"Psi not monotonic at index {i}"

    def test_deployment_phases(self):
        data = load_constitution()
        deploy = data["deployment"]
        assert "phase_0" in deploy
        assert "phase_4" in deploy
        assert deploy["phase_4"]["nodes"] == 100
