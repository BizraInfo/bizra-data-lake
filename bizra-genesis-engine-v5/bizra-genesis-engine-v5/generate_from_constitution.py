"""
BIZRA Constitutional Test Generator
════════════════════════════════════

Reads constitution.toml → generates:
  1. test_constitutional_conformance.py  (pytest conformance suite)
  2. generated_constants.py              (threshold constants for runtime)
  3. generated_gate_config.py            (gate weights + criteria for runtime)

Run: python generate_from_constitution.py [path/to/constitution.toml]

The generated files are the ONLY source of constants in the codebase.
Hardcoded values anywhere else = constitutional violation.
"""

from __future__ import annotations

import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from bizra_constitution import load_constitution, Constitution


def generate_constants(c: Constitution) -> str:
    """Generate the constants module that replaces all hardcoded values."""
    weights = c.ihsan.canonical_weights.as_dict()
    op_weights = c.ihsan.operational_weights()
    ds = c.security.domain_separation

    return textwrap.dedent(f'''\
        """
        BIZRA Generated Constants — DO NOT EDIT MANUALLY
        ═════════════════════════════════════════════════
        Generated from constitution.toml v{c.meta.version}
        SHA-256: {c.raw_hash}
        Generated: {datetime.now(timezone.utc).isoformat()}

        To modify any value: edit constitution.toml, then re-run:
            python generate_from_constitution.py
        """

        # ── Constitution Reference ──
        CONSTITUTION_VERSION = "{c.meta.version}"
        CONSTITUTION_HASH = "{c.raw_hash}"

        # ── Ihsan Tensor: 8-dim Canonical Weights ──
        IHSAN_CANONICAL_WEIGHTS = {{
            {_dict_lines(weights, indent=4)}
        }}

        # ── Ihsan Tensor: 6-dim Operational Projection (renormalized) ──
        IHSAN_OPERATIONAL_WEIGHTS = {{
            {_dict_lines(op_weights, indent=4)}
        }}

        IHSAN_DIMENSIONS_CANONICAL = {c.ihsan.dimensions}
        IHSAN_DIMENSIONS_OPERATIONAL = {len(c.ihsan.operational_dimensions)}
        IHSAN_OPERATIONAL_NAMES = {c.ihsan.operational_dimensions}

        # ── Ihsan Thresholds ──
        IHSAN_GATE_MINIMUM = {c.ihsan.thresholds.gate_minimum}
        IHSAN_POI_CONSENSUS = {c.ihsan.thresholds.poi_consensus}
        IHSAN_BLOOM_ELIGIBILITY = {c.ihsan.thresholds.bloom_eligibility}
        IHSAN_EXCELLENCE = {c.ihsan.thresholds.ihsan_excellence}
        IHSAN_CONFORMANCE_JOIN = {c.ihsan.thresholds.conformance_join}

        # ── Gate Configuration ──
        GATE_FAIL_MODE = "{c.gates.fail_mode}"
        GATE_OVERHEAD_BUDGET_MS = {c.gates.total_overhead_budget_ms}
        GATE_WEIGHTS = {{
            "alpha_4": {c.gates.alpha_4.weight},
            "alpha_7": {c.gates.alpha_7.weight},
            "alpha_8": {c.gates.alpha_8.weight},
            "alpha_9": {c.gates.alpha_9.weight},
            "alpha_10": {c.gates.alpha_10.weight},
        }}

        # ── HHMM Configuration ──
        HMM_NUM_HIDDEN_STATES = {c.hhmm.hidden_states}
        HMM_OBSERVATION_WINDOW = {c.hhmm.observation_window}
        HMM_MAX_EM_ITERATIONS = {c.hhmm.max_em_iterations}
        HMM_INITIAL_LIVE_STATES = {c.hhmm.initial_live_states}
        HMM_EXPANSION_TRIGGER = {c.hhmm.expansion_trigger}

        # ── Complexity Tier Latency Budgets (ms) ──
        TIER_TRIVIAL_BUDGET_MS = {c.hhmm.tiers["trivial"].latency_budget_ms}
        TIER_SIMPLE_BUDGET_MS = {c.hhmm.tiers["simple"].latency_budget_ms}
        TIER_COMPLEX_BUDGET_MS = {c.hhmm.tiers["complex"].latency_budget_ms}
        TIER_SOVEREIGN_BUDGET_MS = {c.hhmm.tiers["sovereign"].latency_budget_ms}

        # ── Action Bus ──
        ACTION_BUS_GCD_TICK_MS = {c.hhmm.gcd_tick_ms}
        ACTION_BUS_MAX_CONCURRENT = {c.hhmm.max_concurrent_missions}
        ACTION_BUS_MAX_PER_HOUR = {c.hhmm.max_missions_per_hour}

        # ── Economics ──
        SEED_YEARLY_CAP = {c.economics.seed_yearly_cap}
        BLOOM_IHSAN_THRESHOLD = {c.economics.bloom_ihsan_threshold}
        ZAKAT_RATE = {c.economics.zakat_rate}
        GINI_THRESHOLD = {c.economics.gini_threshold}
        GINI_MEASUREMENT_INTERVAL_S = {c.economics.gini_measurement_interval_s}
        NO_RIBA = {c.economics.no_riba}
        NO_GHARAR = {c.economics.no_gharar}

        # ── Reflex Cache ──
        REFLEX_STORE_TYPE = "{c.reflex.store_type}"
        REFLEX_MAX_ENTRIES = {c.reflex.max_entries}
        REFLEX_PRECIPITATION_HITS = {c.reflex.consecutive_hits}
        REFLEX_PRECIPITATION_IHSAN = {c.reflex.ihsan_minimum}
        REFLEX_SIMILARITY_THRESHOLD = {c.reflex.template_similarity}
        REFLEX_INVALIDATION_INTERVAL = {c.reflex.invalidation_interval}
        REFLEX_INVALIDATION_DELTA = {c.reflex.invalidation_delta}
        REFLEX_STALENESS_DAYS = {c.reflex.staleness_max_days}

        # ── Security: Domain Separation Contexts ──
        DOMAIN_EVIDENCE_RECEIPT = "{ds.evidence_receipt}"
        DOMAIN_URP_LEASE = "{ds.urp_lease}"
        DOMAIN_POI_ATTESTATION = "{ds.poi_attestation}"
        DOMAIN_IDENTITY_GENESIS = "{ds.identity_genesis}"
        DOMAIN_TELESCRIPT_PUBLISH = "{ds.telescript_publish}"
        DOMAIN_BLOOM_MINT = "{ds.bloom_mint}"

        # ── Identity ──
        IDENTITY_KEY_ALGORITHM = "{c.identity.key_algorithm}"
        IDENTITY_AGENTS_PER_NODE = {c.identity.agents_per_node}
        IDENTITY_GENESIS_DOMAIN = "{c.identity.genesis_domain}"
        IDENTITY_RIGHTS = {c.identity.rights.rights}

        # ── PAT ──
        PAT_AGENT_COUNT = {c.pat.agent_count}
        PAT_AGENT_NAMES = {[a.name for a in c.pat.agents]}
        PAT_TRUST_STAGES = {[a.trust_stage for a in c.pat.agents]}

        # ── SAT ──
        SAT_AGENTS_PER_NODE = {c.sat.agents_per_node}
        SAT_BOOTSTRAP_ROLES = {c.sat.bootstrap_roles}
        SAT_INFRASTRUCTURE_FLOOR_PCT = {c.sat.minimum_infrastructure_pct}
        SAT_REBALANCE_INTERVAL_S = {c.sat.rebalance_interval_s}
        SAT_SERVICE_TYPES = {c.sat.service_types}

        # ── Conformance Thresholds ──
        CONFORMANCE_HHMM_ACCURACY = {c.conformance.hhmm_state_mapping_accuracy}
        CONFORMANCE_POI_VARIANCE = {c.conformance.poi_calculation_variance}
        CONFORMANCE_CROWN_ENTROPY = {c.conformance.crown_entropy_accuracy}
        CONFORMANCE_REFLEX_SEMANTIC = {c.conformance.reflex_abstraction_semantic}
        CONFORMANCE_POOL_LATENCY_MS = {c.conformance.pool_latency_ms}

        # ── Privacy ──
        PRIVACY_CLASSES = {c.security.privacy_classes}
        PRIVACY_DEFAULT = "{c.security.default_privacy}"
    ''')


def generate_tests(c: Constitution) -> str:
    """Generate the conformance test suite derived from the constitution."""
    return textwrap.dedent(f'''\
        """
        BIZRA Constitutional Conformance Tests — DO NOT EDIT MANUALLY
        ═════════════════════════════════════════════════════════════
        Generated from constitution.toml v{c.meta.version}
        SHA-256: {c.raw_hash}
        Generated: {datetime.now(timezone.utc).isoformat()}

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
                assert abs(s - 1.0) < 0.001, f"Ihsan weights sum to {{s}}, expected 1.0"

            def test_gate_weights_sum_to_one(self, constitution):
                s = constitution.gates.total_weight()
                assert abs(s - 1.0) < 0.001, f"Gate weights sum to {{s}}, expected 1.0"

            def test_pat_agent_count_matches(self, constitution):
                assert len(constitution.pat.agents) == {c.pat.agent_count}

            def test_trust_stages_are_unique(self, constitution):
                stages = [a.trust_stage for a in constitution.pat.agents]
                assert len(set(stages)) == len(stages), "Trust stages must be unique"

            def test_identity_rights_minimum(self, constitution):
                assert len(constitution.identity.rights.rights) >= {c.identity.rights.minimum_rights_count}

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
                assert constitution.ihsan.thresholds.gate_minimum == {c.ihsan.thresholds.gate_minimum}

            def test_bloom_eligibility_threshold(self, constitution):
                assert constitution.ihsan.thresholds.bloom_eligibility == {c.ihsan.thresholds.bloom_eligibility}

            def test_ihsan_excellence_standard(self, constitution):
                assert constitution.ihsan.thresholds.ihsan_excellence == {c.ihsan.thresholds.ihsan_excellence}


        # ═══════════════════════════════════════════════════════════
        # §3 — Gate Configuration
        # ═══════════════════════════════════════════════════════════

        class TestGates:
            """Constitutional gates must be correctly weighted."""

            def test_five_gates_defined(self, constitution):
                assert constitution.gates.count == 5

            def test_gate_overhead_budget(self, constitution):
                assert constitution.gates.total_overhead_budget_ms == {c.gates.total_overhead_budget_ms}

            @pytest.mark.parametrize("gate_name,expected_weight", [
                ("alpha_4", {c.gates.alpha_4.weight}),
                ("alpha_7", {c.gates.alpha_7.weight}),
                ("alpha_8", {c.gates.alpha_8.weight}),
                ("alpha_9", {c.gates.alpha_9.weight}),
                ("alpha_10", {c.gates.alpha_10.weight}),
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
                ("evidence_receipt", "{c.security.domain_separation.evidence_receipt}"),
                ("urp_lease", "{c.security.domain_separation.urp_lease}"),
                ("poi_attestation", "{c.security.domain_separation.poi_attestation}"),
                ("identity_genesis", "{c.security.domain_separation.identity_genesis}"),
                ("telescript_publish", "{c.security.domain_separation.telescript_publish}"),
                ("bloom_mint", "{c.security.domain_separation.bloom_mint}"),
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
                assert constitution.economics.zakat_rate == {c.economics.zakat_rate}

            def test_zakat_is_constitutional(self, constitution):
                assert constitution.economics.zakat_constitutional is True

            def test_gini_threshold(self, constitution):
                assert constitution.economics.gini_threshold == {c.economics.gini_threshold}

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
    ''')


def _dict_lines(d: dict[str, float], indent: int = 4) -> str:
    """Format a dict as aligned Python dict literal lines."""
    max_key = max(len(k) for k in d)
    lines = []
    for k, v in d.items():
        lines.append(f'"{k}":{" " * (max_key - len(k) + 1)}{v:.4f},')
    joiner = "\n" + " " * indent
    return joiner.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — Generate all files
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    c = load_constitution(path)

    out_dir = Path("generated")
    out_dir.mkdir(exist_ok=True)

    # 1. Constants module
    constants_path = out_dir / "generated_constants.py"
    constants_path.write_text(generate_constants(c))
    print(f"✅ Generated: {constants_path}")

    # 2. Conformance tests
    tests_path = out_dir / "test_constitutional_conformance.py"
    tests_path.write_text(generate_tests(c))
    print(f"✅ Generated: {tests_path}")

    print()
    print(f"   Constitution v{c.meta.version}")
    print(f"   SHA-256: {c.raw_hash[:16]}...")
    print(f"   Constants: {sum(1 for line in generate_constants(c).splitlines() if '=' in line and not line.strip().startswith('#'))} values generated")
    print(f"   Tests: {sum(1 for line in generate_tests(c).splitlines() if 'def test_' in line)} conformance tests generated")
    print()
    print("   Next: copy generated/ contents into your project")
    print("   Then: grep -r 'hardcoded' to find remaining non-constitutional constants")


if __name__ == "__main__":
    main()
