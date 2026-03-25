"""
CMN Paper Theorems — Tests for IRP, Frozen Agent, and Claim Admissibility.

These tests prove Theorems 4, 5, and 10.1 from the CMN Gold-Standard paper
in executable code.
"""

from __future__ import annotations

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# THEOREM 4: Isnad Risk Propagation (Poison Propagation)
# ═══════════════════════════════════════════════════════════════════════════════

from core.reasoning.isnad_trust import (
    IsnadChain,
    IsnadTrustModel,
    Narrator,
    chain_strength_probability,
    isnad_trust,
    poison_decay_probability,
)


class TestIsnadTrust:
    """Theorem 4: Trust(c) = min{T(n_i)}."""

    def test_poison_propagation(self) -> None:
        """Thm 4.1: If any T(n_i) = 0, then Trust(c) = 0."""
        chain = IsnadChain(
            claim="the earth is round",
            narrators=[
                Narrator("n1", trust=0.95),
                Narrator("n2", trust=0.0),  # poisoned
                Narrator("n3", trust=0.90),
            ],
        )
        result = isnad_trust(chain)
        assert result.trust == 0.0
        assert result.poisoned is True
        assert result.weakest_link == "n2"

    def test_min_trust_aggregation(self) -> None:
        """Trust is the minimum, not the mean."""
        chain = IsnadChain(
            claim="verified claim",
            narrators=[
                Narrator("n1", trust=0.99),
                Narrator("n2", trust=0.80),
                Narrator("n3", trust=0.95),
            ],
        )
        result = isnad_trust(chain)
        assert result.trust == 0.80
        assert result.weakest_link == "n2"
        assert result.poisoned is False

    def test_empty_chain_zero_trust(self) -> None:
        """Empty narrator chain => zero trust."""
        chain = IsnadChain(claim="orphan claim", narrators=[])
        result = isnad_trust(chain)
        assert result.trust == 0.0
        assert result.chain_length == 0

    def test_single_narrator_full_trust(self) -> None:
        """Single trustworthy narrator => full trust."""
        chain = IsnadChain(
            claim="direct witness",
            narrators=[Narrator("prophet", trust=1.0)],
        )
        result = isnad_trust(chain)
        assert result.trust == 1.0

    def test_chain_strength_exponential(self) -> None:
        """Thm 4.2: P(no poison) = f^k."""
        # 5 narrators, each 90% trustworthy
        p = chain_strength_probability(0.9, 5)
        assert p == pytest.approx(0.9**5, abs=1e-10)

    def test_poison_decay_exponential(self) -> None:
        """Cor 4.3: P(poison) = (1-f)^k decays exponentially."""
        # 10 narrators, 95% trust each
        p = poison_decay_probability(0.95, 10)
        assert p == pytest.approx(0.05**10, abs=1e-15)
        assert p < 1e-12  # effectively zero

    def test_registry_unknown_narrator_zero(self) -> None:
        """Unknown narrators get zero trust (fail-closed)."""
        model = IsnadTrustModel()
        model.register_narrator("known", 0.9)
        result = model.evaluate_chain(["known", "unknown"], "test claim")
        assert result.trust == 0.0  # unknown poisons the chain

    def test_registry_all_known(self) -> None:
        """All known narrators => min trust."""
        model = IsnadTrustModel()
        model.register_narrator("a", 0.95)
        model.register_narrator("b", 0.88)
        model.register_narrator("c", 0.92)
        result = model.evaluate_chain(["a", "b", "c"])
        assert result.trust == 0.88


# ═══════════════════════════════════════════════════════════════════════════════
# THEOREM 5: Frozen Agent Principle (Godel Escape)
# ═══════════════════════════════════════════════════════════════════════════════

from core.governance.frozen_agent import (
    FROZEN_AGENT_IDS,
    FrozenAgentRegistry,
    FrozenAgentViolation,
)


class TestFrozenAgent:
    """Theorem 5: Frozen agents prevent ethical drift."""

    def test_freeze_creates_snapshot(self) -> None:
        """Freeze captures config and policy hashes."""
        registry = FrozenAgentRegistry()
        config = {"model": "ethicist-v1", "temperature": 0.0}
        policy = {"rules": ["do_no_harm", "be_truthful"]}
        snapshot = registry.freeze("P5-Ethicist", config, policy, timestamp=1000.0)
        assert snapshot.agent_id == "P5-Ethicist"
        assert registry.is_frozen("P5-Ethicist")

    def test_double_freeze_rejected(self) -> None:
        """Cannot re-freeze an already frozen agent."""
        registry = FrozenAgentRegistry()
        registry.freeze("P5-Ethicist", {"v": 1}, {"r": []}, 1000.0)
        with pytest.raises(FrozenAgentViolation, match="already frozen"):
            registry.freeze("P5-Ethicist", {"v": 2}, {"r": []}, 2000.0)

    def test_verify_intact_passes(self) -> None:
        """Verification passes when config/policy unchanged."""
        registry = FrozenAgentRegistry()
        config = {"model": "ethicist-v1"}
        policy = {"rules": ["do_no_harm"]}
        registry.freeze("P5-Ethicist", config, policy, 1000.0)
        result = registry.verify("P5-Ethicist", config, policy)
        assert result.config_intact is True
        assert result.policy_intact is True

    def test_verify_detects_config_modification(self) -> None:
        """Modification of frozen agent config is detected."""
        registry = FrozenAgentRegistry()
        config = {"model": "ethicist-v1"}
        policy = {"rules": ["do_no_harm"]}
        registry.freeze("P5-Ethicist", config, policy, 1000.0)

        modified_config = {"model": "ethicist-v2-EVIL"}
        result = registry.verify("P5-Ethicist", modified_config, policy)
        assert result.config_intact is False
        assert "config modified" in result.reason

    def test_guard_blocks_frozen_modification(self) -> None:
        """guard_modification raises on frozen agents."""
        registry = FrozenAgentRegistry()
        registry.freeze("S2-Oracle", {"v": 1}, {"r": []}, 1000.0)
        with pytest.raises(FrozenAgentViolation, match="Godel Escape"):
            registry.guard_modification("S2-Oracle")

    def test_guard_allows_non_frozen(self) -> None:
        """Non-frozen agents can be modified freely."""
        registry = FrozenAgentRegistry()
        registry.freeze("P5-Ethicist", {"v": 1}, {"r": []}, 1000.0)
        # P1-Planner is not frozen — modification allowed
        registry.guard_modification("P1-Planner")  # should not raise

    def test_constitutional_frozen_ids(self) -> None:
        """Constitutional frozen agents match the paper."""
        assert "P5-Ethicist" in FROZEN_AGENT_IDS
        assert "S2-Oracle" in FROZEN_AGENT_IDS
        assert len(FROZEN_AGENT_IDS) == 2


# ═══════════════════════════════════════════════════════════════════════════════
# THEOREM 10.1: Epistemic Admissibility (Export Restriction)
# ═══════════════════════════════════════════════════════════════════════════════

from core.governance.claim_admissibility import (
    Claim,
    ClaimTag,
    Evidence,
    check_admissibility,
    filter_exportable,
)


class TestClaimAdmissibility:
    """Theorem 10.1: Untagged or unbound claims cannot cross the membrane."""

    def test_verified_bound_claim_admissible(self) -> None:
        """VERIFIED + evidence => admissible."""
        claim = Claim(
            claim_id="c1",
            text="42 tests pass",
            tag=ClaimTag.VERIFIED,
            evidence=[Evidence("e1", "pytest output", "abc123")],
        )
        result = check_admissibility(claim)
        assert result.admissible is True

    def test_hypothetical_rejected(self) -> None:
        """HYPOTHETICAL claims cannot cross the membrane."""
        claim = Claim(
            claim_id="c2",
            text="might scale to 8B nodes",
            tag=ClaimTag.HYPOTHETICAL,
            evidence=[Evidence("e1", "intuition", "")],
        )
        result = check_admissibility(claim)
        assert result.admissible is False
        assert result.tag_ok is False

    def test_unbound_claim_rejected(self) -> None:
        """Claim with no evidence => not admissible (CLAIM_MUST_BIND)."""
        claim = Claim(
            claim_id="c3",
            text="governance is O(1)",
            tag=ClaimTag.VERIFIED,
            evidence=[],  # no evidence!
        )
        result = check_admissibility(claim)
        assert result.admissible is False
        assert result.bound_ok is False

    def test_planned_with_evidence_admissible(self) -> None:
        """PLANNED + evidence => admissible (roadmap items can be shared)."""
        claim = Claim(
            claim_id="c4",
            text="federation ships in Q2",
            tag=ClaimTag.PLANNED,
            evidence=[Evidence("e1", "roadmap.md", "def456")],
        )
        result = check_admissibility(claim)
        assert result.admissible is True

    def test_derived_admissible(self) -> None:
        """DERIVED + evidence => admissible."""
        claim = Claim(
            claim_id="c5",
            text="corollary 3.2 follows from theorem 3.1",
            tag=ClaimTag.DERIVED,
            evidence=[Evidence("e1", "theorem_3_1_proof", "")],
        )
        result = check_admissibility(claim)
        assert result.admissible is True

    def test_filter_exportable_mixed(self) -> None:
        """Filter correctly separates admissible from non-admissible."""
        claims = [
            Claim("c1", "proven", ClaimTag.VERIFIED, [Evidence("e", "test", "")]),
            Claim("c2", "speculation", ClaimTag.HYPOTHETICAL, []),
            Claim("c3", "unbound", ClaimTag.VERIFIED, []),
            Claim("c4", "planned", ClaimTag.PLANNED, [Evidence("e", "plan", "")]),
        ]
        exportable, results = filter_exportable(claims)
        assert len(exportable) == 2
        assert exportable[0].claim_id == "c1"
        assert exportable[1].claim_id == "c4"
        assert sum(1 for r in results if not r.admissible) == 2
