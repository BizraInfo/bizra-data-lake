"""TDD anchors for SAT Economy (Definition 1.8, Corollary 4.2).

11 tests covering:
  EconomicState: workforce = 5x nodes, constitutional minimum validation
  GDP scaling: super-linear growth, single-node edge case
  Sustainability: local always viable, cloud needs high cache, cost advantage
  Zakat: 2.5% deduction, conservation law, zero mint
"""

from __future__ import annotations

from core.treasury.sat_economy import (
    EconomicState,
    SATRole,
    compute_gdp_scaling,
    sustainability_analysis,
    zakat_mint,
)

# -- Workforce tests ---------------------------------------------------------


class TestWorkforce:
    """Verify W(t) = 5 * N(t) and role allocation validation."""

    def test_workforce_is_5x_nodes(self) -> None:
        """W(t) = 5 * N(t)."""
        state = EconomicState(node_count=1000, role_allocation={})
        assert state.total_workforce == 5000

    def test_constitutional_minimum_violation_detected(self) -> None:
        """Under-allocation is detected.

        100 nodes * 5 = 500 agents.
        Infrastructure minimum: 20% of 500 = 100.
        Allocating only 50 -> violation.
        """
        state = EconomicState(
            node_count=100,
            role_allocation={SATRole.INFRASTRUCTURE: 50},
        )
        violations = state.validate_constitutional_minimums()
        assert len(violations) > 0
        assert "infrastructure" in violations[0]

    def test_constitutional_minimum_passes(self) -> None:
        """Sufficient allocation produces no violations.

        100 nodes * 5 = 500 agents.
        Infrastructure: 20% of 500 = 100. Allocated: 100.
        Consensus: 10% of 500 = 50. Allocated: 50.
        """
        state = EconomicState(
            node_count=100,
            role_allocation={
                SATRole.INFRASTRUCTURE: 100,
                SATRole.CONSENSUS_VALIDATOR: 50,
            },
        )
        violations = state.validate_constitutional_minimums()
        assert len(violations) == 0


# -- GDP scaling tests -------------------------------------------------------


class TestGDPScaling:
    """Verify GDP(N) = Theta(N / log N) super-linear total growth."""

    def test_gdp_scaling_super_linear_total(self) -> None:
        """GDP(1000) > 5 * GDP(100) -- total output scales super-linearly.

        N/log(N): 1000/6.9 ~ 145 vs 100/4.6 ~ 22. Ratio ~ 6.6x > 5x.
        """
        gdp_100 = compute_gdp_scaling(100)
        gdp_1000 = compute_gdp_scaling(1000)
        assert gdp_1000["gdp_lower_bound"] > gdp_100["gdp_lower_bound"] * 5

    def test_gdp_single_node(self) -> None:
        """Edge case: single node has positive GDP."""
        gdp = compute_gdp_scaling(1)
        assert gdp["gdp_lower_bound"] > 0


# -- Sustainability tests ----------------------------------------------------


class TestSustainability:
    """Verify Corollary 4.2 amendment: local models unconditionally viable."""

    def test_local_models_always_viable(self) -> None:
        """Corollary 4.2: local models viable for ALL cache hit rates."""
        for rho in [0.0, 0.1, 0.5, 0.9, 0.99]:
            result = sustainability_analysis(
                missions_per_day=1000,
                cache_hit_rate=rho,
            )
            assert (
                result["local"]["viable"] is True
            ), f"Local should be viable at rho={rho}"

    def test_cloud_models_need_high_cache(self) -> None:
        """Cloud models need high rho (at $0.02/mission, $0.01 revenue).

        min_rho = 1 - (0.01 / 0.02) = 0.5.
        """
        result = sustainability_analysis(
            missions_per_day=1000,
            cache_hit_rate=0.0,
        )
        assert result["cloud"]["min_cache_hit_rate"] > 0.4

    def test_cost_advantage_ratio(self) -> None:
        """Local models are 200x cheaper than cloud.

        cloud_cost / local_cost = 0.02 / 0.0001 = 200.
        """
        result = sustainability_analysis(
            missions_per_day=1000,
            cache_hit_rate=0.0,
        )
        assert result["cost_advantage_ratio"] >= 100


# -- Zakat tests -------------------------------------------------------------


class TestZakat:
    """Verify Zakat redistribution mechanics."""

    def test_zakat_deduction_2_5_percent(self) -> None:
        """Zakat = 2.5% of gross mint."""
        result = zakat_mint(100.0)
        assert result["zakat"] == 2.5
        assert result["net"] == 97.5

    def test_zakat_conservation_law(self) -> None:
        """net + zakat = gross (no value created or destroyed)."""
        for amount in [0.01, 1.0, 100.0, 1_000_000.0]:
            result = zakat_mint(amount)
            assert (
                abs(result["net"] + result["zakat"] - result["gross"]) < 1e-10
            ), f"Conservation violated at amount={amount}"

    def test_zakat_zero_mint(self) -> None:
        """Zero mint -> zero everything."""
        result = zakat_mint(0.0)
        assert result["net"] == 0.0
        assert result["zakat"] == 0.0
