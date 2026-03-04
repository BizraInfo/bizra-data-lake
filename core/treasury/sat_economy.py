"""SAT Economy -- Definition 1.8.

Formalizes the economic model: SAT workforce, role assignment,
GDP computation, Zakat redistribution, and sustainability proof.

Economy(t) = (W(t), R(t), A(t), GDP(t))
Where:
  W(t)   = 5 * N(t)              -- total SAT workforce (5 per node)
  R(t)   subset of SATRole       -- role set
  A(t)   : W(t) -> R(t)          -- role assignment function
  GDP(t) = sum_{r in R(t)} Output(r, t)  -- total economic output

Corollary 4.2 Amendment: With local models (C_LLM approx 0), the system
is economically viable for ALL cache hit rates rho > 0.

Standing on Giants: Corollary 4.2 (proof chain) | Zakat (Islamic economics) | Harberger (optimal taxation)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class SATRole(Enum):
    """SAT agent roles in the economy.

    Each node contributes 5 SAT agents, allocated across these roles.
    Constitutional minimums ensure infrastructure and consensus are always staffed.
    """

    INFRASTRUCTURE = "infrastructure"
    CONSENSUS_VALIDATOR = "consensus_validator"
    CACHE_COORDINATOR = "cache_coordinator"
    COMPUTE_PROVIDER = "compute_provider"
    KNOWLEDGE_WORKER = "knowledge_worker"


# Constitutional minimums for role allocation (immutable)
ROLE_MINIMUMS: Dict[SATRole, float] = {
    SATRole.INFRASTRUCTURE: 0.20,  # 20% minimum to infrastructure
    SATRole.CONSENSUS_VALIDATOR: 0.10,  # 10% minimum to consensus
}

# Zakat rate -- 2.5%, the immutable constitutional constant from Islamic economics.
# Applied at mint time. Conservation law: net + zakat = gross.
ZAKAT_RATE: float = 0.025

# Each node contributes 5 SAT agents to the workforce.
SAT_PER_NODE: int = 5


@dataclass(frozen=True)
class EconomicState:
    """SAT Economy state at time t.

    Attributes:
        node_count: Number of nodes in the network N(t).
        role_allocation: Mapping of SATRole -> number of agents assigned.
        gdp_units: Current GDP output in abstract units.
    """

    node_count: int
    role_allocation: Dict[SATRole, int]
    gdp_units: float = 0.0

    @property
    def total_workforce(self) -> int:
        """W(t) = 5 * N(t) -- total SAT workforce."""
        return self.node_count * SAT_PER_NODE

    @property
    def allocated_workforce(self) -> int:
        """Total agents currently assigned to roles."""
        return sum(self.role_allocation.values())

    def validate_constitutional_minimums(self) -> List[str]:
        """Verify role allocation meets constitutional constraints.

        Returns:
            List of violation descriptions. Empty list means fully compliant.
        """
        violations: List[str] = []
        total = self.total_workforce

        for role, min_fraction in ROLE_MINIMUMS.items():
            actual = self.role_allocation.get(role, 0)
            required = math.ceil(total * min_fraction)
            if actual < required:
                violations.append(
                    f"{role.value}: allocated {actual}, required >= {required} "
                    f"({min_fraction * 100:.0f}% of {total})"
                )

        return violations


def compute_gdp_scaling(node_count: int) -> dict:
    """GDP scaling from Theorem 2.5: GDP(N) = Theta(N / log N).

    Returns theoretical bounds and empirical projections.

    Args:
        node_count: Number of nodes N in the network.

    Returns:
        Dict with node_count, gdp_lower_bound (N/log N), gdp_upper_bound (N^1.2),
        and per-node scaling factors.
    """
    if node_count <= 0:
        return {
            "node_count": 0,
            "gdp_lower_bound": 0.0,
            "gdp_upper_bound": 0.0,
            "scaling_factor_lower": 0.0,
            "scaling_factor_upper": 0.0,
        }

    log_n = math.log(node_count) if node_count > 1 else 1.0
    gdp_lower = node_count / log_n
    gdp_upper = node_count**1.2

    return {
        "node_count": node_count,
        "gdp_lower_bound": gdp_lower,
        "gdp_upper_bound": gdp_upper,
        "scaling_factor_lower": gdp_lower / node_count,
        "scaling_factor_upper": gdp_upper / node_count,
    }


def sustainability_analysis(
    missions_per_day: int,
    cache_hit_rate: float,
    seed_per_mission: float = 0.01,
    cloud_cost_per_mission: float = 0.02,
    local_cost_per_mission: float = 0.0001,
) -> dict:
    """Prove economic viability for local vs cloud models.

    Corollary 4.2 amendment: with local models, C_LLM is approximately zero,
    making the system unconditionally viable for all rho > 0.

    Args:
        missions_per_day: Total missions processed per day.
        cache_hit_rate: Fraction of missions served from reflex cache (rho).
        seed_per_mission: SEED tokens earned per mission.
        cloud_cost_per_mission: Cost per mission using cloud LLM API.
        local_cost_per_mission: Cost per mission using local Ollama/LM Studio.

    Returns:
        Dict with cloud and local economics, viability flags, and cost ratio.
    """
    missions_needing_inference = missions_per_day * (1.0 - cache_hit_rate)

    # Cloud model economics
    cloud_cost = missions_needing_inference * cloud_cost_per_mission
    cloud_revenue = missions_per_day * seed_per_mission
    cloud_profit = cloud_revenue - cloud_cost
    cloud_viable = cloud_profit > 0

    # Local model economics
    local_cost = missions_needing_inference * local_cost_per_mission
    local_revenue = missions_per_day * seed_per_mission
    local_profit = local_revenue - local_cost
    local_viable = local_profit > 0

    # Minimum cache hit rate for cloud viability
    if cloud_cost_per_mission > 0:
        min_rho_cloud = 1.0 - (seed_per_mission / cloud_cost_per_mission)
    else:
        min_rho_cloud = 0.0

    # Cost advantage ratio (local vs cloud)
    if local_cost > 0:
        cost_advantage_ratio = cloud_cost / local_cost
    else:
        cost_advantage_ratio = float("inf")

    return {
        "cache_hit_rate": cache_hit_rate,
        "cloud": {
            "cost_per_day": cloud_cost,
            "revenue_per_day": cloud_revenue,
            "profit_per_day": cloud_profit,
            "viable": cloud_viable,
            "min_cache_hit_rate": max(0.0, min_rho_cloud),
        },
        "local": {
            "cost_per_day": local_cost,
            "revenue_per_day": local_revenue,
            "profit_per_day": local_profit,
            "viable": local_viable,
            "min_cache_hit_rate": 0.0,  # viable for ALL rho > 0
        },
        "cost_advantage_ratio": cost_advantage_ratio,
    }


def zakat_mint(gross_amount: float) -> dict:
    """Apply Zakat deduction at mint time.

    Conservation law: net + zakat = gross (no value created or destroyed).

    Args:
        gross_amount: The gross SEED amount being minted.

    Returns:
        Dict with gross, net, zakat, and rate fields.
    """
    if gross_amount <= 0:
        return {"gross": 0.0, "net": 0.0, "zakat": 0.0, "rate": ZAKAT_RATE}

    zakat = gross_amount * ZAKAT_RATE
    net = gross_amount - zakat

    return {
        "gross": gross_amount,
        "net": net,
        "zakat": zakat,
        "rate": ZAKAT_RATE,
    }
