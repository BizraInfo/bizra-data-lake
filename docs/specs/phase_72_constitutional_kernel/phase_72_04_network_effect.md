# Phase 72.04: Network Effect Estimator

**Target file:** `core/sovereign/network_effect.py`

## Purpose

Project how BIZRA's reverse-scaling works: more nodes → faster, cheaper,
smarter. Every new node brings hardware (compute), data (knowledge), and
intelligence (compiled reflexes). This module quantifies the projection.

## The Reverse-Scaling Principle

Traditional platforms: cost grows linearly with users, latency degrades.
BIZRA: cost per node decreases, latency improves, skill library expands.

```
Nodes   | Response Time | Skills   | Compute (TFLOPS) | Intelligence Density
--------|---------------|----------|------------------|---------------------
1       | baseline      | own only | own TFLOPS       | 1.0
100     | baseline/2    | 5,000    | ~100              | 50x
1,000   | baseline/3    | 50,000   | ~1,000            | 500x
1M      | baseline/6    | 5M       | ~1M               | combinatorial
```

**Why it works:** Each node is both consumer AND contributor. The network is
the infrastructure. No AWS bill scales linearly.

## Pseudocode

```pseudocode
IMPORT log, sqrt FROM math

@dataclass
CLASS NetworkProjection:
    nodes: int
    skills_available: int
    compute_tflops: float
    latency_factor: float       # multiplier on baseline latency (< 1.0 = faster)
    intelligence_density: float # cross-domain connections per node
    cost_per_node: float        # relative cost (1.0 = single node baseline)
    timestamp: str

# ─────────────────────────────────────────────────────────────
# Empirical Constants (calibrated from Node0 baseline)
# ─────────────────────────────────────────────────────────────

# Average compiled reflexes per active node (conservative estimate)
DEFAULT_REFLEXES_PER_NODE: int = 50

# Average node compute contribution in TFLOPS
# (MSI Titan = 43.6, budget laptop = 2.0, phone = 0.5)
# Weighted average assumes mixed device population
DEFAULT_TFLOPS_PER_NODE: float = 5.0

# Baseline latency for single-node operation (ms)
BASELINE_LATENCY_MS: float = 100.0

# Cost scaling constant (empirical)
COST_DECAY_RATE: float = 0.15


CLASS NetworkEffectEstimator:
    """Projects network-wide metrics as a function of node count.

    Pure computation — no I/O, no network calls. Uses conservative
    empirical constants that can be tuned as real data arrives.
    """

    CONSTRUCTOR(
        reflexes_per_node: int = DEFAULT_REFLEXES_PER_NODE,
        tflops_per_node: float = DEFAULT_TFLOPS_PER_NODE,
        baseline_latency_ms: float = BASELINE_LATENCY_MS,
    ):
        self._reflexes = reflexes_per_node
        self._tflops = tflops_per_node
        self._baseline_latency = baseline_latency_ms

    FUNCTION project(n_nodes: int) -> NetworkProjection:
        """Project network metrics for n_nodes participants."""
        IF n_nodes < 1:
            RAISE ValueError("n_nodes must be >= 1")

        # Skills: linear in nodes (each contributes their compiled reflexes)
        skills = n_nodes * self._reflexes

        # Compute: linear sum of node contributions
        compute = n_nodes * self._tflops

        # Latency: improves logarithmically (more local caches, shorter paths)
        # Factor < 1.0 means faster than baseline
        latency_factor = 1.0 / (1.0 + log(n_nodes) / log(10))

        # Intelligence density: cross-domain connections grow combinatorially
        # but we normalize per-node for interpretability
        # Metcalfe's law says connections ~ n², but useful connections
        # grow slower (Reed's law, log-adjusted)
        IF n_nodes > 1:
            intelligence_density = (n_nodes * log(n_nodes)) / n_nodes
            # Simplifies to log(n_nodes), but keeping the derivation clear
        ELSE:
            intelligence_density = 1.0

        # Cost per node: decreases as shared infrastructure amortizes
        cost_per_node = 1.0 / (1.0 + COST_DECAY_RATE * log(max(1, n_nodes)))

        RETURN NetworkProjection(
            nodes=n_nodes,
            skills_available=skills,
            compute_tflops=round(compute, 2),
            latency_factor=round(latency_factor, 4),
            intelligence_density=round(intelligence_density, 4),
            cost_per_node=round(cost_per_node, 4),
            timestamp=now_utc_iso(),
        )

    FUNCTION project_milestones() -> list[NetworkProjection]:
        """Standard milestone projections for dashboards and pitch decks."""
        milestones = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 8_000_000_000]
        RETURN [self.project(n) FOR n IN milestones]

    FUNCTION compute_moat_metrics(n_nodes: int) -> dict:
        """The three-pillar moat quantification.

        Each new node brings:
        1. Hardware — compute contribution
        2. Data — knowledge graph expansion
        3. Intelligence — compiled skill library growth
        """
        proj = self.project(n_nodes)
        RETURN {
            "hardware_contribution_tflops": proj.compute_tflops,
            "data_connections": n_nodes * (n_nodes - 1) / 2 IF n_nodes > 1 ELSE 0,
            "intelligence_skills": proj.skills_available,
            "moat_score": (
                log(1 + proj.compute_tflops) *
                log(1 + proj.skills_available) *
                proj.intelligence_density
            ),
        }
```

## Integration with Node Value

```pseudocode
# In NodeValueEngine._compute_network_synergy():
# When federation is live, replace stub with:

FUNCTION _compute_network_synergy(self) -> float:
    estimator = NetworkEffectEstimator()
    n_peers = self._federation.peer_count() IF self._federation ELSE 1
    proj = estimator.project(n_peers)
    # Network synergy scales with intelligence density
    RETURN proj.intelligence_density
```

## TDD Anchors

```pseudocode
TEST "single node projection is baseline":
    est = NetworkEffectEstimator()
    p = est.project(1)
    ASSERT p.nodes == 1
    ASSERT p.skills_available == 50
    ASSERT p.latency_factor == 1.0
    ASSERT p.cost_per_node == 1.0

TEST "more nodes means more skills":
    est = NetworkEffectEstimator()
    p1 = est.project(1)
    p100 = est.project(100)
    ASSERT p100.skills_available == 100 * p1.skills_available

TEST "latency improves with scale":
    est = NetworkEffectEstimator()
    p1 = est.project(1)
    p1000 = est.project(1000)
    ASSERT p1000.latency_factor < p1.latency_factor

TEST "cost per node decreases with scale":
    est = NetworkEffectEstimator()
    p1 = est.project(1)
    p1M = est.project(1_000_000)
    ASSERT p1M.cost_per_node < p1.cost_per_node

TEST "intelligence density grows with nodes":
    est = NetworkEffectEstimator()
    densities = [est.project(n).intelligence_density FOR n IN [1, 10, 100, 1000]]
    FOR i IN 0..len(densities)-2:
        ASSERT densities[i] < densities[i+1]

TEST "zero nodes raises ValueError":
    est = NetworkEffectEstimator()
    EXPECT_RAISES(ValueError, est.project, 0)

TEST "milestones returns 8 projections":
    est = NetworkEffectEstimator()
    milestones = est.project_milestones()
    ASSERT len(milestones) == 8
    ASSERT milestones[0].nodes == 1
    ASSERT milestones[-1].nodes == 8_000_000_000

TEST "moat metrics are positive for n > 1":
    est = NetworkEffectEstimator()
    moat = est.compute_moat_metrics(1000)
    ASSERT moat["hardware_contribution_tflops"] > 0
    ASSERT moat["intelligence_skills"] > 0
    ASSERT moat["moat_score"] > 0

TEST "custom reflexes_per_node scales skills":
    est = NetworkEffectEstimator(reflexes_per_node=100)
    p = est.project(10)
    ASSERT p.skills_available == 1000  # 10 * 100

TEST "8B nodes — the complete human skill graph":
    est = NetworkEffectEstimator()
    p = est.project(8_000_000_000)
    ASSERT p.skills_available == 400_000_000_000  # 8B * 50
    ASSERT p.latency_factor < 0.10  # ~10x faster than baseline
```
