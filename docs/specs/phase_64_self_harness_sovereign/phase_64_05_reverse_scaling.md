# Phase 64.05 — Reverse Scaling Calibrator

## Purpose

Ensure that technical and economic reverse scaling are COUPLED.
More nodes must mean both better performance AND better economics.
If either axis fails to reverse-scale, the system is unstable.

SEED tracks resource contribution (technical axis).
BLOOM tracks impact created (economic axis).
The calibrator tunes the coupling constant between the two.

Standing on Giants: Shannon (coupled channel capacity) · Lamport (distributed system stability)

## Key Insight

```
TECHNICAL REVERSE SCALING:
  More nodes → more URP capacity → faster inference → better quality

ECONOMIC REVERSE SCALING:
  More nodes → more surplus in URP → more earning opportunities
  → more incentive to join → MORE NODES

PROOF OF COUPLING:
  If technical works but economic doesn't → nodes leave → UNSTABLE
  If economic works but technical doesn't → no value → UNSTABLE
  BOTH must scale together → system is ONLY stable when coupled.
```

## Data Model

```pseudocode
@dataclass
class ScalingSnapshot:
    """A point-in-time measurement of both scaling axes."""
    snapshot_id: str
    timestamp: str
    node_count: int                  # nodes in network (or simulated)
    # Technical axis
    avg_inference_latency_ms: float
    avg_quality_score: float         # Ihsan across network
    total_urp_capacity: float        # aggregate compute available
    urp_utilization: float           # how much URP is actually used
    # Economic axis
    total_seed_minted: float         # cumulative SEED minted
    total_bloom_minted: float        # cumulative BLOOM minted
    avg_daily_seed_per_node: float   # average earnings per node
    seed_velocity: float             # SEED transactions per day
    gini_coefficient: float          # wealth distribution
    # Coupling
    coupling_constant: float         # correlation(technical, economic)
    stability_score: float           # 0.0 (unstable) to 1.0 (stable)


@dataclass
class CouplingAlert:
    """Alert when technical and economic axes diverge."""
    alert_type: str                  # "DECOUPLING", "GINI_DRIFT", "LATENCY_SPIKE"
    severity: str                    # "warning", "critical"
    technical_delta: float           # change in technical axis
    economic_delta: float            # change in economic axis
    divergence: float                # |technical_delta - economic_delta|
    recommendation: str
    timestamp: str
```

## Core Class

```pseudocode
class ScalingCalibrator:
    """Tunes the coupling constant between SEED (technical)
    and BLOOM (economic) reverse scaling axes.

    The exchange rate between SEED and BLOOM IS the coupling
    constant. If well-calibrated, the system is self-reinforcing.
    If miscalibrated, one axis outpaces the other and the system
    becomes unstable.

    The Proof-of-Impact mechanism IS the calibration instrument.
    """

    def __init__(
        self,
        evidence_ledger: "EvidenceLedger",
        *,
        history_window: int = 30,     # days of history to analyze
        coupling_threshold: float = 0.7,  # minimum correlation for stability
        gini_ceiling: float = 0.35,   # ADL_GINI_THRESHOLD
    ):
        self._evidence = evidence_ledger
        self._window = history_window
        self._coupling_threshold = coupling_threshold
        self._gini_ceiling = gini_ceiling
        self._snapshots: List[ScalingSnapshot] = []

    # ── Measurement ────────────────────────────────────────────

    def measure(self, network_state: Dict[str, Any]) -> ScalingSnapshot:
        """Take a scaling measurement from network state.

        PSEUDOCODE:
        1. Extract technical metrics:
            - node_count from network_state
            - avg_latency from recent missions
            - avg_quality from Ihsan scores
            - urp_capacity from aggregate URPSnapshots
        2. Extract economic metrics:
            - total_seed from token ledger
            - total_bloom from impact ledger
            - avg_daily from contribution_ledger
            - gini from SAT rebalancer
        3. Compute coupling_constant:
            - IF len(snapshots) >= 2:
                - tech_trend = slope(latency_improvements over window)
                - econ_trend = slope(seed_earnings over window)
                - coupling = correlation(tech_trend, econ_trend)
            - ELSE: coupling = 1.0 (insufficient data)
        4. stability_score = min(coupling / coupling_threshold, 1.0)
        5. Append snapshot, return it
        """
        ...

    # ── Coupling analysis ──────────────────────────────────────

    def check_coupling(self) -> Optional[CouplingAlert]:
        """Check if technical and economic axes are coupled.

        PSEUDOCODE:
        - IF len(snapshots) < 3: return None (insufficient data)
        - recent = snapshots[-history_window:]
        - tech_deltas = [s2.avg_quality - s1.avg_quality for s1, s2 in pairs]
        - econ_deltas = [s2.avg_daily_seed - s1.avg_daily_seed for s1, s2 in pairs]
        - correlation = pearson(tech_deltas, econ_deltas)
        - IF correlation < coupling_threshold:
            return CouplingAlert(
                type="DECOUPLING",
                severity="critical" if correlation < 0.5 else "warning",
                divergence=coupling_threshold - correlation,
                recommendation="Adjust SEED rates or BLOOM criteria",
            )
        - IF latest.gini > gini_ceiling:
            return CouplingAlert(
                type="GINI_DRIFT",
                severity="critical",
                recommendation="Increase Zakat or activate rebalancer",
            )
        - Return None (healthy)
        """
        ...

    # ── Rate recommendations ───────────────────────────────────

    def recommend_seed_rate(self) -> Dict[str, float]:
        """Recommend SEED-per-unit rates based on scaling data.

        PSEUDOCODE:
        - IF coupling < threshold AND tech > econ:
            # Technical is outpacing economic — increase SEED rates
            adjustment = 1.0 + (threshold - coupling) * 0.5
            return {type: current_rate * adjustment for type in RESOURCE_TYPES}
        - IF coupling < threshold AND econ > tech:
            # Economic is outpacing technical — decrease SEED rates
            adjustment = 1.0 - (threshold - coupling) * 0.3
            return {type: current_rate * adjustment for type in RESOURCE_TYPES}
        - Return current rates (coupled, no change needed)
        """
        ...

    def recommend_bloom_criteria(self) -> Dict[str, Any]:
        """Recommend BLOOM minting criteria based on impact data.

        PSEUDOCODE:
        - Analyze which contributions generate most network value
        - Recommend impact thresholds for BLOOM eligibility
        - Ensure BLOOM tracks REAL impact, not just volume
        """
        ...

    # ── Stability monitoring ───────────────────────────────────

    def stability_report(self) -> Dict[str, Any]:
        """Human-readable scaling stability report.

        Returns:
        {
            "node_count": 1,
            "coupling_constant": 1.0,
            "stability": "HEALTHY",
            "technical_axis": {
                "trend": "improving",
                "latency_ms": 2100,
                "quality": 0.97,
            },
            "economic_axis": {
                "trend": "stable",
                "daily_seed": 4.2,
                "gini": 0.0,
            },
            "alerts": [],
            "rate_recommendation": "no change needed",
            "note": "Single-node mode — coupling analysis requires >= 3 snapshots"
        }
        """
        ...
```

## Single-Node Mode

Phase 64 launches with NODE0 as the only node. The scaling calibrator
operates in single-node mode:

```pseudocode
# Single-node mode:
# - coupling_constant defaults to 1.0 (trivially coupled)
# - No network latency measurements (only local)
# - Gini is always 0.0 (single holder)
# - Economic data comes from local contribution_ledger only
# - All rate recommendations are baseline (no adjustment)

# Multi-node mode (Alpha-100):
# - coupling_constant is computed from network-wide data
# - Latency includes network round-trips
# - Gini is computed across all node balances
# - Economic data aggregated from all contribution_ledgers
# - Rate recommendations are network-aware
```

The transition from single-node to multi-node mode happens automatically
when the node discovers peers through federation. No code change required —
just more data points for the same algorithms.

## Dual-Token Architecture

```
SEED                                    BLOOM
─────────────────────────               ─────────────────────────
Tracks: resource contribution           Tracks: impact created
Earned by: URP capacity provision       Earned by: Proof-of-Impact
Gated by: Ihsan >= 0.95                Gated by: Ihsan >= 0.95
Taxed by: Zakat 2.5%                   Taxed by: Zakat 2.5%
Bounded by: Gini <= 0.35              Bounded by: Gini <= 0.35

        coupling_constant
SEED ◄════════════════════► BLOOM

The coupling constant = correlation between SEED growth
and BLOOM growth over the measurement window. If both
grow together, the system is stable. If they diverge,
the calibrator recommends rate adjustments.
```

## TDD Anchors

```
TEST: measure returns valid ScalingSnapshot with all fields
TEST: single-node mode returns coupling_constant = 1.0
TEST: check_coupling returns None with < 3 snapshots
TEST: check_coupling returns DECOUPLING alert when correlation < threshold
TEST: check_coupling returns GINI_DRIFT when gini > 0.35
TEST: recommend_seed_rate increases rates when tech outpaces econ
TEST: recommend_seed_rate decreases rates when econ outpaces tech
TEST: recommend_seed_rate returns unchanged rates when coupled
TEST: stability_report includes all required fields
TEST: snapshots are bounded by history_window (memory safety)
TEST: coupling_threshold sourced from constants.py (no hardcoded values)
TEST: gini_ceiling matches ADL_GINI_THRESHOLD from constants.py
```
