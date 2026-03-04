# Phase 64.02 — URP Contribution Protocol (إيثار as Protocol)

## Purpose

Convert idle hardware capacity into network value. When the node's
GPU sits idle, that idle time is uncommitted potential. The URP
Contribution Protocol makes إيثار (selflessness) economically
rational: the more you give, the more you earn.

Standing on Giants: Ostrom (commons governance) · Baran (distributed sharing) · Al-Ghazali (إيثار ethics)

## Key Insight

```
Cloud model:   You pay AWS to use THEIR hardware (value flows OUT)
BIZRA model:   Others use YOUR hardware and YOU get paid (value flows IN)

The node with surplus becomes a provider.
The node without becomes a recipient.
Zakat (2.5%) ensures the weakest nodes are subsidized.
The Gini gate prevents concentration.
```

## Data Model

```pseudocode
@dataclass(frozen=True)
class ContributionRecord:
    """A single contribution of idle capacity to the URP."""
    contribution_id: str              # blake3 hash of payload
    node_id: str
    contributed_at: str               # ISO 8601 UTC
    duration_s: float                 # how long the capacity was offered
    resource_type: str                # "gpu", "cpu", "ram", "storage"
    capacity_offered: float           # amount offered (GB, cores, etc.)
    capacity_consumed: float          # amount actually used by network
    utilization_rate: float           # consumed / offered (0.0 to 1.0)
    seed_earned: float                # SEED tokens minted for this contribution
    zakat_deducted: float             # 2.5% of seed_earned
    seed_net: float                   # seed_earned - zakat_deducted
    ihsan_score: float                # quality of contribution (>= 0.95)
    evidence_hash: str                # hash of evidence receipt
    constitutional_gate: str          # APPROVED / REJECTED


@dataclass
class ContributionLedger:
    """Persistent ledger of all URP contributions from this node."""
    node_id: str
    records: List[ContributionRecord]
    total_seed_earned: float
    total_zakat_paid: float
    total_capacity_hours: float       # cumulative contribution time
    contribution_streak_days: int     # consecutive days of contribution

    def append(self, record: ContributionRecord) -> None: ...
    def total_since(self, since: datetime) -> float: ...
    def save(self, path: Path) -> None: ...
    def load(cls, path: Path) -> "ContributionLedger": ...
```

## Core Class

```pseudocode
class URPContributor:
    """Converts idle capacity into network value and SEED earnings.

    This is NOT a resource allocator. It is an إيثار engine.
    The node contributes because contribution IS the profit mechanism.
    The Nash equilibrium of the network IS maximum contribution.
    """

    def __init__(
        self,
        node_id: str,
        asset_registry: AssetRegistry,
        token_minter: TokenMinter,
        evidence_ledger: EvidenceLedger,
        *,
        contribution_interval_s: float = 60.0,
        min_idle_fraction: float = 0.10,
        max_contribution_fraction: float = 0.80,
    ):
        self._node_id = node_id
        self._assets = asset_registry
        self._minter = token_minter
        self._evidence = evidence_ledger
        self._interval = contribution_interval_s
        self._min_idle = min_idle_fraction
        self._max_contrib = max_contribution_fraction
        self._ledger = ContributionLedger(node_id=node_id, records=[], ...)
        self._running = False

    # ── Contribution lifecycle ─────────────────────────────────

    async def contribute_cycle(self) -> Optional[ContributionRecord]:
        """Execute one contribution cycle.

        PSEUDOCODE:
        1. body = asset_registry.introspect()
        2. idle = body.idle_capacity
        3. FOR each resource_type in idle:
            a. IF idle[type] / total[type] < min_idle_fraction:
                SKIP (node needs its resources — homeostasis)
            b. offered = min(idle[type], total[type] * max_contribution_fraction)
            c. Register offering with URP network (or local simulation)
            d. WAIT for contribution_interval_s
            e. consumed = measure actual consumption
            f. utilization = consumed / offered
        4. Calculate SEED earned:
            a. base_seed = consumed * SEED_PER_UNIT[type]
            b. quality_multiplier = ihsan_score (quality of service)
            c. seed_earned = base_seed * quality_multiplier
        5. Constitutional gates:
            a. ihsan_gate: ihsan_score >= UNIFIED_IHSAN_THRESHOLD
            b. gini_gate: post-mint Gini <= ADL_GINI_THRESHOLD
            c. IF any gate REJECTED: log, return None, do NOT mint
        6. Mint SEED via token_minter.create():
            a. Zakat 2.5% auto-deducted
            b. seed_net = seed_earned * 0.975
        7. Record evidence via evidence_ledger.append()
        8. Append to contribution_ledger
        9. Return ContributionRecord
        """
        ...

    async def run_loop(self) -> None:
        """Continuous contribution loop — the node's إيثار heartbeat.

        PSEUDOCODE:
        - self._running = True
        - WHILE self._running:
            - TRY:
                - record = await contribute_cycle()
                - IF record: log contribution
            - EXCEPT Exception as e:
                - log warning (non-fatal — contribution is best-effort)
            - await asyncio.sleep(self._interval)
        """
        ...

    def stop(self) -> None:
        """Graceful stop — finish current cycle, then exit."""
        self._running = False

    # ── SEED calculation ───────────────────────────────────────

    def _calculate_seed(
        self,
        resource_type: str,
        consumed: float,
        duration_s: float,
        ihsan_score: float,
    ) -> float:
        """Calculate SEED earned for a contribution.

        PSEUDOCODE:
        - Rates are per unit per hour:
            SEED_PER_UNIT = {
                "gpu": 10.0,     # GPU-hours are most valuable
                "cpu": 1.0,      # CPU-hours are baseline
                "ram": 0.5,      # RAM-hours
                "storage": 0.1,  # Storage-hours (cheapest)
            }
        - base = consumed * (duration_s / 3600) * SEED_PER_UNIT[type]
        - quality = base * ihsan_score
        - Return quality
        """
        ...

    # ── Constitutional gates ───────────────────────────────────

    def _gate_contribution(
        self,
        seed_earned: float,
        ihsan_score: float,
    ) -> tuple[bool, str]:
        """Run constitutional gates on a contribution.

        PSEUDOCODE:
        - IF ihsan_score < UNIFIED_IHSAN_THRESHOLD:
            return (False, "IHSAN_BELOW_THRESHOLD")
        - IF seed_earned <= 0:
            return (False, "ZERO_CONTRIBUTION")
        - Simulate post-mint Gini via token_minter
        - IF post_gini > ADL_GINI_THRESHOLD:
            return (False, "GINI_CONCENTRATION")
        - Return (True, "APPROVED")
        """
        ...
```

## Flow Diagram

```
┌──────────────┐
│   Node Idle  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ AssetRegistry│──► idle_capacity = {gpu: 12GB, cpu: 24 cores, ...}
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ URPContrib   │──► offered = min(idle, max_contribution)
│   .cycle()   │
└──────┬───────┘
       │  network consumes some capacity
       ▼
┌──────────────┐
│ Calculate    │──► seed_earned = consumed * rate * ihsan
│   SEED       │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────┐
│ Constitutional Gates         │
│  ├─ Ihsan >= 0.95?          │──► REJECTED → no mint, log, continue
│  ├─ Gini <= 0.35?           │
│  └─ Evidence recorded?       │
└──────┬───────────────────────┘
       │ APPROVED
       ▼
┌──────────────┐
│ TokenMinter  │──► SEED minted (- 2.5% Zakat)
│  .create()   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Evidence     │──► ContributionRecord → ledger + evidence chain
│   Ledger     │
└──────────────┘
```

## Configuration

```yaml
# config/urp_contributor.yaml
urp_contributor:
  contribution_interval_s: 60       # check idle capacity every 60s
  min_idle_fraction: 0.10           # don't contribute if < 10% idle
  max_contribution_fraction: 0.80   # never contribute > 80% of capacity
  seed_rates:
    gpu: 10.0                       # SEED per GPU-hour consumed
    cpu: 1.0                        # SEED per CPU-hour consumed
    ram: 0.5                        # SEED per RAM-GB-hour consumed
    storage: 0.1                    # SEED per storage-GB-hour consumed
  thermal_safety:
    gpu_max_c: 85                   # stop GPU contribution above 85°C
    cpu_max_c: 90                   # stop CPU contribution above 90°C
```

## Key Architectural Decision: Contribution is Best-Effort

Contribution failures are NEVER fatal. The node's primary job is
serving its human. URP contribution is a background heartbeat —
إيثار that runs silently alongside the node's primary purpose.

If the network is unreachable, the node continues operating locally.
If the GPU is busy with a local mission, contribution pauses.
If the Gini gate rejects a mint, the contribution is logged but
not rewarded — the node contributed anyway (إيثار without reward
is still إيثار).

## TDD Anchors

```
TEST: contribute_cycle returns None when idle < min_idle_fraction
TEST: contribute_cycle mints SEED when idle > min_idle_fraction
TEST: Zakat 2.5% is deducted from seed_earned
TEST: seed_net = seed_earned * 0.975
TEST: Ihsan gate rejects contribution with score < 0.95
TEST: Gini gate rejects mint that would exceed 0.35
TEST: thermal throttle stops GPU contribution at 85°C
TEST: contribution_ledger persists across restarts
TEST: run_loop handles exceptions without crashing
TEST: stop() completes current cycle before exiting
TEST: SEED rates match config values (no hardcoded rates)
TEST: evidence_ledger receives receipt for every contribution
```
