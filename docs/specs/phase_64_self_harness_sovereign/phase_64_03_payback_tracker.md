# Phase 64.03 — Payback Tracker

## Purpose

Track when the node's hardware investment becomes net-positive.
Current economics: buy device → depreciate → replace (net NEGATIVE).
BIZRA economics: buy device → earn SEED/BLOOM → net POSITIVE.

The payback period is the moment tokens_earned > device_cost.
After that inflection point, the device generates net value.

Standing on Giants: Shannon (information value measurement) · Al-Ghazali (التجارة مع الله — legitimate trade)

## Key Insight

```
When participation is profitable, adoption is inevitable.
BIZRA doesn't need marketing.
It needs the payback period to be shorter than the
device replacement cycle.
```

## Data Model

```pseudocode
@dataclass
class DeviceInvestment:
    """The cost of the hardware that IS this node."""
    device_id: str
    device_description: str          # "MSI laptop i9-14900HX + RTX 4090"
    purchase_cost_usd: float         # what the human paid
    purchase_date: str               # ISO 8601
    expected_lifetime_years: float   # estimated useful life
    depreciation_rate: float         # annual depreciation (0.0-1.0)
    residual_value_usd: float        # value at end of life

    @property
    def current_book_value(self) -> float:
        """Straight-line depreciation from purchase to now."""
        ...

    @property
    def daily_cost(self) -> float:
        """Amortized daily cost = (purchase - residual) / lifetime_days."""
        ...


@dataclass
class PaybackState:
    """Current state of the payback calculation."""
    device: DeviceInvestment
    total_seed_earned: float          # cumulative SEED earned
    total_bloom_earned: float         # cumulative BLOOM earned
    total_zakat_paid: float           # cumulative Zakat deducted
    total_value_usd: float            # estimated USD value of earnings
    seed_to_usd_rate: float           # current SEED → USD conversion
    bloom_to_usd_rate: float          # current BLOOM → USD conversion
    payback_reached: bool             # total_value >= purchase_cost
    payback_date: Optional[str]       # when payback was reached (if ever)
    days_to_payback: Optional[int]    # projected days remaining (if not reached)
    roi_percent: float                # (total_value - purchase_cost) / purchase_cost * 100
    daily_earn_rate: float            # average daily SEED earnings
    contribution_uptime_pct: float    # what % of time the node contributed


@dataclass
class PaybackMilestone:
    """A milestone in the payback journey."""
    milestone_id: str
    milestone_type: str               # "25%", "50%", "75%", "100%", "200%"
    reached_at: str                   # ISO 8601
    total_earned_usd: float
    days_active: int
    evidence_hash: str                # evidence receipt hash
```

## Core Class

```pseudocode
class PaybackTracker:
    """Tracks when the node's hardware investment becomes net-positive.

    This is NOT an accounting system. It is the proof that
    BIZRA's economic thesis works — that devices can become
    income-generating assets instead of depreciating cost centers.
    """

    def __init__(
        self,
        device: DeviceInvestment,
        contribution_ledger: ContributionLedger,
        evidence_ledger: EvidenceLedger,
        *,
        state_path: Optional[Path] = None,
    ):
        self._device = device
        self._contributions = contribution_ledger
        self._evidence = evidence_ledger
        self._state_path = state_path or Path("sovereign_state/payback.json")
        self._milestones: List[PaybackMilestone] = []
        self._state: Optional[PaybackState] = None

    # ── Core calculation ───────────────────────────────────────

    def calculate(self) -> PaybackState:
        """Calculate current payback state.

        PSEUDOCODE:
        1. total_seed = contribution_ledger.total_seed_earned
        2. total_bloom = (computed from impact — separate system)
        3. total_zakat = contribution_ledger.total_zakat_paid
        4. total_value_usd = (total_seed * seed_rate) + (total_bloom * bloom_rate)
        5. payback_reached = total_value_usd >= device.purchase_cost_usd
        6. IF payback_reached AND not previously reached:
            - payback_date = now
            - Record milestone("100%")
            - Emit evidence receipt
        7. IF NOT payback_reached:
            - daily_rate = total_value_usd / days_active
            - remaining = device.purchase_cost_usd - total_value_usd
            - days_to_payback = remaining / daily_rate (if daily_rate > 0)
        8. roi = (total_value_usd - purchase_cost) / purchase_cost * 100
        9. Return PaybackState(...)
        """
        ...

    def check_milestones(self, state: PaybackState) -> List[PaybackMilestone]:
        """Check and record payback milestones.

        PSEUDOCODE:
        - milestones = ["25%", "50%", "75%", "100%", "200%", "500%"]
        - FOR each milestone in milestones:
            - threshold = device.purchase_cost * (pct / 100)
            - IF state.total_value >= threshold AND not already recorded:
                - Record PaybackMilestone
                - Emit evidence receipt
                - IF milestone == "100%":
                    - Log: "PAYBACK REACHED — device is now net-positive"
        - Return new milestones
        """
        ...

    # ── Projections ────────────────────────────────────────────

    def project_payback(self, days_ahead: int = 365) -> Dict[str, Any]:
        """Project payback trajectory.

        PSEUDOCODE:
        - state = calculate()
        - daily_rate = state.daily_earn_rate
        - projection = []
        - FOR day in range(1, days_ahead + 1):
            - projected_total = state.total_value_usd + (daily_rate * day)
            - projection.append({day, projected_total, payback_reached})
        - Return {
            "current": state,
            "projection": projection,
            "estimated_payback_date": ...,
            "estimated_annual_return_pct": daily_rate * 365 / purchase_cost * 100,
          }
        """
        ...

    # ── Persistence ────────────────────────────────────────────

    def save(self) -> None:
        """Persist payback state to sovereign_state/payback.json."""
        ...

    def load(self) -> Optional[PaybackState]:
        """Load persisted payback state."""
        ...

    # ── Dashboard summary ──────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        """Human-readable payback summary for dashboard.

        Returns:
        {
            "device": "MSI i9-14900HX + RTX 4090",
            "purchase_cost": "$2,499",
            "total_earned": "$127.50",
            "progress": "5.1%",
            "daily_rate": "$0.85/day",
            "estimated_payback": "2028-11-15",
            "days_remaining": 973,
            "roi": "-94.9%",
            "status": "EARNING",
            "milestones": ["25% — not reached"],
            "contribution_uptime": "87%",
        }
        """
        ...
```

## Flow Diagram

```
┌─────────────────────┐
│ ContributionLedger   │──► total_seed_earned, total_zakat_paid
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ PaybackTracker      │──► PaybackState
│   .calculate()      │     ├─ total_value_usd
│                     │     ├─ payback_reached?
│                     │     ├─ days_to_payback
│                     │     └─ roi_percent
└──────────┬──────────┘
           │
           ├──► check_milestones() ──► 25%, 50%, 75%, 100%, 200%
           │
           ├──► project_payback() ──► 365-day trajectory
           │
           └──► summary() ──► dashboard display
```

## Configuration

```yaml
# config/payback_tracker.yaml
payback_tracker:
  device:
    description: "Development workstation"
    purchase_cost_usd: 0.0           # user enters actual cost
    purchase_date: "2024-01-01"
    expected_lifetime_years: 5
    depreciation_rate: 0.20
    residual_value_usd: 0.0
  rates:
    seed_to_usd: 0.01               # initial SEED → USD (network sets this)
    bloom_to_usd: 0.05              # initial BLOOM → USD (network sets this)
  milestones: [25, 50, 75, 100, 200, 500]
```

## Important: No Speculation

The payback tracker reports MEASURED values. It does NOT:
- Promise future returns
- Speculate on token prices
- Make investment advice
- Compare to other investment vehicles

It reports: "You contributed X capacity. You earned Y SEED. At current
rates, that's Z USD. Your device cost W USD. Progress: Z/W%."

That's it. No riba. No gharar. Only verified, evidence-backed facts.
التجارة مع الله — honest trade, honest measurement.

## TDD Anchors

```
TEST: calculate returns PaybackState with correct totals
TEST: payback_reached is True when total_value >= purchase_cost
TEST: payback_reached is False when total_value < purchase_cost
TEST: days_to_payback is None when daily_rate is zero
TEST: roi_percent is negative before payback, positive after
TEST: milestone_25 fires at 25% of purchase_cost
TEST: milestone_100 fires exactly at payback point
TEST: milestones are not duplicated on recalculation
TEST: evidence receipt is emitted for each milestone
TEST: state persists across restarts via save/load
TEST: summary formats currency correctly
TEST: project_payback produces correct 365-day trajectory
TEST: zero purchase_cost doesn't cause division by zero
```
