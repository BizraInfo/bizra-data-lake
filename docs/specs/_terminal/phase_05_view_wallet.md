# Phase 05 — View 4: WALLET (Economy)

> **Purpose:** Make the economic engine visible, intelligible, and felt.
> **Status:** PARTIAL — useWallet.ts hook built (170 LOC), sovereign_terminal.py wallet() exists, Rust TUI Treasury tab is placeholder.

## 5.1 Content Spec

```
┌── WALLET ─────────────────────────────────────────────────────┐
│  [LIVE]  Last sync: 3s ago                                    │
│                                                                │
│  SEED     42.5000    Liquid                                   │
│  BLOOM     1.2300    Soulbound (governance)                   │
│  BRANCH    0.0000    Reputation                               │
│  Locked    5.0000    Staking                                  │
│                                                                │
│  Zakat Contributed    1.0897    (2.5% cumulative)             │
│  Community Pool       21.2500   (50% of all minting)          │
│                                                                │
│  Supply Cap:  ████████░░░░░░░░░░░░  42% (420K / 1M)          │
│  Gini:        0.2134  ████████████████░░░░  [<= 0.35 PASS]   │
│                                                                │
│  ── Earning Factors ──────────────────────────────────────     │
│  Sovereignty  0.45  █████████░░░░░░░░░░░░                     │
│  Activation   0.80  ████████████████░░░░░                     │
│  Quality      0.97  ███████████████████░░                     │
│  Compounding  0.40  ████████░░░░░░░░░░░░░                     │
│  Synergy      0.30  ██████░░░░░░░░░░░░░░░                     │
│  ── Composite ────────────────────────────────────────────     │
│  Node Value   0.54  ███████████░░░░░░░░░░  (geometric mean)  │
│                                                                │
│  Last Reward: +2.38 SEED (1.19 node + 1.19 pool + 0.06 zakat)│
└──────────────────────────────────────────────────────────────┘
```

## 5.2 Data Model — Unified EconomicReceipt

From `terminal/economic.ts`, the canonical receipt type:

```pseudocode
struct WalletView:
    // Balances
    seed: float
    bloom: float
    branch: float
    locked_seed: float

    // Redistribution
    zakat_contributed: float     # Cumulative 2.5%
    community_pool_total: float  # Cumulative 50% split

    // Supply
    total_seed: float
    supply_cap: float            # 1,000,000 per year
    supply_utilization: float    # total_seed / cap
    circulating: float

    // Justice
    gini: float                  # Network-wide
    gini_headroom: float         # 0.35 - gini

    // Factors (5-dimensional)
    factors: {
        sovereignty: float,
        activation: float,
        quality: float,
        compounding: float,
        synergy: float,
    }
    composite: float             # Geometric mean

    // Metadata
    live: bool
    last_sync: timestamp | null
    tier: str                    # SEED/SPROUT/TREE/FOREST
    stage: str                   # Seed/Node/.../Catalyst

    // Last reward
    last_receipt: EconomicReceipt | null
```

## 5.3 Existing Implementation

**Frontend (useWallet.ts:81-170):** Complete wallet hook with:
- 30s polling with visibility-based pause/resume
- Offline fallback via `deriveOffline(nodeState)`
- `Promise.all` for 3 API calls (balance, supply, potential)
- Circuit breaker inherited from ApiClient
- 13 tests including 6 hardening (race conditions, partial failures)

**Python (sovereign_terminal.py:297-343):** `wallet()` — Shows SEED, BLOOM, Reflexes, Gini. Missing: factors, supply cap, zakat, community pool.

**Python (bloom.py:155-165):** `WalletState` dataclass — seed_balance, bloom, branch_balance.

**Rust (app.rs):** Treasury tab exists as tab #5 but renders empty.

**economic.ts:** `EconomicReceipt` type — the canonical receipt replacing MissionReceipt, RewardReceipt, and display transforms.

## 5.4 Computation Rules

```pseudocode
// Zakat (2.5% at mint time)
gross_seed = net_seed / (1 - ZAKAT_RATE)
zakat_contributed = gross_seed * ZAKAT_RATE

// Community Pool (50% of all minting — constitutionally locked)
pool_share = gross_seed * COMMUNITY_POOL_SPLIT  // HARDCODED 0.50

// Supply Cap Utilization
supply_utilization = total_seed / SEED_SUPPLY_CAP_PER_YEAR

// Node Value (geometric mean of 5 factors)
composite = (sovereignty * activation * quality * compounding * synergy) ^ (1/5)
// If ANY factor == 0, composite == 0 (no false positives)

// Gini Headroom
gini_headroom = ADL_GINI_THRESHOLD - current_gini  // Must be > 0
```

## 5.5 What to Build

| Component | Surface | LOC Est | Priority |
|-----------|---------|---------|----------|
| Wallet widget (Rust) | Rust | 200 | P0 |
| Factor breakdown bars | Rust | 80 | P0 |
| Supply cap gauge | Both | 40 | P0 |
| Gini headroom display | Both | 30 | P0 |
| Zakat + pool visibility | Both | 40 | P1 |
| Last reward breakdown | Both | 50 | P1 |
| Upgrade Python wallet() | Python | 60 | P1 |

## 5.6 TDD Anchors

```
TEST: wallet_offline_derives_from_node_state
  GIVEN backend unreachable, nodeState={seed:10, bloom:0.5, sovereignty:0.45}
  WHEN wallet view requested
  THEN shows seed=10, bloom=0.5, live=false

TEST: wallet_zakat_computation
  GIVEN balance.seed = 42.5, ZAKAT_RATE = 0.025
  THEN gross_seed = 42.5 / 0.975 = 43.5897...
  AND zakat_contributed = gross_seed * 0.025 = 1.0897...

TEST: wallet_supply_cap_gauge
  GIVEN total_seed = 420_000, cap = 1_000_000
  THEN supply_utilization = 0.42

TEST: wallet_gini_headroom_positive
  GIVEN gini = 0.21
  THEN gini_headroom = 0.14 (green)

TEST: wallet_gini_headroom_negative
  GIVEN gini = 0.38
  THEN gini_headroom = -0.03 (red, VIOLATION)

TEST: wallet_composite_geometric_mean
  GIVEN factors = {0.45, 0.80, 0.97, 0.40, 0.30}
  THEN composite = (0.45*0.80*0.97*0.40*0.30)^0.2 = 0.5247...

TEST: wallet_composite_zero_if_any_factor_zero
  GIVEN factors = {0.90, 0.80, 0.97, 0.0, 0.30}
  THEN composite = 0.0
```
