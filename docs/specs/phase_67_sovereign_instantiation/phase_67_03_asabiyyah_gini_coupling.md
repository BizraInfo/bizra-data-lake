# Phase 67.03 — Asabiyyah-Gini Coupling

## Specification + Pseudocode

**Status:** SPEC-READY
**Priority:** CRITICAL (highest-SNR unwired connection in codebase)
**Estimated LOC:** ~60 new, ~15 modified
**Files touched:** 3 (algorithms.py, ticker.py, constants.py)

---

## 1. Problem Statement

`khaldunian_throttle(gini)` is an open-loop controller. It reads Gini but
ignores social cohesion. `network_asabiyyah()` computes cohesion but the
result is stored in `TickResult` and never fed back into the economic engine.

**Ibn Khaldun's actual thesis:** A society with high asabiyyah (social
cohesion) can tolerate higher inequality without collapse. A fragmented
society cannot tolerate even moderate inequality.

The current implementation contradicts the source material it claims to
embody.

### Evidence from Existing Tests

- `test_khaldunian_throttle_*` (5 tests) — all pass Gini only, no cohesion
- `test_network_asabiyyah_*` — tests cohesion in isolation
- No test anywhere tests the interaction between A4 and A15

---

## 2. Functional Requirements

### FR-1: Cohesion-Adjusted Throttle

The Khaldunian throttle MUST accept both `gini` and `asabiyyah` as inputs.
High asabiyyah relaxes the throttle. Low asabiyyah tightens it.

```
INVARIANT: throttle(gini=0.60, asabiyyah=0.80) > throttle(gini=0.60, asabiyyah=0.20)
INVARIANT: throttle(gini=X, asabiyyah=Y) > 0  for all valid X, Y  (never zero)
INVARIANT: throttle(gini=0, asabiyyah=any) == FP_ONE  (healthy economy unaffected)
```

### FR-2: Ticker Integration

`process_tick()` MUST compute asabiyyah BEFORE minting (move Step 12 before
Step 4), and pass the result into `progressive_mint()`.

### FR-3: Backward Compatibility

- `khaldunian_throttle(gini)` with 1 arg MUST still work (default asabiyyah=0)
- All existing tests MUST pass without modification
- `TickResult` fields unchanged

### FR-4: Constitutional Bounds

The asabiyyah adjustment is bounded:

```
ADJUSTMENT_FLOOR = 0.80  (low cohesion tightens by at most 20%)
ADJUSTMENT_CEIL  = 1.20  (high cohesion relaxes by at most 20%)
```

This prevents asabiyyah from overriding Gini safety gates entirely.

---

## 3. Constants (additions to `core/integration/constants.py`)

```python
# Asabiyyah-Gini Coupling (Phase 67.03)
ASABIYYAH_COUPLING_FLOOR: float = 0.80   # Min adjustment (tighten 20%)
ASABIYYAH_COUPLING_CEIL: float = 1.20    # Max adjustment (relax 20%)
ASABIYYAH_NEUTRAL: float = 0.50          # Asabiyyah score at which adjustment = 1.0
```

**Rationale:** The neutral point (0.50) is the midpoint of the [0, 1] range.
Below 0.50, the network is fragmenting — throttle tightens. Above 0.50,
the network is cohering — throttle relaxes. The +/-20% bounds prevent the
cohesion signal from overwhelming the Gini safety mechanism.

---

## 4. Pseudocode

### 4.1 `asabiyyah_adjustment(asabiyyah: fp) -> fp`

New function in `algorithms.py`:

```
FUNCTION asabiyyah_adjustment(asabiyyah_score):
    # Linear interpolation from FLOOR to CEIL around neutral point
    #
    # asabiyyah = 0.0 -> FLOOR (0.80)
    # asabiyyah = 0.5 -> 1.0 (neutral)
    # asabiyyah = 1.0 -> CEIL (1.20)
    #
    # Formula: FLOOR + (CEIL - FLOOR) * asabiyyah
    # At 0.5: 0.80 + 0.40 * 0.50 = 1.00 (neutral, correct)
    # At 0.0: 0.80 + 0.40 * 0.00 = 0.80 (tighten)
    # At 1.0: 0.80 + 0.40 * 1.00 = 1.20 (relax)

    range = CEIL - FLOOR                                  # 0.40 fixed
    raw   = fp_add(FLOOR_FP, fp_mul(range_fp, asabiyyah)) # linear interp
    RETURN fp_clamp(raw, FLOOR_FP, CEIL_FP)               # belt + suspenders
```

**Properties:**
- Monotonically increasing in asabiyyah (more cohesion = more relaxed)
- Continuous (no discontinuities)
- Bounded [0.80, 1.20] (cannot override Gini safety)
- At neutral (0.50): returns exactly FP_ONE (no change from current behavior)
- All fixed-point arithmetic (no floats)

### 4.2 Modified `khaldunian_throttle(gini, asabiyyah=FP_ZERO) -> fp`

```
FUNCTION khaldunian_throttle(gini, asabiyyah=FP_ZERO):
    # Step 1: Existing Gini-only throttle (unchanged)
    base_throttle = existing_gini_logic(gini)  # lines 244-261

    # Step 2: Apply asabiyyah adjustment (NEW)
    IF asabiyyah > 0:
        adj = asabiyyah_adjustment(asabiyyah)
        adjusted = fp_mul(base_throttle, adj)
    ELSE:
        adjusted = base_throttle  # backward compat: no adjustment

    # Step 3: Floor at fp(0.01) — never zero, even with low cohesion
    RETURN max(adjusted, fp(0.01))
```

**Backward compatibility:** When called with 1 arg (`khaldunian_throttle(gini)`),
`asabiyyah` defaults to `FP_ZERO`, the `IF` branch is skipped, and behavior
is identical to current implementation. All 5 existing throttle tests pass.

### 4.3 Modified `progressive_mint(...)` signature

```
FUNCTION progressive_mint(receipt, ihsan, wallet, network_gini, mean_balance, network_asabiyyah=FP_ZERO):
    base = mint_seed(receipt, ihsan)
    IF base == 0: RETURN 0

    throttle = khaldunian_throttle(network_gini, network_asabiyyah)  # CHANGED: pass asabiyyah
    equity = ghazali_equity_factor(wallet, mean_balance)

    RETURN fp_mul(fp_mul(base, throttle), equity)
```

### 4.4 Modified `process_tick()` — reorder steps

```
FUNCTION process_tick(...):
    # Steps 1-3: UNCHANGED (Intent Gate, Ihsan Score, Compute Gini)

    # Step 3.5 (NEW): Compute Asabiyyah BEFORE minting
    asabiyyah = network_asabiyyah(wallets)

    # Step 4: Progressive Minting — now with asabiyyah
    FOR receipt, ihsan IN scored:
        ...
        minted = progressive_mint(receipt, ihsan, wallet, gini, mean_balance, asabiyyah)
        ...

    # Steps 5-11: UNCHANGED

    # Step 12: Store asabiyyah (already computed — just assign)
    result.network_asabiyyah_score = asabiyyah  # CHANGED: use pre-computed value
    result.network_gini = gini

    RETURN result
```

---

## 5. Edge Cases + Constraints

| Case | Expected Behavior |
|------|-------------------|
| Empty network (0 wallets) | asabiyyah=0, adjustment=0.80, throttle tightens (correct: no community) |
| Single node | asabiyyah=0 (by def), adjustment=0.80 |
| Perfect equality (gini=0) | throttle=1.0 regardless of asabiyyah (FR-1 invariant 3) |
| All nodes max-connected | asabiyyah->1.0, adjustment=1.20, throttle relaxed by 20% |
| 2-node collusion ring | MIN_CONNECTIONS=3 blocks reciprocal, asabiyyah stays low, no relaxation |
| High gini + high asabiyyah | Throttle relaxed (society coheres despite inequality — Khaldun's point) |
| High gini + low asabiyyah | Throttle tightened (fragmentation + inequality = danger) |
| Extreme gini (>0.70) | Base throttle=0.01, even with max asabiyyah: 0.01*1.20=0.012 (minimal) |
| Overflow | fp_clamp prevents; fp_mul truncation is safe; FP_MAX guard in fp_add |

---

## 6. TDD Anchors (12 tests)

```python
class TestAsabiyyahAdjustment:
    # T1: Zero cohesion -> FLOOR (0.80)
    def test_zero_asabiyyah_returns_floor()

    # T2: Neutral cohesion -> FP_ONE (no change)
    def test_neutral_asabiyyah_returns_one()

    # T3: Max cohesion -> CEIL (1.20)
    def test_max_asabiyyah_returns_ceil()

    # T4: Monotonically increasing
    def test_monotonic_increase()

    # T5: Bounded output
    def test_output_within_bounds()


class TestKhaldunianThrottleCoupled:
    # T6: Backward compat — 1-arg call identical to v1
    def test_backward_compatible_single_arg()

    # T7: High asabiyyah relaxes throttle
    def test_high_asabiyyah_relaxes_throttle()

    # T8: Low asabiyyah tightens throttle
    def test_low_asabiyyah_tightens_throttle()

    # T9: Healthy gini unaffected by asabiyyah
    def test_healthy_gini_always_full_throttle()

    # T10: Never-zero invariant maintained
    def test_never_zero_with_coupling()


class TestTickerAsabiyyahCoupling:
    # T11: Asabiyyah computed before minting
    def test_asabiyyah_feeds_into_minting()

    # T12: Anti-collusion prevents relaxation for small rings
    def test_collusion_ring_no_relaxation()
```

---

## 7. Invariant Proofs (informal)

**I1: Never-zero.** Base throttle >= fp(0.01). Adjustment >= 0.80.
Product >= fp(0.01) * 0.80 = fp(0.008). Final `max(_, fp(0.01))` ensures >= fp(0.01).

**I2: Backward compatible.** Default `asabiyyah=FP_ZERO` → `IF asabiyyah > 0` is false →
existing code path. No existing test calls `khaldunian_throttle` with 2 args.

**I3: Bounded influence.** Adjustment in [0.80, 1.20]. A 20% swing on the
throttle means the MOST asabiyyah can do is shift a 0.50 throttle to 0.60
(relax) or 0.40 (tighten). Gini remains the primary driver.

**I4: Anti-collusion preserved.** `asabiyyah_score()` already enforces
`MIN_CONNECTIONS=3`. A 2-node collusion ring gets asabiyyah=0 →
adjustment=0.80 → throttle tightens. Collusion is penalized, not rewarded.

---

## 8. Dependency Graph

```
constants.py ──────────────────────────────┐
  + ASABIYYAH_COUPLING_FLOOR/CEIL/NEUTRAL  │
                                           ▼
algorithms.py ─────────────────────────────┐
  + asabiyyah_adjustment() [NEW]           │
  ~ khaldunian_throttle() [+asabiyyah arg] │
  ~ progressive_mint() [+asabiyyah arg]    │
                                           ▼
ticker.py ─────────────────────────────────┐
  ~ process_tick() [reorder steps]         │
                                           ▼
__init__.py                                │
  + export asabiyyah_adjustment            │
                                           ▼
tests/constitutional/test_algorithms.py    │
  + 10 new tests (T1-T10)                 │
tests/constitutional/test_ticker.py        │
  + 2 new tests (T11-T12)                 │
```

**Zero impact on:** types.py, fixed_point.py, declaration.py, cli.py,
omega_engine.py, or any Rust crate. This is a pure algorithm-layer change.

---

## 9. What We Are NOT Doing

- **Asabiyyah → Gini feedback**: Asabiyyah influences minting rate (throttle),
  which indirectly affects Gini. We do NOT add a direct Asabiyyah → Gini
  calculation path. The coupling is one-directional: A15 → A4.
- **Changing Asabiyyah weights**: 0.4/0.3/0.3 remain as-is. This spec only
  wires the output, not rebalances the input.
- **Nonlinear coupling**: Linear interpolation is sufficient. A quadratic
  or sigmoid coupling adds complexity without proven benefit. Can be
  revisited if simulation data warrants it.
- **Rust sync**: The Rust workspace does not yet implement the constitutional
  ticker. When it does, the coupling must be mirrored. File a follow-up.

---

## 10. Verification Criteria

1. `pytest tests/constitutional/ -v` — all existing + 12 new tests GREEN
2. `ruff check core/constitutional/` — zero lint errors
3. `black --check core/constitutional/` — formatted
4. Property: `khaldunian_throttle(gini)` (1-arg) produces identical output to v1
5. Property: For all valid gini/asabiyyah pairs, `0 < throttle <= FP_ONE * 1.20`
6. The `TickResult.network_asabiyyah_score` field is computed BEFORE minting
   (verify by injecting a mock that asserts call order)
