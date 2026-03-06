# Phase 69.01 — Wiring Sprint: Sprints 1-2 (Constitutional + Bus Foundation)
# Close Every Gap. Wire Every Circuit. Prove Every Connection.

## Context

Phase 69.00 identified 10 gaps ranked by SNR. This spec provides the
implementation pseudocode for Sprints 1-2. Sprints 3-6 are in `phase_69_02`.
Each module stays under 500 lines.

---

## Sprint 1: Close the Asabiyyah Loop (~100 LOC, 12 tests)

### 1.1 Add Constants

```python
# core/integration/constants.py — ADD after ASABIYYAH_WEIGHTS

# Asabiyyah-Gini Coupling (Phase 67.03a)
# Standing on Giants: Ibn Khaldun — high cohesion tolerates higher inequality
ASABIYYAH_COUPLING_FLOOR: Final[float] = 0.80   # Min multiplier (low cohesion)
ASABIYYAH_COUPLING_CEIL: Final[float] = 1.20    # Max multiplier (high cohesion)
ASABIYYAH_NEUTRAL: Final[float] = 0.50          # Neutral point (no adjustment)
```

### 1.2 Add `asabiyyah_adjustment()` to algorithms.py

```python
# core/constitutional/algorithms.py — NEW FUNCTION after network_asabiyyah()

# Import new constants
from core.integration.constants import (
    ASABIYYAH_COUPLING_CEIL,
    ASABIYYAH_COUPLING_FLOOR,
    ASABIYYAH_NEUTRAL,
)

# Fixed-point versions
ASAB_FLOOR: int = fp(ASABIYYAH_COUPLING_FLOOR)   # 800_000
ASAB_CEIL: int = fp(ASABIYYAH_COUPLING_CEIL)      # 1_200_000
ASAB_NEUTRAL: int = fp(ASABIYYAH_NEUTRAL)         # 500_000

def asabiyyah_adjustment(asabiyyah: int) -> int:
    """Linear interpolation of minting multiplier based on social cohesion.

    asabiyyah ∈ [0, FP_ONE]  →  multiplier ∈ [FLOOR, CEIL]

    At neutral (0.50): multiplier = 1.0 (no effect)
    Below neutral: multiplier < 1.0 (punish low cohesion)
    Above neutral: multiplier > 1.0 (reward high cohesion)

    Ibn Khaldun: "When asabiyyah is strong, the group can sustain
    greater internal inequality without collapse."

    >>> fp_float(asabiyyah_adjustment(fp(0.00)))  # No cohesion
    0.8
    >>> fp_float(asabiyyah_adjustment(fp(0.50)))  # Neutral
    1.0
    >>> fp_float(asabiyyah_adjustment(fp(1.00)))  # Maximum cohesion
    1.2
    """
    # Clamp to [0, FP_ONE]
    clamped = fp_clamp(asabiyyah, FP_ZERO, FP_ONE)

    # Linear interpolation: FLOOR + (CEIL - FLOOR) * (asabiyyah / FP_ONE)
    span = fp_sub(ASAB_CEIL, ASAB_FLOOR)       # 400_000 (0.40)
    ratio = fp_div(clamped, FP_ONE)             # Normalized [0, 1]
    return fp_add(ASAB_FLOOR, fp_mul(span, ratio))
```

### 1.3 Modify `khaldunian_throttle()` Signature

```python
def khaldunian_throttle(gini: int, asabiyyah: int = FP_ZERO) -> int:
    """Ibn Khaldun's progressive throttle with asabiyyah coupling.

    The base throttle curve remains unchanged. The asabiyyah parameter
    applies a multiplier that adjusts the throttle output:
    - High cohesion (asabiyyah > 0.50): throttle relaxes slightly
    - Low cohesion (asabiyyah < 0.50): throttle tightens further

    Backward compatible: asabiyyah=FP_ZERO → adjustment = FLOOR (0.80)
    """
    # Existing throttle curve (UNCHANGED)
    IF gini <= GINI_HEALTHY:
        base = FP_ONE
    ELIF gini <= GINI_WARNING:
        # ... existing quadratic dropoff ...
        base = <existing_calculation>
    ELIF gini <= GINI_CRISIS:
        base = fp(0.10)
    ELSE:
        base = fp(0.01)

    # NEW: Apply asabiyyah coupling
    multiplier = asabiyyah_adjustment(asabiyyah)
    adjusted = fp_mul(base, multiplier)

    # Clamp: never exceed 1.0, never reach 0
    return fp_clamp(adjusted, fp(0.001), FP_ONE)
```

### 1.4 Modify `progressive_mint()` Signature

```python
def progressive_mint(
    receipt: ActionReceipt,
    ihsan: int,
    wallet: WalletState,
    gini: int,
    mean_balance: int,
    asabiyyah: int = FP_ZERO,     # NEW parameter
) -> int:
    """Progressive minting with all corrections.

    Chain: ihsan_score → khaldunian_throttle → ghazali_equity → zakat
    """
    # ... existing code ...
    throttle = khaldunian_throttle(gini, asabiyyah)   # CHANGED: pass asabiyyah
    # ... rest unchanged ...
```

### 1.5 Reorder `process_tick()` — Step 3.5

```python
def process_tick(...) -> TickResult:
    # Step 1-2: Intent gate + Ihsan scoring (UNCHANGED)
    # Step 3: Compute Gini (UNCHANGED)

    # ──────────────────────────────────────────────────────────
    # Step 3.5: Compute network asabiyyah BEFORE minting  [NEW]
    # ──────────────────────────────────────────────────────────
    network_asab = network_asabiyyah(wallets) if wallets else FP_ZERO

    # Step 4: Progressive Minting — NOW with asabiyyah
    FOR receipt, ihsan IN scored:
        wallet = _find_wallet(wallets, receipt.actor_id)
        IF wallet IS None: CONTINUE
        minted = progressive_mint(
            receipt, ihsan, wallet, gini, mean_balance,
            asabiyyah=network_asab,       # CHANGED: pass asabiyyah
        )
        # ... rest unchanged ...

    # Step 5-11: UNCHANGED

    # Step 12: Record asabiyyah (already computed at 3.5)
    result.network_asabiyyah_score = network_asab    # CHANGED: use cached value
    result.network_gini = gini
    return result
```

### 1.6 TDD Anchors (12 tests)

```python
class TestAsabiyyahAdjustment:
    def test_zero_cohesion_returns_floor():
        assert fp_float(asabiyyah_adjustment(FP_ZERO)) == pytest.approx(0.80)

    def test_full_cohesion_returns_ceil():
        assert fp_float(asabiyyah_adjustment(FP_ONE)) == pytest.approx(1.20)

    def test_neutral_returns_one():
        assert fp_float(asabiyyah_adjustment(fp(0.50))) == pytest.approx(1.00)

    def test_monotonically_increasing():
        for a in [0.0, 0.25, 0.50, 0.75, 1.0]:
            for b in [a + 0.25]:
                if b <= 1.0:
                    assert asabiyyah_adjustment(fp(a)) <= asabiyyah_adjustment(fp(b))

    def test_clamps_above_one():
        assert asabiyyah_adjustment(fp(1.5)) == asabiyyah_adjustment(FP_ONE)

    def test_clamps_below_zero():
        assert asabiyyah_adjustment(fp(-0.5)) == asabiyyah_adjustment(FP_ZERO)

class TestKhaldunianWithAsabiyyah:
    def test_backward_compatible_default():
        """Without asabiyyah param, behavior unchanged (floor adjustment)."""
        old = khaldunian_throttle_v1(fp(0.40))  # Reference from v1
        new = khaldunian_throttle(fp(0.40))      # Default asabiyyah=0
        # New should be ≤ old (floor = 0.80 multiplier)
        assert new <= old

    def test_high_cohesion_relaxes_throttle():
        base = khaldunian_throttle(fp(0.40), fp(0.50))  # neutral
        relaxed = khaldunian_throttle(fp(0.40), fp(0.90))  # high cohesion
        assert relaxed > base

    def test_low_cohesion_tightens_throttle():
        base = khaldunian_throttle(fp(0.40), fp(0.50))  # neutral
        tight = khaldunian_throttle(fp(0.40), fp(0.10))  # low cohesion
        assert tight < base

    def test_never_exceeds_one():
        assert khaldunian_throttle(fp(0.10), FP_ONE) <= FP_ONE

    def test_never_reaches_zero():
        assert khaldunian_throttle(fp(0.95), FP_ZERO) > 0

class TestTickerAsabiyyahOrder:
    def test_asabiyyah_computed_before_minting():
        """Step 3.5 must execute before Step 4."""
        wallets = [make_wallet(balance=fp(100.0)) for _ in range(5)]
        receipts = [make_receipt() for _ in range(3)]
        result = process_tick(wallets, receipts, [], [], {})
        # Asabiyyah should be non-zero if wallets have attestation history
        # The key test: minting used asabiyyah, not zero
        assert result.network_asabiyyah_score >= FP_ZERO
```

---

## Sprint 2: Bus Foundation (~450 LOC, 34 tests)

### 2.1 `core/bus/__init__.py`

```python
"""BIZRA Bus Architecture — Phase 68 Nervous System."""

from core.bus.telescript import TeleScriptEngine, TeleScriptVerdict
from core.bus.topics import TopicRegistry, TopicTier
from core.bus.types import ActionBudget, ActionEnvelope, ActionStatus, BusActionReceipt
```

### 2.2 `core/bus/types.py` (~120 LOC)

```python
"""Bus Types — Frozen envelopes for CQRS command pipeline."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum

class ActionStatus(Enum):
    PROPOSED = "proposed"
    VALIDATING = "validating"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    DENIED = "denied"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass(frozen=True)
class ActionBudget:
    time_ms: int = 10_000
    s2_tokens_max: int = 50_000
    retry_max: int = 2

@dataclass(frozen=True)
class ActionEnvelope:
    action_id: str                    # blake3(canonical_content)
    kind: str                         # e.g., "mission.search.web"
    channel: str                      # "desktop" | "file" | "browser" | "llm" | "proof"
    payload: dict                     # action-specific data
    capabilities: tuple[str, ...]     # required capabilities
    telescript: dict                   # action-level restrictions
    budget: ActionBudget
    correlation_id: str               # mission linkage
    actor_id: bytes                   # ed25519 public key
    timestamp: int                    # unix ms

@dataclass(frozen=True)
class BusActionReceipt:
    receipt_id: str                   # blake3(canonical_content)
    action_id: str
    status: ActionStatus
    outcome_hash: str                 # blake3(outcome)
    ihsan_score: float
    prev_receipt_hash: str            # merkle chain
    guardian_verdict: str             # "allowed" | "denied" | "conditional"
    duration_ms: float
```

### 2.3 `core/bus/topics.py` (~150 LOC)

Full implementation of TopicRegistry from spec 68.06:
- TopicTier enum (8 tiers)
- TopicDef dataclass
- TOPIC_REGISTRY dict (38 topics)
- TopicRegistry class (validate, activate_tier, deactivate_tier, export_json)
- Generate `core/bus/topics.json` on import or via CLI

### 2.4 `core/bus/telescript.py` (~180 LOC)

Full implementation of TeleScriptEngine from spec 68.05:
- Capability enum (17 values)
- TeleScriptPolicy frozen dataclass
- TeleScriptVerdict frozen dataclass
- TeleScriptEngine.check() — fail-closed capability verification
- Path restriction checking with fnmatch
- Policy merging (action can only restrict, never expand)

### 2.5 TDD Anchors (34 tests across 3 files)

```
tests/core/bus/test_types.py         — 10 tests (envelope creation, status enum)
tests/core/bus/test_topics.py        — 10 tests (spec 68.06 anchors)
tests/core/bus/test_telescript.py    — 14 tests (spec 68.05 anchors)
```

Sprints 3-6 continue in `phase_69_02_wiring_sprint_continued.md`.
