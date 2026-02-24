# 02: HMM Single-Caller Isolation

## Standing on Giants
Rabiner (HMM, 1989) · Lamport (distributed state, 1978) · Shannon (channel isolation, 1948)

## Problem

The HMM engine maintains mutable state (observation history, state probabilities). In Phase 46.1, all MCP callers share a single `HMMEngine` instance inside `Phase46Interface`. During staging, we need to prevent cross-caller state pollution while still collecting telemetry from all callers.

## Design

### Isolation Mode

During staging, HMM observations are accepted from exactly ONE allowed caller. Other callers can read predictions but cannot mutate HMM state.

### Environment Variables (New)

```
BIZRA_PHASE46_HMM_CALLER_MODE=single     # "single" | "multi" | "disabled"
BIZRA_PHASE46_HMM_ALLOWED_CALLER=mcp     # caller_id string
```

### Constants (added to `core/integration/constants.py`)

```python
# Phase 47.1 — HMM Caller Isolation
HMM_CALLER_MODE_DEFAULT: Final[str] = "single"
HMM_ALLOWED_CALLER_DEFAULT: Final[str] = "mcp"
```

## Pseudocode

### `core/rollout/hmm_gate.py`

```
MODULE hmm_gate

IMPORT os, logging
FROM typing IMPORT Optional, Any, Dict
FROM dataclasses IMPORT dataclass, field
FROM datetime IMPORT datetime, timezone

logger = logging.getLogger(__name__)


@dataclass
CLASS HMMCallerStats:
    """Telemetry for HMM caller isolation."""
    accepted_count: int = 0
    dropped_count: int = 0
    dropped_callers: Dict[str, int] = field(default_factory=dict)
    last_accepted: Optional[datetime] = None
    last_dropped: Optional[datetime] = None


CLASS HMMCallerGate:
    """Gates HMM observations to a single allowed caller during staging.

    Modes:
    - "single": Only allowed_caller can observe. Others get read-only access.
    - "multi": All callers can observe (production mode).
    - "disabled": No callers can observe (emergency shutoff).
    """

    FUNCTION __init__(self, hmm_engine: Any):
        self._engine = hmm_engine
        self._mode = os.getenv("BIZRA_PHASE46_HMM_CALLER_MODE", "single")
        self._allowed = os.getenv("BIZRA_PHASE46_HMM_ALLOWED_CALLER", "mcp")
        self._stats = HMMCallerStats()

    FUNCTION observe(self, symbol: str, caller_id: str) -> Optional[Any]:
        """Gated observation — only allowed caller mutates HMM state.

        Args:
            symbol: HMM observation symbol (e.g., "search", "edit")
            caller_id: Identity of the caller (e.g., "mcp", "proactive", "apex")

        Returns:
            PredictionResult if accepted, None if dropped
        """
        IF self._mode == "disabled":
            self._record_drop(caller_id)
            RETURN None

        IF self._mode == "single" AND caller_id != self._allowed:
            self._record_drop(caller_id)
            logger.debug(
                "HMM gate: dropped observation from %s (allowed: %s)",
                caller_id, self._allowed
            )
            RETURN None

        # Accepted — mutate HMM state
        self._stats.accepted_count += 1
        self._stats.last_accepted = datetime.now(timezone.utc)
        RETURN self._engine.observe(symbol)

    FUNCTION predict(self, caller_id: str) -> Optional[Any]:
        """Read-only prediction — always allowed regardless of mode.

        Any caller can read predictions. Only observe() is gated.
        """
        IF self._engine IS None:
            RETURN None
        TRY:
            RETURN self._engine.predict_next()
        EXCEPT Exception AS exc:
            logger.warning("HMM gate predict failed: %s", exc)
            RETURN None

    FUNCTION _record_drop(self, caller_id: str):
        self._stats.dropped_count += 1
        self._stats.dropped_callers[caller_id] = (
            self._stats.dropped_callers.get(caller_id, 0) + 1
        )
        self._stats.last_dropped = datetime.now(timezone.utc)

    @property
    FUNCTION stats(self) -> Dict[str, Any]:
        """Telemetry snapshot for observability."""
        RETURN {
            "mode": self._mode,
            "allowed_caller": self._allowed,
            "accepted_count": self._stats.accepted_count,
            "dropped_count": self._stats.dropped_count,
            "dropped_callers": dict(self._stats.dropped_callers),
            "last_accepted": self._stats.last_accepted.isoformat()
                IF self._stats.last_accepted ELSE None,
            "last_dropped": self._stats.last_dropped.isoformat()
                IF self._stats.last_dropped ELSE None,
        }
```

## TDD Anchors

```python
class TestHMMCallerGate:
    """Tests for core/rollout/hmm_gate.py"""

    def test_single_mode_accepts_allowed_caller(self):
        """Allowed caller's observations mutate HMM state."""
        engine = MockHMMEngine()
        gate = HMMCallerGate(engine)  # default: single mode, allowed=mcp
        result = gate.observe("search", "mcp")
        assert result is not None
        assert gate.stats["accepted_count"] == 1

    def test_single_mode_drops_non_allowed_caller(self):
        """Non-allowed caller's observations are dropped."""
        engine = MockHMMEngine()
        gate = HMMCallerGate(engine)
        result = gate.observe("search", "proactive")
        assert result is None
        assert gate.stats["dropped_count"] == 1
        assert gate.stats["dropped_callers"] == {"proactive": 1}

    def test_multi_mode_accepts_all_callers(self):
        """Multi mode accepts observations from any caller."""
        with patch.dict(os.environ, {"BIZRA_PHASE46_HMM_CALLER_MODE": "multi"}):
            engine = MockHMMEngine()
            gate = HMMCallerGate(engine)
            gate.observe("search", "mcp")
            gate.observe("edit", "proactive")
            gate.observe("test", "apex")
            assert gate.stats["accepted_count"] == 3

    def test_disabled_mode_drops_all(self):
        """Disabled mode drops all observations."""
        with patch.dict(os.environ, {"BIZRA_PHASE46_HMM_CALLER_MODE": "disabled"}):
            engine = MockHMMEngine()
            gate = HMMCallerGate(engine)
            result = gate.observe("search", "mcp")
            assert result is None
            assert gate.stats["dropped_count"] == 1

    def test_predict_always_allowed(self):
        """Any caller can read predictions regardless of mode."""
        with patch.dict(os.environ, {"BIZRA_PHASE46_HMM_CALLER_MODE": "single"}):
            engine = MockHMMEngine()
            gate = HMMCallerGate(engine)
            # Non-allowed caller can still predict
            result = gate.predict("proactive")
            assert result is not None

    def test_stats_tracks_multiple_dropped_callers(self):
        """Stats correctly track drops by caller identity."""
        engine = MockHMMEngine()
        gate = HMMCallerGate(engine)
        gate.observe("search", "proactive")
        gate.observe("edit", "proactive")
        gate.observe("test", "apex")
        assert gate.stats["dropped_callers"] == {"proactive": 2, "apex": 1}
```

## Integration Points

### Where HMM gate hooks in:

1. **`tools/mcp/sovereign_mcp_server.py:Phase46Interface.predict()`** — wrap with `HMMCallerGate.observe(symbol, "mcp")`
2. **`core/living_memory/proactive.py:_observe_hmm()`** — wrap with `HMMCallerGate.observe(symbol, "proactive")`
3. **`core/resonance.py:CognitiveResonance.process()`** — wrap HMM observe with `HMMCallerGate.observe(symbol, "resonance")`

### Pattern:

```python
# BEFORE (Phase 46):
self._hmm_engine.observe(symbol)

# AFTER (Phase 47.1):
from core.rollout.hmm_gate import HMMCallerGate
gate = HMMCallerGate(self._hmm_engine)
gate.observe(symbol, caller_id="mcp")  # Only allowed caller mutates
```
