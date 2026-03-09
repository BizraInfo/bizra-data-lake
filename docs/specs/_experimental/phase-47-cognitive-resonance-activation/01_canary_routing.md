# 01: Deterministic Canary Routing

## Standing on Giants
Fowler (canary releases, 2010) · Google SRE (traffic splitting, 2016) · Nygard (Release It!, 2007)

## Overview

Route a stable, deterministic percentage of requests through Phase 46 components. Same request key always maps to the same routing decision for a fixed percentage — no randomness, fully reproducible.

## Public Interface

### Environment Variables (New)

```
BIZRA_PHASE46_SEARCH_PERCENT=0        # 0-100, granularity 1%
BIZRA_PHASE46_GOT_BRIDGE_PERCENT=0    # 0-100, granularity 1%
BIZRA_PHASE46_HMM_PERCENT=0           # 0-100, granularity 1%
BIZRA_PHASE46_CANARY_SALT=<random>    # Stable salt for hash routing
```

### Constants (added to `core/integration/constants.py`)

```python
# Phase 47.1 — Canary Routing
CANARY_PERCENT_MIN: Final[int] = 0
CANARY_PERCENT_MAX: Final[int] = 100
CANARY_DEFAULT_SALT: Final[str] = "bizra-phase46-canary-v1"
```

## Pseudocode

### `core/rollout/canary.py`

```
MODULE canary

IMPORT hashlib, os, logging
FROM core.integration.constants IMPORT CANARY_DEFAULT_SALT

logger = logging.getLogger(__name__)


CLASS CanaryRouter:
    """Deterministic canary routing using stable hashing.

    Kill switches (boolean flags) ALWAYS take precedence:
    - If BIZRA_PHASE46_SEARCH_ENABLED == "0", search is OFF regardless of percent
    - Percent routing only applies when boolean flag is "1" or unset-and-canaried
    """

    FUNCTION __init__(self, salt: str = None):
        self._salt = salt OR os.getenv("BIZRA_PHASE46_CANARY_SALT", CANARY_DEFAULT_SALT)
        self._cache: Dict[Tuple[str, str, int], bool] = {}  # LRU-bounded

    FUNCTION should_route(self, component: str, request_key: str, percent: int) -> bool:
        """Determine if this request should use the canary path.

        Args:
            component: "search" | "got_bridge" | "hmm"
            request_key: Stable identifier (request_id, query hash, caller_id)
            percent: 0-100 routing percentage

        Returns:
            True if this request should use the Phase 46 component
        """
        # Gate 0: Percent bounds check
        IF percent <= 0:
            RETURN False
        IF percent >= 100:
            RETURN True

        # Gate 1: Kill switch precedence (boolean flags override percent)
        kill_switch = self._check_kill_switch(component)
        IF kill_switch IS NOT None:
            RETURN kill_switch  # Explicit True/False from boolean flag

        # Gate 2: Deterministic hash routing
        hash_input = f"{self._salt}:{component}:{request_key}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()
        bucket = int(hash_value[:8], 16) % 100  # 0-99
        routed = bucket < percent

        logger.debug(
            "canary route: component=%s key=%s percent=%d bucket=%d routed=%s",
            component, request_key[:20], percent, bucket, routed
        )
        RETURN routed

    FUNCTION _check_kill_switch(self, component: str) -> Optional[bool]:
        """Check boolean kill switch. Returns None if not explicitly set."""
        env_map = {
            "search": "BIZRA_PHASE46_SEARCH_ENABLED",
            "got_bridge": "BIZRA_PHASE46_GOT_BRIDGE_ENABLED",
            "hmm": "BIZRA_PHASE46_HMM_ENABLED",
        }
        env_key = env_map.get(component)
        IF env_key IS None:
            RETURN None

        value = os.getenv(env_key)
        IF value IS None:
            RETURN None  # Not set — defer to percent routing
        IF value.lower() IN ("0", "false", "no"):
            RETURN False  # Explicit OFF — overrides percent
        IF value.lower() IN ("1", "true", "yes"):
            RETURN True   # Explicit ON — overrides percent
        RETURN None

    FUNCTION get_active_percents(self) -> Dict[str, int]:
        """Return current canary percentages from env."""
        RETURN {
            "search": int(os.getenv("BIZRA_PHASE46_SEARCH_PERCENT", "0")),
            "got_bridge": int(os.getenv("BIZRA_PHASE46_GOT_BRIDGE_PERCENT", "0")),
            "hmm": int(os.getenv("BIZRA_PHASE46_HMM_PERCENT", "0")),
        }
```

## Kill Switch Precedence Matrix

| Boolean Flag | Percent | Result | Reason |
|-------------|---------|--------|--------|
| `"0"` (OFF) | any | OFF | Kill switch overrides |
| `"1"` (ON) | any | ON | Kill switch overrides |
| not set | `0` | OFF | Percent gate |
| not set | `50` | hash-based | Canary routing |
| not set | `100` | ON | Full rollout |

## TDD Anchors

```python
class TestCanaryRouter:
    """Tests for core/rollout/canary.py"""

    def test_zero_percent_never_routes(self):
        """0% should never route any request."""
        router = CanaryRouter(salt="test-salt")
        for i in range(100):
            assert router.should_route("search", f"req-{i}", 0) is False

    def test_hundred_percent_always_routes(self):
        """100% should always route every request."""
        router = CanaryRouter(salt="test-salt")
        for i in range(100):
            assert router.should_route("search", f"req-{i}", 100) is True

    def test_deterministic_routing(self):
        """Same key + same salt + same percent = same decision."""
        router = CanaryRouter(salt="fixed-salt")
        result1 = router.should_route("search", "stable-key", 50)
        result2 = router.should_route("search", "stable-key", 50)
        assert result1 == result2

    def test_different_salts_different_routing(self):
        """Different salts should produce different routing patterns."""
        r1 = CanaryRouter(salt="salt-a")
        r2 = CanaryRouter(salt="salt-b")
        # Over 100 keys, at least some should differ
        differences = sum(
            1 for i in range(100)
            if r1.should_route("search", f"k-{i}", 50)
               != r2.should_route("search", f"k-{i}", 50)
        )
        assert differences > 10  # Statistically near 50%

    def test_kill_switch_off_overrides_percent(self):
        """ENABLED=0 should block even at 100%."""
        with patch.dict(os.environ, {"BIZRA_PHASE46_SEARCH_ENABLED": "0"}):
            router = CanaryRouter()
            assert router.should_route("search", "any-key", 100) is False

    def test_kill_switch_on_overrides_percent(self):
        """ENABLED=1 should allow even at 0%."""
        with patch.dict(os.environ, {"BIZRA_PHASE46_SEARCH_ENABLED": "1"}):
            router = CanaryRouter()
            assert router.should_route("search", "any-key", 0) is True

    def test_fifty_percent_routes_roughly_half(self):
        """50% should route approximately 50% of keys (within 15% tolerance)."""
        router = CanaryRouter(salt="statistical-test")
        routed = sum(
            1 for i in range(1000)
            if router.should_route("search", f"k-{i}", 50)
        )
        assert 350 < routed < 650  # 35%-65% tolerance

    def test_independent_components(self):
        """Different components can have different routing decisions."""
        router = CanaryRouter(salt="component-test")
        # search at 100%, got_bridge at 0%
        assert router.should_route("search", "key", 100) is True
        assert router.should_route("got_bridge", "key", 0) is False
```

## Integration Points

### Where canary routing hooks in:

1. **`core/sovereign/apex_engine.py:_explore_thoughts()`** — currently checks boolean flag; wrap with canary percent check
2. **`core/living_memory/proactive.py:_init_hmm()`** — currently checks boolean flag; add percent gate
3. **`core/resonance.py:PHASE46_ENABLED`** — module-level boolean; wrap with canary check on process()
4. **`tools/mcp/sovereign_mcp_server.py:Phase46Interface`** — add canary_router to lazy init; route per-tool

### Pattern for existing flag sites:

```python
# BEFORE (Phase 46):
if os.getenv("BIZRA_PHASE46_GOT_BRIDGE_ENABLED", "0").lower() in {"1", "true", "yes"}:
    # use GoT bridge

# AFTER (Phase 47.1):
from core.rollout.canary import CanaryRouter
_canary = CanaryRouter()

if _canary.should_route("got_bridge", request_key,
                         int(os.getenv("BIZRA_PHASE46_GOT_BRIDGE_PERCENT", "0"))):
    # use GoT bridge
```
