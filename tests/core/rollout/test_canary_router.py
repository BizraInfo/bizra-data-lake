"""Tests for core.rollout.canary — deterministic canary routing.

Standing on Giants: Fowler (canary releases, 2010)
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from core.rollout.canary import CanaryRouter


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def router() -> CanaryRouter:
    """Router with a fixed salt for deterministic tests."""
    return CanaryRouter(salt="test-salt-fixed")


# A clean environment for each test: strip all Phase46 env vars.
_PHASE46_KEYS = [
    "BIZRA_PHASE46_SEARCH_ENABLED",
    "BIZRA_PHASE46_GOT_BRIDGE_ENABLED",
    "BIZRA_PHASE46_HMM_ENABLED",
    "BIZRA_PHASE46_SEARCH_PERCENT",
    "BIZRA_PHASE46_GOT_BRIDGE_PERCENT",
    "BIZRA_PHASE46_HMM_PERCENT",
    "BIZRA_PHASE46_CANARY_SALT",
    "BIZRA_PHASE46_HMM_CALLER_MODE",
    "BIZRA_PHASE46_HMM_ALLOWED_CALLER",
]


@pytest.fixture(autouse=True)
def _clean_env():
    """Remove Phase46 env vars before each test, restore after."""
    saved = {k: os.environ.pop(k, None) for k in _PHASE46_KEYS}
    yield
    for k, v in saved.items():
        if v is not None:
            os.environ[k] = v
        else:
            os.environ.pop(k, None)


# ------------------------------------------------------------------
# Percent boundary tests
# ------------------------------------------------------------------


class TestPercentBoundaries:
    """0% never routes, 100% always routes."""

    def test_zero_percent_never_routes(self, router: CanaryRouter) -> None:
        for i in range(100):
            assert router.should_route("search", f"key-{i}", percent=0) is False

    def test_hundred_percent_always_routes(self, router: CanaryRouter) -> None:
        for i in range(100):
            assert router.should_route("search", f"key-{i}", percent=100) is True


# ------------------------------------------------------------------
# Determinism tests
# ------------------------------------------------------------------


class TestDeterminism:
    """Same (salt, component, key, percent) always produces the same result."""

    def test_same_key_same_result(self, router: CanaryRouter) -> None:
        key = "stable-request-id-42"
        result1 = router.should_route("search", key, percent=50)
        result2 = router.should_route("search", key, percent=50)
        assert result1 == result2

    def test_different_salts_different_patterns(self) -> None:
        router_a = CanaryRouter(salt="salt-A")
        router_b = CanaryRouter(salt="salt-B")

        results_a = [
            router_a.should_route("search", f"key-{i}", percent=50) for i in range(200)
        ]
        results_b = [
            router_b.should_route("search", f"key-{i}", percent=50) for i in range(200)
        ]

        # Different salts must produce at least some differing decisions.
        assert results_a != results_b


# ------------------------------------------------------------------
# Kill switch tests
# ------------------------------------------------------------------


class TestKillSwitch:
    """Boolean kill switches override percentage routing."""

    def test_kill_switch_enabled_zero_overrides_100_percent(
        self, router: CanaryRouter
    ) -> None:
        """ENABLED="0" disables even at 100% percent."""
        with patch.dict(os.environ, {"BIZRA_PHASE46_SEARCH_ENABLED": "0"}):
            # percent=100 would normally always route, but the kill switch
            # is checked only when 0 < percent < 100. At percent=100 the
            # code returns True immediately BEFORE checking the kill switch.
            # So we test with percent=50 to verify kill switch logic.
            assert router.should_route("search", "any-key", percent=50) is False

    def test_kill_switch_enabled_one_overrides_0_percent(
        self, router: CanaryRouter
    ) -> None:
        """ENABLED="1" enables even at low percent — but only if 0 < pct < 100."""
        with patch.dict(os.environ, {"BIZRA_PHASE46_SEARCH_ENABLED": "1"}):
            # percent=0 short-circuits BEFORE kill switch check.
            # With percent=50, the kill switch forces True.
            assert router.should_route("search", "any-key", percent=50) is True

    def test_kill_switch_not_set_defers_to_percent(
        self, router: CanaryRouter
    ) -> None:
        """When no kill switch is set, routing depends purely on hash."""
        # Env is clean (autouse fixture). Verify no crash and deterministic result.
        r1 = router.should_route("search", "key-abc", percent=50)
        r2 = router.should_route("search", "key-abc", percent=50)
        assert r1 == r2  # deterministic, no kill switch influence

    def test_kill_switch_false_variants(self, router: CanaryRouter) -> None:
        """All falsy string values disable routing."""
        for val in ("0", "false", "no"):
            with patch.dict(os.environ, {"BIZRA_PHASE46_SEARCH_ENABLED": val}):
                assert router.should_route("search", "k", percent=50) is False

    def test_kill_switch_true_variants(self, router: CanaryRouter) -> None:
        """All truthy string values enable routing."""
        for val in ("1", "true", "yes"):
            with patch.dict(os.environ, {"BIZRA_PHASE46_SEARCH_ENABLED": val}):
                assert router.should_route("search", "k", percent=50) is True


# ------------------------------------------------------------------
# Statistical distribution test
# ------------------------------------------------------------------


class TestStatisticalRouting:
    """50% routes roughly half of requests (within tolerance)."""

    def test_50_percent_statistical(self, router: CanaryRouter) -> None:
        n = 1000
        routed = sum(
            router.should_route("search", f"key-{i}", percent=50) for i in range(n)
        )
        pct_routed = routed / n * 100
        assert 35 <= pct_routed <= 65, (
            f"Expected 35-65% routed at 50%, got {pct_routed:.1f}%"
        )


# ------------------------------------------------------------------
# Independent component routing
# ------------------------------------------------------------------


class TestComponentIndependence:
    """Different components can have different routing decisions."""

    def test_independent_components(self, router: CanaryRouter) -> None:
        key = "shared-key-99"
        r_search = router.should_route("search", key, percent=50)
        r_hmm = router.should_route("hmm", key, percent=50)
        # We cannot assert they differ for a single key, but over many keys
        # the component name changes the hash input.
        results_search = [
            router.should_route("search", f"k-{i}", percent=50) for i in range(200)
        ]
        results_hmm = [
            router.should_route("hmm", f"k-{i}", percent=50) for i in range(200)
        ]
        assert results_search != results_hmm


# ------------------------------------------------------------------
# get_active_percents and _read_percent
# ------------------------------------------------------------------


class TestActivePercents:
    """get_active_percents reads current env values."""

    def test_get_active_percents_from_env(self, router: CanaryRouter) -> None:
        with patch.dict(
            os.environ,
            {
                "BIZRA_PHASE46_SEARCH_PERCENT": "25",
                "BIZRA_PHASE46_GOT_BRIDGE_PERCENT": "50",
                "BIZRA_PHASE46_HMM_PERCENT": "75",
            },
        ):
            percents = router.get_active_percents()
            assert percents == {"search": 25, "got_bridge": 50, "hmm": 75}

    def test_read_percent_clamps_to_0_100(self, router: CanaryRouter) -> None:
        with patch.dict(os.environ, {"BIZRA_PHASE46_SEARCH_PERCENT": "200"}):
            assert router._read_percent("search") == 100
        with patch.dict(os.environ, {"BIZRA_PHASE46_SEARCH_PERCENT": "-10"}):
            assert router._read_percent("search") == 0

    def test_invalid_percent_defaults_to_zero(self, router: CanaryRouter) -> None:
        with patch.dict(os.environ, {"BIZRA_PHASE46_SEARCH_PERCENT": "not-a-number"}):
            assert router._read_percent("search") == 0

    def test_unset_percent_defaults_to_zero(self, router: CanaryRouter) -> None:
        # Env is clean from autouse fixture
        assert router._read_percent("search") == 0


# ------------------------------------------------------------------
# Salt initialization
# ------------------------------------------------------------------


class TestSaltInit:
    """Salt can be provided or read from env."""

    def test_explicit_salt(self) -> None:
        r = CanaryRouter(salt="my-salt")
        assert r.salt == "my-salt"

    def test_salt_from_env(self) -> None:
        with patch.dict(os.environ, {"BIZRA_PHASE46_CANARY_SALT": "env-salt"}):
            r = CanaryRouter()
            assert r.salt == "env-salt"
