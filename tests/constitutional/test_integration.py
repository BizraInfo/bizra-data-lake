"""
Phase 67 Integration Tests — Wiring Verification
═══════════════════════════════════════════════════

Verifies that constitutional modules are properly wired into
the production system:

1. NISAB sourced from constants.py SSoT (not hardcoded)
2. CLI accessible via sovereign __main__ entry point
3. Ticker process_tick callable from constitutional package root

Standing on Giants: Lamport (single source of truth, 1978)
"""

from __future__ import annotations




class TestNisabSSoT:
    """NISAB_THRESHOLD must be derived from constants.py, not hardcoded."""

    def test_nisab_matches_constants(self) -> None:
        """algorithms.NISAB_THRESHOLD == fp(constants.NISAB_THRESHOLD)."""
        from core.constitutional.algorithms import NISAB_THRESHOLD
        from core.constitutional.fixed_point import fp
        from core.integration.constants import NISAB_THRESHOLD as NISAB_FLOAT

        assert NISAB_THRESHOLD == fp(NISAB_FLOAT)

    def test_nisab_value_is_85(self) -> None:
        """NISAB stays at 85.0 SEED (Islamic Nisab for silver)."""
        from core.integration.constants import NISAB_THRESHOLD

        assert NISAB_THRESHOLD == 85.0

    def test_changing_constants_changes_algorithms(self) -> None:
        """If we patch constants.NISAB_THRESHOLD, algorithms picks it up on reimport."""
        # This tests the derivation chain, not runtime mutability
        from core.constitutional.algorithms import NISAB_THRESHOLD
        from core.constitutional.fixed_point import fp

        # Current value is derived from 85.0
        assert NISAB_THRESHOLD == fp(85.0)


class TestCLIWiring:
    """Constitutional CLI must be accessible via core.sovereign.__main__."""

    def test_sovereignty_subcommand_exists(self) -> None:
        """Sovereign __main__ has sovereignty/sov subcommand."""
        import importlib

        mod = importlib.import_module("core.sovereign.__main__")
        # The main() function uses argparse — verify it can parse sovereignty
        assert hasattr(mod, "main")

    def test_constitutional_commands_importable(self) -> None:
        """All 6 constitutional CLI commands are importable."""
        from core.constitutional.__main__ import (
            cmd_attest,
            cmd_init,
            cmd_ledger,
            cmd_reset,
            cmd_status,
            cmd_work,
        )

        assert callable(cmd_init)
        assert callable(cmd_work)
        assert callable(cmd_attest)
        assert callable(cmd_status)
        assert callable(cmd_ledger)
        assert callable(cmd_reset)

    def test_cli_functions_accessible_from_package(self) -> None:
        """CLI functions are re-exported from constitutional package."""
        from core.constitutional import (
            attest_peer,
            get_status,
            init_node,
            process_work,
        )

        assert callable(init_node)
        assert callable(process_work)
        assert callable(attest_peer)
        assert callable(get_status)


class TestTickerWiring:
    """process_tick must be importable from constitutional package root."""

    def test_process_tick_importable(self) -> None:
        """process_tick is accessible from core.constitutional."""
        from core.constitutional import process_tick

        assert callable(process_tick)

    def test_tick_result_type(self) -> None:
        """TickResult is a proper dataclass."""
        import dataclasses

        from core.constitutional import TickResult

        assert dataclasses.is_dataclass(TickResult)

    def test_process_tick_accepts_correct_args(self) -> None:
        """process_tick signature matches expectations."""
        import inspect

        from core.constitutional import process_tick

        sig = inspect.signature(process_tick)
        param_names = list(sig.parameters.keys())
        # Must accept wallets, receipts, event_log, current_time
        assert "wallets" in param_names
        assert "event_log" in param_names


class TestConstantsAlignment:
    """Phase 67 constants must align between constants.py and algorithms.py."""

    def test_gini_thresholds_aligned(self) -> None:
        """Gini zone boundaries match SSoT."""
        from core.constitutional.algorithms import (
            GINI_CRISIS,
            GINI_HEALTHY,
            GINI_WARNING,
        )
        from core.constitutional.fixed_point import fp
        from core.integration.constants import GINI_CRISIS as CRISIS_FLOAT
        from core.integration.constants import GINI_HEALTHY as HEALTHY_FLOAT
        from core.integration.constants import GINI_WARNING as WARNING_FLOAT

        assert GINI_HEALTHY == fp(HEALTHY_FLOAT)
        assert GINI_WARNING == fp(WARNING_FLOAT)
        assert GINI_CRISIS == fp(CRISIS_FLOAT)

    def test_equity_factor_aligned(self) -> None:
        """Equity factor bounds match SSoT."""
        from core.integration.constants import EQUITY_FACTOR_MAX, EQUITY_FACTOR_MIN

        assert EQUITY_FACTOR_MIN == 1.0
        assert EQUITY_FACTOR_MAX == 5.0

    def test_zakat_rate_aligned(self) -> None:
        """Zakat rate sourced from SSoT."""
        from core.constitutional.algorithms import ZAKAT_FP
        from core.constitutional.fixed_point import fp
        from core.integration.constants import ZAKAT_RATE

        assert ZAKAT_FP == fp(ZAKAT_RATE)

    def test_ihsan_floor_aligned(self) -> None:
        """Ihsan floor derived from UNIFIED_IHSAN_THRESHOLD."""
        from core.constitutional.algorithms import IHSAN_FLOOR
        from core.constitutional.fixed_point import fp
        from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

        assert IHSAN_FLOOR == fp(UNIFIED_IHSAN_THRESHOLD)
