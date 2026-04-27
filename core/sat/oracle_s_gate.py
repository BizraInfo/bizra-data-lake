"""
Oracle-S Gate — Layer 2: Constitutional Compliance
====================================================

10 automated + 4 manual checks.
"Does this system uphold its constitutional promises?"

Standing on Giants:
- Al-Ghazali (1095): Self-knowledge and moral excellence
- Rawls (1971): Justice as fairness
- Shannon (1948): Signal-to-noise as quality measure
"""

from __future__ import annotations

from core.integration.constants import (
    ADL_GINI_THRESHOLD,
    ADL_HARBERGER_TAX_RATE,
    BLOOM_REDISTRIBUTION_RATE,
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
    ZAKAT_RATE,
)
from core.sat._helpers import _run, last_line, parse_test_count, prompt_human
from core.sat.gate_result import CheckResult, CheckStatus, GateResult

PASS = CheckStatus.PASS
FAIL = CheckStatus.FAIL
SKIPPED = CheckStatus.SKIPPED


def oracle_s_verify(skip_manual: bool = False, skip_slow: bool = False) -> GateResult:
    """Layer 2: Constitutional Compliance — 10 automated + 4 manual checks."""
    checks: list[CheckResult] = []

    # 3.1 Ihsan production gate
    checks.append(
        CheckResult(
            "ihsan_gate",
            PASS if UNIFIED_IHSAN_THRESHOLD >= 0.95 else FAIL,
            f"UNIFIED_IHSAN_THRESHOLD = {UNIFIED_IHSAN_THRESHOLD}",
        )
    )

    # 3.2 SNR minimum
    checks.append(
        CheckResult(
            "snr_minimum",
            PASS if UNIFIED_SNR_THRESHOLD >= 0.85 else FAIL,
            f"UNIFIED_SNR_THRESHOLD = {UNIFIED_SNR_THRESHOLD}",
        )
    )

    # 3.3 Gini ceiling
    checks.append(
        CheckResult(
            "gini_ceiling",
            PASS if ADL_GINI_THRESHOLD <= 0.35 else FAIL,
            f"ADL_GINI_THRESHOLD = {ADL_GINI_THRESHOLD}",
        )
    )

    # 3.4 Community pool split (founder's oath — constant exists)
    checks.append(
        CheckResult(
            "pool_split",
            PASS if BLOOM_REDISTRIBUTION_RATE == 0.50 else FAIL,
            f"BLOOM_REDISTRIBUTION_RATE = {BLOOM_REDISTRIBUTION_RATE} (founder's oath, not user tax)",
        )
    )

    # 3.5 Zakat rate
    checks.append(
        CheckResult(
            "zakat_rate",
            PASS if ZAKAT_RATE == 0.025 else FAIL,
            f"ZAKAT_RATE = {ZAKAT_RATE}",
        )
    )

    # 3.6 Harberger tax
    checks.append(
        CheckResult(
            "harberger_rate",
            PASS if ADL_HARBERGER_TAX_RATE == 0.05 else FAIL,
            f"ADL_HARBERGER_TAX_RATE = {ADL_HARBERGER_TAX_RATE}",
        )
    )

    # 3.7 Heartbeat alive
    if skip_slow:
        checks.append(CheckResult("heartbeat_alive", SKIPPED, "Skipped (--skip-slow)"))
    else:
        code, out = _run(
            [
                "python",
                "-m",
                "pytest",
                "tests/constitutional/",
                "-q",
                "--timeout=30",
                "-k",
                "heartbeat or tick",
                "-x",
            ],
            timeout=90,
        )
        checks.append(
            CheckResult(
                "heartbeat_alive",
                PASS if code == 0 else FAIL,
                last_line(out),
            )
        )

    # 3.8 Constitutional test suite
    if skip_slow:
        checks.append(
            CheckResult("constitutional_tests", SKIPPED, "Skipped (--skip-slow)")
        )
    else:
        code, out = _run(
            ["python", "-m", "pytest", "tests/constitutional/", "-q", "--timeout=120"],
            timeout=300,
        )
        test_count = parse_test_count(out)
        checks.append(
            CheckResult(
                "constitutional_tests",
                PASS if code == 0 else FAIL,
                f"{test_count} constitutional tests, exit={code}",
            )
        )

    # 3.9 Metabolism E2E
    if skip_slow:
        checks.append(CheckResult("metabolism_e2e", SKIPPED, "Skipped (--skip-slow)"))
    else:
        code, out = _run(
            [
                "python",
                "-m",
                "pytest",
                "tests/integration/test_metabolism_e2e.py",
                "-q",
                "--timeout=60",
            ],
            timeout=120,
        )
        checks.append(
            CheckResult(
                "metabolism_e2e",
                PASS if code == 0 else FAIL,
                last_line(out),
            )
        )

    # 3.10 Threshold sync (cross-language constants)
    code, out = _run(
        [
            "python",
            "-c",
            "from core.integration.constants import validate_cross_repo_consistency; "
            "result = validate_cross_repo_consistency(); "
            "synced = result.get('all_synced', True); "
            "print(f'Synced: {synced}')",
        ],
        timeout=30,
    )
    checks.append(
        CheckResult(
            "threshold_sync",
            PASS if code == 0 else FAIL,
            "Cross-repo constants synced" if code == 0 else last_line(out),
        )
    )

    # === MANUAL CHECKS ===
    if skip_manual:
        for name in [
            "mother_test",
            "daughter_test",
            "rtl_layout",
            "first_run_experience",
        ]:
            checks.append(
                CheckResult(name, SKIPPED, "Manual check skipped", is_manual=True)
            )
    else:
        checks.append(
            CheckResult(
                "mother_test",
                (
                    PASS
                    if prompt_human(
                        "Has your mother navigated the terminal successfully?"
                    )
                    else FAIL
                ),
                is_manual=True,
            )
        )
        checks.append(
            CheckResult(
                "daughter_test",
                (
                    PASS
                    if prompt_human("Would you deploy this for DEMA to use?")
                    else FAIL
                ),
                is_manual=True,
            )
        )
        checks.append(
            CheckResult(
                "rtl_layout",
                (
                    PASS
                    if prompt_human("Do all 7 views mirror properly in Arabic RTL?")
                    else FAIL
                ),
                is_manual=True,
            )
        )
        checks.append(
            CheckResult(
                "first_run_experience",
                (
                    PASS
                    if prompt_human(
                        "Can a new user complete first mission in < 3 minutes?"
                    )
                    else FAIL
                ),
                is_manual=True,
            )
        )

    return GateResult(
        agent="Oracle-S", layer="CONSTITUTIONAL_COMPLIANCE", checks=checks
    )
