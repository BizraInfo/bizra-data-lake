# Phase 79: Oracle-S Gate (Layer 2) — Pseudocode

---

## Module: `core/sat/oracle_s_gate.py`

```pseudocode
IMPORT GateResult, CheckResult, CheckStatus FROM core.sat.gate_result
IMPORT (
    UNIFIED_IHSAN_THRESHOLD,      # 0.95
    UNIFIED_SNR_THRESHOLD,         # 0.85
    ADL_GINI_THRESHOLD,            # 0.35
    ZAKAT_RATE,                    # 0.025
    ADL_HARBERGER_TAX_RATE,        # 0.05 (after FIX-1)
    BLOOM_REDISTRIBUTION_RATE,     # 0.50
) FROM core.integration.constants


FUNCTION oracle_s_verify(skip_manual: bool = False) -> GateResult:
    """Layer 2: Constitutional Compliance — 10 automated + 4 manual checks."""
    checks = []

    # === CONSTITUTIONAL INVARIANTS ===

    # 3.1 Ihsan production gate
    ihsan = UNIFIED_IHSAN_THRESHOLD
    checks.APPEND(CheckResult(
        "ihsan_gate",
        PASS IF ihsan >= 0.95 ELSE FAIL,
        evidence=f"UNIFIED_IHSAN_THRESHOLD = {ihsan}"
    ))

    # 3.2 SNR minimum
    snr = UNIFIED_SNR_THRESHOLD
    checks.APPEND(CheckResult(
        "snr_minimum",
        PASS IF snr >= 0.85 ELSE FAIL,
        evidence=f"UNIFIED_SNR_THRESHOLD = {snr}"
    ))

    # 3.3 Gini ceiling
    gini = ADL_GINI_THRESHOLD
    checks.APPEND(CheckResult(
        "gini_ceiling",
        PASS IF gini <= 0.35 ELSE FAIL,
        evidence=f"ADL_GINI_THRESHOLD = {gini}"
    ))

    # 3.4 Community pool split
    pool = BLOOM_REDISTRIBUTION_RATE
    checks.APPEND(CheckResult(
        "pool_split",
        PASS IF pool == 0.50 ELSE FAIL,
        evidence=f"BLOOM_REDISTRIBUTION_RATE = {pool}"
    ))

    # 3.5 Zakat rate
    zakat = ZAKAT_RATE
    checks.APPEND(CheckResult(
        "zakat_rate",
        PASS IF zakat == 0.025 ELSE FAIL,
        evidence=f"ZAKAT_RATE = {zakat}"
    ))

    # 3.6 Harberger tax
    harberger = ADL_HARBERGER_TAX_RATE
    checks.APPEND(CheckResult(
        "harberger_rate",
        PASS IF harberger == 0.05 ELSE FAIL,
        evidence=f"ADL_HARBERGER_TAX_RATE = {harberger}"
    ))

    # 3.7 Heartbeat alive
    # Verify the heartbeat tick mechanism exists and fires
    code, out = _run(["pytest", "tests/", "-q", "--timeout=30",
                       "-k", "heartbeat or constitutional_tick or process_tick"])
    checks.APPEND(CheckResult(
        "heartbeat_alive",
        PASS IF code == 0 ELSE FAIL,
        evidence=last_line(out)
    ))

    # 3.8 Constitutional test suite (target: 281 tests)
    code, out = _run(["pytest", "tests/constitutional/", "-q", "--timeout=120"], timeout=300)
    test_count = parse_test_count(out)
    checks.APPEND(CheckResult(
        "constitutional_tests",
        PASS IF code == 0 ELSE FAIL,
        evidence=f"{test_count} constitutional tests, exit={code}"
    ))

    # 3.9 Metabolism E2E
    code, out = _run(["pytest", "tests/integration/test_metabolism_e2e.py",
                       "-q", "--timeout=60"])
    checks.APPEND(CheckResult(
        "metabolism_e2e",
        PASS IF code == 0 ELSE FAIL,
        evidence=last_line(out)
    ))

    # 3.10 Threshold sync (cross-language constants)
    code, out = _run(["python", "-c",
        "from core.integration.constants import validate_cross_repo_consistency; "
        "result = validate_cross_repo_consistency(); "
        "assert result['all_synced'], f'Drift: {result}'"
    ])
    checks.APPEND(CheckResult(
        "threshold_sync",
        PASS IF code == 0 ELSE FAIL,
        evidence="Cross-repo constants synced" IF code == 0 ELSE out[-200:]
    ))

    # 3.11 548-day simulation
    code, out = _run(["pytest", "tests/constitutional/test_simulation.py",
                       "-q", "--timeout=120"])
    checks.APPEND(CheckResult(
        "simulation_valid",
        PASS IF code == 0 ELSE FAIL,
        evidence=last_line(out)
    ))

    # === MANUAL CHECKS (human attestation) ===

    IF skip_manual:
        FOR name IN ["mother_test", "daughter_test", "rtl_layout", "first_run_experience"]:
            checks.APPEND(CheckResult(name, SKIPPED, "Manual check skipped", is_manual=True))
    ELSE:
        # 3.12 Mother Test
        checks.APPEND(CheckResult(
            "mother_test",
            PASS IF prompt_human("Has your mother navigated the terminal successfully?") ELSE FAIL,
            is_manual=True
        ))

        # 3.13 Daughter Test
        checks.APPEND(CheckResult(
            "daughter_test",
            PASS IF prompt_human("Would you deploy this for DEMA to use?") ELSE FAIL,
            is_manual=True
        ))

        # 3.14 RTL layout
        checks.APPEND(CheckResult(
            "rtl_layout",
            PASS IF prompt_human("Do all 7 views mirror properly in Arabic RTL?") ELSE FAIL,
            is_manual=True
        ))

        # 3.15 First-run experience
        checks.APPEND(CheckResult(
            "first_run_experience",
            PASS IF prompt_human("Can a new user complete first mission in < 3 minutes?") ELSE FAIL,
            is_manual=True
        ))

    RETURN GateResult(agent="Oracle-S", layer="CONSTITUTIONAL_COMPLIANCE", checks=checks)


FUNCTION prompt_human(question: str) -> bool:
    """Prompt for human attestation. Returns False in non-interactive mode."""
    TRY:
        IF NOT sys.stdin.isatty():
            RETURN False
        response = input(f"\n  [HUMAN ATTESTATION] {question} (y/n): ").strip().lower()
        RETURN response IN ("y", "yes")
    EXCEPT (EOFError, KeyboardInterrupt):
        RETURN False
```

---

## TDD Anchors

```pseudocode
TEST test_oracle_s_constants_verified:
    result = oracle_s_verify(skip_manual=True)
    ihsan_check = find_check(result, "ihsan_gate")
    ASSERT ihsan_check.status == PASS
    ASSERT "0.95" IN ihsan_check.evidence

TEST test_oracle_s_manual_skipped:
    result = oracle_s_verify(skip_manual=True)
    manual_checks = [c FOR c IN result.checks IF c.is_manual]
    ASSERT all(c.status == SKIPPED FOR c IN manual_checks)
    # Manual SKIPPED should not block the gate
    # (only in skip_manual mode — ceremony requires them)

TEST test_oracle_s_harberger_drift_detected:
    # If constants.py still has 0.07, this MUST fail
    result = oracle_s_verify(skip_manual=True)
    harberger = find_check(result, "harberger_rate")
    # Will be FAIL until FIX-1 lands
    ASSERT harberger.evidence CONTAINS str(ADL_HARBERGER_TAX_RATE)

TEST test_oracle_s_all_automated_pass:
    # With mocked subprocesses returning 0
    result = oracle_s_verify(skip_manual=True)
    automated = [c FOR c IN result.checks IF NOT c.is_manual]
    ASSERT all(c.status == PASS FOR c IN automated)
```
