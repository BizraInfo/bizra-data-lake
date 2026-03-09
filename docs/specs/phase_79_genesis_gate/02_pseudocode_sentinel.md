# Phase 79: Sentinel Gate (Layer 1) — Pseudocode

---

## Module: `core/sat/sentinel_gate.py`

```pseudocode
IMPORT subprocess, shutil
IMPORT GateResult, CheckResult, CheckStatus FROM core.sat.gate_result
IMPORT UNIFIED_IHSAN_THRESHOLD, COVERAGE_FLOOR FROM core.integration.constants

# Helper: run shell command, return (exit_code, stdout)
FUNCTION _run(cmd: List[str], timeout: int = 300) -> Tuple[int, str]:
    TRY:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        RETURN (result.returncode, result.stdout + result.stderr)
    EXCEPT TimeoutExpired:
        RETURN (1, "TIMEOUT")
    EXCEPT FileNotFoundError:
        RETURN (-1, "TOOL_NOT_FOUND")

# Helper: check if tool exists
FUNCTION _has_tool(name: str) -> bool:
    RETURN shutil.which(name) IS NOT None


FUNCTION sentinel_verify(skip_slow: bool = False) -> GateResult:
    """Layer 1: Structural Integrity — 12 checks."""
    checks = []

    # === CODE QUALITY GATES ===

    # 1.1 All tests pass
    IF skip_slow:
        code, out = _run(["pytest", "tests/", "-x", "-q", "--timeout=60",
                          "-m", "not slow and not requires_ollama and not requires_gpu"])
    ELSE:
        code, out = _run(["pytest", "tests/", "-x", "-q", "--timeout=60"], timeout=600)
    checks.APPEND(CheckResult(
        "tests_pass", PASS IF code == 0 ELSE FAIL,
        evidence=last_line(out)
    ))

    # 1.2 Zero CRITICAL security findings
    code, out = _run(["bandit", "-r", "core/", "-ll", "-q"])
    IF code == -1:
        checks.APPEND(CheckResult("zero_criticals", NOT_IMPLEMENTED, "bandit not installed"))
    ELSE:
        critical_count = parse_bandit_criticals(out)
        checks.APPEND(CheckResult(
            "zero_criticals", PASS IF critical_count == 0 ELSE FAIL,
            evidence=f"{critical_count} critical findings"
        ))

    # 1.3 Type safety
    code, out = _run(["mypy", "core/", "--ignore-missing-imports"])
    error_count = parse_mypy_errors(out)
    checks.APPEND(CheckResult(
        "type_safe", PASS IF error_count == 0 ELSE PARTIAL,
        evidence=f"{error_count} type errors"
    ))

    # 1.4 Lint clean
    code, out = _run(["ruff", "check", "core/"])
    checks.APPEND(CheckResult(
        "lint_clean", PASS IF code == 0 ELSE FAIL,
        evidence=last_line(out)
    ))

    # 1.5 Coverage floor
    code, out = _run(["pytest", "tests/", "--cov=core", "--cov-report=term-missing",
                       "-q", "--timeout=60", "-m", "not slow"], timeout=600)
    coverage_pct = parse_coverage(out)
    checks.APPEND(CheckResult(
        "coverage_floor", PASS IF coverage_pct >= COVERAGE_FLOOR ELSE FAIL,
        evidence=f"{coverage_pct}% (floor: {COVERAGE_FLOOR}%)"
    ))

    # 1.6 CI pipeline green
    code, out = _run(["gh", "run", "list", "--limit=1", "--json=conclusion"])
    IF code == -1:
        checks.APPEND(CheckResult("ci_green", NOT_IMPLEMENTED, "gh CLI not available"))
    ELSE:
        conclusion = parse_gh_conclusion(out)
        checks.APPEND(CheckResult(
            "ci_green", PASS IF conclusion == "success" ELSE FAIL,
            evidence=f"Last CI: {conclusion}"
        ))

    # === SECURITY GATES ===

    # 2.1 Auth fail-closed
    code, out = _run(["pytest", "tests/core/sovereign/test_endpoint_responses.py",
                       "-q", "--timeout=30"])
    checks.APPEND(CheckResult(
        "auth_closed", PASS IF code == 0 ELSE FAIL,
        evidence=last_line(out)
    ))

    # 2.2 Production auth guard
    # Check that _anonymous_auth_allowed() returns False when BIZRA_ENV=production
    code, out = _run(["python", "-c",
        "import os; os.environ['BIZRA_ENV']='production'; "
        "from core.sovereign.api import _anonymous_auth_allowed; "
        "assert not _anonymous_auth_allowed(), 'FAIL: anon auth allowed in prod'"
    ])
    checks.APPEND(CheckResult(
        "prod_guard", PASS IF code == 0 ELSE FAIL,
        evidence="anon auth blocked in production" IF code == 0 ELSE out[-200:]
    ))

    # 2.3 Atomic writes (static analysis — grep for bare open().write patterns)
    code, out = _run(["grep", "-rn", r"\.write(", "core/", "--include=*.py"])
    # Filter: count writes NOT using tempfile pattern
    unsafe_writes = count_unsafe_writes(out)
    checks.APPEND(CheckResult(
        "atomic_writes", PASS IF unsafe_writes == 0 ELSE PARTIAL,
        evidence=f"{unsafe_writes} potentially non-atomic writes"
    ))

    # 2.4 No hardcoded secrets
    code, out = _run(["git", "grep", "-i", "-n",
                       r"password\|api_key\|secret_key\|private_key",
                       "--", "core/"])
    # Exclude test fixtures and known safe patterns
    real_secrets = filter_false_positives(out)
    checks.APPEND(CheckResult(
        "no_secrets", PASS IF len(real_secrets) == 0 ELSE FAIL,
        evidence=f"{len(real_secrets)} potential secrets found"
    ))

    # 2.5 Ed25519 identity
    code, out = _run(["pytest", "tests/core/sovereign/test_contract_integrity.py",
                       "-q", "--timeout=30", "-k", "ed25519 or identity or genesis"])
    checks.APPEND(CheckResult(
        "identity_works", PASS IF code == 0 ELSE FAIL,
        evidence=last_line(out)
    ))

    # 2.6 Evidence chain integrity
    code, out = _run(["pytest", "tests/", "-q", "--timeout=30",
                       "-k", "evidence_chain or verify_chain or hash_chain"])
    checks.APPEND(CheckResult(
        "chain_valid", PASS IF code == 0 ELSE FAIL,
        evidence=last_line(out)
    ))

    # 2.7 Container signing
    IF _has_tool("cosign"):
        code, out = _run(["cosign", "verify", "--key=cosign.pub",
                           "ghcr.io/bizrainfo/bizra-elite:latest"])
        checks.APPEND(CheckResult("container_signed", PASS IF code == 0 ELSE FAIL))
    ELSE:
        checks.APPEND(CheckResult("container_signed", NOT_IMPLEMENTED, "cosign not installed"))

    # 2.8 SBOM generated
    IF _has_tool("syft"):
        code, out = _run(["syft", "dir:.", "-o", "json"])
        checks.APPEND(CheckResult("sbom_generated", PASS IF code == 0 ELSE FAIL))
    ELSE:
        checks.APPEND(CheckResult("sbom_generated", NOT_IMPLEMENTED, "syft not installed"))

    RETURN GateResult(agent="Sentinel", layer="STRUCTURAL_INTEGRITY", checks=checks)
```

---

## TDD Anchors

```pseudocode
TEST test_sentinel_with_passing_suite:
    # Mock all subprocess calls to return success
    result = sentinel_verify(skip_slow=True)
    ASSERT result.agent == "Sentinel"
    ASSERT len(result.checks) == 12

TEST test_sentinel_fails_on_test_failure:
    # Mock pytest to return exit code 1
    result = sentinel_verify()
    ASSERT result.checks[0].status == FAIL
    ASSERT result.passed == False

TEST test_sentinel_handles_missing_tools:
    # Mock shutil.which to return None for cosign/syft
    result = sentinel_verify()
    cosign_check = find_check(result, "container_signed")
    ASSERT cosign_check.status == NOT_IMPLEMENTED

TEST test_sentinel_prod_guard_blocks_anon:
    # Verify _anonymous_auth_allowed returns False in prod env
    result = sentinel_verify()
    guard_check = find_check(result, "prod_guard")
    ASSERT guard_check.status == PASS
```
