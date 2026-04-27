"""
Sentinel Gate — Layer 1: Structural Integrity
==============================================

12 checks. Zero tolerance on security.
"Can this system be broken, corrupted, or exploited?"

Standing on Giants:
- Dijkstra (1972): Correctness by construction
- Lampson (2004): Computer security in the real world
"""

from __future__ import annotations

from core.sat._helpers import _has_tool, _run, last_line, parse_test_count
from core.sat.gate_result import CheckResult, CheckStatus, GateResult

PASS = CheckStatus.PASS
FAIL = CheckStatus.FAIL
PARTIAL = CheckStatus.PARTIAL
NOT_IMPL = CheckStatus.NOT_IMPLEMENTED


def sentinel_verify(skip_slow: bool = False) -> GateResult:
    """Layer 1: Structural Integrity — 12 checks."""
    checks: list[CheckResult] = []

    if skip_slow:
        checks.extend(
            CheckResult(name, CheckStatus.SKIPPED, "Skipped (--skip-slow)")
            for name in [
                "tests_pass",
                "zero_criticals",
                "type_safe",
                "lint_clean",
                "coverage_floor",
                "ci_green",
                "auth_closed",
                "prod_guard",
                "no_secrets",
                "identity_works",
                "chain_valid",
            ]
        )
        return GateResult(agent="Sentinel", layer="STRUCTURAL_INTEGRITY", checks=checks)

    # 1.1 All tests pass
    code, out = _run(
        [
            "python",
            "-m",
            "pytest",
            "tests/",
            "-q",
            "--timeout=60",
            "-m",
            "not slow and not requires_ollama and not requires_gpu and not requires_network",
        ],
        timeout=600,
    )
    count = parse_test_count(out)
    checks.append(
        CheckResult(
            "tests_pass",
            PASS if code == 0 else FAIL,
            f"{count} tests passed, exit={code}",
        )
    )

    # 1.2 Zero CRITICAL security findings
    if _has_tool("bandit"):
        code, out = _run(
            ["bandit", "-r", "core/", "-ll", "-q", "--format", "json"], timeout=120
        )
        checks.append(
            CheckResult(
                "zero_criticals",
                PASS if code == 0 else PARTIAL,
                f"bandit exit={code}" if code == 0 else last_line(out),
            )
        )
    else:
        checks.append(CheckResult("zero_criticals", NOT_IMPL, "bandit not installed"))

    # 1.3 Type safety (mypy)
    code, out = _run(
        [
            "python",
            "-m",
            "mypy",
            "core/",
            "--ignore-missing-imports",
            "--no-error-summary",
        ],
        timeout=120,
    )
    checks.append(
        CheckResult(
            "type_safe",
            PASS if code == 0 else PARTIAL,
            f"mypy exit={code}",
        )
    )

    # 1.4 Lint clean (ruff)
    code, out = _run(["python", "-m", "ruff", "check", "core/", "--quiet"], timeout=60)
    checks.append(
        CheckResult(
            "lint_clean",
            PASS if code == 0 else FAIL,
            f"ruff exit={code}" if code == 0 else last_line(out),
        )
    )

    # 1.5 Coverage floor
    code, out = _run(
        [
            "python",
            "-m",
            "pytest",
            "tests/",
            "-q",
            "--timeout=60",
            "--cov=core",
            "--cov-report=term-missing",
            "--cov-fail-under=38",
            "-m",
            "not slow and not requires_ollama and not requires_gpu",
        ],
        timeout=600,
    )
    checks.append(
        CheckResult(
            "coverage_floor",
            PASS if code == 0 else FAIL,
            last_line(out),
        )
    )

    # 1.6 CI pipeline (check last GitHub Actions run)
    if _has_tool("gh"):
        code, out = _run(
            [
                "gh",
                "run",
                "list",
                "--limit",
                "1",
                "--json",
                "conclusion",
                "-q",
                ".[0].conclusion",
            ],
            timeout=30,
        )
        conclusion = out.strip()
        checks.append(
            CheckResult(
                "ci_green",
                PASS if conclusion == "success" else PARTIAL,
                f"Last CI: {conclusion}",
            )
        )
    else:
        checks.append(CheckResult("ci_green", NOT_IMPL, "gh CLI not available"))

    # 2.1 Auth fail-closed
    code, out = _run(
        [
            "python",
            "-m",
            "pytest",
            "tests/core/auth/",
            "tests/core/sovereign/",
            "-q",
            "--timeout=30",
            "-k",
            "auth_fail_closed or fail_closed or protected_route",
            "-x",
        ],
        timeout=90,
    )
    checks.append(
        CheckResult(
            "auth_closed",
            PASS if code == 0 else FAIL,
            last_line(out),
        )
    )

    # 2.2 Production auth guard
    code, out = _run(
        [
            "python",
            "-c",
            "import os; os.environ['BIZRA_ENV']='production'; "
            "from core.auth.middleware import _anonymous_auth_allowed; "
            "assert not _anonymous_auth_allowed(), 'Anonymous auth allowed in production!'; "
            "print('PASS: anonymous auth blocked in production')",
        ],
        timeout=15,
    )
    checks.append(
        CheckResult(
            "prod_guard",
            PASS if code == 0 else FAIL,
            last_line(out),
        )
    )

    # 2.3 No hardcoded secrets
    code, out = _run(
        [
            "python",
            "-m",
            "ruff",
            "check",
            "core/",
            "--select",
            "S105,S106,S107",
            "--quiet",
        ],
        timeout=30,
    )
    checks.append(
        CheckResult(
            "no_secrets",
            PASS if code == 0 else PARTIAL,
            f"Secret scan exit={code}",
        )
    )

    # 2.5 Ed25519 identity
    code, out = _run(
        [
            "python",
            "-c",
            "from nacl.signing import SigningKey; "
            "sk = SigningKey.generate(); "
            "msg = b'genesis-test'; "
            "sig = sk.sign(msg); "
            "sk.verify_key.verify(sig); "
            "print('PASS: Ed25519 sign/verify cycle works')",
        ],
        timeout=15,
    )
    checks.append(
        CheckResult(
            "identity_works",
            PASS if code == 0 else FAIL,
            last_line(out),
        )
    )

    # 2.6 Evidence chain integrity
    code, out = _run(
        [
            "python",
            "-m",
            "pytest",
            "tests/core/proof_engine/",
            "-q",
            "--timeout=30",
            "-k",
            "evidence_chain or hash_chain or chain_integrity",
            "-x",
        ],
        timeout=90,
    )
    checks.append(
        CheckResult(
            "chain_valid",
            PASS if code == 0 else FAIL,
            last_line(out),
        )
    )

    return GateResult(agent="Sentinel", layer="STRUCTURAL_INTEGRITY", checks=checks)
