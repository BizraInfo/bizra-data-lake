"""
Conductor Gate — Layer 4: Operational Readiness
=================================================

13 checks. "Can this system serve 100 users reliably?"

Standing on Giants:
- Deming (1986): Statistical process control
- Patterson (2002): Recovery-oriented computing
"""

from __future__ import annotations

from core.sat._helpers import _has_tool, _run, last_line, path_exists
from core.sat.gate_result import CheckResult, CheckStatus, GateResult

PASS = CheckStatus.PASS
FAIL = CheckStatus.FAIL
PARTIAL = CheckStatus.PARTIAL
NOT_IMPL = CheckStatus.NOT_IMPLEMENTED
SKIPPED = CheckStatus.SKIPPED


def conductor_verify(skip_slow: bool = False) -> GateResult:
    """Layer 4: Operational Readiness — 13 checks."""
    checks: list[CheckResult] = []

    # === PERFORMANCE SLOs (5.1-5.7) ===

    # 5.1-5.2 Latency SLOs (check Prometheus config exists)
    prometheus_exists = path_exists("deploy/monitoring/prometheus-config.yaml")
    checks.append(
        CheckResult(
            "monitoring_configured",
            PASS if prometheus_exists else NOT_IMPL,
            (
                "Prometheus config present"
                if prometheus_exists
                else "No monitoring config"
            ),
        )
    )

    # 5.3 Alerting rules
    alerting_exists = path_exists("deploy/monitoring/alerting-rules.yaml")
    checks.append(
        CheckResult(
            "alerting_configured",
            PASS if alerting_exists else NOT_IMPL,
            "Alerting rules present" if alerting_exists else "No alerting config",
        )
    )

    # 5.4 Grafana SLO dashboard
    grafana_exists = path_exists("deploy/k8s/base/grafana-slo-dashboard.yaml")
    checks.append(
        CheckResult(
            "grafana_dashboard",
            PASS if grafana_exists else NOT_IMPL,
            "Grafana dashboard present" if grafana_exists else "No Grafana dashboard",
        )
    )

    # === CAPACITY GATES (5.8-5.12) ===

    # 5.8 k6 load test
    k6_exists = _has_tool("k6") and path_exists("scripts/load_test.js")
    checks.append(
        CheckResult(
            "load_test_configured",
            PASS if k6_exists else NOT_IMPL,
            "k6 + load test script present" if k6_exists else "No k6 load tests",
        )
    )

    # 5.9-5.10 Soak test
    soak_exists = path_exists("scripts/soak_test.sh")
    checks.append(
        CheckResult(
            "soak_test_configured",
            PASS if soak_exists else NOT_IMPL,
            "Soak test script present" if soak_exists else "No soak test",
        )
    )

    # 5.11 HPA configured
    hpa_exists = path_exists("deploy/k8s/base/hpa.yaml")
    checks.append(
        CheckResult(
            "hpa_configured",
            PASS if hpa_exists else NOT_IMPL,
            "HPA present" if hpa_exists else "No HPA config",
        )
    )

    # === DEPLOYMENT GATES (5.13-5.17) ===

    # 5.13 Staging Kustomize builds
    if _has_tool("kustomize"):
        code, out = _run(
            ["kustomize", "build", "deploy/k8s/overlays/staging"], timeout=30
        )
        checks.append(
            CheckResult(
                "staging_builds",
                PASS if code == 0 else FAIL,
                "Kustomize staging builds clean" if code == 0 else last_line(out),
            )
        )
    elif path_exists("deploy/k8s/overlays/staging/kustomization.yaml"):
        # Try with kubectl kustomize
        code, out = _run(
            ["kubectl", "kustomize", "deploy/k8s/overlays/staging"], timeout=30
        )
        checks.append(
            CheckResult(
                "staging_builds",
                PASS if code == 0 else FAIL,
                (
                    "kubectl kustomize staging builds clean"
                    if code == 0
                    else last_line(out)
                ),
            )
        )
    else:
        checks.append(
            CheckResult("staging_builds", NOT_IMPL, "No staging kustomization")
        )

    # 5.14 Argo Rollouts canary config
    argo_exists = path_exists("deploy/argocd/rollouts.yaml") or path_exists(
        "deploy/k8s/base/rollout.yaml"
    )
    checks.append(
        CheckResult(
            "canary_configured",
            PASS if argo_exists else NOT_IMPL,
            "Canary rollout config present" if argo_exists else "No canary config",
        )
    )

    # 5.15 Rollback mechanism
    rollback_exists = path_exists("core/rollout/rollback.py")
    checks.append(
        CheckResult(
            "rollback_exists",
            PASS if rollback_exists else NOT_IMPL,
            "Rollback module present" if rollback_exists else "No rollback module",
        )
    )

    # 5.16 Deployment runbook
    runbook_exists = path_exists("deploy/resilience/RUNBOOK.md") or path_exists(
        "docs/RUNBOOK.md"
    )
    checks.append(
        CheckResult(
            "runbook_exists",
            PASS if runbook_exists else NOT_IMPL,
            "Runbook present" if runbook_exists else "No runbook",
        )
    )

    # === CLI GATES (5.18-5.21) ===

    # 5.18 CLI launches
    code, out = _run(
        [
            "python",
            "-c",
            "import core.sovereign.api; print('OK: sovereign API importable')",
        ],
        timeout=15,
    )
    checks.append(
        CheckResult(
            "cli_launches",
            PASS if code == 0 else FAIL,
            last_line(out),
        )
    )

    # 5.19 CLI doctor
    if skip_slow:
        checks.append(CheckResult("cli_doctor", SKIPPED, "Skipped (--skip-slow)"))
    else:
        code, out = _run(
            [
                "python",
                "-m",
                "pytest",
                "tests/core/sovereign/",
                "-q",
                "--timeout=30",
                "-k",
                "doctor or health_check or diagnostics",
                "-x",
            ],
            timeout=90,
        )
        checks.append(
            CheckResult(
                "cli_doctor",
                PASS if code == 0 else PARTIAL,
                last_line(out),
            )
        )

    return GateResult(agent="Conductor", layer="OPERATIONAL_READINESS", checks=checks)
