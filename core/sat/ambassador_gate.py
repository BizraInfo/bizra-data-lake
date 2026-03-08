"""
Ambassador Gate — Layer 5: Human Verification
===============================================

4 automated + 15 manual checks.
"Will real humans trust, understand, and benefit from this system?"

Standing on Giants:
- Nielsen (1994): Usability heuristics
- Krug (2000): Don't Make Me Think
"""

from __future__ import annotations

from core.sat._helpers import _run, path_exists, prompt_human
from core.sat.gate_result import CheckResult, CheckStatus, GateResult

PASS = CheckStatus.PASS
FAIL = CheckStatus.FAIL
PARTIAL = CheckStatus.PARTIAL
NOT_IMPL = CheckStatus.NOT_IMPLEMENTED
SKIPPED = CheckStatus.SKIPPED


def ambassador_verify(skip_manual: bool = False) -> GateResult:
    """Layer 5: Human Verification — 4 automated + 15 manual checks."""
    checks: list[CheckResult] = []

    # === AUTOMATED TRUST GATES ===

    # 6.11 Network isolation
    netpol_exists = path_exists("deploy/k8s/base/networkpolicy.yaml")
    code, out = _run(
        [
            "python",
            "-m",
            "pytest",
            "tests/",
            "-q",
            "--timeout=30",
            "-k",
            "offline or network_isolation or fail_closed",
        ],
        timeout=60,
    )
    checks.append(
        CheckResult(
            "network_isolation",
            PASS if code == 0 and netpol_exists else PARTIAL,
            f"NetworkPolicy: {'yes' if netpol_exists else 'no'}, tests: exit={code}",
        )
    )

    # 6.12 Receipt export/verify
    evidence_scripts = path_exists(
        "scripts/evidence/build_evidence_package.py"
    ) and path_exists("scripts/evidence/verify_evidence_package.py")
    checks.append(
        CheckResult(
            "receipt_verifiable",
            PASS if evidence_scripts else PARTIAL,
            (
                "Evidence build + verify scripts present"
                if evidence_scripts
                else "Missing evidence scripts"
            ),
        )
    )

    # 6.13 Clean shutdown
    stop_script = path_exists("scripts/stop_proactive.sh")
    checks.append(
        CheckResult(
            "clean_shutdown",
            PASS if stop_script else NOT_IMPL,
            "stop_proactive.sh present" if stop_script else "No shutdown script",
        )
    )

    # 6.14 Clean uninstall
    uninstall_exists = path_exists("scripts/uninstall.sh")
    checks.append(
        CheckResult(
            "clean_uninstall",
            PASS if uninstall_exists else NOT_IMPL,
            (
                "Uninstall script present"
                if uninstall_exists
                else "No uninstall mechanism"
            ),
        )
    )

    # === MANUAL CHECKS (human attestation) ===

    MANUAL_QUESTIONS = [
        ("install_success_rate", "Did >= 9/10 Alpha users install successfully?"),
        ("first_mission_rate", "Did >= 9/10 complete their first mission?"),
        ("time_to_value", "Was median time to first SEED < 5 minutes?"),
        ("comprehension", "Can >= 8/10 users explain what BIZRA did?"),
        ("sovereignty_aware", "Do >= 9/10 users know data is local?"),
        ("woow_moment", "Did >= 5/10 users react to the reflex transition?"),
        ("language_diversity", "Were >= 2 languages tested (Arabic + English)?"),
        ("device_diversity", "Were >= 3 device types tested?"),
        ("technical_diversity", "Did >= 3 non-technical users succeed?"),
        ("geographic_diversity", "Were users from >= 2 countries tested?"),
        (
            "no_data_leakage_manual",
            "Network monitor showed zero unexpected connections?",
        ),
        ("receipt_verified_manual", "Third party verified exported Ed25519 receipt?"),
        (
            "shutdown_verified_manual",
            "User confirmed all processes terminated after stop?",
        ),
        (
            "delete_verified_manual",
            "User confirmed clean slate after ~/.bizra/ deletion?",
        ),
        ("testimonial", "Did at least 1 user provide voluntary positive feedback?"),
    ]

    if skip_manual:
        for name, _ in MANUAL_QUESTIONS:
            checks.append(
                CheckResult(name, SKIPPED, "Manual check skipped", is_manual=True)
            )
    else:
        for name, question in MANUAL_QUESTIONS:
            answer = prompt_human(question)
            checks.append(
                CheckResult(
                    name,
                    PASS if answer else FAIL,
                    f"Human attested: {'YES' if answer else 'NO'}",
                    is_manual=True,
                )
            )

    return GateResult(agent="Ambassador", layer="HUMAN_VERIFICATION", checks=checks)
