import sys

from bizra_kernel.damage_control_engine import DamageControlEngine
from bizra_kernel.snr_tracker import SNRMetrics
from bizra_kernel.got_orchestrator import GoTOrchestrator
from bizra_kernel.giant_protocol import GiantProtocol


damage_control = DamageControlEngine()

def run_test(name, func):
    try:
        details = func()
        print(f"PASS {name}")
        if details:
            print(f"   Details: {details}")
        return True
    except AssertionError as exc:
        print(f"FAIL {name}")
        print(f"   Details: {exc}")
        return False


def test_block_dangerous_command():
    result = damage_control.evaluate_command("rm -rf /")
    assert not result["allowed"], "Expected command to be blocked"
    assert "rm with recursive or force flags" in result["blocked"], result["blocked"]
    return f"Blocked: {result['blocked']}"


def test_allow_safe_command():
    result = damage_control.evaluate_command("ls -la /tmp")
    assert result["allowed"], "Expected command to be allowed"
    return f"Safety SNR: {result['safety_snr']}"


def test_pattern_block_rm_rf():
    command = "rm -rf /tmp/test"
    result = damage_control.evaluate_command(command)
    assert not result["allowed"], "Expected rm -rf to be blocked"
    return f"Command: {command}, Blocked: {not result['allowed']}"


def test_pattern_block_chmod_777():
    command = "chmod 777 /etc/passwd"
    result = damage_control.evaluate_command(command)
    assert not result["allowed"], "Expected chmod 777 to be blocked"
    return f"Command: {command}, Blocked: {not result['allowed']}"


def test_pattern_block_sudo_rm():
    command = "sudo rm -rf /*"
    result = damage_control.evaluate_command(command)
    assert not result["allowed"], "Expected sudo rm -rf to be blocked"
    return f"Command: {command}, Blocked: {not result['allowed']}"


def test_pattern_block_unqualified_sql_delete():
    command = "DELETE FROM users;"
    result = damage_control.evaluate_command(command)
    assert not result["allowed"], "Expected unqualified SQL DELETE to be blocked"
    return f"Command: {command}, Blocked: {not result['allowed']}"


def test_path_protection_zero_access():
    verdict = damage_control.check_path("~/.ssh/id_rsa", operation="read")
    assert not verdict["allowed"], "Expected zero-access path to be blocked"
    return f"Path: ~/.ssh/id_rsa, Allowed: {verdict['allowed']}"


def test_path_protection_read_only_write():
    verdict = damage_control.check_path("/etc/passwd", operation="write")
    assert not verdict["allowed"], "Expected read-only path to be blocked"
    return f"Path: /etc/passwd, Allowed: {verdict['allowed']}"


def test_got_security_lens_integration():
    orchestrator = GoTOrchestrator()
    prompt = "Deploy secure microservices with security layers and ethical AI governance"
    analysis = orchestrator.analyze(prompt, got_links=[("a", "b", "link")])
    assert "Security" in analysis["lenses"], analysis["lenses"]
    assert analysis["cluster_snr"] >= 0.99, analysis["cluster_snr"]
    return f"Cluster SNR: {analysis['cluster_snr']:.3f}, Lenses: {analysis['lenses']}"


def test_snr_safety_compliance_integration():
    safe = SNRMetrics(
        total_tokens=100,
        useful_tokens=90,
        confidence_score=0.95,
        ethical_compliance=0.95,
        safety_compliance=1.0,
        tool_directness=1.0,
        latency_ms=20,
        agent_role="test",
    )
    risky = SNRMetrics(
        total_tokens=100,
        useful_tokens=90,
        confidence_score=0.95,
        ethical_compliance=0.95,
        safety_compliance=0.5,
        tool_directness=1.0,
        latency_ms=20,
        agent_role="test",
    )
    assert safe.snr_score > risky.snr_score, "Risky SNR should be lower"
    return f"Safe SNR: {safe.snr_score:.3f}, Risky SNR: {risky.snr_score:.3f}"


def test_security_giants_alignment():
    giants = GiantProtocol()
    action = "Enforce zero trust and defense in depth across all tiers"
    result = giants.verify_alignment(action, {})
    assert "SECURITY_ZERO_TRUST" in result["principles"], result["principles"]
    assert "SECURITY_DEFENSE_IN_DEPTH" in result["principles"], result["principles"]
    assert abs(result["snr_boost"] - 0.12) < 0.001, result["snr_boost"]
    return f"Principles: {result['principles']}, SNR Boost: {result['snr_boost']}"


def test_peak_masterpiece_elite_integration():
    orchestrator = GoTOrchestrator()
    giants = GiantProtocol()

    prompt = (
        "Deploy secure microservices with multiple security layers, redundancy, "
        "never trust, always verify and ethical AI governance"
    )

    got_analysis = orchestrator.analyze(prompt, got_links=[("a", "b", "link")])
    giant_alignment = giants.verify_alignment(prompt, {})

    print("\n" + "-" * 80)
    print("PEAK MASTERPIECE INTEGRATION DEMONSTRATION")
    print("-" * 80)
    print(f"\nTask: {prompt}")
    print(f"  * Interdisciplinary Lenses: {got_analysis['lenses']}")
    print(f"  * GoT Cluster SNR: {got_analysis['cluster_snr']:.3f}")
    print(f"  * Giant Alignment: {giant_alignment['principles']}")
    print(f"  * Giant SNR Boost: {giant_alignment['snr_boost']:.3f}")

    commands = [
        "kubectl apply -f deployment.yaml",
        "terraform apply -auto-approve",
        "rm -rf /tmp/test",
    ]
    safety_scores = []
    print("\nSecurity Evaluations:")
    for command in commands:
        result = damage_control.evaluate_command(command)
        status = "ALLOWED" if result["allowed"] else "BLOCKED"
        safety_scores.append(result["safety_snr"])
        print(
            f"  * {command:35} {status} (Safety SNR: {result['safety_snr']:.3f})"
        )

    avg_safety = sum(safety_scores) / len(safety_scores)
    elite_score = round(
        (got_analysis["cluster_snr"] * 0.4)
        + (avg_safety * 0.4)
        + (giant_alignment["snr_boost"] * 0.2),
        3,
    )

    print(f"\nElite Performance Score: {elite_score:.3f}")

    assert elite_score >= 0.7, "Elite score below threshold"
    return f"Score: {elite_score:.3f} (Threshold: 0.7)"


def main() -> int:
    print("=" * 80)
    print("BIZRA DAMAGE CONTROL INTEGRATION TEST SUITE")
    print("=" * 80)

    tests = [
        ("Damage Control - Block Dangerous Command", test_block_dangerous_command),
        ("Damage Control - Allow Safe Command", test_allow_safe_command),
        ("Pattern Block - Recursive force delete", test_pattern_block_rm_rf),
        ("Pattern Block - Dangerous permission change", test_pattern_block_chmod_777),
        ("Pattern Block - Sudo recursive delete", test_pattern_block_sudo_rm),
        ("Pattern Block - Unqualified SQL DELETE", test_pattern_block_unqualified_sql_delete),
        ("Path Protection - Zero Access", test_path_protection_zero_access),
        ("Path Protection - Read-Only Write", test_path_protection_read_only_write),
        ("GoT Security Lens Integration", test_got_security_lens_integration),
        ("SNR Safety Compliance Integration", test_snr_safety_compliance_integration),
        ("Security Giants Alignment", test_security_giants_alignment),
        ("Peak Masterpiece Elite Integration", test_peak_masterpiece_elite_integration),
    ]

    passed = 0
    failed = 0

    for name, test in tests:
        if run_test(name, test):
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\nALL TESTS PASSED - ELITE IMPLEMENTATION VALIDATED")
        return 0

    print("\nSOME TESTS FAILED - REVIEW REQUIRED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
