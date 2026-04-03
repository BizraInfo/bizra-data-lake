#!/usr/bin/env python3
"""
SNR Enforcement Demo
====================
Demonstrates SNR threshold enforcement with receipt emission.

Usage:
    python examples/snr_enforcement_demo.py
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from bizra_kernel.snr_enforcer import (
    SNREnforcer,
    EnforcementContext,
    OperationType,
    enforce_snr,
    enforce_snr_async,
)
from bizra_kernel.snr_tracker import SNRMetrics


def demo_basic_enforcement():
    """Demonstrate basic enforcement."""
    print("=" * 70)
    print("DEMO 1: Basic SNR Enforcement")
    print("=" * 70)

    # Test case 1: Pass
    print("\n1. High SNR (should PASS):")
    result = enforce_snr(
        operation_type=OperationType.REASONING,
        agent_id="pat-master-reasoner",
        snr_score=0.97,
        task_id="task-001",
        details={"query": "Analyze code architecture"}
    )

    print(f"   SNR: {result.snr_score:.4f}")
    print(f"   Threshold: {result.threshold:.4f}")
    print(f"   Result: {'✅ PASSED' if result.passed else '❌ REJECTED'}")
    print(f"   Message: {result.message}")

    # Test case 2: Reject
    print("\n2. Low SNR (should REJECT):")
    result = enforce_snr(
        operation_type=OperationType.SYNTHESIS,
        agent_id="creative-synthesizer",
        snr_score=0.92,
        task_id="task-002",
        details={"query": "Generate documentation"}
    )

    print(f"   SNR: {result.snr_score:.4f}")
    print(f"   Threshold: {result.threshold:.4f}")
    print(f"   Result: {'✅ PASSED' if result.passed else '❌ REJECTED'}")
    print(f"   Message: {result.message}")
    if result.receipt_id:
        print(f"   Receipt ID: {result.receipt_id}")
        print(f"   Rejection Code: {result.rejection_code}")


def demo_custom_enforcer():
    """Demonstrate custom enforcer with SNR tracker."""
    print("\n" + "=" * 70)
    print("DEMO 2: Custom Enforcer with SNR Tracker")
    print("=" * 70)

    from bizra_kernel.snr_tracker import SNRTracker
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create custom enforcer
        tracker = SNRTracker()
        enforcer = SNREnforcer(
            constitution_path="constitution/pat_enforcement_v1.yaml",
            snr_tracker=tracker,
            emit_receipts=True,
            receipt_dir=tmpdir,
        )

        print(f"\nEnforcer Configuration:")
        print(f"  Target SNR: {enforcer.thresholds.target_snr}")
        print(f"  Minimum SNR: {enforcer.thresholds.minimum_snr}")
        print(f"  Escalate Below: {enforcer.thresholds.escalate_below}")
        print(f"  Receipt Directory: {tmpdir}")

        # Record some metrics
        print("\nRecording metrics:")
        for i, (total, useful, agent) in enumerate([
            (1000, 950, "agent-alpha"),
            (800, 720, "agent-beta"),
            (1200, 1080, "agent-gamma"),
        ], 1):
            metrics = SNRMetrics(
                total_tokens=total,
                useful_tokens=useful,
                confidence_score=0.95,
                ethical_compliance=0.97,
                tool_directness=1.0,
                latency_ms=200,
                agent_role=agent,
            )
            enforcer.record_metrics(metrics)
            print(f"  {i}. {agent}: SNR={metrics.snr_score:.4f}")

        # Perform enforcements
        print("\nEnforcing thresholds:")
        test_cases = [
            (OperationType.REASONING, "agent-alpha", 0.96),
            (OperationType.VALIDATION, "agent-beta", 0.93),
            (OperationType.SYNTHESIS, "agent-gamma", 0.98),
        ]

        for op_type, agent_id, snr in test_cases:
            context = EnforcementContext(
                operation_type=op_type,
                agent_id=agent_id,
                snr_score=snr,
            )
            result = enforcer.enforce(context)
            status = "✅ PASS" if result.passed else "❌ REJECT"
            print(f"  {agent_id} ({op_type.value}): {snr:.4f} → {status}")

        # Show statistics
        print("\nEnforcement Statistics:")
        stats = enforcer.get_statistics()
        print(f"  Total Enforcements: {stats['enforcements']}")
        print(f"  Rejections: {stats['rejections']}")
        print(f"  Rejection Rate: {stats['rejection_rate']:.1%}")
        print(f"  Receipts Emitted: {stats['receipts_emitted']}")

        print("\nTracker Statistics:")
        tracker_stats = stats['tracker_stats']
        print(f"  Average SNR: {tracker_stats['average_snr']:.4f}")
        print(f"  Current SNR: {tracker_stats['current_snr']:.4f}")
        print(f"  Meets Target: {tracker_stats['meets_target']}")


async def demo_async_enforcement():
    """Demonstrate async enforcement."""
    print("\n" + "=" * 70)
    print("DEMO 3: Async SNR Enforcement")
    print("=" * 70)

    print("\nExecuting async enforcements:")

    tasks = [
        ("reasoning", "async-agent-1", 0.97),
        ("synthesis", "async-agent-2", 0.94),
        ("validation", "async-agent-3", 0.96),
    ]

    results = await asyncio.gather(*[
        enforce_snr_async(
            operation_type=op_type,
            agent_id=agent_id,
            snr_score=snr,
            task_id=f"async-task-{i}",
        )
        for i, (op_type, agent_id, snr) in enumerate(tasks, 1)
    ])

    for (op_type, agent_id, snr), result in zip(tasks, results):
        status = "✅ PASS" if result.passed else "❌ REJECT"
        print(f"  {agent_id} ({op_type}): {snr:.4f} → {status}")


def demo_operation_types():
    """Demonstrate different operation types."""
    print("\n" + "=" * 70)
    print("DEMO 4: Operation Type Thresholds")
    print("=" * 70)

    print("\nTesting various operation types:")

    operation_types = [
        (OperationType.REASONING, 0.96),
        (OperationType.SYNTHESIS, 0.97),
        (OperationType.VALIDATION, 0.95),
        (OperationType.RETRIEVAL, 0.94),
        (OperationType.GENERATION, 0.96),
        (OperationType.PAT_EXECUTION, 0.97),
        (OperationType.SAT_VALIDATION, 0.96),
        (OperationType.SAPE_PROBE, 0.98),
    ]

    for op_type, snr in operation_types:
        result = enforce_snr(
            operation_type=op_type,
            agent_id=f"test-{op_type.value}",
            snr_score=snr,
        )

        status = "✅ PASS" if result.passed else "❌ REJECT"
        delta = snr - result.threshold
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"

        print(f"  {op_type.value:20s}: {snr:.4f} vs {result.threshold:.4f} "
              f"({delta_str}) → {status}")


def demo_edge_cases():
    """Demonstrate edge cases."""
    print("\n" + "=" * 70)
    print("DEMO 5: Edge Cases")
    print("=" * 70)

    print("\nTesting edge cases:")

    test_cases = [
        ("Exact threshold", 0.95),
        ("Just above threshold", 0.9501),
        ("Just below threshold", 0.9499),
        ("Perfect SNR", 1.0),
        ("Zero SNR", 0.0),
        ("Very low SNR", 0.50),
    ]

    for name, snr in test_cases:
        result = enforce_snr(
            operation_type=OperationType.DEFAULT,
            agent_id="edge-case-test",
            snr_score=snr,
        )

        status = "✅ PASS" if result.passed else "❌ REJECT"
        print(f"  {name:25s}: {snr:.4f} → {status}")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("SNR ENFORCEMENT SYSTEM DEMO")
    print("=" * 70)
    print("\nThis demo showcases the SNR enforcement system:")
    print("  - Constitutional threshold loading")
    print("  - Fail-closed enforcement semantics")
    print("  - Receipt emission on rejection")
    print("  - Integration with SNR tracker")
    print("  - Async enforcement support")

    try:
        # Run demos
        demo_basic_enforcement()
        demo_custom_enforcer()
        asyncio.run(demo_async_enforcement())
        demo_operation_types()
        demo_edge_cases()

        print("\n" + "=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        print("\nKey Takeaways:")
        print("  ✓ SNR enforcement is fail-closed (rejects on low SNR)")
        print("  ✓ Rejections emit receipts with full evidence")
        print("  ✓ Thresholds are loaded from constitution")
        print("  ✓ Per-operation-type thresholds are supported")
        print("  ✓ Integration with SNR tracker for metrics")
        print("\nNext Steps:")
        print("  1. Review receipts in docs/evidence/receipts/snr/")
        print("  2. Check enforcement statistics")
        print("  3. Monitor rejection rates")
        print("  4. Adjust thresholds in constitution if needed")
        print("\nDocumentation: docs/SNR_ENFORCEMENT.md")

    except Exception as e:
        print(f"\n❌ Error during demo: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
