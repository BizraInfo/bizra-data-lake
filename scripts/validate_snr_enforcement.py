#!/usr/bin/env python3
"""
SNR Enforcement Validation Script
==================================
Validates that SNR enforcement is working correctly.

Usage:
    python scripts/validate_snr_enforcement.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")

    try:
        from bizra_kernel.snr_enforcer import (
            SNREnforcer,
            SNRThresholds,
            EnforcementContext,
            EnforcementResult,
            OperationType,
            enforce_snr,
            get_snr_enforcer,
        )
        print("  ✅ SNR enforcer imports successful")
    except ImportError as e:
        print(f"  ❌ Failed to import SNR enforcer: {e}")
        return False

    try:
        from bizra_kernel.snr_tracker import SNRMetrics, SNRTracker
        print("  ✅ SNR tracker imports successful")
    except ImportError as e:
        print(f"  ❌ Failed to import SNR tracker: {e}")
        return False

    try:
        from core.pci import RejectCode, reject_snr
        print("  ✅ PCI reject codes imports successful")
    except ImportError as e:
        print(f"  ⚠️  PCI imports failed (optional): {e}")

    return True


def test_constitution_loading():
    """Test loading thresholds from constitution."""
    print("\nTesting constitution loading...")

    from bizra_kernel.snr_enforcer import SNRThresholds

    # Test with actual constitution
    constitution_path = Path("constitution/pat_enforcement_v1.yaml")
    if constitution_path.exists():
        thresholds = SNRThresholds.from_constitution(constitution_path)
        print(f"  ✅ Constitution loaded")
        print(f"     target_snr: {thresholds.target_snr}")
        print(f"     minimum_snr: {thresholds.minimum_snr}")
        print(f"     escalate_below: {thresholds.escalate_below}")

        # Validate thresholds
        if thresholds.target_snr >= 0.95 and thresholds.minimum_snr >= 0.90:
            print("  ✅ Thresholds are valid")
            return True
        else:
            print("  ❌ Thresholds are invalid")
            return False
    else:
        print(f"  ⚠️  Constitution not found at {constitution_path}, using defaults")
        thresholds = SNRThresholds()
        print(f"     target_snr: {thresholds.target_snr}")
        print(f"     minimum_snr: {thresholds.minimum_snr}")
        return True


def test_enforcement_logic():
    """Test basic enforcement logic."""
    print("\nTesting enforcement logic...")

    from bizra_kernel.snr_enforcer import enforce_snr, OperationType

    # Test 1: Should pass (SNR above threshold)
    result = enforce_snr(
        operation_type=OperationType.REASONING,
        agent_id="test-agent",
        snr_score=0.97,
    )

    if result.passed:
        print("  ✅ High SNR correctly passed")
    else:
        print(f"  ❌ High SNR incorrectly rejected: {result.message}")
        return False

    # Test 2: Should reject (SNR below threshold)
    result = enforce_snr(
        operation_type=OperationType.REASONING,
        agent_id="test-agent",
        snr_score=0.92,
    )

    if not result.passed and result.rejection_code == 7:
        print("  ✅ Low SNR correctly rejected")
        print(f"     Rejection code: {result.rejection_code}")
        if result.receipt_id:
            print(f"     Receipt ID: {result.receipt_id}")
    else:
        print(f"  ❌ Low SNR not rejected properly")
        return False

    # Test 3: Edge case - exact threshold
    result = enforce_snr(
        operation_type=OperationType.DEFAULT,
        agent_id="test-agent",
        snr_score=0.95,  # Exactly at minimum_snr
    )

    if result.passed:
        print("  ✅ Exact threshold correctly passed")
    else:
        print(f"  ❌ Exact threshold incorrectly rejected")
        return False

    return True


def test_receipt_directory():
    """Test that receipt directory exists and is writable."""
    print("\nTesting receipt directory...")

    receipt_dir = Path("docs/evidence/receipts/snr")

    if receipt_dir.exists():
        print(f"  ✅ Receipt directory exists: {receipt_dir}")

        # Test writability
        test_file = receipt_dir / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
            print("  ✅ Receipt directory is writable")
            return True
        except Exception as e:
            print(f"  ❌ Receipt directory not writable: {e}")
            return False
    else:
        print(f"  ⚠️  Receipt directory does not exist: {receipt_dir}")
        print("     Attempting to create...")
        try:
            receipt_dir.mkdir(parents=True, exist_ok=True)
            print("  ✅ Receipt directory created")
            return True
        except Exception as e:
            print(f"  ❌ Failed to create receipt directory: {e}")
            return False


def test_pci_integration():
    """Test PCI gate chain integration."""
    print("\nTesting PCI integration...")

    try:
        from core.pci.gates import GateChain
        print("  ✅ PCI gates module imported")

        # Check if SNR enforcer is available
        try:
            from bizra_kernel.snr_enforcer import get_snr_enforcer
            enforcer = get_snr_enforcer()
            print("  ✅ SNR enforcer available for PCI integration")
            return True
        except Exception as e:
            print(f"  ⚠️  SNR enforcer not available: {e}")
            return True  # Not critical

    except ImportError as e:
        print(f"  ⚠️  PCI gates not available (optional): {e}")
        return True  # Not critical for enforcer functionality


def test_statistics():
    """Test enforcement statistics."""
    print("\nTesting statistics...")

    from bizra_kernel.snr_enforcer import get_snr_enforcer

    enforcer = get_snr_enforcer()
    stats = enforcer.get_statistics()

    if isinstance(stats, dict):
        print("  ✅ Statistics retrieved successfully")
        print(f"     Enforcements: {stats.get('enforcements', 0)}")
        print(f"     Rejections: {stats.get('rejections', 0)}")
        print(f"     Rejection rate: {stats.get('rejection_rate', 0):.1%}")
        return True
    else:
        print("  ❌ Statistics not in expected format")
        return False


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("SNR ENFORCEMENT VALIDATION")
    print("=" * 70)

    results = {
        "Imports": test_imports(),
        "Constitution Loading": test_constitution_loading(),
        "Enforcement Logic": test_enforcement_logic(),
        "Receipt Directory": test_receipt_directory(),
        "PCI Integration": test_pci_integration(),
        "Statistics": test_statistics(),
    }

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:25s}: {status}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✅ ALL TESTS PASSED - SNR enforcement is working correctly!")
        print("\nNext steps:")
        print("  1. Run full test suite: pytest tests/test_snr_enforcer.py -v")
        print("  2. Run demo script: python examples/snr_enforcement_demo.py")
        print("  3. Check documentation: docs/SNR_ENFORCEMENT.md")
        return 0
    else:
        print("\n❌ SOME TESTS FAILED - Please review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
