"""
DaughterTest - Continuous Integrity Verification

Implements continuous integrity verification for BIZRA operations.
Named after the concept: "Trust, but verify - as a parent verifies their daughter's wellbeing."

Key Features:
- Continuous monitoring of operation integrity
- Determinism verification (same input -> same output)
- Checksum validation (SHA-256)
- Temporal consistency checks
- Evidence chain verification
- Real-time violation detection

NO external dependencies - pure stdlib implementation.
"""

import hashlib
import json
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass
from datetime import datetime
import uuid
import threading


@dataclass
class IntegrityCheck:
    """Represents a single integrity check."""

    check_id: str
    timestamp: str
    check_type: str  # "determinism", "checksum", "temporal", "chain"
    subject: str  # What is being checked
    expected: Any
    actual: Any
    passed: bool
    metadata: Dict[str, Any]
    integrity_hash: str


@dataclass
class ViolationAlert:
    """Alert for integrity violation."""

    alert_id: str
    timestamp: str
    severity: str  # "critical", "high", "medium", "low"
    check_type: str
    subject: str
    message: str
    expected: Any
    actual: Any
    metadata: Dict[str, Any]


class DaughterTest:
    """
    Continuous integrity verification system.

    Performs ongoing verification of operation integrity, determinism,
    and evidence chain consistency.
    """

    DOMAIN_PREFIX = "bizra-pci-v1:"

    def __init__(self, auto_start: bool = False):
        """
        Initialize DaughterTest.

        Args:
            auto_start: Start continuous monitoring immediately
        """
        self.checks: List[IntegrityCheck] = []
        self.violations: List[ViolationAlert] = []
        self.monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Registry of monitored operations
        self.operation_registry: Dict[str, Dict[str, Any]] = {}

        if auto_start:
            self.start_monitoring()

    def verify_determinism(
        self,
        operation: Callable,
        inputs: Tuple[Any, ...],
        iterations: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IntegrityCheck:
        """
        Verify that operation produces deterministic outputs.

        Args:
            operation: Function to test
            inputs: Input arguments for the function
            iterations: Number of times to run (default 3)
            metadata: Additional context

        Returns:
            IntegrityCheck with determinism verification results
        """
        outputs = []
        hashes = []

        # Run operation multiple times
        for i in range(iterations):
            result = operation(*inputs)
            outputs.append(result)

            # Hash the result
            result_json = json.dumps(result, sort_keys=True)
            result_hash = hashlib.sha256(result_json.encode("utf-8")).hexdigest()
            hashes.append(result_hash)

        # Check if all hashes are identical
        passed = len(set(hashes)) == 1

        check = self._create_check(
            check_type="determinism",
            subject=operation.__name__,
            expected=hashes[0],
            actual=hashes,
            passed=passed,
            metadata={
                "iterations": iterations,
                "unique_outputs": len(set(hashes)),
                **(metadata or {}),
            },
        )

        if not passed:
            self._raise_violation(
                severity="critical",
                check_type="determinism",
                subject=operation.__name__,
                message=f"Operation produced non-deterministic outputs: {len(set(hashes))} unique results",
                expected=hashes[0],
                actual=hashes,
                metadata=metadata or {},
            )

        return check

    def verify_checksum(
        self,
        data: Any,
        expected_hash: str,
        algorithm: str = "sha256",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IntegrityCheck:
        """
        Verify data integrity using checksum.

        Args:
            data: Data to verify
            expected_hash: Expected hash value
            algorithm: Hash algorithm (default sha256)
            metadata: Additional context

        Returns:
            IntegrityCheck with checksum verification results
        """
        # Compute hash
        if isinstance(data, (dict, list)):
            data_json = json.dumps(data, sort_keys=True)
            data_bytes = data_json.encode("utf-8")
        elif isinstance(data, str):
            data_bytes = data.encode("utf-8")
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            data_bytes = str(data).encode("utf-8")

        hasher = hashlib.new(algorithm)
        hasher.update(data_bytes)
        actual_hash = hasher.hexdigest()

        passed = actual_hash == expected_hash

        check = self._create_check(
            check_type="checksum",
            subject=f"data_{algorithm}",
            expected=expected_hash,
            actual=actual_hash,
            passed=passed,
            metadata={
                "algorithm": algorithm,
                "data_size": len(data_bytes),
                **(metadata or {}),
            },
        )

        if not passed:
            self._raise_violation(
                severity="critical",
                check_type="checksum",
                subject=f"data_{algorithm}",
                message=f"Checksum mismatch: {algorithm} verification failed",
                expected=expected_hash,
                actual=actual_hash,
                metadata=metadata or {},
            )

        return check

    def verify_temporal_consistency(
        self,
        current_state: Any,
        previous_state: Any,
        consistency_rules: Dict[str, Callable[[Any, Any], bool]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IntegrityCheck:
        """
        Verify temporal consistency between states.

        Args:
            current_state: Current system state
            previous_state: Previous system state
            consistency_rules: Dict of rule_name -> validation_function
            metadata: Additional context

        Returns:
            IntegrityCheck with temporal consistency results
        """
        violations_found = []

        for rule_name, rule_func in consistency_rules.items():
            try:
                is_consistent = rule_func(previous_state, current_state)
                if not is_consistent:
                    violations_found.append(rule_name)
            except Exception as e:
                violations_found.append(f"{rule_name}_error:{str(e)}")

        passed = len(violations_found) == 0

        check = self._create_check(
            check_type="temporal",
            subject="state_consistency",
            expected="no_violations",
            actual=violations_found if violations_found else "consistent",
            passed=passed,
            metadata={
                "rules_checked": len(consistency_rules),
                "violations": violations_found,
                **(metadata or {}),
            },
        )

        if not passed:
            self._raise_violation(
                severity="high",
                check_type="temporal",
                subject="state_consistency",
                message=f"Temporal consistency violations: {', '.join(violations_found)}",
                expected="no_violations",
                actual=violations_found,
                metadata=metadata or {},
            )

        return check

    def verify_evidence_chain(
        self, chain: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None
    ) -> IntegrityCheck:
        """
        Verify integrity of evidence chain.

        Args:
            chain: List of evidence entries with 'hash' and 'prev_hash' fields
            metadata: Additional context

        Returns:
            IntegrityCheck with chain verification results
        """
        if not chain:
            check = self._create_check(
                check_type="chain",
                subject="evidence_chain",
                expected="non_empty",
                actual="empty",
                passed=False,
                metadata=metadata or {},
            )
            self._raise_violation(
                severity="high",
                check_type="chain",
                subject="evidence_chain",
                message="Evidence chain is empty",
                expected="non_empty",
                actual="empty",
                metadata=metadata or {},
            )
            return check

        chain_valid = True
        broken_links = []

        # Verify first entry (genesis)
        if chain[0].get("prev_hash") is not None:
            chain_valid = False
            broken_links.append("genesis_has_prev_hash")

        # Verify chain links
        for i in range(1, len(chain)):
            current = chain[i]
            previous = chain[i - 1]

            # Check if current's prev_hash matches previous's hash
            if current.get("prev_hash") != previous.get("hash"):
                chain_valid = False
                broken_links.append(f"link_{i}_broken")

        check = self._create_check(
            check_type="chain",
            subject="evidence_chain",
            expected="valid_chain",
            actual="valid" if chain_valid else f"broken:{broken_links}",
            passed=chain_valid,
            metadata={
                "chain_length": len(chain),
                "broken_links": broken_links,
                **(metadata or {}),
            },
        )

        if not chain_valid:
            self._raise_violation(
                severity="critical",
                check_type="chain",
                subject="evidence_chain",
                message=f"Evidence chain integrity violated at: {', '.join(broken_links)}",
                expected="valid_chain",
                actual=broken_links,
                metadata=metadata or {},
            )

        return check

    def register_operation(
        self,
        operation_id: str,
        operation_func: Callable,
        baseline_inputs: Tuple[Any, ...],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register operation for continuous monitoring.

        Args:
            operation_id: Unique identifier for operation
            operation_func: Function to monitor
            baseline_inputs: Baseline inputs for determinism checks
            metadata: Additional context
        """
        # Run baseline
        baseline_output = operation_func(*baseline_inputs)
        baseline_hash = hashlib.sha256(
            json.dumps(baseline_output, sort_keys=True).encode("utf-8")
        ).hexdigest()

        self.operation_registry[operation_id] = {
            "function": operation_func,
            "baseline_inputs": baseline_inputs,
            "baseline_hash": baseline_hash,
            "baseline_output": baseline_output,
            "last_check": datetime.utcnow().isoformat(),
            "check_count": 0,
            "metadata": metadata or {},
        }

    def check_registered_operation(self, operation_id: str) -> IntegrityCheck:
        """
        Verify integrity of registered operation.

        Args:
            operation_id: ID of registered operation

        Returns:
            IntegrityCheck with verification results
        """
        if operation_id not in self.operation_registry:
            raise ValueError(f"Operation '{operation_id}' not registered")

        reg = self.operation_registry[operation_id]

        # Run operation with baseline inputs
        current_output = reg["function"](*reg["baseline_inputs"])
        current_hash = hashlib.sha256(
            json.dumps(current_output, sort_keys=True).encode("utf-8")
        ).hexdigest()

        passed = current_hash == reg["baseline_hash"]

        # Update registry
        reg["last_check"] = datetime.utcnow().isoformat()
        reg["check_count"] += 1

        check = self._create_check(
            check_type="determinism",
            subject=operation_id,
            expected=reg["baseline_hash"],
            actual=current_hash,
            passed=passed,
            metadata={"check_count": reg["check_count"], **reg["metadata"]},
        )

        if not passed:
            self._raise_violation(
                severity="critical",
                check_type="determinism",
                subject=operation_id,
                message=f"Registered operation '{operation_id}' produced different output",
                expected=reg["baseline_hash"],
                actual=current_hash,
                metadata=reg["metadata"],
            )

        return check

    def start_monitoring(self, interval_seconds: int = 60) -> None:
        """
        Start continuous monitoring of registered operations.

        Args:
            interval_seconds: Check interval in seconds
        """
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self._stop_event.clear()

        def monitor_loop():
            while not self._stop_event.is_set():
                # Check all registered operations
                for operation_id in list(self.operation_registry.keys()):
                    try:
                        self.check_registered_operation(operation_id)
                    except Exception as e:
                        self._raise_violation(
                            severity="high",
                            check_type="monitoring",
                            subject=operation_id,
                            message=f"Monitoring error: {str(e)}",
                            expected="success",
                            actual=f"error:{type(e).__name__}",
                            metadata={"error": str(e)},
                        )

                # Wait for interval or stop signal
                self._stop_event.wait(interval_seconds)

        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        if not self.monitoring_active:
            return

        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)

        self.monitoring_active = False

    def get_integrity_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive integrity report.

        Returns:
            Dictionary with integrity statistics
        """
        total_checks = len(self.checks)
        passed_checks = sum(1 for c in self.checks if c.passed)

        # Count by type
        checks_by_type = {}
        for check in self.checks:
            checks_by_type[check.check_type] = (
                checks_by_type.get(check.check_type, 0) + 1
            )

        # Count violations by severity
        violations_by_severity = {}
        for violation in self.violations:
            violations_by_severity[violation.severity] = (
                violations_by_severity.get(violation.severity, 0) + 1
            )

        return {
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": total_checks - passed_checks,
            "pass_rate": passed_checks / total_checks if total_checks > 0 else 0.0,
            "checks_by_type": checks_by_type,
            "total_violations": len(self.violations),
            "violations_by_severity": violations_by_severity,
            "registered_operations": len(self.operation_registry),
            "monitoring_active": self.monitoring_active,
        }

    def _create_check(
        self,
        check_type: str,
        subject: str,
        expected: Any,
        actual: Any,
        passed: bool,
        metadata: Dict[str, Any],
    ) -> IntegrityCheck:
        """Create integrity check with hash."""
        check_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        # Create check data
        check_data = {
            "check_id": check_id,
            "timestamp": timestamp,
            "check_type": check_type,
            "subject": subject,
            "expected": str(expected),
            "actual": str(actual),
            "passed": passed,
            "metadata": metadata,
        }

        # Generate integrity hash
        check_json = json.dumps(check_data, sort_keys=True)
        integrity_hash = hashlib.sha256(
            (self.DOMAIN_PREFIX + check_json).encode("utf-8")
        ).hexdigest()

        check = IntegrityCheck(
            check_id=check_id,
            timestamp=timestamp,
            check_type=check_type,
            subject=subject,
            expected=expected,
            actual=actual,
            passed=passed,
            metadata=metadata,
            integrity_hash=integrity_hash,
        )

        self.checks.append(check)
        return check

    def _raise_violation(
        self,
        severity: str,
        check_type: str,
        subject: str,
        message: str,
        expected: Any,
        actual: Any,
        metadata: Dict[str, Any],
    ) -> ViolationAlert:
        """Create and store violation alert."""
        alert = ViolationAlert(
            alert_id=str(uuid.uuid4()),
            timestamp=datetime.utcnow().isoformat(),
            severity=severity,
            check_type=check_type,
            subject=subject,
            message=message,
            expected=expected,
            actual=actual,
            metadata=metadata,
        )

        self.violations.append(alert)
        return alert


def main():
    """Demo DaughterTest functionality."""
    print("DaughterTest - Continuous Integrity Verification")
    print("=" * 60)

    tester = DaughterTest()

    # Test determinism
    print("\n1. Testing determinism verification...")

    def deterministic_func(x: int) -> int:
        return x * 2

    check1 = tester.verify_determinism(
        operation=deterministic_func, inputs=(5,), iterations=3
    )
    print(f"Deterministic check passed: {check1.passed}")

    # Test checksum
    print("\n2. Testing checksum verification...")
    data = "BIZRA sovereignty module"
    expected_hash = hashlib.sha256(data.encode("utf-8")).hexdigest()

    check2 = tester.verify_checksum(data, expected_hash)
    print(f"Checksum check passed: {check2.passed}")

    # Test evidence chain
    print("\n3. Testing evidence chain verification...")
    chain = [
        {"hash": "abc123", "prev_hash": None, "data": "genesis"},
        {"hash": "def456", "prev_hash": "abc123", "data": "block1"},
        {"hash": "ghi789", "prev_hash": "def456", "data": "block2"},
    ]

    check3 = tester.verify_evidence_chain(chain)
    print(f"Chain check passed: {check3.passed}")

    # Register operation for monitoring
    print("\n4. Registering operation for monitoring...")
    tester.register_operation(
        operation_id="double_op",
        operation_func=deterministic_func,
        baseline_inputs=(10,),
        metadata={"purpose": "test_monitoring"},
    )

    check4 = tester.check_registered_operation("double_op")
    print(f"Registered operation check passed: {check4.passed}")

    # Generate report
    print("\n5. Integrity report:")
    report = tester.get_integrity_report()
    for key, value in report.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
