"""
Constitution - Compliance Verification System

Implements constitutional compliance verification for BIZRA operations.
Enforces Ihsān (excellence) thresholds and generates compliance receipts.

Key Features:
- 0.95 minimum Ihsān threshold (95% excellence)
- Domain-separated verification: "bizra-pci-v1:"
- Cryptographic integrity checks (SHA-256)
- Compliance receipts for audit trails
- Threshold enforcement with detailed violations

NO external dependencies - pure stdlib implementation.
"""

import hashlib
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid


@dataclass
class ConstitutionalPrinciple:
    """Represents a single constitutional principle."""

    name: str
    description: str
    threshold: float  # Minimum compliance score (0.0 to 1.0)
    weight: float = 1.0  # Weight in overall scoring
    required: bool = True  # Must pass for overall compliance


@dataclass
class ComplianceViolation:
    """Represents a compliance violation."""

    principle: str
    threshold: float
    actual: float
    severity: str  # "critical", "major", "minor"
    message: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ComplianceReceipt:
    """Compliance verification receipt."""

    receipt_id: str
    timestamp: str
    domain: str
    operation: str
    overall_score: float
    threshold: float
    compliant: bool
    principles_checked: int
    principles_passed: int
    violations: List[ComplianceViolation]
    metadata: Dict[str, Any]
    integrity_hash: str


class Constitution:
    """
    Constitutional compliance verification system.

    Enforces Ihsān (excellence) thresholds across all BIZRA operations.
    """

    DOMAIN_PREFIX = "bizra-pci-v1:"
    DEFAULT_THRESHOLD = 0.95  # 95% minimum Ihsān

    # Core constitutional principles
    CORE_PRINCIPLES = [
        ConstitutionalPrinciple(
            name="ihsan",
            description="Excellence in execution (Ihsān)",
            threshold=0.95,
            weight=2.0,  # Double weight for core principle
            required=True,
        ),
        ConstitutionalPrinciple(
            name="sovereignty",
            description="Offline operation capability (HYPER LOOPBACK)",
            threshold=1.0,  # Must be 100% sovereign
            weight=2.0,
            required=True,
        ),
        ConstitutionalPrinciple(
            name="transparency",
            description="Complete auditability and evidence",
            threshold=0.95,
            weight=1.5,
            required=True,
        ),
        ConstitutionalPrinciple(
            name="integrity",
            description="Cryptographic integrity verification",
            threshold=1.0,  # Must be 100% verifiable
            weight=1.5,
            required=True,
        ),
        ConstitutionalPrinciple(
            name="determinism",
            description="Reproducible operations",
            threshold=1.0,  # Must be 100% deterministic
            weight=1.0,
            required=True,
        ),
        ConstitutionalPrinciple(
            name="efficiency",
            description="Resource optimization",
            threshold=0.90,
            weight=1.0,
            required=False,
        ),
    ]

    def __init__(
        self,
        principles: Optional[List[ConstitutionalPrinciple]] = None,
        global_threshold: float = DEFAULT_THRESHOLD,
    ):
        """
        Initialize Constitution.

        Args:
            principles: Custom principles (uses CORE_PRINCIPLES if None)
            global_threshold: Minimum overall compliance score
        """
        self.principles = principles or self.CORE_PRINCIPLES
        self.global_threshold = global_threshold
        self.verification_history: List[ComplianceReceipt] = []

    def verify(
        self,
        operation: str,
        scores: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ComplianceReceipt:
        """
        Verify constitutional compliance for an operation.

        Args:
            operation: Name of the operation being verified
            scores: Dictionary of principle_name -> score (0.0 to 1.0)
            metadata: Additional context for the receipt

        Returns:
            ComplianceReceipt with verification results
        """
        violations: List[ComplianceViolation] = []
        principle_scores: List[Tuple[ConstitutionalPrinciple, float]] = []

        # Check each principle
        for principle in self.principles:
            if principle.name not in scores:
                # Missing score for principle
                if principle.required:
                    violations.append(
                        ComplianceViolation(
                            principle=principle.name,
                            threshold=principle.threshold,
                            actual=0.0,
                            severity="critical",
                            message=f"Required principle '{principle.name}' not evaluated",
                        )
                    )
                continue

            score = scores[principle.name]
            principle_scores.append((principle, score))

            # Check threshold
            if score < principle.threshold:
                severity = "critical" if principle.required else "major"
                violations.append(
                    ComplianceViolation(
                        principle=principle.name,
                        threshold=principle.threshold,
                        actual=score,
                        severity=severity,
                        message=(
                            f"{principle.description} below threshold: "
                            f"{score:.4f} < {principle.threshold:.4f}"
                        ),
                    )
                )

        # Calculate weighted overall score
        if principle_scores:
            total_weight = sum(p.weight for p, _ in principle_scores)
            weighted_sum = sum(p.weight * s for p, s in principle_scores)
            overall_score = weighted_sum / total_weight
        else:
            overall_score = 0.0

        # Check global threshold
        compliant = overall_score >= self.global_threshold and not any(
            v.severity == "critical" for v in violations
        )

        # Generate receipt
        receipt = self._generate_receipt(
            operation=operation,
            overall_score=overall_score,
            compliant=compliant,
            principles_checked=len(principle_scores),
            principles_passed=len(principle_scores) - len(violations),
            violations=violations,
            metadata=metadata or {},
        )

        # Store in history
        self.verification_history.append(receipt)

        return receipt

    def verify_with_enforcement(
        self,
        operation: str,
        scores: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ComplianceReceipt:
        """
        Verify compliance and raise exception if non-compliant.

        Args:
            operation: Name of the operation
            scores: Principle scores
            metadata: Additional context

        Returns:
            ComplianceReceipt if compliant

        Raises:
            ConstitutionalViolationError: If operation is non-compliant
        """
        receipt = self.verify(operation, scores, metadata)

        if not receipt.compliant:
            raise ConstitutionalViolationError(
                f"Operation '{operation}' violates constitution", receipt
            )

        return receipt

    def get_principle(self, name: str) -> Optional[ConstitutionalPrinciple]:
        """Get principle by name."""
        for principle in self.principles:
            if principle.name == name:
                return principle
        return None

    def add_principle(self, principle: ConstitutionalPrinciple) -> None:
        """Add new constitutional principle."""
        # Check for duplicates
        existing = self.get_principle(principle.name)
        if existing:
            raise ValueError(f"Principle '{principle.name}' already exists")

        self.principles.append(principle)

    def get_compliance_summary(self) -> Dict[str, Any]:
        """
        Get summary of compliance verification history.

        Returns:
            Dictionary with compliance statistics
        """
        if not self.verification_history:
            return {
                "total_verifications": 0,
                "compliant": 0,
                "non_compliant": 0,
                "compliance_rate": 0.0,
                "average_score": 0.0,
            }

        compliant_count = sum(1 for r in self.verification_history if r.compliant)
        total = len(self.verification_history)
        avg_score = sum(r.overall_score for r in self.verification_history) / total

        # Count violations by severity
        violation_counts = {"critical": 0, "major": 0, "minor": 0}
        for receipt in self.verification_history:
            for violation in receipt.violations:
                violation_counts[violation.severity] += 1

        return {
            "total_verifications": total,
            "compliant": compliant_count,
            "non_compliant": total - compliant_count,
            "compliance_rate": compliant_count / total,
            "average_score": avg_score,
            "violation_counts": violation_counts,
            "global_threshold": self.global_threshold,
        }

    def _generate_receipt(
        self,
        operation: str,
        overall_score: float,
        compliant: bool,
        principles_checked: int,
        principles_passed: int,
        violations: List[ComplianceViolation],
        metadata: Dict[str, Any],
    ) -> ComplianceReceipt:
        """Generate compliance receipt with integrity hash."""
        receipt_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        # Convert violations to dict for hashing
        violations_dict = [asdict(v) for v in violations]

        # Create receipt data (without integrity hash)
        receipt_data = {
            "receipt_id": receipt_id,
            "timestamp": timestamp,
            "domain": self.DOMAIN_PREFIX,
            "operation": operation,
            "overall_score": overall_score,
            "threshold": self.global_threshold,
            "compliant": compliant,
            "principles_checked": principles_checked,
            "principles_passed": principles_passed,
            "violations": violations_dict,
            "metadata": metadata,
        }

        # Generate integrity hash
        receipt_json = json.dumps(receipt_data, sort_keys=True)
        integrity_hash = hashlib.sha256(
            (self.DOMAIN_PREFIX + receipt_json).encode("utf-8")
        ).hexdigest()

        return ComplianceReceipt(
            receipt_id=receipt_id,
            timestamp=timestamp,
            domain=self.DOMAIN_PREFIX,
            operation=operation,
            overall_score=overall_score,
            threshold=self.global_threshold,
            compliant=compliant,
            principles_checked=principles_checked,
            principles_passed=principles_passed,
            violations=violations,
            metadata=metadata,
            integrity_hash=integrity_hash,
        )

    def verify_receipt_integrity(self, receipt: ComplianceReceipt) -> bool:
        """
        Verify integrity hash of a compliance receipt.

        Args:
            receipt: Receipt to verify

        Returns:
            True if integrity hash is valid
        """
        # Recreate receipt data without hash
        violations_dict = [asdict(v) for v in receipt.violations]
        receipt_data = {
            "receipt_id": receipt.receipt_id,
            "timestamp": receipt.timestamp,
            "domain": receipt.domain,
            "operation": receipt.operation,
            "overall_score": receipt.overall_score,
            "threshold": receipt.threshold,
            "compliant": receipt.compliant,
            "principles_checked": receipt.principles_checked,
            "principles_passed": receipt.principles_passed,
            "violations": violations_dict,
            "metadata": receipt.metadata,
        }

        # Recompute hash
        receipt_json = json.dumps(receipt_data, sort_keys=True)
        computed_hash = hashlib.sha256(
            (self.DOMAIN_PREFIX + receipt_json).encode("utf-8")
        ).hexdigest()

        return computed_hash == receipt.integrity_hash

    def export_receipt(self, receipt: ComplianceReceipt, path: str) -> None:
        """Export receipt to JSON file."""
        receipt_dict = asdict(receipt)
        with open(path, "w") as f:
            json.dump(receipt_dict, f, indent=2)

    def import_receipt(self, path: str) -> ComplianceReceipt:
        """Import receipt from JSON file and verify integrity."""
        with open(path, "r") as f:
            data = json.load(f)

        # Reconstruct violations
        violations = [ComplianceViolation(**v) for v in data["violations"]]

        receipt = ComplianceReceipt(
            receipt_id=data["receipt_id"],
            timestamp=data["timestamp"],
            domain=data["domain"],
            operation=data["operation"],
            overall_score=data["overall_score"],
            threshold=data["threshold"],
            compliant=data["compliant"],
            principles_checked=data["principles_checked"],
            principles_passed=data["principles_passed"],
            violations=violations,
            metadata=data["metadata"],
            integrity_hash=data["integrity_hash"],
        )

        # Verify integrity
        if not self.verify_receipt_integrity(receipt):
            raise ValueError("Receipt integrity verification failed")

        return receipt


class ConstitutionalViolationError(Exception):
    """Raised when constitutional compliance is violated."""

    def __init__(self, message: str, receipt: ComplianceReceipt):
        super().__init__(message)
        self.receipt = receipt


def main():
    """Demo Constitution functionality."""
    print("Constitution - Compliance Verification System")
    print("=" * 60)

    constitution = Constitution()

    print(f"Global Ihsān threshold: {constitution.global_threshold}")
    print(f"Core principles: {len(constitution.principles)}")

    # Test compliant operation
    print("\n1. Testing compliant operation...")
    scores = {
        "ihsan": 0.98,
        "sovereignty": 1.0,
        "transparency": 0.97,
        "integrity": 1.0,
        "determinism": 1.0,
        "efficiency": 0.92,
    }

    receipt = constitution.verify(
        operation="winter_proof_embedding",
        scores=scores,
        metadata={"test": "compliant_case"},
    )

    print(f"Compliant: {receipt.compliant}")
    print(f"Overall score: {receipt.overall_score:.4f}")
    print(f"Violations: {len(receipt.violations)}")
    print(f"Receipt ID: {receipt.receipt_id}")

    # Test non-compliant operation
    print("\n2. Testing non-compliant operation...")
    bad_scores = {
        "ihsan": 0.85,  # Below threshold
        "sovereignty": 0.90,  # Below threshold
        "transparency": 0.97,
        "integrity": 1.0,
        "determinism": 1.0,
    }

    receipt2 = constitution.verify(
        operation="external_api_call",
        scores=bad_scores,
        metadata={"test": "non_compliant_case"},
    )

    print(f"Compliant: {receipt2.compliant}")
    print(f"Overall score: {receipt2.overall_score:.4f}")
    print(f"Violations: {len(receipt2.violations)}")

    for violation in receipt2.violations:
        print(f"  - {violation.severity.upper()}: {violation.message}")

    # Test integrity verification
    print("\n3. Testing receipt integrity...")
    is_valid = constitution.verify_receipt_integrity(receipt)
    print(f"Receipt integrity valid: {is_valid}")

    # Get compliance summary
    print("\n4. Compliance summary:")
    summary = constitution.get_compliance_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
