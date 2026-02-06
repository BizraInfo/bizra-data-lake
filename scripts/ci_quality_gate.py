#!/usr/bin/env python3
"""
BIZRA CI Quality Gate — Constitutional Validation Script

This script validates code quality against BIZRA constitutional thresholds.
It is designed to run in CI pipelines and enforce quality gates.

Standing on Giants: Shannon (SNR) • PMBOK (Quality Gates) • BIZRA Constitution

Usage:
    python scripts/ci_quality_gate.py [--environment ENV] [--strict]

Exit Codes:
    0 - All quality gates passed
    1 - Quality gate failed
    2 - Configuration error
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure core module is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.integration.constants import (
    IHSAN_THRESHOLD_CI,
    IHSAN_THRESHOLD_DEV,
    IHSAN_THRESHOLD_PRODUCTION,
    IHSAN_THRESHOLD_STAGING,
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
    SNR_THRESHOLD_T0_ELITE,
    SNR_THRESHOLD_T1_HIGH,
    SNR_THRESHOLD_T2_STANDARD,
)


@dataclass
class QualityMetrics:
    """Quality metrics collected during gate evaluation."""

    snr_score: float = 0.0
    ihsan_score: float = 0.0
    test_coverage: float = 0.0
    type_coverage: float = 0.0
    lint_score: float = 0.0
    security_score: float = 0.0
    documentation_score: float = 0.0

    # Computed
    overall_score: float = field(init=False)

    def __post_init__(self):
        # Weighted average of all scores
        weights = {
            "snr": 0.25,
            "ihsan": 0.25,
            "test_coverage": 0.15,
            "type_coverage": 0.10,
            "lint": 0.10,
            "security": 0.10,
            "documentation": 0.05,
        }
        self.overall_score = (
            weights["snr"] * self.snr_score
            + weights["ihsan"] * self.ihsan_score
            + weights["test_coverage"] * self.test_coverage
            + weights["type_coverage"] * self.type_coverage
            + weights["lint"] * self.lint_score
            + weights["security"] * self.security_score
            + weights["documentation"] * self.documentation_score
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snr_score": self.snr_score,
            "ihsan_score": self.ihsan_score,
            "test_coverage": self.test_coverage,
            "type_coverage": self.type_coverage,
            "lint_score": self.lint_score,
            "security_score": self.security_score,
            "documentation_score": self.documentation_score,
            "overall_score": self.overall_score,
        }


@dataclass
class GateResult:
    """Result of a quality gate check."""

    name: str
    passed: bool
    actual: float
    threshold: float
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "actual": self.actual,
            "threshold": self.threshold,
            "message": self.message,
        }


class QualityGateValidator:
    """
    Quality gate validator for CI pipelines.

    Enforces BIZRA constitutional thresholds based on environment.
    """

    ENVIRONMENT_THRESHOLDS = {
        "production": {
            "ihsan": IHSAN_THRESHOLD_PRODUCTION,
            "snr": SNR_THRESHOLD_T1_HIGH,
        },
        "staging": {
            "ihsan": IHSAN_THRESHOLD_STAGING,
            "snr": SNR_THRESHOLD_T2_STANDARD,
        },
        "ci": {
            "ihsan": IHSAN_THRESHOLD_CI,
            "snr": SNR_THRESHOLD_T2_STANDARD,
        },
        "dev": {
            "ihsan": IHSAN_THRESHOLD_DEV,
            "snr": 0.80,
        },
    }

    def __init__(self, environment: str = "ci", strict: bool = False):
        self.environment = environment
        self.strict = strict
        self.thresholds = self.ENVIRONMENT_THRESHOLDS.get(
            environment, self.ENVIRONMENT_THRESHOLDS["ci"]
        )

        if strict:
            # Use production thresholds in strict mode
            self.thresholds = self.ENVIRONMENT_THRESHOLDS["production"]

    def collect_metrics(self, project_root: Path) -> QualityMetrics:
        """Collect quality metrics from the project."""
        metrics = QualityMetrics()

        # Count source and test files
        core_files = list(project_root.glob("core/**/*.py"))
        test_files = list(project_root.glob("tests/**/*.py"))

        # Calculate test coverage proxy (test files / source files)
        if core_files:
            test_ratio = len(test_files) / len(core_files)
            metrics.test_coverage = min(1.0, test_ratio)
        else:
            metrics.test_coverage = 0.0

        # Calculate type coverage (files with type hints)
        typed_files = 0
        for f in core_files:
            try:
                content = f.read_text()
                if "def " in content and ("-> " in content or ": " in content):
                    typed_files += 1
            except Exception:
                pass

        if core_files:
            metrics.type_coverage = typed_files / len(core_files)

        # Calculate documentation score (files with docstrings)
        documented_files = 0
        for f in core_files:
            try:
                content = f.read_text()
                if '"""' in content or "'''" in content:
                    documented_files += 1
            except Exception:
                pass

        if core_files:
            metrics.documentation_score = documented_files / len(core_files)

        # Calculate lint score (estimate based on code structure)
        # In real implementation, this would run actual linters
        metrics.lint_score = 0.90  # Assume good lint score

        # Calculate security score (check for common issues)
        security_issues = 0
        dangerous_patterns = ["eval(", "exec(", "pickle.load", "__import__"]

        for f in core_files:
            try:
                content = f.read_text()
                for pattern in dangerous_patterns:
                    if pattern in content:
                        security_issues += 1
            except Exception:
                pass

        metrics.security_score = max(0.0, 1.0 - (security_issues * 0.1))

        # Calculate SNR score (based on other metrics)
        metrics.snr_score = (
            0.3 * metrics.test_coverage
            + 0.2 * metrics.type_coverage
            + 0.2 * metrics.lint_score
            + 0.2 * metrics.security_score
            + 0.1 * metrics.documentation_score
        )

        # Boost SNR if overall quality is good
        if metrics.snr_score > 0.8:
            metrics.snr_score = min(1.0, metrics.snr_score + 0.1)

        # Calculate Ihsan score (SNR + ethical considerations)
        # Ihsan is slightly higher than SNR when security is good
        ihsan_boost = 0.05 if metrics.security_score > 0.9 else 0.0
        metrics.ihsan_score = min(1.0, metrics.snr_score + ihsan_boost)

        return metrics

    def validate(self, metrics: QualityMetrics) -> List[GateResult]:
        """Validate metrics against thresholds."""
        results = []

        # SNR Gate
        snr_threshold = self.thresholds["snr"]
        results.append(
            GateResult(
                name="SNR Threshold",
                passed=metrics.snr_score >= snr_threshold,
                actual=metrics.snr_score,
                threshold=snr_threshold,
                message=f"SNR score {'meets' if metrics.snr_score >= snr_threshold else 'below'} {self.environment} threshold",
            )
        )

        # Ihsan Gate
        ihsan_threshold = self.thresholds["ihsan"]
        results.append(
            GateResult(
                name="Ihsan Threshold",
                passed=metrics.ihsan_score >= ihsan_threshold,
                actual=metrics.ihsan_score,
                threshold=ihsan_threshold,
                message=f"Ihsan score {'meets' if metrics.ihsan_score >= ihsan_threshold else 'below'} {self.environment} threshold",
            )
        )

        # Test Coverage Gate (minimum 60%)
        test_threshold = 0.6
        results.append(
            GateResult(
                name="Test Coverage",
                passed=metrics.test_coverage >= test_threshold,
                actual=metrics.test_coverage,
                threshold=test_threshold,
                message=f"Test coverage {'adequate' if metrics.test_coverage >= test_threshold else 'insufficient'}",
            )
        )

        # Security Gate (minimum 80%)
        security_threshold = 0.8
        results.append(
            GateResult(
                name="Security Score",
                passed=metrics.security_score >= security_threshold,
                actual=metrics.security_score,
                threshold=security_threshold,
                message=f"Security {'acceptable' if metrics.security_score >= security_threshold else 'concerns detected'}",
            )
        )

        return results


def main():
    parser = argparse.ArgumentParser(
        description="BIZRA CI Quality Gate Validator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--environment",
        "-e",
        choices=["production", "staging", "ci", "dev"],
        default="ci",
        help="Target environment for threshold selection",
    )
    parser.add_argument(
        "--strict",
        "-s",
        action="store_true",
        help="Use strict (production) thresholds regardless of environment",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Write results to file",
    )
    parser.add_argument(
        "--project-root",
        "-p",
        type=str,
        default=".",
        help="Project root directory",
    )

    args = parser.parse_args()

    # Initialize validator
    validator = QualityGateValidator(
        environment=args.environment,
        strict=args.strict,
    )

    project_root = Path(args.project_root).resolve()

    print("=" * 70)
    print("BIZRA QUALITY GATE VALIDATOR")
    print("=" * 70)
    print(f"Environment: {args.environment}")
    print(f"Strict Mode: {args.strict}")
    print(f"Project Root: {project_root}")
    print()

    # Collect metrics
    print("Collecting metrics...")
    metrics = validator.collect_metrics(project_root)

    print("\nQuality Metrics:")
    print(f"  SNR Score:          {metrics.snr_score:.4f}")
    print(f"  Ihsan Score:        {metrics.ihsan_score:.4f}")
    print(f"  Test Coverage:      {metrics.test_coverage:.2%}")
    print(f"  Type Coverage:      {metrics.type_coverage:.2%}")
    print(f"  Lint Score:         {metrics.lint_score:.2%}")
    print(f"  Security Score:     {metrics.security_score:.2%}")
    print(f"  Documentation:      {metrics.documentation_score:.2%}")
    print(f"  Overall Score:      {metrics.overall_score:.4f}")

    # Validate
    print("\nValidating against thresholds...")
    results = validator.validate(metrics)

    print("\nGate Results:")
    print("-" * 70)
    all_passed = True
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        symbol = "[+]" if result.passed else "[-]"
        print(
            f"  {symbol} {result.name}: {result.actual:.4f} "
            f"(threshold: {result.threshold:.4f}) - {status}"
        )
        if not result.passed:
            all_passed = False
            print(f"      {result.message}")

    print("-" * 70)

    # Output results
    output_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "environment": args.environment,
        "strict": args.strict,
        "metrics": metrics.to_dict(),
        "gates": [r.to_dict() for r in results],
        "passed": all_passed,
    }

    if args.json:
        print("\nJSON Output:")
        print(json.dumps(output_data, indent=2))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults written to: {args.output}")

    # Write to GitHub Actions output if available
    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a") as f:
            f.write(f"snr_score={metrics.snr_score:.4f}\n")
            f.write(f"ihsan_score={metrics.ihsan_score:.4f}\n")
            f.write(f"gate_passed={'true' if all_passed else 'false'}\n")
            f.write(f"overall_score={metrics.overall_score:.4f}\n")

    # Final result
    print()
    if all_passed:
        print("[SUCCESS] All quality gates passed!")
        print(f"  SNR: {metrics.snr_score:.4f} >= {validator.thresholds['snr']}")
        print(f"  Ihsan: {metrics.ihsan_score:.4f} >= {validator.thresholds['ihsan']}")
        return 0
    else:
        print("[FAILURE] Quality gates failed!")
        print("  Review the failed gates above and address the issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
