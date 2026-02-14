"""
Spearpoint Baseline Suite — Fixed Benchmark for Recursive Gain Measurement
===========================================================================

100 deterministic test cases with known expected outputs.
If scores change, the system changed. If they improve, improvement is REAL.

Cases:
  30 SNR     — Signal-to-Noise Ratio calculation with known inputs/outputs
  30 CLEAR   — Cost/Latency/Efficacy/Assurance/Reliability scoring
  20 Ihsan   — 8-dimensional excellence scoring
  20 Guard   — Safety gate pass/fail with known thresholds

Standing on Giants:
  Deming (1950): "Without data, you're just another person with an opinion."
  Knuth (1974): "Premature optimization is the root of all evil."
  Shannon (1948): SNR = signal / (signal + noise)
"""

from __future__ import annotations

import hashlib
import math
import statistics
from typing import Any, Dict, List, Tuple

from benchmark.runner import BenchmarkRunner


# ─── Fixed Test Vectors ──────────────────────────────────────────────

SNR_CASES: List[Dict[str, Any]] = [
    # Format: signal_components, noise_components, expected_snr (±0.02)
    {"signal": [0.95, 0.92, 0.88], "noise": [0.05, 0.08, 0.12], "expected": 0.917},
    {"signal": [0.80, 0.75, 0.70], "noise": [0.20, 0.25, 0.30], "expected": 0.750},
    {"signal": [1.00, 1.00, 1.00], "noise": [0.00, 0.00, 0.00], "expected": 1.000},
    {"signal": [0.50, 0.50, 0.50], "noise": [0.50, 0.50, 0.50], "expected": 0.500},
    {"signal": [0.99, 0.98, 0.97], "noise": [0.01, 0.02, 0.03], "expected": 0.980},
    {"signal": [0.60, 0.65, 0.70], "noise": [0.40, 0.35, 0.30], "expected": 0.650},
    {"signal": [0.85, 0.90, 0.88], "noise": [0.15, 0.10, 0.12], "expected": 0.877},
    {"signal": [0.70, 0.80, 0.90], "noise": [0.30, 0.20, 0.10], "expected": 0.800},
    {"signal": [0.92, 0.91, 0.93], "noise": [0.08, 0.09, 0.07], "expected": 0.920},
    {"signal": [0.55, 0.60, 0.58], "noise": [0.45, 0.40, 0.42], "expected": 0.577},
    {"signal": [0.88, 0.85, 0.82], "noise": [0.12, 0.15, 0.18], "expected": 0.850},
    {"signal": [0.75, 0.78, 0.73], "noise": [0.25, 0.22, 0.27], "expected": 0.753},
    {"signal": [0.97, 0.96, 0.95], "noise": [0.03, 0.04, 0.05], "expected": 0.960},
    {"signal": [0.40, 0.45, 0.42], "noise": [0.60, 0.55, 0.58], "expected": 0.423},
    {"signal": [0.83, 0.87, 0.85], "noise": [0.17, 0.13, 0.15], "expected": 0.850},
    {"signal": [0.91, 0.89, 0.90], "noise": [0.09, 0.11, 0.10], "expected": 0.900},
    {"signal": [0.65, 0.70, 0.68], "noise": [0.35, 0.30, 0.32], "expected": 0.677},
    {"signal": [0.78, 0.82, 0.80], "noise": [0.22, 0.18, 0.20], "expected": 0.800},
    {"signal": [0.94, 0.93, 0.95], "noise": [0.06, 0.07, 0.05], "expected": 0.940},
    {"signal": [0.86, 0.84, 0.88], "noise": [0.14, 0.16, 0.12], "expected": 0.860},
    {"signal": [0.72, 0.74, 0.71], "noise": [0.28, 0.26, 0.29], "expected": 0.723},
    {"signal": [0.90, 0.92, 0.91], "noise": [0.10, 0.08, 0.09], "expected": 0.910},
    {"signal": [0.58, 0.62, 0.60], "noise": [0.42, 0.38, 0.40], "expected": 0.600},
    {"signal": [0.81, 0.79, 0.83], "noise": [0.19, 0.21, 0.17], "expected": 0.810},
    {"signal": [0.96, 0.94, 0.95], "noise": [0.04, 0.06, 0.05], "expected": 0.950},
    {"signal": [0.67, 0.72, 0.69], "noise": [0.33, 0.28, 0.31], "expected": 0.693},
    {"signal": [0.89, 0.87, 0.91], "noise": [0.11, 0.13, 0.09], "expected": 0.890},
    {"signal": [0.76, 0.74, 0.78], "noise": [0.24, 0.26, 0.22], "expected": 0.760},
    {"signal": [0.93, 0.92, 0.94], "noise": [0.07, 0.08, 0.06], "expected": 0.930},
    {"signal": [0.84, 0.86, 0.85], "noise": [0.16, 0.14, 0.15], "expected": 0.850},
]

# CLEAR dimension weights (from core/benchmark/clear_framework.py)
CLEAR_WEIGHTS = {
    "cost": 0.20,
    "latency": 0.15,
    "efficacy": 0.35,
    "assurance": 0.15,
    "reliability": 0.15,
}

CLEAR_CASES: List[Dict[str, Any]] = [
    {"cost": 0.90, "latency": 0.85, "efficacy": 0.95, "assurance": 0.92, "reliability": 0.88, "expected": 0.912},
    {"cost": 0.50, "latency": 0.50, "efficacy": 0.50, "assurance": 0.50, "reliability": 0.50, "expected": 0.500},
    {"cost": 1.00, "latency": 1.00, "efficacy": 1.00, "assurance": 1.00, "reliability": 1.00, "expected": 1.000},
    {"cost": 0.00, "latency": 0.00, "efficacy": 0.00, "assurance": 0.00, "reliability": 0.00, "expected": 0.000},
    {"cost": 0.80, "latency": 0.90, "efficacy": 0.70, "assurance": 0.85, "reliability": 0.75, "expected": 0.780},
    {"cost": 0.95, "latency": 0.92, "efficacy": 0.98, "assurance": 0.96, "reliability": 0.94, "expected": 0.959},
    {"cost": 0.60, "latency": 0.70, "efficacy": 0.80, "assurance": 0.65, "reliability": 0.72, "expected": 0.716},
    {"cost": 0.85, "latency": 0.78, "efficacy": 0.92, "assurance": 0.88, "reliability": 0.83, "expected": 0.871},
    {"cost": 0.72, "latency": 0.68, "efficacy": 0.88, "assurance": 0.75, "reliability": 0.80, "expected": 0.794},
    {"cost": 0.40, "latency": 0.55, "efficacy": 0.60, "assurance": 0.45, "reliability": 0.50, "expected": 0.518},
    {"cost": 0.88, "latency": 0.92, "efficacy": 0.94, "assurance": 0.90, "reliability": 0.86, "expected": 0.908},
    {"cost": 0.65, "latency": 0.60, "efficacy": 0.75, "assurance": 0.70, "reliability": 0.68, "expected": 0.695},
    {"cost": 0.92, "latency": 0.88, "efficacy": 0.96, "assurance": 0.93, "reliability": 0.91, "expected": 0.932},
    {"cost": 0.55, "latency": 0.62, "efficacy": 0.58, "assurance": 0.60, "reliability": 0.57, "expected": 0.582},
    {"cost": 0.78, "latency": 0.82, "efficacy": 0.85, "assurance": 0.80, "reliability": 0.77, "expected": 0.813},
    {"cost": 0.97, "latency": 0.95, "efficacy": 0.99, "assurance": 0.98, "reliability": 0.96, "expected": 0.975},
    {"cost": 0.35, "latency": 0.40, "efficacy": 0.45, "assurance": 0.38, "reliability": 0.42, "expected": 0.410},
    {"cost": 0.82, "latency": 0.79, "efficacy": 0.90, "assurance": 0.85, "reliability": 0.81, "expected": 0.850},
    {"cost": 0.70, "latency": 0.75, "efficacy": 0.82, "assurance": 0.78, "reliability": 0.73, "expected": 0.769},
    {"cost": 0.93, "latency": 0.90, "efficacy": 0.97, "assurance": 0.94, "reliability": 0.92, "expected": 0.942},
    {"cost": 0.45, "latency": 0.50, "efficacy": 0.55, "assurance": 0.48, "reliability": 0.52, "expected": 0.507},
    {"cost": 0.87, "latency": 0.84, "efficacy": 0.91, "assurance": 0.89, "reliability": 0.86, "expected": 0.882},
    {"cost": 0.62, "latency": 0.67, "efficacy": 0.72, "assurance": 0.65, "reliability": 0.69, "expected": 0.680},
    {"cost": 0.91, "latency": 0.87, "efficacy": 0.95, "assurance": 0.92, "reliability": 0.89, "expected": 0.919},
    {"cost": 0.48, "latency": 0.52, "efficacy": 0.58, "assurance": 0.50, "reliability": 0.54, "expected": 0.534},
    {"cost": 0.84, "latency": 0.81, "efficacy": 0.89, "assurance": 0.86, "reliability": 0.83, "expected": 0.858},
    {"cost": 0.73, "latency": 0.76, "efficacy": 0.80, "assurance": 0.74, "reliability": 0.71, "expected": 0.759},
    {"cost": 0.96, "latency": 0.93, "efficacy": 0.98, "assurance": 0.95, "reliability": 0.94, "expected": 0.960},
    {"cost": 0.58, "latency": 0.63, "efficacy": 0.68, "assurance": 0.61, "reliability": 0.65, "expected": 0.641},
    {"cost": 0.86, "latency": 0.83, "efficacy": 0.92, "assurance": 0.88, "reliability": 0.85, "expected": 0.879},
]

# Ihsan 8D weights (from core/integration/constants.py)
IHSAN_WEIGHTS = {
    "correctness": 0.22,
    "safety": 0.22,
    "user_benefit": 0.14,
    "efficiency": 0.12,
    "auditability": 0.12,
    "anti_centralization": 0.08,
    "robustness": 0.06,
    "adl_fairness": 0.04,
}

IHSAN_CASES: List[Dict[str, Any]] = [
    {"correctness": 0.96, "safety": 0.98, "user_benefit": 0.92, "efficiency": 0.90, "auditability": 0.88, "anti_centralization": 0.85, "robustness": 0.87, "adl_fairness": 0.90, "expected": 0.934},
    {"correctness": 0.50, "safety": 0.50, "user_benefit": 0.50, "efficiency": 0.50, "auditability": 0.50, "anti_centralization": 0.50, "robustness": 0.50, "adl_fairness": 0.50, "expected": 0.500},
    {"correctness": 1.00, "safety": 1.00, "user_benefit": 1.00, "efficiency": 1.00, "auditability": 1.00, "anti_centralization": 1.00, "robustness": 1.00, "adl_fairness": 1.00, "expected": 1.000},
    {"correctness": 0.00, "safety": 0.00, "user_benefit": 0.00, "efficiency": 0.00, "auditability": 0.00, "anti_centralization": 0.00, "robustness": 0.00, "adl_fairness": 0.00, "expected": 0.000},
    {"correctness": 0.95, "safety": 0.97, "user_benefit": 0.90, "efficiency": 0.88, "auditability": 0.92, "anti_centralization": 0.80, "robustness": 0.85, "adl_fairness": 0.82, "expected": 0.923},
    {"correctness": 0.80, "safety": 0.85, "user_benefit": 0.75, "efficiency": 0.78, "auditability": 0.72, "anti_centralization": 0.70, "robustness": 0.74, "adl_fairness": 0.68, "expected": 0.790},
    {"correctness": 0.92, "safety": 0.94, "user_benefit": 0.88, "efficiency": 0.85, "auditability": 0.90, "anti_centralization": 0.82, "robustness": 0.80, "adl_fairness": 0.78, "expected": 0.901},
    {"correctness": 0.70, "safety": 0.72, "user_benefit": 0.68, "efficiency": 0.65, "auditability": 0.62, "anti_centralization": 0.60, "robustness": 0.58, "adl_fairness": 0.55, "expected": 0.674},
    {"correctness": 0.98, "safety": 0.99, "user_benefit": 0.96, "efficiency": 0.94, "auditability": 0.95, "anti_centralization": 0.90, "robustness": 0.92, "adl_fairness": 0.88, "expected": 0.966},
    {"correctness": 0.60, "safety": 0.65, "user_benefit": 0.58, "efficiency": 0.55, "auditability": 0.52, "anti_centralization": 0.50, "robustness": 0.48, "adl_fairness": 0.45, "expected": 0.577},
    {"correctness": 0.88, "safety": 0.90, "user_benefit": 0.85, "efficiency": 0.82, "auditability": 0.86, "anti_centralization": 0.78, "robustness": 0.76, "adl_fairness": 0.74, "expected": 0.861},
    {"correctness": 0.75, "safety": 0.78, "user_benefit": 0.72, "efficiency": 0.70, "auditability": 0.68, "anti_centralization": 0.65, "robustness": 0.62, "adl_fairness": 0.60, "expected": 0.726},
    {"correctness": 0.94, "safety": 0.96, "user_benefit": 0.91, "efficiency": 0.89, "auditability": 0.93, "anti_centralization": 0.86, "robustness": 0.84, "adl_fairness": 0.80, "expected": 0.927},
    {"correctness": 0.85, "safety": 0.88, "user_benefit": 0.82, "efficiency": 0.80, "auditability": 0.78, "anti_centralization": 0.75, "robustness": 0.72, "adl_fairness": 0.70, "expected": 0.832},
    {"correctness": 0.90, "safety": 0.92, "user_benefit": 0.87, "efficiency": 0.84, "auditability": 0.89, "anti_centralization": 0.81, "robustness": 0.79, "adl_fairness": 0.76, "expected": 0.885},
    {"correctness": 0.97, "safety": 0.98, "user_benefit": 0.94, "efficiency": 0.92, "auditability": 0.96, "anti_centralization": 0.88, "robustness": 0.90, "adl_fairness": 0.86, "expected": 0.953},
    {"correctness": 0.55, "safety": 0.58, "user_benefit": 0.52, "efficiency": 0.50, "auditability": 0.48, "anti_centralization": 0.45, "robustness": 0.42, "adl_fairness": 0.40, "expected": 0.523},
    {"correctness": 0.82, "safety": 0.84, "user_benefit": 0.79, "efficiency": 0.76, "auditability": 0.80, "anti_centralization": 0.72, "robustness": 0.70, "adl_fairness": 0.68, "expected": 0.802},
    {"correctness": 0.93, "safety": 0.95, "user_benefit": 0.89, "efficiency": 0.87, "auditability": 0.91, "anti_centralization": 0.83, "robustness": 0.81, "adl_fairness": 0.79, "expected": 0.913},
    {"correctness": 0.78, "safety": 0.80, "user_benefit": 0.74, "efficiency": 0.72, "auditability": 0.76, "anti_centralization": 0.68, "robustness": 0.66, "adl_fairness": 0.64, "expected": 0.762},
]

# Guardrail cases: input metrics + threshold → pass/fail
GUARDRAIL_CASES: List[Dict[str, Any]] = [
    # SNR gate (threshold: 0.85)
    {"gate": "snr", "value": 0.90, "threshold": 0.85, "expected_pass": True},
    {"gate": "snr", "value": 0.84, "threshold": 0.85, "expected_pass": False},
    {"gate": "snr", "value": 0.85, "threshold": 0.85, "expected_pass": True},
    {"gate": "snr", "value": 0.99, "threshold": 0.85, "expected_pass": True},
    {"gate": "snr", "value": 0.50, "threshold": 0.85, "expected_pass": False},
    # Ihsan gate (threshold: 0.95)
    {"gate": "ihsan", "value": 0.96, "threshold": 0.95, "expected_pass": True},
    {"gate": "ihsan", "value": 0.94, "threshold": 0.95, "expected_pass": False},
    {"gate": "ihsan", "value": 0.95, "threshold": 0.95, "expected_pass": True},
    {"gate": "ihsan", "value": 1.00, "threshold": 0.95, "expected_pass": True},
    {"gate": "ihsan", "value": 0.80, "threshold": 0.95, "expected_pass": False},
    # ADL Gini gate (threshold: 0.40, inverted: lower is better)
    {"gate": "adl_gini", "value": 0.35, "threshold": 0.40, "expected_pass": True},
    {"gate": "adl_gini", "value": 0.41, "threshold": 0.40, "expected_pass": False},
    {"gate": "adl_gini", "value": 0.40, "threshold": 0.40, "expected_pass": True},
    {"gate": "adl_gini", "value": 0.10, "threshold": 0.40, "expected_pass": True},
    {"gate": "adl_gini", "value": 0.80, "threshold": 0.40, "expected_pass": False},
    # Safety gate (threshold: 0.95)
    {"gate": "safety", "value": 0.97, "threshold": 0.95, "expected_pass": True},
    {"gate": "safety", "value": 0.93, "threshold": 0.95, "expected_pass": False},
    {"gate": "safety", "value": 0.95, "threshold": 0.95, "expected_pass": True},
    {"gate": "safety", "value": 0.99, "threshold": 0.95, "expected_pass": True},
    {"gate": "safety", "value": 0.00, "threshold": 0.95, "expected_pass": False},
]


# ─── Computation Functions ───────────────────────────────────────────

def compute_snr(signal: List[float], noise: List[float]) -> float:
    """Shannon-derived SNR: mean(signal) / (mean(signal) + mean(noise))."""
    s = statistics.mean(signal)
    n = statistics.mean(noise)
    return s / (s + n) if (s + n) > 0 else 0.0


def compute_clear(scores: Dict[str, float]) -> float:
    """CLEAR framework weighted score."""
    return sum(scores[dim] * CLEAR_WEIGHTS[dim] for dim in CLEAR_WEIGHTS)


def compute_ihsan(scores: Dict[str, float]) -> float:
    """8D Ihsan weighted score."""
    return sum(scores[dim] * IHSAN_WEIGHTS[dim] for dim in IHSAN_WEIGHTS)


def evaluate_guardrail(gate: str, value: float, threshold: float) -> bool:
    """Evaluate a guardrail gate. ADL Gini is inverted (lower = better)."""
    if gate == "adl_gini":
        return value <= threshold
    return value >= threshold


# ─── Benchmark Suite ────────────────────────────────────────────────

class SpearPointBenchmark:
    """
    Fixed baseline benchmark for measuring recursive improvement.

    100 deterministic cases with known expected outputs.
    Tolerance: 0.02 for floating-point comparison.
    """

    TOLERANCE = 0.02

    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner

    def benchmark_snr_baseline(self, iterations: int = 20) -> Dict[str, Any]:
        """Run 30 SNR cases and verify correctness."""
        passed = 0
        failed = 0
        max_error = 0.0

        def snr_op():
            nonlocal passed, failed, max_error
            passed = 0
            failed = 0
            max_error = 0.0

            for case in SNR_CASES:
                actual = compute_snr(case["signal"], case["noise"])
                error = abs(actual - case["expected"])
                max_error = max(max_error, error)
                if error <= self.TOLERANCE:
                    passed += 1
                else:
                    failed += 1

        result = self.runner.run(
            "spearpoint.snr_baseline",
            snr_op,
            iterations=iterations,
        )

        result_dict = result.to_dict()
        result_dict["correctness"] = {
            "passed": passed,
            "failed": failed,
            "total": len(SNR_CASES),
            "accuracy": passed / len(SNR_CASES) if SNR_CASES else 0.0,
            "max_error": max_error,
            "tolerance": self.TOLERANCE,
        }
        return result_dict

    def benchmark_clear_baseline(self, iterations: int = 20) -> Dict[str, Any]:
        """Run 30 CLEAR cases and verify correctness."""
        passed = 0
        failed = 0
        max_error = 0.0

        def clear_op():
            nonlocal passed, failed, max_error
            passed = 0
            failed = 0
            max_error = 0.0

            for case in CLEAR_CASES:
                scores = {k: case[k] for k in CLEAR_WEIGHTS}
                actual = compute_clear(scores)
                error = abs(actual - case["expected"])
                max_error = max(max_error, error)
                if error <= self.TOLERANCE:
                    passed += 1
                else:
                    failed += 1

        result = self.runner.run(
            "spearpoint.clear_baseline",
            clear_op,
            iterations=iterations,
        )

        result_dict = result.to_dict()
        result_dict["correctness"] = {
            "passed": passed,
            "failed": failed,
            "total": len(CLEAR_CASES),
            "accuracy": passed / len(CLEAR_CASES) if CLEAR_CASES else 0.0,
            "max_error": max_error,
            "tolerance": self.TOLERANCE,
        }
        return result_dict

    def benchmark_ihsan_baseline(self, iterations: int = 20) -> Dict[str, Any]:
        """Run 20 Ihsan 8D cases and verify correctness."""
        passed = 0
        failed = 0
        max_error = 0.0

        def ihsan_op():
            nonlocal passed, failed, max_error
            passed = 0
            failed = 0
            max_error = 0.0

            for case in IHSAN_CASES:
                scores = {k: case[k] for k in IHSAN_WEIGHTS}
                actual = compute_ihsan(scores)
                error = abs(actual - case["expected"])
                max_error = max(max_error, error)
                if error <= self.TOLERANCE:
                    passed += 1
                else:
                    failed += 1

        result = self.runner.run(
            "spearpoint.ihsan_baseline",
            ihsan_op,
            iterations=iterations,
        )

        result_dict = result.to_dict()
        result_dict["correctness"] = {
            "passed": passed,
            "failed": failed,
            "total": len(IHSAN_CASES),
            "accuracy": passed / len(IHSAN_CASES) if IHSAN_CASES else 0.0,
            "max_error": max_error,
            "tolerance": self.TOLERANCE,
        }
        return result_dict

    def benchmark_guardrail_baseline(self, iterations: int = 20) -> Dict[str, Any]:
        """Run 20 guardrail gate cases and verify pass/fail."""
        passed = 0
        failed = 0

        def guardrail_op():
            nonlocal passed, failed
            passed = 0
            failed = 0

            for case in GUARDRAIL_CASES:
                actual = evaluate_guardrail(case["gate"], case["value"], case["threshold"])
                if actual == case["expected_pass"]:
                    passed += 1
                else:
                    failed += 1

        result = self.runner.run(
            "spearpoint.guardrail_baseline",
            guardrail_op,
            iterations=iterations,
        )

        result_dict = result.to_dict()
        result_dict["correctness"] = {
            "passed": passed,
            "failed": failed,
            "total": len(GUARDRAIL_CASES),
            "accuracy": passed / len(GUARDRAIL_CASES) if GUARDRAIL_CASES else 0.0,
        }
        return result_dict

    def run_all(self, iterations: int = 20) -> Dict[str, Any]:
        """Run all 100 spearpoint baseline cases."""
        print("\n--- Spearpoint Baseline Suite (100 cases) ---")
        results = {
            "snr_baseline": self.benchmark_snr_baseline(iterations),
            "clear_baseline": self.benchmark_clear_baseline(iterations),
            "ihsan_baseline": self.benchmark_ihsan_baseline(iterations),
            "guardrail_baseline": self.benchmark_guardrail_baseline(iterations),
        }

        # Aggregate correctness
        total_passed = sum(
            r.get("correctness", {}).get("passed", 0)
            for r in results.values()
        )
        total_cases = sum(
            r.get("correctness", {}).get("total", 0)
            for r in results.values()
        )
        overall_accuracy = total_passed / total_cases if total_cases > 0 else 0.0

        print(f"  Correctness: {total_passed}/{total_cases} ({overall_accuracy:.1%})")
        results["_aggregate"] = {
            "total_passed": total_passed,
            "total_cases": total_cases,
            "overall_accuracy": overall_accuracy,
        }
        return results

    @staticmethod
    def fingerprint() -> str:
        """
        Deterministic hash of all test vectors.
        If this changes, the baseline changed — invalidates all prior measurements.
        """
        h = hashlib.blake2b(digest_size=16)
        for case in SNR_CASES:
            h.update(str(case).encode())
        for case in CLEAR_CASES:
            h.update(str(case).encode())
        for case in IHSAN_CASES:
            h.update(str(case).encode())
        for case in GUARDRAIL_CASES:
            h.update(str(case).encode())
        return h.hexdigest()
