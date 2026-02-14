"""Tests for SpearPoint Baseline benchmark suite — verifies fixed test vectors."""

from benchmark.runner import BenchmarkRunner
from benchmark.suites.spearpoint import (
    CLEAR_CASES,
    CLEAR_WEIGHTS,
    GUARDRAIL_CASES,
    IHSAN_CASES,
    IHSAN_WEIGHTS,
    SNR_CASES,
    SpearPointBenchmark,
    compute_clear,
    compute_ihsan,
    compute_snr,
    evaluate_guardrail,
)


class TestComputeFunctions:
    """Verify the computation functions match expected outputs."""

    def test_snr_perfect_signal(self):
        assert compute_snr([1.0, 1.0, 1.0], [0.0, 0.0, 0.0]) == 1.0

    def test_snr_equal_signal_noise(self):
        assert compute_snr([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) == 0.5

    def test_snr_zero_everything(self):
        assert compute_snr([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]) == 0.0

    def test_clear_all_ones(self):
        scores = {k: 1.0 for k in CLEAR_WEIGHTS}
        assert abs(compute_clear(scores) - 1.0) < 0.001

    def test_clear_all_zeros(self):
        scores = {k: 0.0 for k in CLEAR_WEIGHTS}
        assert abs(compute_clear(scores)) < 0.001

    def test_clear_weights_sum_to_one(self):
        assert abs(sum(CLEAR_WEIGHTS.values()) - 1.0) < 0.001

    def test_ihsan_all_ones(self):
        scores = {k: 1.0 for k in IHSAN_WEIGHTS}
        assert abs(compute_ihsan(scores) - 1.0) < 0.001

    def test_ihsan_all_zeros(self):
        scores = {k: 0.0 for k in IHSAN_WEIGHTS}
        assert abs(compute_ihsan(scores)) < 0.001

    def test_ihsan_weights_sum_to_one(self):
        assert abs(sum(IHSAN_WEIGHTS.values()) - 1.0) < 0.001

    def test_guardrail_snr_pass(self):
        assert evaluate_guardrail("snr", 0.90, 0.85) is True

    def test_guardrail_snr_fail(self):
        assert evaluate_guardrail("snr", 0.80, 0.85) is False

    def test_guardrail_snr_boundary(self):
        assert evaluate_guardrail("snr", 0.85, 0.85) is True

    def test_guardrail_gini_pass(self):
        assert evaluate_guardrail("adl_gini", 0.35, 0.40) is True

    def test_guardrail_gini_fail(self):
        assert evaluate_guardrail("adl_gini", 0.45, 0.40) is False

    def test_guardrail_gini_boundary(self):
        assert evaluate_guardrail("adl_gini", 0.40, 0.40) is True


class TestFixedVectors:
    """Verify all 100 test vectors produce correct results."""

    TOLERANCE = 0.02

    def test_snr_case_count(self):
        assert len(SNR_CASES) == 30

    def test_clear_case_count(self):
        assert len(CLEAR_CASES) == 30

    def test_ihsan_case_count(self):
        assert len(IHSAN_CASES) == 20

    def test_guardrail_case_count(self):
        assert len(GUARDRAIL_CASES) == 20

    def test_total_100_cases(self):
        total = len(SNR_CASES) + len(CLEAR_CASES) + len(IHSAN_CASES) + len(GUARDRAIL_CASES)
        assert total == 100

    def test_all_snr_cases(self):
        for i, case in enumerate(SNR_CASES):
            actual = compute_snr(case["signal"], case["noise"])
            error = abs(actual - case["expected"])
            assert error <= self.TOLERANCE, (
                f"SNR case {i}: expected {case['expected']}, got {actual}, error {error}"
            )

    def test_all_clear_cases(self):
        for i, case in enumerate(CLEAR_CASES):
            scores = {k: case[k] for k in CLEAR_WEIGHTS}
            actual = compute_clear(scores)
            error = abs(actual - case["expected"])
            assert error <= self.TOLERANCE, (
                f"CLEAR case {i}: expected {case['expected']}, got {actual}, error {error}"
            )

    def test_all_ihsan_cases(self):
        for i, case in enumerate(IHSAN_CASES):
            scores = {k: case[k] for k in IHSAN_WEIGHTS}
            actual = compute_ihsan(scores)
            error = abs(actual - case["expected"])
            assert error <= self.TOLERANCE, (
                f"Ihsan case {i}: expected {case['expected']}, got {actual}, error {error}"
            )

    def test_all_guardrail_cases(self):
        for i, case in enumerate(GUARDRAIL_CASES):
            actual = evaluate_guardrail(case["gate"], case["value"], case["threshold"])
            assert actual == case["expected_pass"], (
                f"Guardrail case {i}: gate={case['gate']}, "
                f"value={case['value']}, expected={case['expected_pass']}, got={actual}"
            )


class TestBenchmarkSuite:
    """Test the benchmark runner integration."""

    def test_run_all_returns_results(self):
        runner = BenchmarkRunner(warmup=1, verbose=False)
        bench = SpearPointBenchmark(runner)
        results = bench.run_all(iterations=3)
        assert "snr_baseline" in results
        assert "clear_baseline" in results
        assert "ihsan_baseline" in results
        assert "guardrail_baseline" in results
        assert "_aggregate" in results

    def test_100_percent_accuracy(self):
        runner = BenchmarkRunner(warmup=1, verbose=False)
        bench = SpearPointBenchmark(runner)
        results = bench.run_all(iterations=1)
        agg = results["_aggregate"]
        assert agg["total_cases"] == 100
        assert agg["overall_accuracy"] == 1.0

    def test_fingerprint_deterministic(self):
        fp1 = SpearPointBenchmark.fingerprint()
        fp2 = SpearPointBenchmark.fingerprint()
        assert fp1 == fp2
        assert len(fp1) == 32  # blake2b with 16 digest bytes = 32 hex chars
