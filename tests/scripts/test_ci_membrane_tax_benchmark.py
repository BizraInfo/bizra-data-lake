from __future__ import annotations

import sys
import types

import scripts.ci_membrane_tax_benchmark as membrane_tax
from scripts.ci_membrane_tax_benchmark import compute_membrane_tax, evaluate_report


def test_compute_membrane_tax_summarizes_governance_vs_work() -> None:
    report = {
        "stages": [
            {"rss_growth_mb": 10.0},
            {"rss_growth_mb": 5.5},
            {"rss_growth_mb": 2.0},
        ],
        "benchmark_results": {
            "vrg_receipt_build_ms": 12.0,
            "eventbus_emission_ms": 3.0,
            "got_bridge_reason_ms": 50.0,
            "node0_breathe_ms": 25.0,
        },
    }

    summary = compute_membrane_tax(report)

    assert summary["governance_tax_ms"] == 15.0
    assert summary["task_work_ms"] == 75.0
    assert summary["governance_tax_ratio"] == 0.1667
    assert summary["rss_growth_mb"] == 17.5


def test_compute_membrane_tax_clamps_negative_governance_inputs() -> None:
    report = {
        "stages": [
            {"rss_growth_mb": 1.0},
            {"rss_growth_mb": 2.0},
        ],
        "benchmark_results": {
            "vrg_receipt_build_ms": 10.0,
            "eventbus_emission_ms": 0.0,
            "got_bridge_reason_ms": 20.0,
            "node0_breathe_ms": 30.0,
        },
    }

    summary = compute_membrane_tax(report)

    assert summary["governance_tax_ms"] == 10.0
    assert summary["task_work_ms"] == 50.0
    assert summary["governance_tax_ratio"] == 0.1667
    assert summary["rss_growth_mb"] == 3.0


def test_evaluate_report_passes_when_within_default_gates() -> None:
    report = {
        "membrane_tax": {
            "governance_tax_ms": 25.0,
            "governance_tax_ratio": 0.1,
            "rss_growth_mb": 32.0,
        }
    }

    verdict = evaluate_report(report, strict=False)

    assert verdict["passed"] is True
    assert verdict["failed_metrics"] == []


def test_evaluate_report_fails_when_strict_gate_exceeded() -> None:
    report = {
        "membrane_tax": {
            "governance_tax_ms": 120.0,
            "governance_tax_ratio": 0.25,
            "rss_growth_mb": 200.0,
        }
    }

    verdict = evaluate_report(report, strict=True)

    assert verdict["passed"] is False
    assert "governance_tax_ms" in verdict["failed_metrics"]
    assert "governance_tax_ratio" in verdict["failed_metrics"]


def test_got_bridge_compatibility_falls_back_to_async_reason(monkeypatch) -> None:
    class _DummyResult:
        converged = True

    class _DummyBridge:
        async def reason(self, query: str, context: dict | None = None) -> _DummyResult:
            return _DummyResult()

    def _raise_interface_mismatch() -> dict:
        raise AttributeError("reason_and_verify")

    monkeypatch.setattr(membrane_tax, "benchmark_got_bridge", _raise_interface_mismatch)
    monkeypatch.setitem(
        sys.modules,
        "core.reasoning.got_bridge",
        types.SimpleNamespace(GoTBridge=_DummyBridge),
    )

    result = membrane_tax._benchmark_got_bridge_compatible()

    assert result["got_bridge_available"] is True
    assert result["got_bridge_mode"] == "async_reason"
    assert result["got_bridge_converged"] is True
    assert result["got_bridge_reason_ms"] >= 0.0


def test_normalize_benchmark_results_records_negative_metrics() -> None:
    normalized, sanity = membrane_tax._normalize_benchmark_results(
        {
            "vrg_receipt_build_ms": 10.0,
            "eventbus_emission_ms": -12.5,
            "got_bridge_reason_ms": 8.0,
            "node0_breathe_ms": 30.0,
        }
    )

    assert normalized["eventbus_emission_ms"] == 0.0
    assert sanity["clamped_negative_metrics"] == {"eventbus_emission_ms": -12.5}
