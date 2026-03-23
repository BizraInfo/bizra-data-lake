from __future__ import annotations

from scripts.ci_benchmark_summary import render_summary


def test_render_summary_includes_both_benchmark_planes() -> None:
    canonical_report = {
        "benchmark_results": {
            "got_bridge_init_ms": 80.0,
            "got_bridge_reason_ms": 15.0,
            "vrg_receipt_build_ms": 0.1,
            "organism_boot_ms": 120.0,
            "node0_breathe_ms": 6.0,
            "eventbus_emission_ms": 0.02,
            "full_spine_ms": 1600.0,
        },
        "gate_verdict": {
            "passed": True,
            "checks": {
                "got_bridge_init_ms": {"gate": 500.0},
                "got_bridge_reason_ms": {"gate": 200.0},
                "vrg_receipt_build_ms": {"gate": 100.0},
                "organism_boot_ms": {"gate": 5000.0},
                "node0_breathe_ms": {"gate": 50.0},
                "eventbus_emission_ms": {"gate": 10.0},
                "full_spine_ms": {"gate": 6000.0},
            },
        },
    }
    membrane_report = {
        "membrane_tax": {
            "governance_tax_ms": 0.08,
            "governance_tax_ratio": 0.0036,
            "rss_growth_mb": 30.28,
        },
        "gate_verdict": {
            "passed": True,
            "checks": {
                "governance_tax_ms": {"gate": 250.0},
                "governance_tax_ratio": {"gate": 0.35},
                "rss_growth_mb": {"gate": 512.0},
            },
        },
        "benchmark_sanity": {"clamped_negative_metrics": {}},
    }

    summary = render_summary(
        canonical_report=canonical_report,
        membrane_report=membrane_report,
    )

    assert "## Performance Proof Plane" in summary
    assert "- Canonical E2E: PASS" in summary
    assert "- Membrane Tax: PASS" in summary
    assert "| got_bridge_init_ms | 80.00 | 500.00 |" in summary
    assert "| governance_tax_ratio | 0.0036 | 0.3500 |" in summary
    assert "No clamped negative metrics detected." in summary
