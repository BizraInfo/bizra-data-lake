from __future__ import annotations

from scripts.ci_release_readiness_bundle import (
    build_release_readiness_bundle,
    render_release_readiness_markdown,
)


def _minimal_program() -> dict:
    return {
        "program_id": "bizra-delivery",
        "version": "2026-03-23",
        "status": "active",
        "north_star": "Protect the canonical receipt spine.",
        "top_next_step": {
            "id": "NEXT-EXEC-002",
            "title": "Eliminate mutable release refs",
            "reason": "Reproducibility drift still weakens public proof.",
        },
        "workstreams": [
            {"id": "W1", "priority": "P0"},
            {"id": "W2", "priority": "P1"},
        ],
        "scorecard": [
            {"dimension": "canonical_execution", "status": "proven"},
            {"dimension": "release_reproducibility", "status": "partial"},
        ],
    }


def test_build_release_readiness_bundle_passes_when_all_planes_are_green() -> None:
    bundle = build_release_readiness_bundle(
        program=_minimal_program(),
        program_issues=[],
        canonical_report={
            "benchmark_results": {
                "full_spine_ms": 1200.0,
                "node0_breathe_ms": 6.0,
                "eventbus_emission_ms": 0.03,
            },
            "gate_verdict": {"passed": True, "failed_metrics": []},
        },
        membrane_report={
            "membrane_tax": {
                "governance_tax_ratio": 0.004,
                "rss_growth_mb": 42.0,
            },
            "benchmark_sanity": {"clamped_negative_metrics": {}},
            "gate_verdict": {"passed": True, "failed_metrics": []},
        },
        boundary_report={
            "boundary_signal": {
                "boundary_quality_multiplier": 0.98,
                "boundary_error_receipts": 2,
                "boundary_retries": 2,
                "boundary_degradations": 0,
            },
            "gate_verdict": {
                "passed": True,
                "checks": {
                    "degradation_receipt_emitted": True,
                    "boundary_multiplier_applied": True,
                },
            },
        },
    )

    assert bundle["overall_verdict"]["passed"] is True
    assert bundle["planes"]["delivery_program"]["passed"] is True
    assert bundle["runtime_signals"]["boundary_quality_multiplier"] == 0.98
    assert bundle["program"]["scorecard_status_counts"]["proven"] == 1


def test_build_release_readiness_bundle_marks_failed_planes() -> None:
    bundle = build_release_readiness_bundle(
        program=_minimal_program(),
        program_issues=["roadmap.next_7_days must be non-empty"],
        canonical_report={
            "benchmark_results": {},
            "gate_verdict": {"passed": False, "failed_metrics": ["full_spine_ms"]},
        },
        membrane_report={
            "membrane_tax": {},
            "benchmark_sanity": {
                "clamped_negative_metrics": {"eventbus_emission_ms": 1}
            },
            "gate_verdict": {"passed": True, "failed_metrics": []},
        },
        boundary_report={
            "boundary_signal": {},
            "gate_verdict": {
                "passed": False,
                "checks": {"boundary_multiplier_applied": False},
            },
        },
    )

    assert bundle["overall_verdict"]["passed"] is False
    assert bundle["overall_verdict"]["failed_planes"] == [
        "delivery_program",
        "canonical_e2e",
        "boundary_quality",
    ]


def test_render_release_readiness_markdown_includes_plane_table() -> None:
    bundle = build_release_readiness_bundle(
        program=_minimal_program(),
        program_issues=[],
        canonical_report={
            "benchmark_results": {
                "full_spine_ms": 1200.0,
                "node0_breathe_ms": 6.0,
                "eventbus_emission_ms": 0.03,
            },
            "gate_verdict": {"passed": True, "failed_metrics": []},
        },
        membrane_report={
            "membrane_tax": {
                "governance_tax_ratio": 0.004,
                "rss_growth_mb": 42.0,
            },
            "benchmark_sanity": {"clamped_negative_metrics": {}},
            "gate_verdict": {"passed": True, "failed_metrics": []},
        },
        boundary_report={
            "boundary_signal": {
                "boundary_quality_multiplier": 0.98,
                "boundary_error_receipts": 2,
                "boundary_retries": 2,
                "boundary_degradations": 0,
            },
            "gate_verdict": {
                "passed": True,
                "checks": {"boundary_multiplier_applied": True},
            },
        },
    )

    markdown = render_release_readiness_markdown(bundle)

    assert "# BIZRA Release Readiness Bundle" in markdown
    assert "| `delivery_program` | `PASS` | 0 issue(s) |" in markdown
    assert "| `boundary_quality_multiplier` | `0.9800` |" in markdown
    assert "## Top Next Step" in markdown
