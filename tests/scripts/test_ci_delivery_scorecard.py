from __future__ import annotations

import json
from pathlib import Path

from scripts.ci_delivery_scorecard import (
    build_scorecard_markdown,
    load_program,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _minimal_program() -> dict:
    return {
        "program_id": "bizra-delivery",
        "version": "2026-03-22",
        "status": "active",
        "north_star": "Protect the canonical spine.",
        "operating_graph": ["research_corpus", "constitution", "persisted_proof"],
        "scorecard": [
            {
                "dimension": "canonical_execution",
                "status": "proven",
                "measure": "runtime_owned_organism_authority",
            }
        ],
        "workstreams": [
            {
                "id": "W1",
                "priority": "P0",
                "name": "Canonical Evidence Plane",
                "current_truth": "proven_in_core",
            }
        ],
        "roadmap": {
            "next_7_days": ["remove_remaining_latest_refs"],
            "next_30_days": ["typed_exception_taxonomy"],
            "next_60_days": ["raise_type_ratchets"],
            "next_90_days": ["publish_public_technical_note"],
        },
        "top_next_step": {
            "id": "NEXT-EXEC-001",
            "title": "Promote the blueprint into machine-driven execution artifacts",
            "reason": "Manual-only governance drifts.",
        },
        "risk_register": [
            {
                "id": "R1",
                "priority": "P0",
                "risk": "adjacent_cognition_bypasses_receipts",
                "mitigation": "receipt_native_cognition_contract",
            },
            {
                "id": "R2",
                "priority": "P2",
                "risk": "federation_claims_outrun_proof",
                "mitigation": "staged_labeling",
            },
        ],
    }


def test_load_program_reads_json(tmp_path: Path) -> None:
    path = tmp_path / "docs" / "program" / "bizra_delivery_program.json"
    payload = _minimal_program()
    _write_json(path, payload)

    loaded = load_program(path)

    assert loaded["program_id"] == "bizra-delivery"


def test_build_scorecard_markdown_renders_key_sections() -> None:
    markdown = build_scorecard_markdown(_minimal_program())

    assert "# BIZRA Delivery Scorecard" in markdown
    assert "## Scorecard" in markdown
    assert "`canonical_execution`" in markdown
    assert "## Workstreams" in markdown
    assert "`W1`" in markdown
    assert "## Next Horizons" in markdown
    assert "`remove_remaining_latest_refs`" in markdown
    assert "## Top Next Step" in markdown
    assert "## P0/P1 Risks" in markdown
    assert "`R1`" in markdown
    assert "`R2`" not in markdown


def test_build_scorecard_markdown_renders_boundary_runtime_signals() -> None:
    markdown = build_scorecard_markdown(
        _minimal_program(),
        boundary_report={
            "boundary_signal": {
                "boundary_error_receipts": 2,
                "boundary_degradations": 1,
                "boundary_retries": 1,
                "pre_boundary_ihsan_composite": 0.95,
                "post_boundary_ihsan_composite": 0.91,
                "boundary_quality_multiplier": 0.96,
            },
            "gate_verdict": {"passed": True},
        },
    )

    assert "## Live Runtime Signals" in markdown
    assert "Boundary Quality Probe" in markdown
    assert "`boundary_quality_multiplier`" in markdown
    assert "`0.9600`" in markdown
