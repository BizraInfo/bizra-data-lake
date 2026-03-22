from __future__ import annotations

import json
from pathlib import Path

from scripts.ci_delivery_program_gate import validate_delivery_program


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
        "ethical_invariants": [
            {"id": "ihsan", "rule": "Visible excellence."},
            {"id": "amanah", "rule": "Entrusted truth."},
        ],
        "workstreams": [
            {
                "id": "W1",
                "name": "Canonical Evidence Plane",
                "priority": "P0",
                "objective": "Unify evidence.",
                "current_truth": "proven",
                "dependencies": [],
                "kpis": ["event_delivery_success_rate"],
                "deliverables": ["delivery_receipts"],
                "risks": ["state_drift"],
            },
            {
                "id": "W2",
                "name": "Governed Self-Improvement",
                "priority": "P1",
                "objective": "Bounded replayable improvement.",
                "current_truth": "staged",
                "dependencies": ["W1"],
                "kpis": ["reward_verified_improvement_rate"],
                "deliverables": ["bounded_delta_replay"],
                "risks": ["opaque_rewards"],
            },
        ],
        "delivery_gates": [
            {"id": "G1", "name": "Static Quality", "checks": ["ruff"]},
            {"id": "G2", "name": "Proof and Replay", "checks": ["canonical_spearpoint"]},
        ],
        "roadmap": {
            "next_7_days": ["remove_mutable_refs"],
            "next_30_days": ["typed_exception_taxonomy"],
            "next_60_days": ["raise_type_ratchets"],
            "next_90_days": ["publish_public_technical_note"],
        },
        "scorecard": [
            {
                "dimension": "canonical_execution",
                "status": "proven",
                "measure": "runtime_owned_organism_authority",
            },
            {
                "dimension": "knowledge_governance",
                "status": "staged",
                "measure": "research_asset_registry",
            },
        ],
        "risk_register": [
            {
                "id": "R1",
                "risk": "adjacent_cognition_bypasses_receipts",
                "impact": "unverifiable_behavior",
                "mitigation": "receipt_native_cognition_contract",
                "priority": "P0",
            }
        ],
        "top_next_step": {
            "id": "NEXT-EXEC-001",
            "title": "Promote the blueprint into machine-driven execution artifacts",
            "reason": "Manual-only governance drifts.",
        },
    }


def test_delivery_program_gate_passes_for_consistent_manifest(tmp_path: Path) -> None:
    path = tmp_path / "docs" / "program" / "bizra_delivery_program.json"
    _write_json(path, _minimal_program())

    assert validate_delivery_program(path) == []


def test_delivery_program_gate_reports_bad_dependency(tmp_path: Path) -> None:
    path = tmp_path / "docs" / "program" / "bizra_delivery_program.json"
    payload = _minimal_program()
    payload["workstreams"][1]["dependencies"] = ["W9"]
    _write_json(path, payload)

    issues = validate_delivery_program(path)

    assert any("dependency 'W9'" in issue for issue in issues)


def test_delivery_program_gate_reports_invalid_scorecard_status(tmp_path: Path) -> None:
    path = tmp_path / "docs" / "program" / "bizra_delivery_program.json"
    payload = _minimal_program()
    payload["scorecard"][0]["status"] = "legendary"
    _write_json(path, payload)

    issues = validate_delivery_program(path)

    assert any("invalid status" in issue for issue in issues)
