from __future__ import annotations

import json
from pathlib import Path

from scripts.ops.masterpiece_program_board import (
    build_program_board,
    load_config,
    render_markdown,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _seed_config(root: Path) -> Path:
    config_path = root / "config" / "masterpiece_program_board.json"
    _write_json(
        config_path,
        {
            "program": {
                "id": "masterpiece_program_board",
                "version": "1.0.0",
            },
            "source_artifacts": {
                "genesis_execution_framework": "config/genesis_execution_framework.json",
                "mastery_framework": "config/mastery_framework_roadmap.json",
                "optimization_blueprint": "config/bizra_unified_optimization_blueprint.json",
            },
            "score_weights": {
                "blueprint": 0.35,
                "autonomous": 0.25,
                "snr": 0.15,
                "empirical": 0.25,
            },
            "thresholds": {
                "min_board_score": 0.93,
                "require_blueprint_gate": True,
                "require_autonomous_gate": True,
                "require_canonical_empirical_gate": True,
            },
            "max_top_workstreams": 5,
            "giants_protocol": [
                "Shannon:SNR maximization",
                "PMBOK:lifecycle governance",
            ],
        },
    )
    return config_path


def _seed_source_artifacts(root: Path) -> None:
    _write_json(
        root / "config" / "genesis_execution_framework.json",
        {
            "workstreams": [
                {
                    "id": "ws1",
                    "priority": "P0",
                    "name": "terminal_contract_completion",
                    "deliverables": ["event_native_timeline"],
                    "acceptance_gates": ["terminal_lock_checklist"],
                }
            ]
        },
    )
    _write_json(
        root / "config" / "mastery_framework_roadmap.json",
        {
            "waves": [
                {
                    "name": "Foundation Closure",
                    "items": [
                        {
                            "priority": "P1",
                            "title": "Single quality spine in CI",
                            "owner": "JARVIS",
                            "success_gate": "one_path",
                            "snr_gain": 0.015,
                        }
                    ],
                }
            ]
        },
    )
    _write_json(
        root / "config" / "bizra_unified_optimization_blueprint.json",
        {
            "workstreams": [
                {
                    "id": "ws2",
                    "name": "ci_cd_evidence_and_release_integrity",
                    "priority": "P0",
                    "focus": ["proof_pack_generation"],
                }
            ]
        },
    )


def _blueprint_report(gate_passed: bool = True) -> dict:
    return {
        "program_id": "elite_fullstack_masterpiece",
        "gate_passed": gate_passed,
        "weighted_score": 0.98 if gate_passed else 0.62,
        "snr": {"normalized": 0.99 if gate_passed else 0.41},
        "control_planes": [
            {
                "id": "security",
                "score": 0.98 if gate_passed else 0.61,
                "status": "PASS" if gate_passed else "BLOCKED",
                "failed_checks": 0 if gate_passed else 2,
                "check_count": 5,
            }
        ],
        "interdisciplinary_lenses": {
            "architecture": 1.0 if gate_passed else 0.6,
            "devops": 1.0 if gate_passed else 0.6,
            "security": 1.0 if gate_passed else 0.5,
        },
        "ethical_integrity_posture": {
            "ihsan": {
                "score": 0.99 if gate_passed else 0.7,
                "status": "PASS" if gate_passed else "BLOCKED",
            },
            "adl": {
                "score": 0.98 if gate_passed else 0.68,
                "status": "PASS" if gate_passed else "BLOCKED",
            },
            "amanah": {
                "score": 0.97 if gate_passed else 0.72,
                "status": "PASS" if gate_passed else "BLOCKED",
            },
            "overall": {
                "score": 0.98 if gate_passed else 0.7,
                "status": "PASS" if gate_passed else "BLOCKED",
            },
        },
        "implementation_strategy": {
            "current_phase": (
                "promote_release_evidence"
                if gate_passed
                else "stabilize_truth_and_trust"
            ),
            "phase_objective": (
                "Promote synchronized release evidence."
                if gate_passed
                else "Resolve blueprint blockers first."
            ),
        },
        "risk_register": (
            []
            if gate_passed
            else [
                {
                    "risk_id": "R001",
                    "priority": "P0",
                    "dimension": "devops",
                    "owner": "devops",
                    "cascade": "Broken orchestration allows incomplete evidence to ship.",
                }
            ]
        ),
        "standing_on_giants_protocol": [
            "Shannon:SNR maximization",
            "Deming:PDCA quality discipline",
        ],
        "graph_of_thought": {
            "nodes": [
                {"id": "files", "score": 1.0, "status": "PASS"},
                {
                    "id": "release_readiness",
                    "score": 1.0 if gate_passed else 0.0,
                    "status": "PASS" if gate_passed else "BLOCKED",
                },
            ],
            "edges": [{"from": "files", "to": "release_readiness"}],
        },
        "optimization_roadmap": (
            []
            if gate_passed
            else [
                {
                    "priority": "P0",
                    "owner": "devops",
                    "check": "pipeline:missing",
                    "action": "Fix CI/CD orchestration and job dependency chain.",
                }
            ]
        ),
    }


def _autonomous_report(gate_passed: bool = True) -> dict:
    return {
        "program": {"id": "autonomous_engine_gate"},
        "gate_passed": gate_passed,
        "metrics": {
            "score": 0.94 if gate_passed else 0.52,
            "prompt_snr": 0.95 if gate_passed else 0.35,
            "rlvr_snr": 0.93 if gate_passed else 0.30,
        },
        "standing_on_giants_protocol": [
            "Besta:graph-of-thought reasoning topology",
            "PMBOK:lifecycle governance",
        ],
        "graph_of_thought": {
            "nodes": [
                {"id": "prompt_engine", "score": 0.95 if gate_passed else 0.35},
                {"id": "release_decision", "score": 0.94 if gate_passed else 0.52},
            ],
            "edges": [{"from": "prompt_engine", "to": "release_decision"}],
        },
        "autonomous_next_step": {
            "priority": "P0" if gate_passed else "P1",
            "owner": "autonomous-engine",
            "action": (
                "Promote autonomous profile to protected release pipeline."
                if gate_passed
                else "Recalibrate autonomous gate."
            ),
        },
    }


def _canonical_empirical_report(gate_passed: bool = True) -> dict:
    return {
        "program": {"id": "canonical_empirical_validation"},
        "gate_passed": gate_passed,
        "canonical_status": "CANONICAL" if gate_passed else "DEGRADED",
        "metrics": {
            "score": 1.0 if gate_passed else 0.72,
            "empirical_pass_rate": 1.0 if gate_passed else 0.8,
        },
        "standing_on_giants_protocol": [
            "Lamport:compositional invariant verification",
            "Al-Ghazali:Ihsan as hard floor",
        ],
        "graph_of_thought": {
            "nodes": [
                {
                    "id": "canonical_empirical_status",
                    "score": 1.0 if gate_passed else 0.72,
                    "status": "PASS" if gate_passed else "DEGRADED",
                },
                {
                    "id": "flagship_metabolism",
                    "score": 1.0 if gate_passed else 0.0,
                    "status": "PASS" if gate_passed else "BLOCKED",
                },
            ],
            "edges": [
                {
                    "from": "flagship_metabolism",
                    "to": "canonical_empirical_status",
                }
            ],
        },
        "autonomous_next_step": {
            "priority": "P0" if gate_passed else "P1",
            "owner": "release-evidence" if gate_passed else "validation-lane",
            "action": (
                "Promote canonical empirical packet into protected CI and release evidence artifacts."
                if gate_passed
                else "Repair failing empirical proof planes."
            ),
        },
    }


def test_masterpiece_program_board_passes_and_emits_graph(tmp_path: Path) -> None:
    config_path = _seed_config(tmp_path)
    _seed_source_artifacts(tmp_path)
    config = load_config(config_path)

    board = build_program_board(
        blueprint_report=_blueprint_report(True),
        autonomous_report=_autonomous_report(True),
        canonical_empirical_report=_canonical_empirical_report(True),
        config=config,
        genesis_execution_framework=json.loads(
            (tmp_path / "config" / "genesis_execution_framework.json").read_text(
                encoding="utf-8"
            )
        ),
        mastery_framework=json.loads(
            (tmp_path / "config" / "mastery_framework_roadmap.json").read_text(
                encoding="utf-8"
            )
        ),
        optimization_blueprint=json.loads(
            (
                tmp_path / "config" / "bizra_unified_optimization_blueprint.json"
            ).read_text(encoding="utf-8")
        ),
    )

    assert board["gate_passed"] is True
    assert board["metrics"]["board_score"] >= 0.90
    assert board["metrics"]["empirical_score"] >= 1.0
    assert board["metrics"]["composite_snr"] >= 0.90
    assert board["ethical_integrity_posture"]["overall"]["status"] == "PASS"
    assert (
        board["implementation_strategy"]["current_phase"] == "promote_release_evidence"
    )
    node_ids = {node["id"] for node in board["graph_of_thought"]["nodes"]}
    assert "blueprint:release_readiness" in node_ids
    assert "autonomous:release_decision" in node_ids
    assert "empirical:canonical_empirical_status" in node_ids
    assert "board:program_board" in node_ids
    assert board["top_workstreams"][0]["priority"] == "P0"
    assert (
        "Besta:graph-of-thought reasoning topology"
        in board["standing_on_giants_protocol"]
    )
    assert board["source_reports"]["canonical_empirical"]["status"] == "CANONICAL"


def test_masterpiece_program_board_fails_when_blueprint_gate_fails(
    tmp_path: Path,
) -> None:
    config_path = _seed_config(tmp_path)
    _seed_source_artifacts(tmp_path)
    config = load_config(config_path)

    board = build_program_board(
        blueprint_report=_blueprint_report(False),
        autonomous_report=_autonomous_report(True),
        canonical_empirical_report=_canonical_empirical_report(True),
        config=config,
        genesis_execution_framework=json.loads(
            (tmp_path / "config" / "genesis_execution_framework.json").read_text(
                encoding="utf-8"
            )
        ),
        mastery_framework=json.loads(
            (tmp_path / "config" / "mastery_framework_roadmap.json").read_text(
                encoding="utf-8"
            )
        ),
        optimization_blueprint=json.loads(
            (
                tmp_path / "config" / "bizra_unified_optimization_blueprint.json"
            ).read_text(encoding="utf-8")
        ),
    )

    assert board["gate_passed"] is False
    assert board["constraints"]["blueprint_gate"] is False
    assert board["autonomous_next_step"]["priority"] == "P0"
    assert "Fix CI/CD orchestration" in board["autonomous_next_step"]["action"]
    assert board["risk_register"][0]["priority"] == "P0"


def test_masterpiece_program_board_fails_when_empirical_gate_fails(
    tmp_path: Path,
) -> None:
    config_path = _seed_config(tmp_path)
    _seed_source_artifacts(tmp_path)
    config = load_config(config_path)

    board = build_program_board(
        blueprint_report=_blueprint_report(True),
        autonomous_report=_autonomous_report(True),
        canonical_empirical_report=_canonical_empirical_report(False),
        config=config,
        genesis_execution_framework=json.loads(
            (tmp_path / "config" / "genesis_execution_framework.json").read_text(
                encoding="utf-8"
            )
        ),
        mastery_framework=json.loads(
            (tmp_path / "config" / "mastery_framework_roadmap.json").read_text(
                encoding="utf-8"
            )
        ),
        optimization_blueprint=json.loads(
            (
                tmp_path / "config" / "bizra_unified_optimization_blueprint.json"
            ).read_text(encoding="utf-8")
        ),
    )

    assert board["gate_passed"] is False
    assert board["constraints"]["canonical_empirical_gate"] is False
    assert board["autonomous_next_step"]["priority"] == "P1"
    assert (
        "Repair failing empirical proof planes"
        in board["autonomous_next_step"]["action"]
    )


def test_masterpiece_program_board_markdown_contains_next_step(tmp_path: Path) -> None:
    config_path = _seed_config(tmp_path)
    _seed_source_artifacts(tmp_path)
    config = load_config(config_path)

    board = build_program_board(
        blueprint_report=_blueprint_report(True),
        autonomous_report=_autonomous_report(True),
        canonical_empirical_report=_canonical_empirical_report(True),
        config=config,
        genesis_execution_framework=json.loads(
            (tmp_path / "config" / "genesis_execution_framework.json").read_text(
                encoding="utf-8"
            )
        ),
        mastery_framework=json.loads(
            (tmp_path / "config" / "mastery_framework_roadmap.json").read_text(
                encoding="utf-8"
            )
        ),
        optimization_blueprint=json.loads(
            (
                tmp_path / "config" / "bizra_unified_optimization_blueprint.json"
            ).read_text(encoding="utf-8")
        ),
    )
    rendered = render_markdown(board)

    assert "BIZRA Masterpiece Program Board" in rendered
    assert "Autonomous Next Step" in rendered
    assert "Ethical Integrity" in rendered
    assert "Implementation Strategy" in rendered
    assert "terminal_contract_completion" in rendered
    assert "Empirical score" in rendered
