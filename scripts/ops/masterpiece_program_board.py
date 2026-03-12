"""
Masterpiece Program Board

Synthesizes the repo's strongest execution control planes into one machine-readable
program board:
1. Elite full-stack blueprint audit
2. Autonomous engine gate
3. Genesis execution framework
4. Unified mastery framework
5. Unified optimization blueprint

The output is a single artifact with:
- board_score
- composite_snr
- empirical_score
- graph_of_thought
- standing_on_giants_protocol
- interdisciplinary lenses
- prioritized workstreams
- autonomous next step
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BoardConfig:
    source_artifacts: dict[str, Path]
    score_weights: dict[str, float]
    min_board_score: float
    require_blueprint_gate: bool
    require_autonomous_gate: bool
    require_canonical_empirical_gate: bool
    max_top_workstreams: int
    giants_protocol: list[str]
    program: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _normalize_weights(raw: dict[str, Any]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for key, value in raw.items():
        try:
            parsed[key] = max(0.0, float(value))
        except (TypeError, ValueError):
            parsed[key] = 0.0
    total = sum(parsed.values())
    if total <= 0.0:
        return {
            "blueprint": 0.35,
            "autonomous": 0.25,
            "snr": 0.15,
            "empirical": 0.25,
        }
    return {key: value / total for key, value in parsed.items()}


def load_config(path: Path) -> BoardConfig:
    payload = _load_json(path)
    source_artifacts_raw = payload.get("source_artifacts") or {}
    source_artifacts = {
        key: Path(str(value))
        for key, value in source_artifacts_raw.items()
    }
    thresholds = payload.get("thresholds") or {}
    return BoardConfig(
        source_artifacts=source_artifacts,
        score_weights=_normalize_weights(payload.get("score_weights") or {}),
        min_board_score=float(thresholds.get("min_board_score", 0.90)),
        require_blueprint_gate=bool(thresholds.get("require_blueprint_gate", True)),
        require_autonomous_gate=bool(thresholds.get("require_autonomous_gate", True)),
        require_canonical_empirical_gate=bool(
            thresholds.get("require_canonical_empirical_gate", True)
        ),
        max_top_workstreams=max(1, int(payload.get("max_top_workstreams", 10))),
        giants_protocol=[str(x) for x in (payload.get("giants_protocol") or [])],
        program=payload.get("program") or {},
    )


def _priority_rank(priority: str) -> int:
    return {"P0": 0, "P1": 1, "P2": 2, "P3": 3}.get(priority, 9)


def _dedupe_workstreams(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        key = (
            str(item.get("priority", "")),
            str(item.get("title", item.get("action", ""))).strip().lower(),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _extract_genesis_workstreams(payload: dict[str, Any]) -> list[dict[str, Any]]:
    workstreams: list[dict[str, Any]] = []
    for item in payload.get("workstreams") or []:
        workstreams.append(
            {
                "source": "genesis_execution_framework",
                "id": item.get("id"),
                "priority": str(item.get("priority", "P2")),
                "owner": "program-management",
                "title": str(item.get("name", item.get("id", "genesis_workstream"))),
                "detail": ", ".join(str(x) for x in (item.get("deliverables") or [])),
                "success_gate": ", ".join(
                    str(x) for x in (item.get("acceptance_gates") or [])
                ),
            }
        )
    return workstreams


def _extract_mastery_workstreams(payload: dict[str, Any]) -> list[dict[str, Any]]:
    workstreams: list[dict[str, Any]] = []
    for wave in payload.get("waves") or []:
        for item in wave.get("items") or []:
            workstreams.append(
                {
                    "source": "mastery_framework",
                    "id": item.get("title"),
                    "priority": str(item.get("priority", "P2")),
                    "owner": str(item.get("owner", "team")),
                    "title": str(item.get("title", "mastery_item")),
                    "detail": f"wave={wave.get('name', wave.get('id', 'wave'))}",
                    "success_gate": str(item.get("success_gate", "")),
                    "snr_gain": float(item.get("snr_gain", 0.0) or 0.0),
                }
            )
    return workstreams


def _extract_optimization_workstreams(payload: dict[str, Any]) -> list[dict[str, Any]]:
    workstreams: list[dict[str, Any]] = []
    for item in payload.get("workstreams") or []:
        workstreams.append(
            {
                "source": "optimization_blueprint",
                "id": item.get("id"),
                "priority": str(item.get("priority", "P2")),
                "owner": "systems-integration",
                "title": str(item.get("name", item.get("id", "optimization_workstream"))),
                "detail": ", ".join(str(x) for x in (item.get("focus") or [])),
                "success_gate": "",
            }
        )
    return workstreams


def _extract_blueprint_roadmap(report: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in report.get("optimization_roadmap") or []:
        items.append(
            {
                "source": "elite_blueprint_audit",
                "id": row.get("check"),
                "priority": str(row.get("priority", "P2")),
                "owner": str(row.get("owner", "team")),
                "title": str(row.get("action", row.get("check", "blueprint_action"))),
                "detail": str(row.get("check", "")),
                "success_gate": "failed check resolved",
            }
        )
    return items


def _extract_autonomous_action(report: dict[str, Any]) -> list[dict[str, Any]]:
    next_step = report.get("autonomous_next_step") or {}
    if not next_step:
        return []
    return [
        {
            "source": "autonomous_engine_gate",
            "id": "autonomous_next_step",
            "priority": str(next_step.get("priority", "P1")),
            "owner": str(next_step.get("owner", "autonomous-engine")),
            "title": str(next_step.get("action", "Advance autonomous engine")),
            "detail": "",
            "success_gate": "autonomous gate satisfied",
        }
    ]


def _extract_empirical_action(report: dict[str, Any]) -> list[dict[str, Any]]:
    next_step = report.get("autonomous_next_step") or {}
    if not next_step:
        return []
    return [
        {
            "source": "canonical_empirical_validation",
            "id": "canonical_empirical_next_step",
            "priority": str(next_step.get("priority", "P1")),
            "owner": str(next_step.get("owner", "release-evidence")),
            "title": str(
                next_step.get("action", "Promote canonical empirical evidence lane")
            ),
            "detail": f"status={report.get('canonical_status', 'UNKNOWN')}",
            "success_gate": "canonical empirical validation satisfied",
        }
    ]


def _composite_snr(blueprint_report: dict[str, Any], autonomous_report: dict[str, Any]) -> float:
    blueprint_snr = float((blueprint_report.get("snr") or {}).get("normalized", 0.0))
    metrics = autonomous_report.get("metrics") or {}
    prompt_snr = float(metrics.get("prompt_snr", 0.0))
    rlvr_snr = float(metrics.get("rlvr_snr", 0.0))
    return round((blueprint_snr + prompt_snr + rlvr_snr) / 3.0, 4)


def _build_graph_of_thought(
    blueprint_report: dict[str, Any],
    autonomous_report: dict[str, Any],
    canonical_empirical_report: dict[str, Any],
    top_workstreams: list[dict[str, Any]],
    board_score: float,
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for node in (blueprint_report.get("graph_of_thought") or {}).get("nodes") or []:
        nodes.append(
            {
                "id": f"blueprint:{node.get('id')}",
                "score": node.get("score", 0.0),
                "status": node.get("status", "UNKNOWN"),
            }
        )
    for edge in (blueprint_report.get("graph_of_thought") or {}).get("edges") or []:
        edges.append(
            {
                "from": f"blueprint:{edge.get('from')}",
                "to": f"blueprint:{edge.get('to')}",
            }
        )

    for node in (autonomous_report.get("graph_of_thought") or {}).get("nodes") or []:
        nodes.append(
            {
                "id": f"autonomous:{node.get('id')}",
                "score": node.get("score", 0.0),
                "status": "PASS"
                if float(node.get("score", 0.0) or 0.0) >= 0.78
                else "DEGRADED",
            }
        )
    for edge in (autonomous_report.get("graph_of_thought") or {}).get("edges") or []:
        edges.append(
            {
                "from": f"autonomous:{edge.get('from')}",
                "to": f"autonomous:{edge.get('to')}",
            }
        )

    for node in (canonical_empirical_report.get("graph_of_thought") or {}).get("nodes") or []:
        nodes.append(
            {
                "id": f"empirical:{node.get('id')}",
                "score": node.get("score", 0.0),
                "status": node.get("status", "UNKNOWN"),
            }
        )
    for edge in (canonical_empirical_report.get("graph_of_thought") or {}).get("edges") or []:
        edges.append(
            {
                "from": f"empirical:{edge.get('from')}",
                "to": f"empirical:{edge.get('to')}",
            }
        )

    nodes.extend(
        [
            {"id": "framework:genesis_execution_framework", "score": 1.0, "status": "PASS"},
            {"id": "framework:mastery_framework", "score": 1.0, "status": "PASS"},
            {"id": "framework:optimization_blueprint", "score": 1.0, "status": "PASS"},
            {
                "id": "board:signal_fusion",
                "score": board_score,
                "status": "PASS" if board_score >= 0.90 else "DEGRADED",
            },
            {
                "id": "board:program_board",
                "score": board_score,
                "status": "PASS" if board_score >= 0.90 else "DEGRADED",
            },
        ]
    )

    edges.extend(
        [
            {"from": "blueprint:release_readiness", "to": "board:signal_fusion"},
            {"from": "autonomous:release_decision", "to": "board:signal_fusion"},
            {
                "from": "empirical:canonical_empirical_status",
                "to": "board:signal_fusion",
            },
            {"from": "framework:genesis_execution_framework", "to": "board:program_board"},
            {"from": "framework:mastery_framework", "to": "board:program_board"},
            {"from": "framework:optimization_blueprint", "to": "board:program_board"},
            {"from": "board:signal_fusion", "to": "board:program_board"},
        ]
    )

    for index, workstream in enumerate(top_workstreams[:3], start=1):
        node_id = f"board:workstream:{index}"
        nodes.append(
            {
                "id": node_id,
                "score": 1.0 - (0.05 * (index - 1)),
                "status": workstream.get("priority", "P2"),
                "title": workstream.get("title"),
            }
        )
        edges.append({"from": "board:program_board", "to": node_id})

    return {"nodes": nodes, "edges": edges}


def _merge_giants(
    config: BoardConfig,
    blueprint_report: dict[str, Any],
    autonomous_report: dict[str, Any],
    canonical_empirical_report: dict[str, Any],
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for source in [
        config.giants_protocol,
        blueprint_report.get("standing_on_giants_protocol") or [],
        autonomous_report.get("standing_on_giants_protocol") or [],
        canonical_empirical_report.get("standing_on_giants_protocol") or [],
    ]:
        for item in source:
            normalized = str(item)
            if normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _build_interdisciplinary_lenses(
    blueprint_report: dict[str, Any],
    autonomous_report: dict[str, Any],
    canonical_empirical_report: dict[str, Any],
    board_score: float,
) -> dict[str, float]:
    blueprint_lenses = blueprint_report.get("interdisciplinary_lenses") or {}
    autonomous_score = float((autonomous_report.get("metrics") or {}).get("score", 0.0))
    empirical_score = float(
        (canonical_empirical_report.get("metrics") or {}).get("score", 0.0)
    )
    return {
        **{str(k): float(v) for k, v in blueprint_lenses.items()},
        "autonomy": round(autonomous_score, 4),
        "empirical_validation": round(empirical_score, 4),
        "program_synthesis": round(board_score, 4),
    }


def _derive_board_next_step(
    gate_passed: bool,
    blueprint_report: dict[str, Any],
    autonomous_report: dict[str, Any],
    canonical_empirical_report: dict[str, Any],
    top_workstreams: list[dict[str, Any]],
) -> dict[str, str]:
    if not gate_passed and not bool(canonical_empirical_report.get("gate_passed", False)):
        next_step = canonical_empirical_report.get("autonomous_next_step") or {}
        return {
            "priority": str(next_step.get("priority", "P1")),
            "owner": str(next_step.get("owner", "validation-lane")),
            "action": str(
                next_step.get("action", "repair canonical empirical validation failures")
            ),
        }
    if not gate_passed and (blueprint_report.get("optimization_roadmap") or []):
        top = (blueprint_report.get("optimization_roadmap") or [])[0]
        return {
            "priority": str(top.get("priority", "P1")),
            "owner": str(top.get("owner", "team")),
            "action": str(top.get("action", top.get("check", "resolve blueprint failures"))),
        }
    if not gate_passed:
        next_step = autonomous_report.get("autonomous_next_step") or {}
        return {
            "priority": str(next_step.get("priority", "P1")),
            "owner": str(next_step.get("owner", "autonomous-engine")),
            "action": str(next_step.get("action", "recalibrate autonomous gate")),
        }
    if top_workstreams:
        top = top_workstreams[0]
        return {
            "priority": str(top.get("priority", "P0")),
            "owner": str(top.get("owner", "team")),
            "action": str(top.get("title", "advance next masterpiece workstream")),
        }
    return {
        "priority": "P0",
        "owner": "release-management",
        "action": "Promote the synchronized masterpiece board into release evidence.",
    }


def build_program_board(
    blueprint_report: dict[str, Any],
    autonomous_report: dict[str, Any],
    canonical_empirical_report: dict[str, Any],
    config: BoardConfig,
    genesis_execution_framework: dict[str, Any],
    mastery_framework: dict[str, Any],
    optimization_blueprint: dict[str, Any],
) -> dict[str, Any]:
    composite_snr = _composite_snr(blueprint_report, autonomous_report)
    blueprint_score = float(blueprint_report.get("weighted_score", 0.0))
    autonomous_score = float((autonomous_report.get("metrics") or {}).get("score", 0.0))
    empirical_score = float(
        (canonical_empirical_report.get("metrics") or {}).get("score", 0.0)
    )
    board_score = round(
        (
            config.score_weights.get("blueprint", 0.35) * blueprint_score
            + config.score_weights.get("autonomous", 0.25) * autonomous_score
            + config.score_weights.get("snr", 0.15) * composite_snr
            + config.score_weights.get("empirical", 0.25) * empirical_score
        ),
        4,
    )

    blueprint_gate = bool(blueprint_report.get("gate_passed", False))
    autonomous_gate = bool(autonomous_report.get("gate_passed", False))
    canonical_empirical_gate = bool(canonical_empirical_report.get("gate_passed", False))
    gate_constraints = {
        "blueprint_gate": blueprint_gate or (not config.require_blueprint_gate),
        "autonomous_gate": autonomous_gate or (not config.require_autonomous_gate),
        "canonical_empirical_gate": canonical_empirical_gate
        or (not config.require_canonical_empirical_gate),
        "board_score": board_score >= config.min_board_score,
    }
    gate_passed = all(gate_constraints.values())

    workstreams = _dedupe_workstreams(
        _extract_blueprint_roadmap(blueprint_report)
        + _extract_autonomous_action(autonomous_report)
        + _extract_empirical_action(canonical_empirical_report)
        + _extract_genesis_workstreams(genesis_execution_framework)
        + _extract_mastery_workstreams(mastery_framework)
        + _extract_optimization_workstreams(optimization_blueprint)
    )
    workstreams.sort(
        key=lambda item: (
            _priority_rank(str(item.get("priority", "P9"))),
            -float(item.get("snr_gain", 0.0) or 0.0),
            str(item.get("title", "")),
        )
    )
    top_workstreams = workstreams[: config.max_top_workstreams]

    next_step = _derive_board_next_step(
        gate_passed,
        blueprint_report,
        autonomous_report,
        canonical_empirical_report,
        top_workstreams,
    )
    graph_of_thought = _build_graph_of_thought(
        blueprint_report,
        autonomous_report,
        canonical_empirical_report,
        top_workstreams,
        board_score,
    )
    giants_protocol = _merge_giants(
        config, blueprint_report, autonomous_report, canonical_empirical_report
    )
    lenses = _build_interdisciplinary_lenses(
        blueprint_report, autonomous_report, canonical_empirical_report, board_score
    )

    return {
        "program": config.program,
        "gate_passed": gate_passed,
        "thresholds": {
            "min_board_score": config.min_board_score,
            "require_blueprint_gate": config.require_blueprint_gate,
            "require_autonomous_gate": config.require_autonomous_gate,
            "require_canonical_empirical_gate": config.require_canonical_empirical_gate,
        },
        "constraints": gate_constraints,
        "metrics": {
            "board_score": board_score,
            "blueprint_score": round(blueprint_score, 4),
            "autonomous_score": round(autonomous_score, 4),
            "empirical_score": round(empirical_score, 4),
            "empirical_pass_rate": round(
                float(
                    (canonical_empirical_report.get("metrics") or {}).get(
                        "empirical_pass_rate", 0.0
                    )
                ),
                4,
            ),
            "composite_snr": composite_snr,
            "prompt_snr": round(
                float((autonomous_report.get("metrics") or {}).get("prompt_snr", 0.0)),
                4,
            ),
            "rlvr_snr": round(
                float((autonomous_report.get("metrics") or {}).get("rlvr_snr", 0.0)),
                4,
            ),
        },
        "control_planes": blueprint_report.get("control_planes") or [],
        "interdisciplinary_lenses": lenses,
        "ethical_integrity_posture": blueprint_report.get(
            "ethical_integrity_posture"
        )
        or {},
        "implementation_strategy": blueprint_report.get("implementation_strategy")
        or {},
        "risk_register": blueprint_report.get("risk_register") or [],
        "standing_on_giants_protocol": giants_protocol,
        "graph_of_thought": graph_of_thought,
        "top_workstreams": top_workstreams,
        "autonomous_next_step": next_step,
        "source_reports": {
            "blueprint": {
                "program_id": blueprint_report.get("program_id"),
                "gate_passed": blueprint_gate,
            },
            "autonomous": {
                "program_id": (autonomous_report.get("program") or {}).get("id"),
                "gate_passed": autonomous_gate,
            },
            "canonical_empirical": {
                "program_id": (canonical_empirical_report.get("program") or {}).get("id"),
                "gate_passed": canonical_empirical_gate,
                "status": canonical_empirical_report.get("canonical_status"),
            },
        },
    }


def render_markdown(board: dict[str, Any]) -> str:
    metrics = board.get("metrics") or {}
    next_step = board.get("autonomous_next_step") or {}
    workstreams = board.get("top_workstreams") or []
    posture = board.get("ethical_integrity_posture") or {}
    strategy = board.get("implementation_strategy") or {}
    risks = board.get("risk_register") or []

    lines = [
        "# BIZRA Masterpiece Program Board",
        "",
        f"- Gate passed: `{board.get('gate_passed', False)}`",
        f"- Board score: `{metrics.get('board_score', 0.0):.4f}`",
        f"- Composite SNR: `{metrics.get('composite_snr', 0.0):.4f}`",
        f"- Blueprint score: `{metrics.get('blueprint_score', 0.0):.4f}`",
        f"- Autonomous score: `{metrics.get('autonomous_score', 0.0):.4f}`",
        f"- Empirical score: `{metrics.get('empirical_score', 0.0):.4f}`",
        f"- Empirical pass rate: `{metrics.get('empirical_pass_rate', 0.0):.4f}`",
        "",
        "## Ethical Integrity",
        "",
        f"- Ihsan: `{((posture.get('ihsan') or {}).get('status', 'UNKNOWN'))}` / `{((posture.get('ihsan') or {}).get('score', 0.0)):.4f}`",
        f"- Adl: `{((posture.get('adl') or {}).get('status', 'UNKNOWN'))}` / `{((posture.get('adl') or {}).get('score', 0.0)):.4f}`",
        f"- Amanah: `{((posture.get('amanah') or {}).get('status', 'UNKNOWN'))}` / `{((posture.get('amanah') or {}).get('score', 0.0)):.4f}`",
        "",
        "## Autonomous Next Step",
        "",
        f"- Priority: `{next_step.get('priority', 'P1')}`",
        f"- Owner: `{next_step.get('owner', 'team')}`",
        f"- Action: {next_step.get('action', 'n/a')}",
        "",
        "## Implementation Strategy",
        "",
        f"- Current phase: `{strategy.get('current_phase', 'unknown')}`",
        f"- Objective: {strategy.get('phase_objective', 'n/a')}",
        "",
        "## Top Workstreams",
        "",
        "| Priority | Owner | Title | Source |",
        "|---|---|---|---|",
    ]

    for item in workstreams:
        lines.append(
            f"| {item.get('priority', '')} | {item.get('owner', '')} | "
            f"{item.get('title', '')} | {item.get('source', '')} |"
        )

    if risks:
        lines.extend(
            [
                "",
                "## Priority Risks",
                "",
                "| Priority | Dimension | Owner | Cascade |",
                "|---|---|---|---|",
            ]
        )
        for risk in risks[:5]:
            lines.append(
                f"| {risk.get('priority', '')} | {risk.get('dimension', '')} | "
                f"{risk.get('owner', '')} | {risk.get('cascade', '')} |"
            )

    lines.extend(
        [
            "",
            "## Standing On Giants",
            "",
        ]
    )
    for giant in board.get("standing_on_giants_protocol") or []:
        lines.append(f"- {giant}")
    return "\n".join(lines) + "\n"


def _write_github_outputs(path: Path, board: dict[str, Any]) -> None:
    metrics = board.get("metrics") or {}
    next_step = board.get("autonomous_next_step") or {}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"masterpiece_board_passed={str(board.get('gate_passed', False)).lower()}\n")
        fh.write(f"masterpiece_board_score={metrics.get('board_score', 0.0)}\n")
        fh.write(f"masterpiece_composite_snr={metrics.get('composite_snr', 0.0)}\n")
        fh.write(f"masterpiece_empirical_score={metrics.get('empirical_score', 0.0)}\n")
        fh.write(
            f"masterpiece_empirical_pass_rate={metrics.get('empirical_pass_rate', 0.0)}\n"
        )
        fh.write(
            "masterpiece_empirical_status="
            f"{(board.get('source_reports') or {}).get('canonical_empirical', {}).get('status', 'UNKNOWN')}\n"
        )
        fh.write(f"masterpiece_next_priority={next_step.get('priority', 'P1')}\n")
        fh.write(f"masterpiece_next_owner={next_step.get('owner', 'team')}\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build masterpiece program board.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/masterpiece_program_board.json"),
        help="Masterpiece board config JSON path.",
    )
    parser.add_argument(
        "--blueprint-report",
        type=Path,
        required=True,
        help="Elite blueprint audit report JSON path.",
    )
    parser.add_argument(
        "--autonomous-report",
        type=Path,
        required=True,
        help="Autonomous engine gate report JSON path.",
    )
    parser.add_argument(
        "--canonical-empirical-report",
        type=Path,
        required=True,
        help="Canonical empirical validation report JSON path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional output JSON report path.",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=None,
        help="Optional output Markdown report path.",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        default=None,
        help="Optional GitHub output file path.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    blueprint_report = _load_json(args.blueprint_report)
    autonomous_report = _load_json(args.autonomous_report)
    canonical_empirical_report = _load_json(args.canonical_empirical_report)
    genesis_execution_framework = _load_json(
        config.source_artifacts["genesis_execution_framework"]
    )
    mastery_framework = _load_json(config.source_artifacts["mastery_framework"])
    optimization_blueprint = _load_json(
        config.source_artifacts["optimization_blueprint"]
    )

    board = build_program_board(
        blueprint_report=blueprint_report,
        autonomous_report=autonomous_report,
        canonical_empirical_report=canonical_empirical_report,
        config=config,
        genesis_execution_framework=genesis_execution_framework,
        mastery_framework=mastery_framework,
        optimization_blueprint=optimization_blueprint,
    )
    encoded = json.dumps(board, indent=2)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    if args.markdown_report is not None:
        args.markdown_report.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_report.write_text(render_markdown(board), encoding="utf-8")
    if args.github_output is not None:
        _write_github_outputs(args.github_output, board)

    print(encoded)
    return 0 if board.get("gate_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
