#!/usr/bin/env python3
"""
CI Delivery Program Gate — validates the machine-readable BIZRA program manifest.

Checks:
1. Required top-level sections exist.
2. Workstreams have unique ids, valid priorities, and valid dependencies.
3. Delivery gates are unique and structurally complete.
4. Roadmap horizons exist and are non-empty.
5. Scorecard rows and risk register entries are internally consistent.

Standing on Giants: Deming (PDCA, 1950) — treat the program plan as an executable
quality surface, not just prose.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROGRAM_PATH = ROOT / "docs" / "program" / "bizra_delivery_program.json"
REQUIRED_TOP_LEVEL_KEYS = {
    "program_id",
    "version",
    "status",
    "north_star",
    "operating_graph",
    "ethical_invariants",
    "workstreams",
    "delivery_gates",
    "roadmap",
    "scorecard",
    "risk_register",
    "top_next_step",
}
VALID_PRIORITIES = {"P0", "P1", "P2"}
VALID_SCORECARD_STATUSES = {
    "proven",
    "proven_at_spearpoint_scale",
    "partial",
    "staged",
    "aspirational",
}
REQUIRED_ROADMAP_KEYS = {
    "next_7_days",
    "next_30_days",
    "next_60_days",
    "next_90_days",
}


def _load_program(path: Path = DEFAULT_PROGRAM_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _is_non_empty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_program_structure(program: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    missing = sorted(REQUIRED_TOP_LEVEL_KEYS - set(program.keys()))
    if missing:
        issues.append(f"Missing top-level keys: {', '.join(missing)}")

    if not _is_non_empty_string(program.get("program_id")):
        issues.append("program_id must be a non-empty string")
    if not _is_non_empty_string(program.get("version")):
        issues.append("version must be a non-empty string")
    if not _is_non_empty_string(program.get("north_star")):
        issues.append("north_star must be a non-empty string")

    operating_graph = program.get("operating_graph")
    if not isinstance(operating_graph, list) or not operating_graph:
        issues.append("operating_graph must be a non-empty list")

    ethical_invariants = program.get("ethical_invariants")
    if not isinstance(ethical_invariants, list) or not ethical_invariants:
        issues.append("ethical_invariants must be a non-empty list")
    else:
        for idx, invariant in enumerate(ethical_invariants):
            if not isinstance(invariant, dict):
                issues.append(f"ethical_invariants[{idx}] must be an object")
                continue
            if not _is_non_empty_string(invariant.get("id")):
                issues.append(f"ethical_invariants[{idx}].id must be non-empty")
            if not _is_non_empty_string(invariant.get("rule")):
                issues.append(f"ethical_invariants[{idx}].rule must be non-empty")

    return issues


def _validate_workstreams(program: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    workstreams = program.get("workstreams")
    if not isinstance(workstreams, list) or not workstreams:
        return ["workstreams must be a non-empty list"]

    ids: list[str] = []
    for idx, workstream in enumerate(workstreams):
        if not isinstance(workstream, dict):
            issues.append(f"workstreams[{idx}] must be an object")
            continue
        wid = workstream.get("id")
        if not _is_non_empty_string(wid):
            issues.append(f"workstreams[{idx}].id must be non-empty")
            continue
        ids.append(str(wid))

        if workstream.get("priority") not in VALID_PRIORITIES:
            issues.append(
                f"workstreams[{idx}] has invalid priority: {workstream.get('priority')}"
            )

        for field_name in ("name", "objective", "current_truth"):
            if not _is_non_empty_string(workstream.get(field_name)):
                issues.append(f"workstreams[{idx}].{field_name} must be non-empty")

        for list_field in ("dependencies", "kpis", "deliverables", "risks"):
            value = workstream.get(list_field)
            if not isinstance(value, list):
                issues.append(f"workstreams[{idx}].{list_field} must be a list")

    if len(set(ids)) != len(ids):
        issues.append("workstream ids must be unique")

    known_ids = set(ids)
    for idx, workstream in enumerate(workstreams):
        if not isinstance(workstream, dict):
            continue
        for dep in workstream.get("dependencies", []):
            if dep not in known_ids:
                issues.append(
                    f"workstreams[{idx}] dependency '{dep}' does not match any workstream id"
                )

    return issues


def _validate_delivery_gates(program: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    gates = program.get("delivery_gates")
    if not isinstance(gates, list) or not gates:
        return ["delivery_gates must be a non-empty list"]

    ids: list[str] = []
    for idx, gate in enumerate(gates):
        if not isinstance(gate, dict):
            issues.append(f"delivery_gates[{idx}] must be an object")
            continue
        gid = gate.get("id")
        if not _is_non_empty_string(gid):
            issues.append(f"delivery_gates[{idx}].id must be non-empty")
        else:
            ids.append(str(gid))
        if not _is_non_empty_string(gate.get("name")):
            issues.append(f"delivery_gates[{idx}].name must be non-empty")
        checks = gate.get("checks")
        if not isinstance(checks, list) or not checks:
            issues.append(f"delivery_gates[{idx}].checks must be a non-empty list")

    if len(set(ids)) != len(ids):
        issues.append("delivery gate ids must be unique")
    return issues


def _validate_roadmap(program: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    roadmap = program.get("roadmap")
    if not isinstance(roadmap, dict):
        return ["roadmap must be an object"]

    missing = sorted(REQUIRED_ROADMAP_KEYS - set(roadmap.keys()))
    if missing:
        issues.append(f"roadmap missing keys: {', '.join(missing)}")

    for key in REQUIRED_ROADMAP_KEYS:
        value = roadmap.get(key)
        if not isinstance(value, list) or not value:
            issues.append(f"roadmap.{key} must be a non-empty list")
    return issues


def _validate_scorecard(program: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    scorecard = program.get("scorecard")
    if not isinstance(scorecard, list) or not scorecard:
        return ["scorecard must be a non-empty list"]

    seen_dimensions: set[str] = set()
    for idx, row in enumerate(scorecard):
        if not isinstance(row, dict):
            issues.append(f"scorecard[{idx}] must be an object")
            continue
        dimension = row.get("dimension")
        if not _is_non_empty_string(dimension):
            issues.append(f"scorecard[{idx}].dimension must be non-empty")
        else:
            if str(dimension) in seen_dimensions:
                issues.append(f"scorecard dimension '{dimension}' must be unique")
            seen_dimensions.add(str(dimension))
        if row.get("status") not in VALID_SCORECARD_STATUSES:
            issues.append(f"scorecard[{idx}] has invalid status: {row.get('status')}")
        if not _is_non_empty_string(row.get("measure")):
            issues.append(f"scorecard[{idx}].measure must be non-empty")
    return issues


def _validate_risk_register(program: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    risks = program.get("risk_register")
    if not isinstance(risks, list) or not risks:
        return ["risk_register must be a non-empty list"]

    ids: set[str] = set()
    for idx, risk in enumerate(risks):
        if not isinstance(risk, dict):
            issues.append(f"risk_register[{idx}] must be an object")
            continue
        rid = risk.get("id")
        if not _is_non_empty_string(rid):
            issues.append(f"risk_register[{idx}].id must be non-empty")
        else:
            rid_text = str(rid)
            if rid_text in ids:
                issues.append(f"risk id '{rid_text}' must be unique")
            ids.add(rid_text)
        if risk.get("priority") not in VALID_PRIORITIES:
            issues.append(
                f"risk_register[{idx}] has invalid priority: {risk.get('priority')}"
            )
        for field_name in ("risk", "impact", "mitigation"):
            if not _is_non_empty_string(risk.get(field_name)):
                issues.append(f"risk_register[{idx}].{field_name} must be non-empty")
    return issues


def _validate_top_next_step(program: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    item = program.get("top_next_step")
    if not isinstance(item, dict):
        return ["top_next_step must be an object"]
    for field_name in ("id", "title", "reason"):
        if not _is_non_empty_string(item.get(field_name)):
            issues.append(f"top_next_step.{field_name} must be non-empty")
    return issues


def validate_delivery_program(path: Path = DEFAULT_PROGRAM_PATH) -> list[str]:
    program = _load_program(path)
    issues: list[str] = []
    issues.extend(_validate_program_structure(program))
    issues.extend(_validate_workstreams(program))
    issues.extend(_validate_delivery_gates(program))
    issues.extend(_validate_roadmap(program))
    issues.extend(_validate_scorecard(program))
    issues.extend(_validate_risk_register(program))
    issues.extend(_validate_top_next_step(program))
    return issues


def main() -> int:
    issues = validate_delivery_program()
    if issues:
        print("CI DELIVERY PROGRAM GATE FAILED")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("CI DELIVERY PROGRAM GATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
