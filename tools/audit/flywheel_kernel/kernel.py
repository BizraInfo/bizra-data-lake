"""BIZRA Autonomous Flywheel Kernel v1.

This module turns audit artifacts into a repeatable engineering flywheel:

    Signal -> Root Cause -> Fix -> Test -> Validate -> Document -> Encode -> Repeat

The kernel is intentionally non-destructive. It reads artifacts, evaluates
guards, chooses the next priority, and emits a machine-readable report. It does
not mutate source code, publish claims, rotate secrets, or run network actions.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


DEFAULT_REGISTRY = Path(__file__).with_name("patterns.json")


@dataclass(frozen=True)
class GuardResult:
    guard_id: str
    status: str
    signal: str
    evidence: list[str] = field(default_factory=list)
    next_action: str = ""


@dataclass(frozen=True)
class PriorityDecision:
    priority_id: str
    title: str
    rationale: str
    next_actions: list[str]
    blocked_by: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FlywheelState:
    audit_dir: str
    summary_counts: dict[str, int]
    secret_count: int
    claim_counts: dict[str, int]
    code_risk_counts: dict[str, int]
    dep_gaps: list[str]
    website_captures: list[dict[str, Any]]
    missing_artifacts: list[str]


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def _count_by(items: Iterable[Any], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        value = "UNKNOWN"
        if isinstance(item, dict):
            value = str(item.get(key) or "UNKNOWN")
        counts[value] = counts.get(value, 0) + 1
    return counts


def _int_map(values: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for key, value in values.items():
        try:
            out[str(key)] = int(value)
        except (TypeError, ValueError):
            continue
    return out


def load_patterns(registry_path: Path | None = None) -> list[dict[str, Any]]:
    """Load the pattern registry.

    Missing or malformed registries degrade to an empty pattern list so the
    priority engine can still operate from audit artifacts.
    """

    path = registry_path or DEFAULT_REGISTRY
    data = _read_json(path, {"patterns": []})
    patterns = data.get("patterns", []) if isinstance(data, dict) else []
    return [p for p in patterns if isinstance(p, dict)]


def load_audit_state(audit_dir: Path | str) -> FlywheelState:
    """Read an omni-audit artifact directory into normalized state."""

    base = Path(audit_dir)
    required = [
        "audit_summary.json",
        "secret_findings.json",
        "claims_register.json",
        "code_risks.json",
        "dependencies.json",
    ]
    missing = [name for name in required if not (base / name).exists()]

    summary = _read_json(base / "audit_summary.json", {})
    secret_findings = _read_json(base / "secret_findings.json", [])
    claims = _read_json(base / "claims_register.json", [])
    code_risks = _read_json(base / "code_risks.json", [])
    dependencies = _read_json(base / "dependencies.json", {})

    summary_counts = _int_map(summary.get("counts", {}) if isinstance(summary, dict) else {})
    secret_count = summary_counts.get("secrets")
    if secret_count is None:
        secret_count = len(secret_findings) if isinstance(secret_findings, list) else 0

    dep_gaps = dependencies.get("gaps", []) if isinstance(dependencies, dict) else []
    if not isinstance(dep_gaps, list):
        dep_gaps = []

    website_captures = summary.get("website_captures", []) if isinstance(summary, dict) else []
    if not isinstance(website_captures, list):
        website_captures = []

    return FlywheelState(
        audit_dir=str(base),
        summary_counts=summary_counts,
        secret_count=int(secret_count),
        claim_counts=_count_by(claims if isinstance(claims, list) else [], "classification"),
        code_risk_counts=_count_by(
            code_risks if isinstance(code_risks, list) else [], "rule"
        ),
        dep_gaps=[str(gap) for gap in dep_gaps],
        website_captures=[c for c in website_captures if isinstance(c, dict)],
        missing_artifacts=missing,
    )


def evaluate_guards(state: FlywheelState) -> list[GuardResult]:
    """Evaluate pre-execution and promotion guards from audit state."""

    guards: list[GuardResult] = []

    if state.missing_artifacts:
        guards.append(
            GuardResult(
                guard_id="G-FW-001",
                status="BLOCK",
                signal="Required audit artifacts are missing.",
                evidence=state.missing_artifacts,
                next_action="Run the omni audit before making priority decisions.",
            )
        )
    else:
        guards.append(
            GuardResult(
                guard_id="G-FW-001",
                status="PASS",
                signal="Required audit artifacts are present.",
                next_action="Continue guard evaluation.",
            )
        )

    if state.secret_count > 0:
        guards.append(
            GuardResult(
                guard_id="G-FW-002",
                status="BLOCK",
                signal=f"{state.secret_count} secret-pattern finding(s) remain.",
                evidence=["secret_findings.json"],
                next_action="Triage or remove secret findings before launch work.",
            )
        )
    else:
        guards.append(
            GuardResult(
                guard_id="G-FW-002",
                status="PASS",
                signal="Secret-pattern findings are zero.",
                evidence=["secret_findings.json"],
                next_action="Shift to the next exposed constraint.",
            )
        )

    prohibited = state.claim_counts.get("PROHIBITED", 0)
    needs_rewrite = state.claim_counts.get("NEEDS_REWRITE", 0)
    proof_required = state.claim_counts.get("PROOF_REQUIRED", 0)
    if prohibited or needs_rewrite:
        guards.append(
            GuardResult(
                guard_id="G-FW-003",
                status="BLOCK",
                signal=(
                    f"Public claim debt remains: {prohibited} prohibited, "
                    f"{needs_rewrite} needs rewrite, {proof_required} proof required."
                ),
                evidence=["claims_register.json"],
                next_action="Remove, soften, or receipt-link public claims.",
            )
        )
    elif proof_required:
        guards.append(
            GuardResult(
                guard_id="G-FW-003",
                status="WARN",
                signal=f"{proof_required} proof-required claim(s) remain.",
                evidence=["claims_register.json"],
                next_action="Publish receipts or convert to directional wording.",
            )
        )
    else:
        guards.append(
            GuardResult(
                guard_id="G-FW-003",
                status="PASS",
                signal="No public claim blockers detected.",
                evidence=["claims_register.json"],
            )
        )

    if state.dep_gaps:
        guards.append(
            GuardResult(
                guard_id="G-FW-004",
                status="WARN",
                signal=f"{len(state.dep_gaps)} dependency reproducibility gap(s).",
                evidence=state.dep_gaps,
                next_action="Add lockfiles, SBOMs, and dependency policy gates.",
            )
        )
    else:
        guards.append(
            GuardResult(
                guard_id="G-FW-004",
                status="PASS",
                signal="No dependency reproducibility gaps detected.",
            )
        )

    risky_rules = {
        "PY_SHELL_TRUE": state.code_risk_counts.get("PY_SHELL_TRUE", 0),
        "RS_PANIC": state.code_risk_counts.get("RS_PANIC", 0),
        "RS_UNWRAP": state.code_risk_counts.get("RS_UNWRAP", 0),
        "PY_EVAL_EXEC": state.code_risk_counts.get("PY_EVAL_EXEC", 0),
    }
    if any(risky_rules.values()):
        guards.append(
            GuardResult(
                guard_id="G-FW-005",
                status="WARN",
                signal=f"Runtime hardening findings remain: {risky_rules}.",
                evidence=["code_risks.json"],
                next_action="Triage by hot path and replace unsafe failure behavior.",
            )
        )
    else:
        guards.append(
            GuardResult(
                guard_id="G-FW-005",
                status="PASS",
                signal="No high-priority runtime hardening findings detected.",
            )
        )

    return guards


def decide_priority(state: FlywheelState, guards: list[GuardResult]) -> PriorityDecision:
    """Choose the next system constraint from current audit state."""

    blocked = [g.guard_id for g in guards if g.status == "BLOCK"]
    prohibited = state.claim_counts.get("PROHIBITED", 0)
    needs_rewrite = state.claim_counts.get("NEEDS_REWRITE", 0)
    proof_required = state.claim_counts.get("PROOF_REQUIRED", 0)

    if state.missing_artifacts:
        return PriorityDecision(
            priority_id="P-BOOTSTRAP-AUDIT",
            title="Generate audit artifacts",
            rationale="The flywheel cannot rank constraints without current audit state.",
            next_actions=[
                "Run the omni audit in no-network mode.",
                "Commit or archive the generated audit_summary, findings, and registers.",
            ],
            blocked_by=blocked,
        )

    if state.secret_count > 0:
        return PriorityDecision(
            priority_id="P0_SECRET_TRIAGE",
            title="Secret-pattern triage",
            rationale="Secrets remain the highest immediate blast-radius risk.",
            next_actions=[
                "Classify every secret finding as real, placeholder, substitution, or scanner noise.",
                "Rotate and remove any real credential.",
                "Encode false-positive fixes as scanner tests.",
            ],
            blocked_by=blocked,
        )

    if prohibited or needs_rewrite or proof_required:
        return PriorityDecision(
            priority_id="P1_TRUTH_INTEGRITY",
            title="Public claim truth alignment",
            rationale=(
                "Secret risk is clear, so the next exposed constraint is "
                "claim evidence integrity."
            ),
            next_actions=[
                "Remove or receipt-link prohibited and exact public claims.",
                "Rewrite claims that need evidence into directional language.",
                "Publish receipt-backed proof links for proof-required claims.",
            ],
            blocked_by=blocked,
        )

    if state.dep_gaps:
        return PriorityDecision(
            priority_id="P2_SUPPLY_CHAIN_TRUST",
            title="Supply-chain reproducibility",
            rationale="Claims are clean enough to shift to build attestation.",
            next_actions=[
                "Add missing lockfiles for releasable workspaces.",
                "Generate SBOM artifacts in release CI.",
                "Add dependency advisory and license policy gates.",
            ],
            blocked_by=blocked,
        )

    if any(
        state.code_risk_counts.get(rule, 0)
        for rule in ("PY_SHELL_TRUE", "RS_PANIC", "RS_UNWRAP", "PY_EVAL_EXEC")
    ):
        return PriorityDecision(
            priority_id="P3_RUNTIME_HARDENING",
            title="Runtime failure-mode hardening",
            rationale="Trust surfaces are clear enough to reduce production failure modes.",
            next_actions=[
                "Triage code_risks by hot path.",
                "Replace shell execution and panic surfaces with typed failures.",
                "Add regression tests for rare paths.",
            ],
            blocked_by=blocked,
        )

    return PriorityDecision(
        priority_id="P4_MONITOR_AND_RELOOP",
        title="Monitor and re-loop",
        rationale="No current audit blocker dominates the priority stack.",
        next_actions=[
            "Schedule the next no-network audit.",
            "Watch changed paths for pattern-triggered re-audits.",
        ],
        blocked_by=blocked,
    )


def should_trigger_audit(
    changed_paths: Iterable[str],
    patterns: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Return pattern-triggered audit reasons for changed paths."""

    active_patterns = patterns if patterns is not None else load_patterns()
    triggers: list[dict[str, str]] = []
    for path in changed_paths:
        normalized = path.replace("\\", "/")
        for pattern in active_patterns:
            for glob in pattern.get("trigger_globs", []):
                if fnmatch.fnmatch(normalized, glob):
                    triggers.append(
                        {
                            "path": normalized,
                            "pattern_id": str(pattern.get("pattern_id", "")),
                            "pattern": str(pattern.get("name", "")),
                            "glob": str(glob),
                        }
                    )
                    break
    return triggers


def build_report(
    audit_dir: Path | str,
    registry_path: Path | None = None,
    changed_paths: Iterable[str] = (),
) -> dict[str, Any]:
    """Build a deterministic Flywheel Kernel report."""

    patterns = load_patterns(registry_path)
    state = load_audit_state(audit_dir)
    guards = evaluate_guards(state)
    decision = decide_priority(state, guards)
    triggers = should_trigger_audit(changed_paths, patterns)

    return {
        "schema": "bizra.flywheel.kernel_report.v1",
        "loop": [
            "Signal",
            "Root Cause",
            "Fix",
            "Test",
            "Validate",
            "Document",
            "Encode",
            "Repeat",
        ],
        "state": asdict(state),
        "guards": [asdict(g) for g in guards],
        "priority": asdict(decision),
        "triggered_patterns": triggers,
        "pattern_count": len(patterns),
        "active_pattern_ids": [str(p.get("pattern_id", "")) for p in patterns],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bizra-flywheel-kernel")
    parser.add_argument("--audit-dir", required=True)
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--changed-path", action="append", default=[])
    parser.add_argument("--out")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    report = build_report(
        audit_dir=args.audit_dir,
        registry_path=Path(args.registry),
        changed_paths=args.changed_path,
    )

    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    print(text)

    if args.strict and any(g["status"] == "BLOCK" for g in report["guards"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
