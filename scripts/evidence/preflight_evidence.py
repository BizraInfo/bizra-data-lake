#!/usr/bin/env python3
"""Fast readiness scanner for evidence package stage/tier execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from build_evidence_package import (
        collect_research,
        evaluate_gates,
        load_artifact_entries,
    )
    from common import (
        DEFAULT_CONFIG_PATH,
        DEFAULT_GATE_CONFIG_PATH,
        REPO_ROOT,
        load_yaml,
        resolve_path,
    )
except ModuleNotFoundError:
    from scripts.evidence.build_evidence_package import (
        collect_research,
        evaluate_gates,
        load_artifact_entries,
    )
    from scripts.evidence.common import (
        DEFAULT_CONFIG_PATH,
        DEFAULT_GATE_CONFIG_PATH,
        REPO_ROOT,
        load_yaml,
        resolve_path,
    )


def run(
    *,
    config_path: Path,
    gate_config_path: Path,
    stage: str,
    tier: str,
    repo_root: Path,
    strict: bool,
) -> int:
    cfg = load_yaml(config_path)
    gate_cfg = load_yaml(gate_config_path)
    policy_version = str(cfg.get("policy_version", "evidence-v1.0"))

    raw_entries = load_artifact_entries(cfg)
    entries = []
    for e in raw_entries:
        source = resolve_path(repo_root, str(e["source"]))
        discovery_rule = str(e.get("discovery_rule", "canonical_repo_path"))
        if discovery_rule == "source_lock":
            source = repo_root / str(e["logical_path"])
        row = dict(e)
        row["source_exists"] = source.exists() and source.is_file()
        entries.append(row)

    research = collect_research(cfg.get("research_policy", {}), repo_root)
    gate_report = evaluate_gates(
        stage=stage,
        tier=tier,
        entries=entries,
        research=research,
        gate_cfg=gate_cfg,
        policy_version=policy_version,
    )

    summary = {
        "stage": stage,
        "tier": tier,
        "policy_version": policy_version,
        "expected_pass": gate_report["passed"],
        "required_gates": gate_report["required_gates"],
        "gate_results": gate_report["gate_results"],
        "required_artifacts_missing": gate_report["required_artifacts_missing"],
        "reasons": gate_report["reasons"],
        "research_discovered_count": gate_report["research_discovered_count"],
        "research_indexed_count": gate_report["research_indexed_count"],
        "research_unindexed_count": gate_report["research_unindexed_count"],
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if strict and not gate_report["passed"]:
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preflight readiness scanner for evidence package"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--gate-config", type=Path, default=DEFAULT_GATE_CONFIG_PATH)
    parser.add_argument("--stage", choices=["scaffold", "final"], default="scaffold")
    parser.add_argument(
        "--tier", choices=["private_full", "public_redacted"], default="private_full"
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    raise SystemExit(
        run(
            config_path=args.config,
            gate_config_path=args.gate_config,
            stage=args.stage,
            tier=args.tier,
            repo_root=args.repo_root,
            strict=args.strict,
        )
    )


if __name__ == "__main__":
    main()
