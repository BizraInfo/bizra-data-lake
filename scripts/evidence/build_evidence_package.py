#!/usr/bin/env python3
"""Build canonical BIZRA evidence package tiers with fail-closed gate reports."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

try:
    from common import (
        DEFAULT_CONFIG_PATH,
        DEFAULT_GATE_CONFIG_PATH,
        DEFAULT_PACKAGE_ROOT,
        REPO_ROOT,
        ensure_package_layout,
        hash_file_blake3,
        hash_file_sha256,
        load_yaml,
        manifest_content_hash,
        path_matches_patterns,
        rel_posix,
        resolve_path,
        utc_now_iso,
        write_json,
    )
except ModuleNotFoundError:
    from scripts.evidence.common import (
        DEFAULT_CONFIG_PATH,
        DEFAULT_GATE_CONFIG_PATH,
        DEFAULT_PACKAGE_ROOT,
        REPO_ROOT,
        ensure_package_layout,
        hash_file_blake3,
        hash_file_sha256,
        load_yaml,
        manifest_content_hash,
        path_matches_patterns,
        rel_posix,
        resolve_path,
        utc_now_iso,
        write_json,
    )

FOUNDING_LOGICAL_PATHS = {
    "00_GENESIS/01_ARABIC_FOUNDING/al_risalah_original.pdf",
    "00_GENESIS/01_ARABIC_FOUNDING/al_bazrah_original.pdf",
}
THEOLOGICAL_LOGICAL_PATH = "00_GENESIS/03_SPIRITUAL_TECHNICAL/ihsan_as_architecture.md"


def _required_for_stage(entry: dict[str, Any], stage: str) -> bool:
    if stage == "scaffold":
        return bool(entry.get("required_scaffold", False))
    if stage == "final":
        return bool(entry.get("required_final", False))
    return False


def _entry_with_defaults(raw: dict[str, Any], is_founding: bool) -> dict[str, Any]:
    entry = dict(raw)
    entry.setdefault("visibility", "both")
    entry.setdefault("public_mode", "full")
    entry.setdefault("classification", "general_artifact")
    entry.setdefault("discovery_rule", "canonical_repo_path")
    entry.setdefault("required_scaffold", False)
    entry.setdefault("required_final", False)
    entry.setdefault("duplicate_of", None)
    if is_founding:
        entry.setdefault("classification", "founding_document")
        entry.setdefault("discovery_rule", "source_lock")
    return entry


def load_artifact_entries(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for item in cfg.get("founding_import", {}).get("assets", []):
        entries.append(_entry_with_defaults(item, is_founding=True))
    for item in cfg.get("artifacts", []):
        entries.append(_entry_with_defaults(item, is_founding=False))
    entries.sort(key=lambda x: str(x["logical_path"]))
    return entries


def collect_research(policy: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    approved_roots = [
        resolve_path(repo_root, str(p)) for p in policy.get("approved_roots", [])
    ]
    exclude_patterns = [str(p) for p in policy.get("exclude_patterns", [])]

    discovered = 0
    indexed = 0
    unindexed = 0
    errors: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    first_path_by_hash: dict[str, str] = {}

    for root in approved_roots:
        if not root.exists():
            continue
        for fp in sorted(root.rglob("*"), key=lambda x: x.as_posix()):
            if not fp.is_file():
                continue
            if fp.suffix.lower() not in {".pdf", ".md", ".txt"}:
                continue

            rel_repo = rel_posix(fp, repo_root)
            rel_root = rel_posix(fp, root)
            if path_matches_patterns(
                rel_repo, exclude_patterns
            ) or path_matches_patterns(rel_root, exclude_patterns):
                continue

            discovered += 1
            try:
                sha256 = hash_file_sha256(fp)
                blake3 = hash_file_blake3(fp)
            except OSError as exc:
                unindexed += 1
                errors.append({"path": fp.as_posix(), "error": str(exc)})
                continue

            duplicate_of = first_path_by_hash.get(blake3)
            if duplicate_of is None:
                first_path_by_hash[blake3] = rel_repo

            indexed += 1
            records.append(
                {
                    "logical_path": rel_repo,
                    "source_root": root.as_posix(),
                    "source_path": fp.as_posix(),
                    "source_rel_to_root": rel_root,
                    "discovery_rule": "research_policy.scan",
                    "classification": "research_document",
                    "visibility": "both",
                    "public_mode": "hash_metadata",
                    "duplicate_of": duplicate_of,
                    "sha256": sha256,
                    "blake3": blake3,
                    "size_bytes": fp.stat().st_size,
                }
            )

    return {
        "gate_mode": policy.get("gate_mode", "manifest_completeness"),
        "approved_roots": [p.as_posix() for p in approved_roots],
        "exclude_patterns": exclude_patterns,
        "discovered_count": discovered,
        "indexed_count": indexed,
        "unindexed_count": unindexed,
        "errors": errors,
        "records": sorted(records, key=lambda r: r["logical_path"]),
    }


def write_research_outputs(package_tier_root: Path, research: dict[str, Any]) -> None:
    manifest_dir = package_tier_root / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_json(manifest_dir / "research_index.json", research)

    index_md = package_tier_root / "01_FOUNDATION" / "02_RESEARCH_CORPUS" / "INDEX.md"
    index_md.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Research Corpus Index",
        "",
        f"- Gate mode: `{research.get('gate_mode')}`",
        f"- Discovered: `{research.get('discovered_count', 0)}`",
        f"- Indexed: `{research.get('indexed_count', 0)}`",
        f"- Unindexed: `{research.get('unindexed_count', 0)}`",
        "",
        "## Roots",
    ]
    for root in research.get("approved_roots", []):
        lines.append(f"- `{root}`")

    lines.extend(
        [
            "",
            "## Records (first 200)",
            "",
            "| Path | SHA256 | BLAKE3 | Duplicate Of |",
            "|---|---|---|---|",
        ]
    )

    for rec in research.get("records", [])[:200]:
        dup = rec.get("duplicate_of") or "-"
        lines.append(
            f"| `{rec['logical_path']}` | `{rec['sha256'][:16]}...` | `{rec['blake3'][:16]}...` | `{dup}` |"
        )

    index_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_gates(
    *,
    stage: str,
    tier: str,
    entries: list[dict[str, Any]],
    research: dict[str, Any],
    gate_cfg: dict[str, Any],
    policy_version: str,
) -> dict[str, Any]:
    by_path = {e["logical_path"]: e for e in entries}
    required_missing = [
        e["logical_path"]
        for e in entries
        if _required_for_stage(e, stage) and not bool(e.get("source_exists", False))
    ]

    founding_docs_present = all(
        bool(by_path.get(path, {}).get("source_exists", False))
        for path in FOUNDING_LOGICAL_PATHS
    )
    theological_bridge_present = bool(
        by_path.get(THEOLOGICAL_LOGICAL_PATH, {}).get("source_exists", False)
    )
    required_artifacts_present = len(required_missing) == 0

    discovered = int(research.get("discovered_count", 0))
    indexed = int(research.get("indexed_count", 0))
    unindexed = int(research.get("unindexed_count", 0))
    research_manifest_complete = discovered == indexed and unindexed == 0

    gate_results = {
        "FOUNDING_DOCS_PRESENT": founding_docs_present,
        "REQUIRED_ARTIFACTS_PRESENT": required_artifacts_present,
        "THEOLOGICAL_BRIDGE_PRESENT": theological_bridge_present,
        "RESEARCH_MANIFEST_COMPLETE": research_manifest_complete,
    }

    stage_cfg = gate_cfg.get("stages", {}).get(stage, {})
    required_gates = [str(g) for g in stage_cfg.get("required_gates", [])]
    fail_closed = bool(gate_cfg.get("fail_closed", True))

    reasons: list[str] = []
    for gate in required_gates:
        if not gate_results.get(gate, False):
            reasons.append(f"GATE_FAILED:{gate}")

    if required_missing:
        reasons.append("MISSING_REQUIRED_ARTIFACTS")

    passed = all(gate_results.get(g, False) for g in required_gates)
    if fail_closed and not passed:
        passed = False

    return {
        "generated_at": utc_now_iso(),
        "stage": stage,
        "tier": tier,
        "policy_version": policy_version,
        "required_gates": required_gates,
        "gate_results": gate_results,
        "passed": passed,
        "reasons": reasons,
        "required_artifacts_missing": sorted(required_missing),
        "research_discovered_count": discovered,
        "research_indexed_count": indexed,
        "research_unindexed_count": unindexed,
        "founding_docs_present": founding_docs_present,
        "theological_bridge_present": theological_bridge_present,
    }


def run(
    *,
    config_path: Path,
    gate_config_path: Path,
    stage: str,
    tier: str,
    repo_root: Path,
    package_root: Path,
    allow_fail: bool,
    json_stdout: bool,
) -> int:
    cfg = load_yaml(config_path)
    gate_cfg = load_yaml(gate_config_path)
    policy_version = str(
        cfg.get("policy_version", gate_cfg.get("policy_version", "evidence-v1.0"))
    )

    ensure_package_layout(package_root)
    tier_root = package_root / tier
    tier_root.mkdir(parents=True, exist_ok=True)

    entries_cfg = load_artifact_entries(cfg)
    manifest_entries: list[dict[str, Any]] = []

    for raw in entries_cfg:
        logical_path = str(raw["logical_path"])
        origin_source_path = resolve_path(repo_root, str(raw["source"]))
        discovery_rule = str(raw.get("discovery_rule", "canonical_repo_path"))
        canonical_source_path = (
            repo_root / logical_path
            if discovery_rule == "source_lock"
            else origin_source_path
        )
        source_exists = (
            canonical_source_path.exists() and canonical_source_path.is_file()
        )

        entry: dict[str, Any] = {
            "logical_path": logical_path,
            "source_path": canonical_source_path.as_posix(),
            "origin_source_path": origin_source_path.as_posix(),
            "source_root": str(
                raw.get("source_root", canonical_source_path.parent.as_posix())
            ),
            "discovery_rule": discovery_rule,
            "visibility": str(raw.get("visibility", "both")),
            "public_mode": str(raw.get("public_mode", "full")),
            "classification": str(raw.get("classification", "general_artifact")),
            "duplicate_of": raw.get("duplicate_of"),
            "required_scaffold": bool(raw.get("required_scaffold", False)),
            "required_final": bool(raw.get("required_final", False)),
            "source_exists": source_exists,
            "copied_path": None,
            "size_bytes": None,
            "sha256": None,
            "blake3": None,
        }

        if source_exists:
            entry["size_bytes"] = canonical_source_path.stat().st_size
            entry["sha256"] = hash_file_sha256(canonical_source_path)
            entry["blake3"] = hash_file_blake3(canonical_source_path)

            should_copy = False
            if tier == "private_full":
                should_copy = True
            elif tier == "public_redacted":
                should_copy = entry["public_mode"] == "full" and entry[
                    "visibility"
                ] in {"both", "public_only"}

            if should_copy:
                dest = tier_root / logical_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(canonical_source_path, dest)
                entry["copied_path"] = rel_posix(dest, tier_root)

        manifest_entries.append(entry)

    manifest_entries.sort(key=lambda e: e["logical_path"])

    research_policy = cfg.get("research_policy", {})
    research = collect_research(research_policy, repo_root)
    write_research_outputs(tier_root, research)

    gate_report = evaluate_gates(
        stage=stage,
        tier=tier,
        entries=manifest_entries,
        research=research,
        gate_cfg=gate_cfg,
        policy_version=policy_version,
    )

    manifest = {
        "generated_at": utc_now_iso(),
        "stage": stage,
        "tier": tier,
        "policy_version": policy_version,
        "manifest_content_hash": manifest_content_hash(
            stage, tier, policy_version, manifest_entries
        ),
        "entries": manifest_entries,
    }

    manifest_path = tier_root / "manifest" / "evidence_manifest.json"
    gate_path = tier_root / "gate_reports" / f"{stage}_{tier}_gate_report.json"
    latest_gate_path = tier_root / "gate_reports" / "latest_gate_report.json"

    write_json(manifest_path, manifest)
    write_json(gate_path, gate_report)
    write_json(latest_gate_path, gate_report)

    output = {
        "manifest_path": manifest_path.as_posix(),
        "gate_report_path": gate_path.as_posix(),
        "passed": gate_report["passed"],
        "reasons": gate_report["reasons"],
        "research_discovered_count": gate_report["research_discovered_count"],
        "research_indexed_count": gate_report["research_indexed_count"],
        "research_unindexed_count": gate_report["research_unindexed_count"],
    }

    if json_stdout:
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(f"Manifest: {manifest_path}")
        print(f"Gate report: {gate_path}")
        print(f"Passed: {gate_report['passed']}")
        if gate_report["reasons"]:
            print("Reasons:")
            for reason in gate_report["reasons"]:
                print(f"- {reason}")

    if not gate_report["passed"] and not allow_fail:
        return 1
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Build BIZRA evidence package tier")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--gate-config", type=Path, default=DEFAULT_GATE_CONFIG_PATH)
    parser.add_argument("--stage", choices=["scaffold", "final"], default="scaffold")
    parser.add_argument(
        "--tier", choices=["private_full", "public_redacted"], default="private_full"
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--package-root", type=Path, default=DEFAULT_PACKAGE_ROOT)
    parser.add_argument("--allow-fail", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    raise SystemExit(
        run(
            config_path=args.config,
            gate_config_path=args.gate_config,
            stage=args.stage,
            tier=args.tier,
            repo_root=args.repo_root,
            package_root=args.package_root,
            allow_fail=args.allow_fail,
            json_stdout=args.json,
        )
    )


if __name__ == "__main__":
    main()
