"""
Elite full-stack blueprint audit.

Validates that repository implementation aligns with:
- PMBOK execution structure
- DevOps and CI/CD automation
- Phase65 quality/security/performance thresholds
- Documentation and community visibility requirements
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _check_files(
    repo_root: Path, rel_paths: list[str]
) -> tuple[float, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    passed = 0
    for rel in rel_paths:
        exists = (repo_root / rel).exists()
        checks.append(
            {
                "name": f"file:{rel}",
                "passed": exists,
                "expected": "exists",
                "actual": "exists" if exists else "missing",
            }
        )
        passed += int(exists)
    score = (passed / len(rel_paths)) if rel_paths else 1.0
    return score, checks


def _check_jobs(
    repo_root: Path, job_cfg: dict[str, list[str]]
) -> tuple[float, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    total = 0
    passed = 0
    for rel_file, jobs in job_cfg.items():
        path = repo_root / rel_file
        if not path.exists():
            for job in jobs:
                checks.append(
                    {
                        "name": f"job:{rel_file}:{job}",
                        "passed": False,
                        "expected": "present",
                        "actual": "workflow_missing",
                    }
                )
                total += 1
            continue

        payload = _load_yaml(path)
        declared_jobs = set((payload.get("jobs") or {}).keys())
        for job in jobs:
            ok = job in declared_jobs
            checks.append(
                {
                    "name": f"job:{rel_file}:{job}",
                    "passed": ok,
                    "expected": "present",
                    "actual": "present" if ok else "missing",
                }
            )
            total += 1
            passed += int(ok)

    score = (passed / total) if total else 1.0
    return score, checks


def _check_readme_patterns(
    repo_root: Path, patterns: list[str]
) -> tuple[float, list[dict[str, Any]]]:
    readme = repo_root / "README.md"
    text = readme.read_text(encoding="utf-8") if readme.exists() else ""
    checks: list[dict[str, Any]] = []
    passed = 0
    for pattern in patterns:
        ok = pattern in text
        checks.append(
            {
                "name": f"readme:{pattern}",
                "passed": ok,
                "expected": "present",
                "actual": "present" if ok else "missing",
            }
        )
        passed += int(ok)
    score = (passed / len(patterns)) if patterns else 1.0
    return score, checks


def _check_phase65_thresholds(
    repo_root: Path,
    expected: dict[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    cfg_path = repo_root / "config/phase65_masterpiece_roadmap.yaml"
    checks: list[dict[str, Any]] = []
    if not cfg_path.exists():
        for key, value in expected.items():
            checks.append(
                {
                    "name": f"phase65:{key}",
                    "passed": False,
                    "expected": value,
                    "actual": "phase65_config_missing",
                }
            )
        return 0.0, checks

    payload = _load_yaml(cfg_path)
    required = (payload.get("quality_gates") or {}).get("required") or {}
    scoring = (payload.get("quality_gates") or {}).get("scoring") or {}

    observed = {
        "min_snr_score": scoring.get("min_snr_score"),
        "min_avg_ihsan": required.get("min_avg_ihsan"),
        "min_speedup_system1_vs_system2": required.get(
            "min_speedup_system1_vs_system2"
        ),
        "max_avg_latency_ms": required.get("max_avg_latency_ms"),
        "signed_receipts_required": required.get("signed_receipts_required"),
    }

    passed = 0
    total = len(expected)
    for key, expected_value in expected.items():
        actual = observed.get(key)
        ok = actual == expected_value
        checks.append(
            {
                "name": f"phase65:{key}",
                "passed": ok,
                "expected": expected_value,
                "actual": actual,
            }
        )
        passed += int(ok)

    return (passed / total) if total else 1.0, checks


def _check_pmbok_artifacts(
    repo_root: Path, pmbok_cfg: dict[str, list[str]]
) -> tuple[float, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    total = 0
    passed = 0
    for domain, rel_paths in pmbok_cfg.items():
        for rel in rel_paths:
            exists = (repo_root / rel).exists()
            checks.append(
                {
                    "name": f"pmbok:{domain}:{rel}",
                    "passed": exists,
                    "expected": "exists",
                    "actual": "exists" if exists else "missing",
                }
            )
            total += 1
            passed += int(exists)
    return (passed / total) if total else 1.0, checks


def _check_pipeline_automation(
    repo_root: Path, pipeline_cfg: dict[str, Any]
) -> tuple[float, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    total = 0
    passed = 0

    for rel in pipeline_cfg.get("workflows") or []:
        exists = (repo_root / rel).exists()
        checks.append(
            {
                "name": f"pipeline:workflow:{rel}",
                "passed": exists,
                "expected": "exists",
                "actual": "exists" if exists else "missing",
            }
        )
        total += 1
        passed += int(exists)

    dep_cfg = pipeline_cfg.get("required_job_dependencies") or {}
    for rel_file, spec in dep_cfg.items():
        path = repo_root / rel_file
        job_name = str(spec.get("job", ""))
        if not path.exists():
            checks.append(
                {
                    "name": f"pipeline:deps:{rel_file}:{job_name}",
                    "passed": False,
                    "expected": "workflow and job present",
                    "actual": "workflow_missing",
                }
            )
            total += 1
            continue

        payload = _load_yaml(path)
        jobs = payload.get("jobs") or {}
        job_payload = jobs.get(job_name)
        if not isinstance(job_payload, dict):
            checks.append(
                {
                    "name": f"pipeline:deps:{rel_file}:{job_name}",
                    "passed": False,
                    "expected": "job present",
                    "actual": "job_missing",
                }
            )
            total += 1
            continue

        needs_raw = job_payload.get("needs", [])
        if isinstance(needs_raw, str):
            needs = [needs_raw]
        elif isinstance(needs_raw, list):
            needs = [str(x) for x in needs_raw]
        else:
            needs = []

        needs_all = [str(x) for x in (spec.get("needs_all_of") or [])]
        needs_any = [str(x) for x in (spec.get("needs_any_of") or [])]

        if needs_all:
            ok_all = all(req in needs for req in needs_all)
            checks.append(
                {
                    "name": f"pipeline:deps_all:{rel_file}:{job_name}",
                    "passed": ok_all,
                    "expected": needs_all,
                    "actual": needs,
                }
            )
            total += 1
            passed += int(ok_all)

        if needs_any:
            ok_any = any(req in needs for req in needs_any)
            checks.append(
                {
                    "name": f"pipeline:deps_any:{rel_file}:{job_name}",
                    "passed": ok_any,
                    "expected": f"any_of:{needs_any}",
                    "actual": needs,
                }
            )
            total += 1
            passed += int(ok_any)

    pattern_cfg = pipeline_cfg.get("required_patterns") or {}
    for rel_file, patterns in pattern_cfg.items():
        path = repo_root / rel_file
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        for pattern in patterns or []:
            ok = pattern in text
            checks.append(
                {
                    "name": f"pipeline:pattern:{rel_file}:{pattern}",
                    "passed": ok,
                    "expected": "present",
                    "actual": "present" if ok else "missing",
                }
            )
            total += 1
            passed += int(ok)

    return (passed / total) if total else 1.0, checks


def _check_qa_controls(
    repo_root: Path, qa_cfg: dict[str, Any]
) -> tuple[float, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    total = 0
    passed = 0

    for rel in qa_cfg.get("required_test_targets") or []:
        exists = (repo_root / rel).exists()
        checks.append(
            {
                "name": f"qa:test:{rel}",
                "passed": exists,
                "expected": "exists",
                "actual": "exists" if exists else "missing",
            }
        )
        total += 1
        passed += int(exists)

    for rel in qa_cfg.get("required_quality_scripts") or []:
        exists = (repo_root / rel).exists()
        checks.append(
            {
                "name": f"qa:script:{rel}",
                "passed": exists,
                "expected": "exists",
                "actual": "exists" if exists else "missing",
            }
        )
        total += 1
        passed += int(exists)

    min_test_files = qa_cfg.get("min_test_files")
    if isinstance(min_test_files, int) and min_test_files >= 0:
        observed = len(list((repo_root / "tests").rglob("test_*.py")))
        ok = observed >= min_test_files
        checks.append(
            {
                "name": "qa:min_test_files",
                "passed": ok,
                "expected": min_test_files,
                "actual": observed,
            }
        )
        total += 1
        passed += int(ok)

    return (passed / total) if total else 1.0, checks


def _check_ethical_integrity(
    repo_root: Path, ethics_cfg: dict[str, Any]
) -> tuple[float, list[dict[str, Any]]]:
    if not ethics_cfg:
        return 1.0, []

    checks: list[dict[str, Any]] = []
    total = 0
    passed = 0

    required_invariants = [str(x) for x in (ethics_cfg.get("required_invariants") or [])]
    rel_source_raw = ethics_cfg.get("source_file")
    if not required_invariants and rel_source_raw is None:
        return 1.0, []

    rel_source = str(rel_source_raw or "core/integration/constants.py")
    source_path = repo_root / rel_source
    text = source_path.read_text(encoding="utf-8") if source_path.exists() else ""

    checks.append(
        {
            "name": f"ethics:source:{rel_source}",
            "passed": source_path.exists(),
            "expected": "exists",
            "actual": "exists" if source_path.exists() else "missing",
        }
    )
    total += 1
    passed += int(source_path.exists())

    for invariant in required_invariants:
        ok = invariant in text
        checks.append(
            {
                "name": f"ethics:invariant:{invariant}",
                "passed": ok,
                "expected": "present",
                "actual": "present" if ok else "missing",
            }
        )
        total += 1
        passed += int(ok)

    return (passed / total) if total else 1.0, checks


def _recommendation_from_check(check_name: str) -> dict[str, str]:
    if check_name.startswith("phase65:") or check_name.startswith("ethics:"):
        return {
            "priority": "P0",
            "owner": "governance",
            "action": "Restore constitutional thresholds/invariants before any release.",
        }
    if check_name.startswith("pipeline:") or check_name.startswith("job:"):
        return {
            "priority": "P1",
            "owner": "devops",
            "action": "Fix CI/CD orchestration and job dependency chain.",
        }
    if check_name.startswith("qa:"):
        return {
            "priority": "P1",
            "owner": "quality",
            "action": "Recover required test and quality-control coverage.",
        }
    if check_name.startswith("pmbok:"):
        return {
            "priority": "P2",
            "owner": "program-management",
            "action": "Restore PMBOK artifact traceability for lifecycle control.",
        }
    if check_name.startswith("file:"):
        return {
            "priority": "P2",
            "owner": "architecture",
            "action": "Restore missing blueprint dependencies (files/scripts/docs).",
        }
    return {
        "priority": "P3",
        "owner": "docs",
        "action": "Fix visibility/README hygiene requirements.",
    }


def _build_optimization_roadmap(
    failed_checks: list[dict[str, Any]]
) -> list[dict[str, str]]:
    roadmap: list[dict[str, str]] = []
    priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    for check in failed_checks:
        rec = _recommendation_from_check(str(check.get("name", "")))
        roadmap.append(
            {
                "priority": rec["priority"],
                "owner": rec["owner"],
                "check": str(check.get("name", "")),
                "action": rec["action"],
            }
        )
    roadmap.sort(key=lambda item: priority_order.get(item["priority"], 99))
    return roadmap


def audit_repo(repo_root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    checks_cfg = cfg.get("checks") or {}
    scoring_cfg = cfg.get("scoring") or {}
    weights = scoring_cfg.get("weights") or {}

    wf_score, wf_checks = _check_files(
        repo_root, (checks_cfg.get("files") or {}).get("workflows") or []
    )
    script_score, script_checks = _check_files(
        repo_root, (checks_cfg.get("files") or {}).get("scripts") or []
    )
    doc_score, doc_checks = _check_files(
        repo_root, (checks_cfg.get("files") or {}).get("docs") or []
    )
    file_score = (wf_score + script_score + doc_score) / 3.0

    job_score, job_checks = _check_jobs(repo_root, checks_cfg.get("jobs") or {})
    readme_score, readme_checks = _check_readme_patterns(
        repo_root, (checks_cfg.get("readme") or {}).get("required_patterns") or []
    )
    threshold_score, threshold_checks = _check_phase65_thresholds(
        repo_root, checks_cfg.get("phase65_thresholds") or {}
    )
    pmbok_score, pmbok_checks = _check_pmbok_artifacts(
        repo_root, checks_cfg.get("pmbok_artifacts") or {}
    )
    pipeline_score, pipeline_checks = _check_pipeline_automation(
        repo_root, checks_cfg.get("pipeline_automation") or {}
    )
    qa_score, qa_checks = _check_qa_controls(repo_root, checks_cfg.get("qa") or {})
    ethics_score, ethics_checks = _check_ethical_integrity(
        repo_root, checks_cfg.get("ethical_integrity") or {}
    )

    sections = {
        "files": {
            "score": round(file_score, 4),
            "checks": wf_checks + script_checks + doc_checks,
        },
        "jobs": {"score": round(job_score, 4), "checks": job_checks},
        "readme": {"score": round(readme_score, 4), "checks": readme_checks},
        "thresholds": {"score": round(threshold_score, 4), "checks": threshold_checks},
        "pmbok": {"score": round(pmbok_score, 4), "checks": pmbok_checks},
        "pipeline": {"score": round(pipeline_score, 4), "checks": pipeline_checks},
        "qa": {"score": round(qa_score, 4), "checks": qa_checks},
        "ethics": {"score": round(ethics_score, 4), "checks": ethics_checks},
    }

    weighted_score = 0.0
    for section_name, section_payload in sections.items():
        weighted_score += section_payload["score"] * float(weights.get(section_name, 0.0))
    min_score = float(scoring_cfg.get("min_score", 0.0))
    weights_total = round(sum(float(v) for v in weights.values()), 4)

    all_checks: list[dict[str, Any]] = []
    for section in sections.values():
        all_checks.extend(section["checks"])
    hard_fail = any(not c["passed"] for c in all_checks)
    gate_passed = (not hard_fail) and (weighted_score >= min_score)
    failed_checks = [c for c in all_checks if not c["passed"]]

    return {
        "program_id": (cfg.get("program") or {}).get("id"),
        "program_version": (cfg.get("program") or {}).get("version"),
        "gate_passed": gate_passed,
        "hard_fail": hard_fail,
        "weighted_score": round(weighted_score, 4),
        "min_score": min_score,
        "weights_total": weights_total,
        "sections": sections,
        "failed_checks": failed_checks,
        "optimization_roadmap": _build_optimization_roadmap(failed_checks),
        "pmbok_domains": cfg.get("pmbok_domains") or [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run elite full-stack blueprint audit."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/elite_fullstack_blueprint.yaml"),
        help="Blueprint audit config YAML path.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Repository root path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional output JSON report path.",
    )
    args = parser.parse_args()

    cfg = _load_yaml(args.config)
    report = audit_repo(args.repo_root.resolve(), cfg)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
