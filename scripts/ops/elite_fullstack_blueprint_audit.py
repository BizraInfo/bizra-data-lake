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

    sections = {
        "files": {
            "score": round(file_score, 4),
            "checks": wf_checks + script_checks + doc_checks,
        },
        "jobs": {"score": round(job_score, 4), "checks": job_checks},
        "readme": {"score": round(readme_score, 4), "checks": readme_checks},
        "thresholds": {"score": round(threshold_score, 4), "checks": threshold_checks},
    }

    weighted_score = (
        sections["files"]["score"] * float(weights.get("files", 0.0))
        + sections["jobs"]["score"] * float(weights.get("jobs", 0.0))
        + sections["readme"]["score"] * float(weights.get("readme", 0.0))
        + sections["thresholds"]["score"] * float(weights.get("thresholds", 0.0))
    )
    min_score = float(scoring_cfg.get("min_score", 0.0))

    all_checks = (
        sections["files"]["checks"]
        + sections["jobs"]["checks"]
        + sections["readme"]["checks"]
        + sections["thresholds"]["checks"]
    )
    hard_fail = any(not c["passed"] for c in all_checks)
    gate_passed = (not hard_fail) and (weighted_score >= min_score)

    return {
        "program_id": (cfg.get("program") or {}).get("id"),
        "program_version": (cfg.get("program") or {}).get("version"),
        "gate_passed": gate_passed,
        "hard_fail": hard_fail,
        "weighted_score": round(weighted_score, 4),
        "min_score": min_score,
        "sections": sections,
        "failed_checks": [c for c in all_checks if not c["passed"]],
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
