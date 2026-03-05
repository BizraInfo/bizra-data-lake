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
from dataclasses import dataclass, field
from math import isclose
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RepoReader:
    """Repository reader with memoized filesystem access."""

    root: Path
    _exists_cache: dict[str, bool] = field(default_factory=dict)
    _text_cache: dict[str, str] = field(default_factory=dict)
    _yaml_cache: dict[str, dict[str, Any]] = field(default_factory=dict)

    def _path(self, rel: str) -> Path:
        return self.root / rel

    def exists(self, rel: str) -> bool:
        if rel not in self._exists_cache:
            self._exists_cache[rel] = self._path(rel).exists()
        return self._exists_cache[rel]

    def text(self, rel: str) -> str:
        if rel not in self._text_cache:
            if self.exists(rel):
                self._text_cache[rel] = self._path(rel).read_text(encoding="utf-8")
            else:
                self._text_cache[rel] = ""
        return self._text_cache[rel]

    def yaml(self, rel: str) -> dict[str, Any]:
        if rel in self._yaml_cache:
            return self._yaml_cache[rel]
        if not self.exists(rel):
            self._yaml_cache[rel] = {}
            return self._yaml_cache[rel]
        payload = yaml.safe_load(self.text(rel))
        if isinstance(payload, dict):
            self._yaml_cache[rel] = payload
        else:
            self._yaml_cache[rel] = {}
        return self._yaml_cache[rel]


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _check_files(
    reader: RepoReader, rel_paths: list[str]
) -> tuple[float, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    passed = 0
    for rel in rel_paths:
        exists = reader.exists(rel)
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
    reader: RepoReader, job_cfg: dict[str, list[str]]
) -> tuple[float, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    total = 0
    passed = 0
    for rel_file, jobs in job_cfg.items():
        if not reader.exists(rel_file):
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

        payload = reader.yaml(rel_file)
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
    reader: RepoReader, patterns: list[str]
) -> tuple[float, list[dict[str, Any]]]:
    text = reader.text("README.md")
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
    reader: RepoReader,
    expected: dict[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    cfg_rel = "config/phase65_masterpiece_roadmap.yaml"
    checks: list[dict[str, Any]] = []
    if not reader.exists(cfg_rel):
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

    payload = reader.yaml(cfg_rel)
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
    reader: RepoReader, pmbok_cfg: dict[str, list[str]]
) -> tuple[float, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    total = 0
    passed = 0
    for domain, rel_paths in pmbok_cfg.items():
        for rel in rel_paths:
            exists = reader.exists(rel)
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
    reader: RepoReader, pipeline_cfg: dict[str, Any]
) -> tuple[float, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    total = 0
    passed = 0

    for rel in pipeline_cfg.get("workflows") or []:
        exists = reader.exists(rel)
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
        job_name = str(spec.get("job", ""))
        if not reader.exists(rel_file):
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

        payload = reader.yaml(rel_file)
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
        text = reader.text(rel_file)
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
    reader: RepoReader, qa_cfg: dict[str, Any]
) -> tuple[float, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    total = 0
    passed = 0

    for rel in qa_cfg.get("required_test_targets") or []:
        exists = reader.exists(rel)
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
        exists = reader.exists(rel)
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
        observed = len(list((reader.root / "tests").rglob("test_*.py")))
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
    reader: RepoReader, ethics_cfg: dict[str, Any]
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
    source_exists = reader.exists(rel_source)
    text = reader.text(rel_source)

    checks.append(
        {
            "name": f"ethics:source:{rel_source}",
            "passed": source_exists,
            "expected": "exists",
            "actual": "exists" if source_exists else "missing",
        }
    )
    total += 1
    passed += int(source_exists)

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


def _normalize_weights(raw_weights: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    parsed: dict[str, float] = {}
    non_numeric: list[str] = []
    for key, value in raw_weights.items():
        try:
            parsed[key] = float(value)
        except (TypeError, ValueError):
            non_numeric.append(str(key))
    return parsed, non_numeric


def _check_config_integrity(
    pmbok_domains: list[Any],
    min_score_value: Any,
    parsed_weights: dict[str, float],
    non_numeric_weights: list[str],
) -> tuple[float, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    passed = 0
    total = 0

    pmbok_ok = len(pmbok_domains) > 0
    checks.append(
        {
            "name": "config:pmbok_domains_present",
            "passed": pmbok_ok,
            "expected": "non-empty",
            "actual": len(pmbok_domains),
        }
    )
    passed += int(pmbok_ok)
    total += 1

    min_score_ok = False
    min_score_actual: float | str
    try:
        min_score_actual = float(min_score_value)
        min_score_ok = 0.0 <= min_score_actual <= 1.0
    except (TypeError, ValueError):
        min_score_actual = "invalid"
    checks.append(
        {
            "name": "config:min_score_range",
            "passed": min_score_ok,
            "expected": "[0.0, 1.0]",
            "actual": min_score_actual,
        }
    )
    passed += int(min_score_ok)
    total += 1

    non_numeric_ok = len(non_numeric_weights) == 0
    checks.append(
        {
            "name": "config:weights_numeric",
            "passed": non_numeric_ok,
            "expected": "all numeric",
            "actual": "all_numeric" if non_numeric_ok else non_numeric_weights,
        }
    )
    passed += int(non_numeric_ok)
    total += 1

    non_negative_ok = all(v >= 0.0 for v in parsed_weights.values())
    checks.append(
        {
            "name": "config:weights_non_negative",
            "passed": non_negative_ok,
            "expected": ">= 0.0",
            "actual": parsed_weights,
        }
    )
    passed += int(non_negative_ok)
    total += 1

    weights_total = sum(parsed_weights.values())
    weights_sum_ok = isclose(weights_total, 1.0, abs_tol=1e-6)
    checks.append(
        {
            "name": "config:weights_sum",
            "passed": weights_sum_ok,
            "expected": 1.0,
            "actual": round(weights_total, 6),
        }
    )
    passed += int(weights_sum_ok)
    total += 1

    return (passed / total) if total else 1.0, checks


def _build_snr(total_checks: int, failed_checks: int) -> dict[str, Any]:
    passed_checks = max(total_checks - failed_checks, 0)
    # Keep noise floor at 1 to avoid divide-by-zero and preserve monotonic behavior.
    snr_raw = passed_checks / (failed_checks + 1)
    snr_normalized = snr_raw / (snr_raw + 1.0)
    return {
        "signal": passed_checks,
        "noise": failed_checks,
        "raw": round(snr_raw, 4),
        "normalized": round(snr_normalized, 4),
    }


def _build_graph_of_thought(
    sections: dict[str, dict[str, Any]],
    gate_passed: bool,
) -> dict[str, Any]:
    nodes = [
        {
            "id": name,
            "score": section["score"],
            "status": "PASS" if section["score"] >= 0.999 else "DEGRADED",
        }
        for name, section in sections.items()
    ]
    nodes.append(
        {
            "id": "release_readiness",
            "score": 1.0 if gate_passed else 0.0,
            "status": "PASS" if gate_passed else "BLOCKED",
        }
    )

    edges = [
        {"from": "pmbok", "to": "files"},
        {"from": "files", "to": "jobs"},
        {"from": "files", "to": "pipeline"},
        {"from": "jobs", "to": "thresholds"},
        {"from": "pipeline", "to": "qa"},
        {"from": "qa", "to": "thresholds"},
        {"from": "thresholds", "to": "ethics"},
        {"from": "ethics", "to": "release_readiness"},
        {"from": "readme", "to": "release_readiness"},
    ]
    return {"nodes": nodes, "edges": edges}


def _build_interdisciplinary_lenses(
    sections: dict[str, dict[str, Any]]
) -> dict[str, float]:
    def _avg(*names: str) -> float:
        values = [float(sections[name]["score"]) for name in names if name in sections]
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "architecture": _avg("files", "jobs"),
        "devops": _avg("pipeline", "jobs"),
        "quality": _avg("qa", "thresholds"),
        "governance": _avg("pmbok", "ethics"),
        "documentation": _avg("readme"),
        "performance": _avg("thresholds", "qa"),
    }


def _derive_autonomous_next_step(
    gate_passed: bool, optimization_roadmap: list[dict[str, str]]
) -> dict[str, str]:
    if gate_passed:
        return {
            "priority": "P0",
            "owner": "release-management",
            "action": (
                "Promote to protected branch, run Phase65 + Phase56 gates, "
                "then publish evidence artifacts."
            ),
        }
    if optimization_roadmap:
        top = optimization_roadmap[0]
        return {
            "priority": top["priority"],
            "owner": top["owner"],
            "action": top["action"],
        }
    return {
        "priority": "P1",
        "owner": "engineering",
        "action": "Investigate failed gate and rebuild verification evidence.",
    }


def audit_repo(repo_root: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    reader = RepoReader(repo_root)
    checks_cfg = cfg.get("checks") or {}
    scoring_cfg = cfg.get("scoring") or {}
    raw_weights = scoring_cfg.get("weights") or {}
    parsed_weights, non_numeric_weights = _normalize_weights(raw_weights)
    pmbok_domains = cfg.get("pmbok_domains") or []

    wf_score, wf_checks = _check_files(
        reader, (checks_cfg.get("files") or {}).get("workflows") or []
    )
    script_score, script_checks = _check_files(
        reader, (checks_cfg.get("files") or {}).get("scripts") or []
    )
    doc_score, doc_checks = _check_files(
        reader, (checks_cfg.get("files") or {}).get("docs") or []
    )
    file_score = (wf_score + script_score + doc_score) / 3.0

    job_score, job_checks = _check_jobs(reader, checks_cfg.get("jobs") or {})
    readme_score, readme_checks = _check_readme_patterns(
        reader, (checks_cfg.get("readme") or {}).get("required_patterns") or []
    )
    threshold_score, threshold_checks = _check_phase65_thresholds(
        reader, checks_cfg.get("phase65_thresholds") or {}
    )
    pmbok_score, pmbok_checks = _check_pmbok_artifacts(
        reader, checks_cfg.get("pmbok_artifacts") or {}
    )
    pipeline_score, pipeline_checks = _check_pipeline_automation(
        reader, checks_cfg.get("pipeline_automation") or {}
    )
    qa_score, qa_checks = _check_qa_controls(reader, checks_cfg.get("qa") or {})
    ethics_score, ethics_checks = _check_ethical_integrity(
        reader, checks_cfg.get("ethical_integrity") or {}
    )
    config_score, config_checks = _check_config_integrity(
        pmbok_domains=pmbok_domains,
        min_score_value=scoring_cfg.get("min_score"),
        parsed_weights=parsed_weights,
        non_numeric_weights=non_numeric_weights,
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
        "config": {"score": round(config_score, 4), "checks": config_checks},
    }

    weighted_score = 0.0
    for section_name, section_payload in sections.items():
        weighted_score += section_payload["score"] * float(
            parsed_weights.get(section_name, 0.0)
        )
    try:
        min_score = float(scoring_cfg.get("min_score", 0.0))
    except (TypeError, ValueError):
        min_score = 1.0
    weights_total = round(sum(parsed_weights.values()), 4)

    all_checks: list[dict[str, Any]] = []
    for section in sections.values():
        all_checks.extend(section["checks"])
    hard_fail = any(not c["passed"] for c in all_checks)
    gate_passed = (not hard_fail) and (weighted_score >= min_score)
    failed_checks = [c for c in all_checks if not c["passed"]]
    optimization_roadmap = _build_optimization_roadmap(failed_checks)
    snr = _build_snr(len(all_checks), len(failed_checks))
    graph_of_thought = _build_graph_of_thought(sections, gate_passed)
    interdisciplinary_lenses = _build_interdisciplinary_lenses(sections)
    autonomous_next_step = _derive_autonomous_next_step(
        gate_passed, optimization_roadmap
    )
    giants_protocol = (cfg.get("program") or {}).get("giants_protocol") or [
        "Shannon:SNR maximization",
        "Deming:PDCA quality discipline",
        "Lamport:deterministic evidence ordering",
        "PMBOK:lifecycle governance",
        "Al-Ghazali:Ihsan as hard floor",
    ]

    return {
        "program_id": (cfg.get("program") or {}).get("id"),
        "program_version": (cfg.get("program") or {}).get("version"),
        "gate_passed": gate_passed,
        "hard_fail": hard_fail,
        "weighted_score": round(weighted_score, 4),
        "min_score": min_score,
        "weights_total": weights_total,
        "snr": snr,
        "graph_of_thought": graph_of_thought,
        "interdisciplinary_lenses": interdisciplinary_lenses,
        "standing_on_giants_protocol": giants_protocol,
        "autonomous_next_step": autonomous_next_step,
        "sections": sections,
        "failed_checks": failed_checks,
        "optimization_roadmap": optimization_roadmap,
        "pmbok_domains": pmbok_domains,
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
