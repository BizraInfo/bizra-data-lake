"""
BIZRA Agentic Self Harness Engine.

Proactively scans the codebase for performance/reliability best-practice
gaps and produces a prioritized action report with a normalized score.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


@dataclass
class HarnessRule:
    id: str
    category: str
    severity: str
    description: str
    file_globs: List[str]
    patterns: List[str]
    recommendation: str


@dataclass
class HarnessProfile:
    profile_name: str = "bizra-agentic-self-harness"
    profile_version: str = "1.0.0"
    cache_ttl_s: int = 45
    max_file_size_bytes: int = 1_500_000
    max_findings_per_rule: int = 200
    include_paths: List[str] = field(default_factory=list)
    exclude_path_fragments: List[str] = field(default_factory=list)
    penalties: Dict[str, float] = field(default_factory=dict)
    rules: List[HarnessRule] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "HarnessProfile":
        if not data:
            return cls(
                include_paths=[
                    "core",
                    "tests",
                    "scripts",
                    "deploy",
                    "config",
                    ".claude",
                    "tools",
                ],
                exclude_path_fragments=[
                    ".git/",
                    ".venv/",
                    ".venv-linux/",
                    "node_modules/",
                    "04_GOLD/",
                    "99_QUARANTINE/",
                    "__pycache__/",
                ],
                penalties={
                    "critical": 0.045,
                    "high": 0.020,
                    "medium": 0.010,
                    "low": 0.004,
                },
            )

        rules: List[HarnessRule] = []
        for raw in data.get("rules", []) or []:
            if not isinstance(raw, dict):
                continue
            rules.append(
                HarnessRule(
                    id=str(raw.get("id", "unknown_rule")),
                    category=str(raw.get("category", "general")),
                    severity=str(raw.get("severity", "low")).lower(),
                    description=str(raw.get("description", "")),
                    file_globs=[
                        str(x) for x in (raw.get("file_globs", []) or ["*.py"])
                    ],
                    patterns=[str(x) for x in (raw.get("patterns", []) or [])],
                    recommendation=str(raw.get("recommendation", "")),
                )
            )

        return cls(
            profile_name=str(data.get("profile_name", "bizra-agentic-self-harness")),
            profile_version=str(data.get("profile_version", "1.0.0")),
            cache_ttl_s=max(1, int(data.get("cache_ttl_s", 45))),
            max_file_size_bytes=max(
                10_000, int(data.get("max_file_size_bytes", 1_500_000))
            ),
            max_findings_per_rule=max(1, int(data.get("max_findings_per_rule", 200))),
            include_paths=[
                str(x)
                for x in (data.get("include_paths", []) or ["core", "tests", "scripts"])
            ],
            exclude_path_fragments=[
                str(x) for x in (data.get("exclude_path_fragments", []) or [])
            ],
            penalties={
                "critical": float(
                    (data.get("penalties", {}) or {}).get("critical", 0.045)
                ),
                "high": float((data.get("penalties", {}) or {}).get("high", 0.020)),
                "medium": float((data.get("penalties", {}) or {}).get("medium", 0.010)),
                "low": float((data.get("penalties", {}) or {}).get("low", 0.004)),
            },
            rules=rules,
        )


@dataclass
class HarnessFinding:
    rule_id: str
    category: str
    severity: str
    path: str
    line: int
    snippet: str
    recommendation: str


class SelfHarnessEngine:
    """Agentic self harness for proactive performance best-practice enforcement."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        profile_path: Optional[Path] = None,
    ):
        self.project_root = project_root or self._resolve_project_root()
        self.profile_path = profile_path or self._resolve_profile_path()
        self.profile = self._load_profile(self.profile_path)

        self._last_report: Optional[Dict[str, Any]] = None
        self._last_scan_ts: float = 0.0

    def _resolve_project_root(self) -> Path:
        env_root = os.environ.get("BIZRA_PROJECT_ROOT")
        if env_root:
            return Path(env_root)

        cwd = Path.cwd()
        if (cwd / "core").exists() and (cwd / "config").exists():
            return cwd

        fallback = Path("/mnt/c/BIZRA-DATA-LAKE")
        if fallback.exists():
            return fallback

        return cwd

    def _resolve_profile_path(self) -> Path:
        env_path = os.environ.get("BIZRA_SELF_HARNESS_PROFILE")
        if env_path:
            return Path(env_path)
        return self.project_root / "config" / "self_harness_profile.yaml"

    def _load_profile(self, profile_path: Path) -> HarnessProfile:
        if not profile_path.exists():
            return HarnessProfile.from_dict(None)

        try:
            raw = profile_path.read_text(encoding="utf-8")
            data = yaml.safe_load(raw) or {}
            return HarnessProfile.from_dict(data)
        except Exception:
            return HarnessProfile.from_dict(None)

    def _is_excluded(self, path: Path) -> bool:
        rel = str(path.relative_to(self.project_root)).replace("\\", "/")
        return any(fragment in rel for fragment in self.profile.exclude_path_fragments)

    def _iter_candidate_files(self, file_globs: List[str]) -> Iterable[Path]:
        seen: set[str] = set()
        for include in self.profile.include_paths:
            root = self.project_root / include
            if not root.exists():
                continue

            for pattern in file_globs:
                for path in root.rglob(pattern):
                    if not path.is_file():
                        continue
                    if self._is_excluded(path):
                        continue
                    if path.stat().st_size > self.profile.max_file_size_bytes:
                        continue
                    key = str(path)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield path

    def _scan_rule(self, rule: HarnessRule) -> List[HarnessFinding]:
        regexes = [re.compile(pat) for pat in rule.patterns]
        findings: List[HarnessFinding] = []

        for path in self._iter_candidate_files(rule.file_globs):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            lines = text.splitlines()
            for idx, line in enumerate(lines, start=1):
                if any(r.search(line) for r in regexes):
                    findings.append(
                        HarnessFinding(
                            rule_id=rule.id,
                            category=rule.category,
                            severity=rule.severity,
                            path=str(path.relative_to(self.project_root)),
                            line=idx,
                            snippet=line.strip()[:200],
                            recommendation=rule.recommendation,
                        )
                    )
                    if len(findings) >= self.profile.max_findings_per_rule:
                        return findings

        return findings

    def _score(self, findings: List[HarnessFinding]) -> float:
        score = 1.0
        penalties = self.profile.penalties
        for f in findings:
            score -= penalties.get(f.severity, penalties.get("low", 0.004))

        return round(max(0.0, min(1.0, score)), 6)

    def _top_actions(
        self, findings: List[HarnessFinding], limit: int
    ) -> List[Dict[str, Any]]:
        bucket: Dict[str, Dict[str, Any]] = {}
        for f in findings:
            key = f.rule_id
            if key not in bucket:
                bucket[key] = {
                    "rule_id": f.rule_id,
                    "severity": f.severity,
                    "category": f.category,
                    "recommendation": f.recommendation,
                    "count": 0,
                    "examples": [],
                }
            bucket[key]["count"] += 1
            if len(bucket[key]["examples"]) < 3:
                bucket[key]["examples"].append(f"{f.path}:{f.line}")

        ranked = sorted(
            bucket.values(),
            key=lambda x: (
                {"critical": 4, "high": 3, "medium": 2, "low": 1}.get(x["severity"], 0),
                x["count"],
            ),
            reverse=True,
        )
        return ranked[: max(1, min(limit, 50))]

    def run(
        self,
        include_findings: bool = False,
        findings_limit: int = 200,
        force: bool = False,
    ) -> Dict[str, Any]:
        now = time.time()
        if (
            not force
            and self._last_report is not None
            and (now - self._last_scan_ts) <= self.profile.cache_ttl_s
        ):
            if include_findings:
                return self._last_report
            compact = dict(self._last_report)
            compact.pop("findings", None)
            return compact

        all_findings: List[HarnessFinding] = []
        by_rule: Dict[str, int] = {}
        by_severity: Dict[str, int] = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        for rule in self.profile.rules:
            rule_findings = self._scan_rule(rule)
            by_rule[rule.id] = len(rule_findings)
            all_findings.extend(rule_findings)
            for f in rule_findings:
                by_severity[f.severity] = by_severity.get(f.severity, 0) + 1

        score = self._score(all_findings)
        report: Dict[str, Any] = {
            "profile_name": self.profile.profile_name,
            "profile_version": self.profile.profile_version,
            "profile_path": str(self.profile_path),
            "project_root": str(self.project_root),
            "harness_score": score,
            "total_findings": len(all_findings),
            "by_severity": by_severity,
            "by_rule": by_rule,
            "top_actions": self._top_actions(all_findings, limit=10),
            "scanned_at": int(now),
        }

        if include_findings:
            report["findings"] = [
                {
                    "rule_id": f.rule_id,
                    "category": f.category,
                    "severity": f.severity,
                    "path": f.path,
                    "line": f.line,
                    "snippet": f.snippet,
                    "recommendation": f.recommendation,
                }
                for f in all_findings[: max(1, min(findings_limit, 5000))]
            ]

        self._last_report = report
        self._last_scan_ts = now
        return report

    def peek_report(self, include_findings: bool = False) -> Dict[str, Any]:
        """
        Return cached self-harness report without triggering a cold filesystem scan.

        Used by lightweight status endpoints where blocking full-repo scans are
        undesirable.
        """
        if self._last_report is None:
            return {
                "profile_name": self.profile.profile_name,
                "profile_version": self.profile.profile_version,
                "profile_path": str(self.profile_path),
                "project_root": str(self.project_root),
                "status": "not_scanned",
                "harness_score": None,
                "total_findings": None,
                "by_severity": {},
                "by_rule": {},
                "top_actions": [],
                "scanned_at": None,
            }

        if include_findings:
            return dict(self._last_report)

        compact = dict(self._last_report)
        compact.pop("findings", None)
        return compact
