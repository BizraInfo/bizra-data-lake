"""
BIZRA Resource Fabric — Unified Tooling/Persona/Memory Surface Scanner.

This module turns scattered resources (agents, commands, hooks, skills,
memory modules, MCP tool surfaces, profile configs) into one scored fabric
view so orchestration can be proactive instead of fragmented.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class FabricSource:
    name: str
    category: str
    path: str
    patterns: List[str] = field(default_factory=list)
    weight: float = 1.0


@dataclass
class ResourceFabricProfile:
    profile_name: str = "bizra-proactive-pco-fabric"
    profile_version: str = "1.0.0"
    cache_ttl_s: int = 60
    category_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "agents": 0.20,
            "commands": 0.15,
            "hooks": 0.15,
            "skills": 0.15,
            "memory": 0.15,
            "mcp": 0.12,
            "profiles": 0.08,
        }
    )
    expected_minimums: Dict[str, int] = field(
        default_factory=lambda: {
            "agents": 20,
            "commands": 20,
            "hooks": 5,
            "skills": 5,
            "memory": 10,
            "mcp": 5,
            "profiles": 5,
        }
    )
    sources: List[FabricSource] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "ResourceFabricProfile":
        if not data:
            return cls()

        sources: List[FabricSource] = []
        for raw in data.get("sources", []) or []:
            if not isinstance(raw, dict):
                continue
            sources.append(
                FabricSource(
                    name=str(raw.get("name", "unknown")),
                    category=str(raw.get("category", "profiles")),
                    path=str(raw.get("path", ".")),
                    patterns=[str(p) for p in (raw.get("patterns", []) or [])],
                    weight=max(0.0, float(raw.get("weight", 1.0))),
                )
            )

        profile = cls(
            profile_name=str(data.get("profile_name", "bizra-proactive-pco-fabric")),
            profile_version=str(data.get("profile_version", "1.0.0")),
            cache_ttl_s=max(1, int(data.get("cache_ttl_s", 60))),
            category_weights=(
                data.get("category_weights")
                if isinstance(data.get("category_weights"), dict)
                else cls().category_weights
            ),
            expected_minimums=(
                data.get("expected_minimums")
                if isinstance(data.get("expected_minimums"), dict)
                else cls().expected_minimums
            ),
            sources=sources,
        )

        if not profile.sources:
            profile.sources = cls().sources

        profile._normalize_category_weights()
        return profile

    def _normalize_category_weights(self):
        if not self.category_weights:
            self.category_weights = ResourceFabricProfile().category_weights
            return

        cleaned: Dict[str, float] = {}
        total = 0.0
        for key, val in self.category_weights.items():
            try:
                f = max(0.0, float(val))
            except Exception:
                f = 0.0
            cleaned[str(key)] = f
            total += f

        if total <= 0:
            self.category_weights = ResourceFabricProfile().category_weights
            return

        self.category_weights = {k: v / total for k, v in cleaned.items()}


class ResourceFabric:
    """Scans and scores BIZRA's unified operational resource fabric."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        profile_path: Optional[Path] = None,
    ):
        self.project_root = project_root or self._resolve_project_root()
        self.profile_path = profile_path or self._resolve_profile_path()
        self.profile = self._load_profile(self.profile_path)

        self._last_snapshot: Optional[Dict[str, Any]] = None
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
        env_path = os.environ.get("BIZRA_RESOURCE_FABRIC_PROFILE")
        if env_path:
            return Path(env_path)
        return self.project_root / "config" / "resource_fabric_profile.yaml"

    def _load_profile(self, profile_path: Path) -> ResourceFabricProfile:
        if not profile_path.exists():
            return ResourceFabricProfile()

        try:
            raw = profile_path.read_text(encoding="utf-8")
            data = yaml.safe_load(raw) or {}
            return ResourceFabricProfile.from_dict(data)
        except Exception:
            return ResourceFabricProfile()

    def get_profile(self) -> Dict[str, Any]:
        return {
            "profile_name": self.profile.profile_name,
            "profile_version": self.profile.profile_version,
            "profile_path": str(self.profile_path),
            "cache_ttl_s": self.profile.cache_ttl_s,
            "category_weights": dict(self.profile.category_weights),
            "expected_minimums": dict(self.profile.expected_minimums),
            "source_count": len(self.profile.sources),
        }

    def _scan_source(self, source: FabricSource) -> List[Dict[str, Any]]:
        source_path = self.project_root / source.path
        if not source_path.exists():
            return []

        patterns = source.patterns or ["*"]
        assets: List[Dict[str, Any]] = []

        for pattern in patterns:
            for p in source_path.rglob(pattern):
                if not p.is_file():
                    continue
                rel = p.relative_to(self.project_root)
                assets.append(
                    {
                        "name": p.name,
                        "path": str(rel),
                        "source": source.name,
                        "category": source.category,
                        "weight": float(source.weight),
                    }
                )

        # deduplicate by relative path
        seen = set()
        deduped = []
        for a in assets:
            key = a["path"]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(a)
        return deduped

    def _compute_coverage_score(self, counts_by_category: Dict[str, int]) -> float:
        score = 0.0
        for category, weight in self.profile.category_weights.items():
            expected = max(1, int(self.profile.expected_minimums.get(category, 1)))
            observed = counts_by_category.get(category, 0)
            ratio = min(1.0, observed / expected)
            score += float(weight) * ratio

        return round(max(0.0, min(1.0, score)), 6)

    def snapshot(
        self,
        limit: int = 25,
        include_assets: bool = True,
        force: bool = False,
    ) -> Dict[str, Any]:
        now = time.time()
        ttl = max(1, int(self.profile.cache_ttl_s))
        if (
            not force
            and self._last_snapshot is not None
            and (now - self._last_scan_ts) <= ttl
        ):
            if include_assets:
                return self._last_snapshot

            compact = dict(self._last_snapshot)
            compact.pop("top_assets", None)
            compact.pop("sources_missing", None)
            return compact

        all_assets: List[Dict[str, Any]] = []
        sources_missing: List[str] = []
        by_source: Dict[str, int] = {}
        by_category: Dict[str, int] = {}

        for source in self.profile.sources:
            source_path = self.project_root / source.path
            if not source_path.exists():
                sources_missing.append(source.name)
                by_source[source.name] = 0
                continue

            assets = self._scan_source(source)
            by_source[source.name] = len(assets)
            all_assets.extend(assets)

        for asset in all_assets:
            cat = asset["category"]
            by_category[cat] = by_category.get(cat, 0) + 1

        for asset in all_assets:
            cat_weight = float(
                self.profile.category_weights.get(asset["category"], 0.05)
            )
            asset["composite_weight"] = round(asset["weight"] * cat_weight, 6)

        all_assets.sort(
            key=lambda x: (x["composite_weight"], x["source"], x["path"]),
            reverse=True,
        )

        coverage = self._compute_coverage_score(by_category)
        active_sources = sum(1 for v in by_source.values() if v > 0)
        source_health = (
            active_sources / max(1, len(self.profile.sources))
            if self.profile.sources
            else 0.0
        )

        snapshot: Dict[str, Any] = {
            "profile": self.get_profile(),
            "project_root": str(self.project_root),
            "total_assets": len(all_assets),
            "active_sources": active_sources,
            "source_health": round(source_health, 6),
            "coverage_score": coverage,
            "fabric_score": round((coverage * 0.7) + (source_health * 0.3), 6),
            "by_category": by_category,
            "by_source": by_source,
            "last_scan_at": int(now),
        }

        if include_assets:
            snapshot["top_assets"] = all_assets[: max(1, min(limit, 200))]
            snapshot["sources_missing"] = sources_missing

        self._last_snapshot = snapshot
        self._last_scan_ts = now
        return snapshot
