"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA SKILLS — Skill Registry                                             ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Load, register, and manage skills from .claude/skills/ directory.         ║
║   Each skill has a SKILL.md manifest with YAML frontmatter.                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Standing on Giants:
- Eric Evans (2003): Domain-Driven Design (registry as aggregate root)
- Martin Fowler (2004): Plugin architecture patterns
"""

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from core.elite.self_harness_engine import SelfHarnessEngine
from core.integration.constants import UNIFIED_IHSAN_THRESHOLD
from core.skills.resource_fabric import ResourceFabric

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════


class SkillStatus(str, Enum):
    """Skill lifecycle status."""

    AVAILABLE = "available"  # Ready to invoke
    LOADING = "loading"  # Currently loading
    ACTIVE = "active"  # Currently executing
    SUSPENDED = "suspended"  # Temporarily disabled
    ERROR = "error"  # Failed to load


class SkillContext(str, Enum):
    """Skill execution context."""

    FORK = "fork"  # Run in separate context (sub-agent)
    INLINE = "inline"  # Run in current context


# ═══════════════════════════════════════════════════════════════════════════════
# SKILL MANIFEST
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SkillManifest:
    """
    Parsed SKILL.md manifest.

    Extracted from YAML frontmatter:
    ---
    name: skill-name
    description: What the skill does
    context: fork
    agent: sovereign-planner
    tags: [tag1, tag2]
    ---
    """

    # Identity
    name: str
    description: str
    version: str = "1.0.0"
    author: str = "BIZRA"

    # Execution
    context: SkillContext = SkillContext.FORK
    agent: str = "general-purpose"

    # Classification
    tags: List[str] = field(default_factory=list)

    # Inputs/Outputs (from SKILL.md 'inputs' section if present)
    required_inputs: List[str] = field(default_factory=list)
    optional_inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)

    # Tool requirements (from 'inputs.runtime_gate' or inferred)
    mcp_tools: List[str] = field(default_factory=list)

    # Quality constraints
    ihsan_floor: float = UNIFIED_IHSAN_THRESHOLD

    # The full content of SKILL.md (for reference)
    raw_content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
            "author": self.author,
            "context": self.context.value,
            "agent": self.agent,
            "tags": self.tags,
            "required_inputs": self.required_inputs,
            "optional_inputs": self.optional_inputs,
            "outputs": self.outputs,
            "mcp_tools": self.mcp_tools,
            "ihsan_floor": self.ihsan_floor,
        }

    @classmethod
    def from_frontmatter(
        cls, frontmatter: Dict[str, Any], raw: str = ""
    ) -> "SkillManifest":
        """Create manifest from parsed YAML frontmatter."""
        # Parse context
        ctx_str = frontmatter.get("context", "fork")
        context = SkillContext.FORK
        if ctx_str == "inline":
            context = SkillContext.INLINE

        # Parse inputs if present
        inputs = frontmatter.get("inputs", {})
        required_inputs = []
        optional_inputs = []
        if isinstance(inputs, dict):
            required_inputs = inputs.get("required", [])
            optional_inputs = inputs.get("optional", [])

        return cls(
            name=frontmatter.get("name", "unknown"),
            description=frontmatter.get("description", ""),
            version=frontmatter.get("version", "1.0.0"),
            author=frontmatter.get("author", "BIZRA"),
            context=context,
            agent=frontmatter.get("agent", "general-purpose"),
            tags=frontmatter.get("tags", []),
            required_inputs=required_inputs,
            optional_inputs=optional_inputs,
            outputs=frontmatter.get("outputs", []),
            mcp_tools=frontmatter.get("mcp_tools", []),
            ihsan_floor=float(frontmatter.get("ihsan_floor", UNIFIED_IHSAN_THRESHOLD)),
            raw_content=raw,
        )


@dataclass
class SkillPerformanceProfile:
    """
    Runtime scoring profile for ranking top skills by execution quality.

    The profile is configurable via YAML and combines reliability, latency,
    usage confidence, Ihsān floor, and strategic-tag boost into one score.
    """

    profile_name: str = "elite-performance-v1"
    profile_version: str = "1.0.0"
    top_n_default: int = 10
    min_invocations_for_confidence: int = 20
    latency_target_ms: float = 1200.0
    max_tag_boost: float = 0.20

    # Score weights (normalized on load)
    weight_success_rate: float = 0.40
    weight_latency: float = 0.20
    weight_usage_confidence: float = 0.15
    weight_ihsan_floor: float = 0.15
    weight_tag_boost: float = 0.10

    # Domain-level strategic boosts
    tag_boosts: Dict[str, float] = field(
        default_factory=lambda: {
            "security": 0.06,
            "performance": 0.05,
            "reasoning": 0.05,
            "architecture": 0.04,
            "integration": 0.03,
            "documentation": 0.02,
            "testing": 0.03,
            "reliability": 0.04,
            "research": 0.03,
            "autonomous": 0.05,
        }
    )

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SkillPerformanceProfile":
        """Create profile from YAML dict with safe defaults and normalized weights."""
        if not data:
            profile = cls()
            profile.normalize_weights()
            return profile

        weights = (
            data.get("weights", {}) if isinstance(data.get("weights"), dict) else {}
        )
        profile = cls(
            profile_name=str(data.get("profile_name", "elite-performance-v1")),
            profile_version=str(data.get("profile_version", "1.0.0")),
            top_n_default=max(1, int(data.get("top_n_default", 10))),
            min_invocations_for_confidence=max(
                1, int(data.get("min_invocations_for_confidence", 20))
            ),
            latency_target_ms=max(1.0, float(data.get("latency_target_ms", 1200.0))),
            max_tag_boost=max(0.0, float(data.get("max_tag_boost", 0.20))),
            weight_success_rate=float(
                weights.get("success_rate", data.get("weight_success_rate", 0.40))
            ),
            weight_latency=float(
                weights.get("latency", data.get("weight_latency", 0.20))
            ),
            weight_usage_confidence=float(
                weights.get(
                    "usage_confidence",
                    data.get("weight_usage_confidence", 0.15),
                )
            ),
            weight_ihsan_floor=float(
                weights.get("ihsan_floor", data.get("weight_ihsan_floor", 0.15))
            ),
            weight_tag_boost=float(
                weights.get("tag_boost", data.get("weight_tag_boost", 0.10))
            ),
            tag_boosts=(
                data.get("tag_boosts")
                if isinstance(data.get("tag_boosts"), dict)
                else cls().tag_boosts
            ),
        )
        profile.normalize_weights()
        return profile

    def normalize_weights(self):
        """Normalize all score weights so their sum is exactly 1.0."""
        total = (
            self.weight_success_rate
            + self.weight_latency
            + self.weight_usage_confidence
            + self.weight_ihsan_floor
            + self.weight_tag_boost
        )
        if total <= 0:
            self.weight_success_rate = 0.40
            self.weight_latency = 0.20
            self.weight_usage_confidence = 0.15
            self.weight_ihsan_floor = 0.15
            self.weight_tag_boost = 0.10
            total = 1.0

        self.weight_success_rate /= total
        self.weight_latency /= total
        self.weight_usage_confidence /= total
        self.weight_ihsan_floor /= total
        self.weight_tag_boost /= total

    def to_dict(self) -> Dict[str, Any]:
        """Serialize profile for diagnostics and API exposure."""
        return {
            "profile_name": self.profile_name,
            "profile_version": self.profile_version,
            "top_n_default": self.top_n_default,
            "min_invocations_for_confidence": self.min_invocations_for_confidence,
            "latency_target_ms": self.latency_target_ms,
            "max_tag_boost": self.max_tag_boost,
            "weights": {
                "success_rate": self.weight_success_rate,
                "latency": self.weight_latency,
                "usage_confidence": self.weight_usage_confidence,
                "ihsan_floor": self.weight_ihsan_floor,
                "tag_boost": self.weight_tag_boost,
            },
            "tag_boosts": dict(self.tag_boosts),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTERED SKILL
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RegisteredSkill:
    """
    A skill registered in the runtime.

    Tracks usage metrics and status.
    """

    manifest: SkillManifest
    path: str  # Path to SKILL.md
    status: SkillStatus = SkillStatus.AVAILABLE

    # Usage metrics
    invocation_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    total_duration_ms: float = 0.0

    # Timestamps
    registered_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_invoked: Optional[str] = None

    @property
    def success_rate(self) -> float:
        """Compute success rate."""
        if self.invocation_count == 0:
            return 1.0
        return self.success_count / self.invocation_count

    @property
    def avg_duration_ms(self) -> float:
        """Average invocation duration."""
        if self.invocation_count == 0:
            return 0.0
        return self.total_duration_ms / self.invocation_count

    def record_invocation(self, success: bool, duration_ms: float):
        """Record an invocation."""
        self.invocation_count += 1
        self.total_duration_ms += duration_ms
        self.last_invoked = datetime.now(timezone.utc).isoformat()

        if success:
            self.success_count += 1
        else:
            self.failure_count += 1

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "manifest": self.manifest.to_dict(),
            "path": self.path,
            "status": self.status.value,
            "invocation_count": self.invocation_count,
            "success_rate": self.success_rate,
            "avg_duration_ms": self.avg_duration_ms,
            "registered_at": self.registered_at,
            "last_invoked": self.last_invoked,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SKILL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════


class SkillRegistry:
    """
    Central registry for all skills.

    Responsibilities:
    - Load skills from .claude/skills/
    - Track skill usage and success rates
    - Provide skill lookup by name, tag, agent
    - Enforce Ihsān floor requirements
    """

    # YAML frontmatter pattern
    FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---", re.DOTALL)

    def __init__(self, skills_dir: Optional[str] = None):
        """
        Initialize registry.

        Args:
            skills_dir: Path to skills directory. Defaults to .claude/skills/
        """
        if skills_dir:
            self.skills_dir = Path(skills_dir)
        else:
            # Try to find .claude/skills/ relative to repo root
            self.skills_dir = self._find_skills_dir()

        self._skills: Dict[str, RegisteredSkill] = {}
        self._by_tag: Dict[str, List[str]] = {}  # tag -> [skill_names]
        self._by_agent: Dict[str, List[str]] = {}  # agent -> [skill_names]
        self._profile_path = self._resolve_profile_path()
        self._performance_profile = self._load_performance_profile(self._profile_path)
        self._resource_fabric = self._init_resource_fabric()
        self._self_harness = self._init_self_harness()

    def _find_skills_dir(self) -> Path:
        """Find the skills directory."""
        # Try common locations
        candidates = [
            Path(".claude/skills"),
            Path("/mnt/c/BIZRA-DATA-LAKE/.claude/skills"),
            Path.home() / ".claude/skills",
        ]

        for path in candidates:
            if path.exists():
                return path

        # Fallback to first candidate
        return candidates[0]

    def _resolve_profile_path(self) -> Path:
        """
        Resolve the skill performance profile YAML path.

        Order:
        1. BIZRA_SKILL_PROFILE_PATH env var
        2. ./config/skill_performance_profile.yaml
        """
        env_path = os.environ.get("BIZRA_SKILL_PROFILE_PATH")
        if env_path:
            return Path(env_path)
        return Path("config/skill_performance_profile.yaml")

    def _init_resource_fabric(self) -> ResourceFabric:
        """Initialize cross-surface resource fabric scanner."""
        project_root = self.skills_dir.parent
        if project_root.name == ".claude":
            project_root = project_root.parent
        return ResourceFabric(project_root=project_root)

    def _init_self_harness(self) -> SelfHarnessEngine:
        """Initialize proactive self-harness engine."""
        project_root = self.skills_dir.parent
        if project_root.name == ".claude":
            project_root = project_root.parent
        return SelfHarnessEngine(project_root=project_root)

    def _load_performance_profile(self, profile_path: Path) -> SkillPerformanceProfile:
        """Load performance ranking profile from YAML, fallback to defaults."""
        if not profile_path.exists():
            logger.info(
                "Skill performance profile not found at %s, using defaults",
                profile_path,
            )
            return SkillPerformanceProfile()

        try:
            raw = profile_path.read_text(encoding="utf-8")
            data = yaml.safe_load(raw) or {}
            profile = SkillPerformanceProfile.from_dict(data)
            logger.info(
                "Loaded skill profile '%s' from %s",
                profile.profile_name,
                profile_path,
            )
            return profile
        except (OSError, ValueError) as exc:  # SEC-003 — file_io boundary
            logger.warning(
                "Failed to load skill profile from %s: %s. Using defaults.",
                profile_path,
                exc,
            )
            return SkillPerformanceProfile()

    def get_performance_profile(self) -> Dict[str, Any]:
        """Return active performance profile config."""
        payload = self._performance_profile.to_dict()
        payload["profile_path"] = str(self._profile_path)
        return payload

    def get_resource_fabric_summary(
        self,
        limit: int = 25,
        include_assets: bool = False,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Return unified resource fabric status across agents/commands/hooks/memory."""
        try:
            return self._resource_fabric.snapshot(
                limit=limit,
                include_assets=include_assets,
                force=force,
            )
        except Exception as exc:  # noqa: BLE001 — boundary boundary
            return {
                "error": str(exc),
                "profile": self._resource_fabric.get_profile(),
            }

    def get_self_harness_report(
        self,
        include_findings: bool = False,
        findings_limit: int = 200,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Run proactive self-harness and return report."""
        try:
            return self._self_harness.run(
                include_findings=include_findings,
                findings_limit=findings_limit,
                force=force,
            )
        except Exception as exc:  # noqa: BLE001 — boundary boundary
            return {
                "error": str(exc),
                "profile_name": "bizra-agentic-self-harness",
            }

    def _latency_score(self, avg_duration_ms: float) -> float:
        """
        Convert latency to [0,1] where lower is better.

        Formula is smooth and bounded:
            score = target / (target + latency)
        """
        if avg_duration_ms <= 0:
            return 1.0

        target = max(1.0, self._performance_profile.latency_target_ms)
        return max(0.0, min(1.0, target / (target + avg_duration_ms)))

    def _usage_confidence_score(self, invocation_count: int) -> float:
        """Convert invocation count to [0,1] confidence score."""
        denom = max(1, self._performance_profile.min_invocations_for_confidence)
        return max(0.0, min(1.0, invocation_count / denom))

    def _tag_boost_score(self, tags: List[str]) -> float:
        """Compute normalized tag boost score in [0,1]."""
        if not tags:
            return 0.0

        raw_boost = 0.0
        for tag in tags:
            raw_boost += float(self._performance_profile.tag_boosts.get(tag, 0.0))

        capped = min(self._performance_profile.max_tag_boost, raw_boost)
        if self._performance_profile.max_tag_boost <= 0:
            return 0.0
        return max(0.0, min(1.0, capped / self._performance_profile.max_tag_boost))

    def _compute_performance_score(
        self, skill: RegisteredSkill
    ) -> tuple[float, Dict[str, float]]:
        """Compute weighted performance score and per-component breakdown."""
        profile = self._performance_profile
        success = max(0.0, min(1.0, skill.success_rate))
        latency = self._latency_score(skill.avg_duration_ms)
        usage = self._usage_confidence_score(skill.invocation_count)
        ihsan_floor = max(0.0, min(1.0, skill.manifest.ihsan_floor))
        tag_boost = self._tag_boost_score(skill.manifest.tags)

        score = (
            profile.weight_success_rate * success
            + profile.weight_latency * latency
            + profile.weight_usage_confidence * usage
            + profile.weight_ihsan_floor * ihsan_floor
            + profile.weight_tag_boost * tag_boost
        )

        components = {
            "success_rate": round(success, 6),
            "latency": round(latency, 6),
            "usage_confidence": round(usage, 6),
            "ihsan_floor": round(ihsan_floor, 6),
            "tag_boost": round(tag_boost, 6),
        }
        return round(score, 6), components

    def get_top_skills(
        self,
        limit: Optional[int] = None,
        ihsan_score: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Return highest-quality skills ranked by performance profile.

        Only skills currently invokable at the provided Ihsān level are included.
        """
        if limit is None:
            limit = self._performance_profile.top_n_default
        limit = max(1, min(int(limit), 100))

        ranked: List[Dict[str, Any]] = []
        for skill in self.get_all():
            if not self.can_invoke(skill.manifest.name, ihsan_score):
                continue

            score, components = self._compute_performance_score(skill)
            ranked.append(
                {
                    "name": skill.manifest.name,
                    "description": skill.manifest.description,
                    "status": skill.status.value,
                    "agent": skill.manifest.agent,
                    "tags": list(skill.manifest.tags),
                    "performance_score": score,
                    "score_components": components,
                    "invocation_count": skill.invocation_count,
                    "success_rate": round(skill.success_rate, 6),
                    "avg_duration_ms": round(skill.avg_duration_ms, 3),
                    "ihsan_floor": round(skill.manifest.ihsan_floor, 6),
                }
            )

        ranked.sort(
            key=lambda item: (
                item["performance_score"],
                item["success_rate"],
                -item["avg_duration_ms"],
                item["invocation_count"],
            ),
            reverse=True,
        )
        return ranked[:limit]

    def load_all(self) -> int:
        """
        Load all skills from the skills directory.

        Returns:
            Number of skills loaded
        """
        if not self.skills_dir.exists():
            logger.warning(f"Skills directory not found: {self.skills_dir}")
            return 0

        loaded = 0
        for skill_path in self.skills_dir.iterdir():
            if skill_path.is_dir():
                manifest_path = skill_path / "SKILL.md"
                if manifest_path.exists():
                    try:
                        skill = self._load_skill(manifest_path)
                        if skill:
                            self._register(skill)
                            loaded += 1
                    except Exception as e:  # noqa: BLE001 — boundary boundary
                        logger.error(f"Failed to load skill {skill_path.name}: {e}")

        logger.info(f"Loaded {loaded} skills from {self.skills_dir}")
        return loaded

    def _load_skill(self, manifest_path: Path) -> Optional[RegisteredSkill]:
        """Load a single skill from its SKILL.md."""
        raw = manifest_path.read_text(encoding="utf-8")

        # Extract YAML frontmatter
        match = self.FRONTMATTER_RE.match(raw)
        if not match:
            logger.warning(f"No frontmatter in {manifest_path}")
            return None

        frontmatter_yaml = match.group(1)
        try:
            frontmatter = yaml.safe_load(frontmatter_yaml) or {}
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in {manifest_path}: {e}")
            return None

        manifest = SkillManifest.from_frontmatter(frontmatter, raw)

        return RegisteredSkill(
            manifest=manifest,
            path=str(manifest_path),
            status=SkillStatus.AVAILABLE,
        )

    def _register(self, skill: RegisteredSkill):
        """Register a skill in the indexes."""
        name = skill.manifest.name
        self._skills[name] = skill

        # Index by tag
        for tag in skill.manifest.tags:
            if tag not in self._by_tag:
                self._by_tag[tag] = []
            if name not in self._by_tag[tag]:
                self._by_tag[tag].append(name)

        # Index by agent
        agent = skill.manifest.agent
        if agent not in self._by_agent:
            self._by_agent[agent] = []
        if name not in self._by_agent[agent]:
            self._by_agent[agent].append(name)

    def get(self, name: str) -> Optional[RegisteredSkill]:
        """Get a skill by name."""
        return self._skills.get(name)

    def get_all(self) -> List[RegisteredSkill]:
        """Get all registered skills."""
        return list(self._skills.values())

    def find_by_tag(self, tag: str) -> List[RegisteredSkill]:
        """Find skills with a specific tag."""
        names = self._by_tag.get(tag, [])
        return [self._skills[n] for n in names if n in self._skills]

    def find_by_agent(self, agent: str) -> List[RegisteredSkill]:
        """Find skills that use a specific agent."""
        names = self._by_agent.get(agent, [])
        return [self._skills[n] for n in names if n in self._skills]

    def can_invoke(self, name: str, ihsan_score: float) -> bool:
        """
        Check if a skill can be invoked given current Ihsān score.

        Args:
            name: Skill name
            ihsan_score: Current Ihsān score

        Returns:
            True if skill can be invoked
        """
        skill = self._skills.get(name)
        if not skill:
            return False

        if skill.status not in (SkillStatus.AVAILABLE, SkillStatus.ACTIVE):
            return False

        return ihsan_score >= skill.manifest.ihsan_floor

    def get_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        total = len(self._skills)
        by_status: Dict[str, int] = {}
        for skill in self._skills.values():
            status = skill.status.value
            by_status[status] = by_status.get(status, 0) + 1

        total_invocations = sum(s.invocation_count for s in self._skills.values())
        total_success = sum(s.success_count for s in self._skills.values())

        return {
            "total_skills": total,
            "by_status": by_status,
            "by_agent": {k: len(v) for k, v in self._by_agent.items()},
            "total_tags": len(self._by_tag),
            "total_invocations": total_invocations,
            "overall_success_rate": total_success / max(total_invocations, 1),
            "skills_dir": str(self.skills_dir),
            "performance_profile": self.get_performance_profile(),
            "resource_fabric": self.get_resource_fabric_summary(
                include_assets=False,
                force=False,
            ),
            # Keep stats non-blocking: do not trigger a cold full-repo scan.
            "self_harness": self._self_harness.peek_report(include_findings=False),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_registry: Optional[SkillRegistry] = None


def get_skill_registry(skills_dir: Optional[str] = None) -> SkillRegistry:
    """
    Get the global skill registry.

    Creates and loads on first call.
    """
    global _registry
    if _registry is None:
        _registry = SkillRegistry(skills_dir)
        _registry.load_all()
    return _registry
