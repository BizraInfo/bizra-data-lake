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
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

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
                    except Exception as e:
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
