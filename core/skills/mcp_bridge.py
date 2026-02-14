"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA SKILLS — MCP Tool Bridge                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Maps skills to their required MCP tools and manages permissions.          ║
╚══════════════════════════════════════════════════════════════════════════════╝

Standing on Giants:
- Alistair Cockburn (2005): Ports & Adapters (MCP as port)
- Robert C. Martin (2008): Dependency Inversion (skill doesn't know MCP details)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ═══════════════════════════════════════════════════════════════════════════════


class MCPPermission(str, Enum):
    """MCP tool permission levels."""

    READ = "read"  # Can read files, fetch data
    WRITE = "write"  # Can create/edit files
    EXECUTE = "execute"  # Can run commands
    NETWORK = "network"  # Can access network
    DESTRUCTIVE = "destructive"  # Can delete, stop processes


class MCPToolCategory(str, Enum):
    """MCP tool categories."""

    CORE = "core"  # bash, view, edit, create, grep, glob
    GITHUB = "github"  # github-mcp-server-*
    WEB = "web"  # web_fetch, web_search
    FLOW_NEXUS = "flow_nexus"  # flow-nexus-*
    AGENT = "agent"  # task, skill
    MEMORY = "memory"  # store_memory, agentdb-*


# ═══════════════════════════════════════════════════════════════════════════════
# SKILL-TOOL MAPPING
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SkillToolMapping:
    """
    Maps a skill to its MCP tool requirements.

    Defines what tools a skill needs and what permissions it requires.
    """

    skill_name: str
    required_tools: List[str] = field(default_factory=list)
    optional_tools: List[str] = field(default_factory=list)
    permissions: List[MCPPermission] = field(default_factory=list)
    categories: List[MCPToolCategory] = field(default_factory=list)

    # Cost estimation (tokens per invocation)
    estimated_token_cost: int = 1000

    # Risk level (for audit)
    risk_level: str = "low"  # low, medium, high, critical

    def requires_permission(self, perm: MCPPermission) -> bool:
        """Check if skill requires a permission."""
        return perm in self.permissions

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "skill_name": self.skill_name,
            "required_tools": self.required_tools,
            "optional_tools": self.optional_tools,
            "permissions": [p.value for p in self.permissions],
            "categories": [c.value for c in self.categories],
            "estimated_token_cost": self.estimated_token_cost,
            "risk_level": self.risk_level,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SKILL-TOOL MAPPINGS
# ═══════════════════════════════════════════════════════════════════════════════

SKILL_TOOL_MAP: Dict[str, SkillToolMapping] = {
    # Benchmark & Evaluation
    "true-spearpoint": SkillToolMapping(
        skill_name="true-spearpoint",
        required_tools=["bash", "view", "edit", "create"],
        optional_tools=[
            "github-mcp-server-list_workflows",
            "github-mcp-server-get_job_logs",
        ],
        permissions=[MCPPermission.READ, MCPPermission.WRITE, MCPPermission.EXECUTE],
        categories=[MCPToolCategory.CORE, MCPToolCategory.GITHUB],
        estimated_token_cost=5000,
        risk_level="medium",
    ),
    # Research
    "deep-research": SkillToolMapping(
        skill_name="deep-research",
        required_tools=["web_search", "web_fetch", "grep", "view"],
        optional_tools=[
            "github-mcp-server-search_code",
            "github-mcp-server-search_repositories",
        ],
        permissions=[MCPPermission.READ, MCPPermission.NETWORK],
        categories=[MCPToolCategory.WEB, MCPToolCategory.CORE, MCPToolCategory.GITHUB],
        estimated_token_cost=3000,
        risk_level="low",
    ),
    # Code Review
    "guardian-review": SkillToolMapping(
        skill_name="guardian-review",
        required_tools=["view", "grep", "glob"],
        optional_tools=[
            "github-mcp-server-get_commit",
            "github-mcp-server-pull_request_read",
        ],
        permissions=[MCPPermission.READ],
        categories=[MCPToolCategory.CORE, MCPToolCategory.GITHUB],
        estimated_token_cost=2000,
        risk_level="low",
    ),
    "github-code-review": SkillToolMapping(
        skill_name="github-code-review",
        required_tools=[
            "view",
            "grep",
            "github-mcp-server-pull_request_read",
            "github-mcp-server-get_commit",
        ],
        optional_tools=["github-mcp-server-get_file_contents"],
        permissions=[MCPPermission.READ],
        categories=[MCPToolCategory.CORE, MCPToolCategory.GITHUB],
        estimated_token_cost=2500,
        risk_level="low",
    ),
    # Implementation
    "implement": SkillToolMapping(
        skill_name="implement",
        required_tools=["view", "edit", "create", "bash", "grep", "glob"],
        optional_tools=["github-mcp-server-list_branches"],
        permissions=[MCPPermission.READ, MCPPermission.WRITE, MCPPermission.EXECUTE],
        categories=[MCPToolCategory.CORE],
        estimated_token_cost=5000,
        risk_level="medium",
    ),
    "sparc-methodology": SkillToolMapping(
        skill_name="sparc-methodology",
        required_tools=["view", "edit", "create", "bash", "grep", "glob"],
        optional_tools=["task"],
        permissions=[MCPPermission.READ, MCPPermission.WRITE, MCPPermission.EXECUTE],
        categories=[MCPToolCategory.CORE, MCPToolCategory.AGENT],
        estimated_token_cost=8000,
        risk_level="medium",
    ),
    # Swarm & Orchestration
    "swarm-advanced": SkillToolMapping(
        skill_name="swarm-advanced",
        required_tools=["bash", "task"],
        optional_tools=["flow-nexus-swarm", "ruv-swarm"],
        permissions=[MCPPermission.EXECUTE, MCPPermission.NETWORK],
        categories=[
            MCPToolCategory.CORE,
            MCPToolCategory.AGENT,
            MCPToolCategory.FLOW_NEXUS,
        ],
        estimated_token_cost=10000,
        risk_level="high",
    ),
    "flow-nexus-swarm": SkillToolMapping(
        skill_name="flow-nexus-swarm",
        required_tools=["bash"],
        optional_tools=["web_fetch"],
        permissions=[MCPPermission.EXECUTE, MCPPermission.NETWORK],
        categories=[MCPToolCategory.FLOW_NEXUS, MCPToolCategory.CORE],
        estimated_token_cost=8000,
        risk_level="high",
    ),
    # Memory & Learning
    "claude-flow-memory": SkillToolMapping(
        skill_name="claude-flow-memory",
        required_tools=["store_memory", "view"],
        optional_tools=["grep"],
        permissions=[MCPPermission.READ, MCPPermission.WRITE],
        categories=[MCPToolCategory.MEMORY, MCPToolCategory.CORE],
        estimated_token_cost=1000,
        risk_level="low",
    ),
    # Verification
    "Verification": SkillToolMapping(
        skill_name="Verification",
        required_tools=["view", "grep", "bash"],
        optional_tools=["edit"],
        permissions=[MCPPermission.READ, MCPPermission.EXECUTE],
        categories=[MCPToolCategory.CORE],
        estimated_token_cost=3000,
        risk_level="medium",
    ),
    # SNR Check
    "snr-check": SkillToolMapping(
        skill_name="snr-check",
        required_tools=["view"],
        optional_tools=[],
        permissions=[MCPPermission.READ],
        categories=[MCPToolCategory.CORE],
        estimated_token_cost=500,
        risk_level="low",
    ),
    # Sovereign Query
    "sovereign-query": SkillToolMapping(
        skill_name="sovereign-query",
        required_tools=["view", "grep", "glob"],
        optional_tools=["web_search"],
        permissions=[MCPPermission.READ, MCPPermission.NETWORK],
        categories=[MCPToolCategory.CORE, MCPToolCategory.WEB],
        estimated_token_cost=2000,
        risk_level="low",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# MCP BRIDGE
# ═══════════════════════════════════════════════════════════════════════════════


class MCPBridge:
    """
    Bridge between Skills and MCP tools.

    Responsibilities:
    - Resolve skill tool requirements
    - Check tool availability
    - Track tool usage per skill
    - Enforce permission policies
    """

    def __init__(self, available_tools: Optional[Set[str]] = None):
        """
        Initialize MCP bridge.

        Args:
            available_tools: Set of available MCP tool names.
                             If None, assumes all tools available.
        """
        self.available_tools = available_tools
        self._usage: Dict[str, Dict[str, int]] = {}  # skill -> {tool: count}

    def get_mapping(self, skill_name: str) -> Optional[SkillToolMapping]:
        """Get tool mapping for a skill."""
        return SKILL_TOOL_MAP.get(skill_name)

    def get_required_tools(self, skill_name: str) -> List[str]:
        """Get required tools for a skill."""
        mapping = SKILL_TOOL_MAP.get(skill_name)
        if not mapping:
            return []
        return mapping.required_tools

    def get_all_tools(self, skill_name: str) -> List[str]:
        """Get all tools (required + optional) for a skill."""
        mapping = SKILL_TOOL_MAP.get(skill_name)
        if not mapping:
            return []
        return mapping.required_tools + mapping.optional_tools

    def check_availability(self, skill_name: str) -> Dict[str, bool]:
        """
        Check which required tools are available for a skill.

        Returns:
            Dict mapping tool name to availability
        """
        if self.available_tools is None:
            # Assume all available
            return {t: True for t in self.get_required_tools(skill_name)}

        result = {}
        for tool in self.get_required_tools(skill_name):
            # Check exact match or pattern match
            available = tool in self.available_tools or any(
                t.startswith(tool.split("*")[0])
                for t in self.available_tools
                if "*" in tool
            )
            result[tool] = available

        return result

    def can_execute(self, skill_name: str) -> bool:
        """Check if all required tools are available."""
        availability = self.check_availability(skill_name)
        return all(availability.values())

    def get_permissions(self, skill_name: str) -> List[MCPPermission]:
        """Get required permissions for a skill."""
        mapping = SKILL_TOOL_MAP.get(skill_name)
        if not mapping:
            return []
        return mapping.permissions

    def has_destructive_permission(self, skill_name: str) -> bool:
        """Check if skill requires destructive permission."""
        return MCPPermission.DESTRUCTIVE in self.get_permissions(skill_name)

    def estimate_cost(self, skill_name: str) -> int:
        """Estimate token cost for skill."""
        mapping = SKILL_TOOL_MAP.get(skill_name)
        if not mapping:
            return 1000  # Default
        return mapping.estimated_token_cost

    def get_risk_level(self, skill_name: str) -> str:
        """Get risk level for skill."""
        mapping = SKILL_TOOL_MAP.get(skill_name)
        if not mapping:
            return "unknown"
        return mapping.risk_level

    def record_usage(self, skill_name: str, tool_name: str):
        """Record tool usage for a skill."""
        if skill_name not in self._usage:
            self._usage[skill_name] = {}
        if tool_name not in self._usage[skill_name]:
            self._usage[skill_name][tool_name] = 0
        self._usage[skill_name][tool_name] += 1

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get tool usage statistics."""
        return {
            "by_skill": self._usage,
            "total_invocations": sum(
                sum(tools.values()) for tools in self._usage.values()
            ),
            "mapped_skills": len(SKILL_TOOL_MAP),
        }

    def get_skills_by_category(self, category: MCPToolCategory) -> List[str]:
        """Get skills that use tools in a category."""
        return [
            name
            for name, mapping in SKILL_TOOL_MAP.items()
            if category in mapping.categories
        ]

    def get_high_risk_skills(self) -> List[str]:
        """Get skills with high or critical risk level."""
        return [
            name
            for name, mapping in SKILL_TOOL_MAP.items()
            if mapping.risk_level in ("high", "critical")
        ]
