"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA SKILLS — Runtime Skill Registry & Invocation                        ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Skill Registry: Load, track, and invoke skills from .claude/skills/       ║
║   MCP Bridge: Map skills to required MCP tools                              ║
║   Invocation Engine: Execute skills with FATE gate validation               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Standing on Giants:
- Alistair Cockburn (2005): Hexagonal Architecture (Ports & Adapters)
- Eric Evans (2003): Domain-Driven Design (skill as bounded context)
- Anthropic (2023): Constitutional AI (Ihsān enforcement)

Created: 2026-02-08 | BIZRA Skill Infrastructure v1.0.0
"""

from .mcp_bridge import (
    SKILL_TOOL_MAP,
    MCPBridge,
    SkillToolMapping,
)
from .registry import (
    RegisteredSkill,
    SkillManifest,
    SkillRegistry,
    SkillStatus,
    get_skill_registry,
)
from .router import (
    SkillInvocationResult,
    SkillRouter,
)

__all__ = [
    # Registry
    "SkillManifest",
    "SkillStatus",
    "RegisteredSkill",
    "SkillRegistry",
    "get_skill_registry",
    # Router
    "SkillRouter",
    "SkillInvocationResult",
    # MCP Bridge
    "SkillToolMapping",
    "SKILL_TOOL_MAP",
    "MCPBridge",
]

__version__ = "1.0.0"
__giants__ = [
    "Alistair Cockburn (2005): Hexagonal Architecture",
    "Eric Evans (2003): Domain-Driven Design",
    "Anthropic (2023): Constitutional AI constraints",
]
