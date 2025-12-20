# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Islamic Masterminds Agentic Constellation v2.0
# ═══════════════════════════════════════════════════════════════════════════════
"""
BIZRA Constellation - Multi-Agent System for the Islamic Masterminds

This package implements a production-ready multi-agent orchestration system
featuring 27 historical Islamic masterminds + 2 meta-agents, with SNR-tier
routing and GoT/ToT/CoT reasoning controls.

═══════════════════════════════════════════════════════════════════════════════
ARCHITECTURE v2.0
═══════════════════════════════════════════════════════════════════════════════

Core Components:
- agents/roster.yaml: 29 agent definitions with SNR targets and output contracts
- teams/configurations.yaml: 8 cross-pollination team presets
- router/policy.yaml: SNR-tier routing and reasoning mode escalation
- evaluation/gates.yaml: Evidence gates and verification requirements
- prompts/system_prompts.yaml: LangGraph-ready system prompts for all agents
- orchestrator.py: Python orchestrator implementing full execution flow

Subsystems:
- memory/: HyperGraphRAG connector + multi-tier memory system
- hooks/: Event-driven hook system for agent lifecycle
- triggers/: Condition-based automatic agent activation
- commands/: Slash command interface (/agent, /team, /recall, etc.)
- skills/: Modular skill system + sub-agent orchestration
- protocols/: MCP server + A2A inter-agent protocol
- audit/: Auto-audit integration with CodeRabbit

Execution Flow:
1. INTAKE: Parse task, stakes, domains, evidence requirements
2. PLAN: Select team, choose reasoning mode (CoT/ToT/GoT)
3. WORK: Agents produce candidate solutions + evidence bundles
4. VERIFY: Verifiers challenge assumptions, check sources
5. SYNTHESIZE: Polymath Integrator produces unified deliverable
6. DELIVER: Output with 'what we know / assume / test next'

Usage:
    from constellation import ConstellationOrchestrator
    
    orchestrator = ConstellationOrchestrator()
    result = orchestrator.execute(
        "Design an ethical AI governance framework",
        context={"stakes": "high"}
    )
    print(result.executive_summary)
    
Advanced Usage (with Runtime):
    from constellation import initialize_constellation
    
    runtime = await initialize_constellation()
    await runtime.process_input("/agent ibn-sina Diagnose symptoms")
"""

from pathlib import Path

# Package metadata
__version__ = "2.0.0"
__author__ = "BIZRA"
__description__ = "Islamic Masterminds Agentic Constellation with Autonomous Infrastructure"

# Constellation path
CONSTELLATION_PATH = Path(__file__).parent

# ─────────────────────────────────────────────────────────────────────────────
# LAZY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

def get_orchestrator():
    """Get the constellation orchestrator instance."""
    from .orchestrator import ConstellationOrchestrator
    return ConstellationOrchestrator(CONSTELLATION_PATH)


def get_loader():
    """Get the constellation loader for accessing configurations."""
    from .orchestrator import ConstellationLoader
    loader = ConstellationLoader(CONSTELLATION_PATH)
    loader.load_all()
    return loader


def get_runtime(config=None):
    """Get the constellation runtime instance."""
    from .runtime import get_runtime as _get_runtime
    return _get_runtime(config)


async def initialize_constellation(config=None):
    """Initialize and return the constellation runtime."""
    from .runtime import initialize_constellation as _init
    return await _init(config)


# ─────────────────────────────────────────────────────────────────────────────
# SUBSYSTEM ACCESSORS
# ─────────────────────────────────────────────────────────────────────────────

def get_memory_manager():
    """Get the memory manager."""
    from .memory import MemoryManager
    return MemoryManager()


def get_knowledge_connector():
    """Get the HyperGraphRAG knowledge connector."""
    from .memory import HyperGraphRAGConnector
    return HyperGraphRAGConnector()


def get_hook_registry():
    """Get the hook registry."""
    from .hooks import get_hook_registry as _get
    return _get()


def get_trigger_engine():
    """Get the trigger engine."""
    from .triggers import get_trigger_engine as _get
    return _get()


def get_command_registry():
    """Get the command registry."""
    from .commands import get_command_registry as _get
    return _get()


def get_skill_registry():
    """Get the skill registry."""
    from .skills import get_skill_registry as _get
    return _get()


def get_mcp_server():
    """Get the MCP server."""
    from .protocols import get_mcp_server as _get
    return _get()


def get_a2a_router():
    """Get the A2A router."""
    from .protocols import get_a2a_router as _get
    return _get()


def get_audit_engine():
    """Get the auto-audit engine."""
    from .audit import get_audit_engine as _get
    return _get()


# ─────────────────────────────────────────────────────────────────────────────
# CONVENIENCE EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Core
    "get_orchestrator",
    "get_loader",
    "get_runtime",
    "initialize_constellation",
    "CONSTELLATION_PATH",
    "__version__",
    # Subsystems
    "get_memory_manager",
    "get_knowledge_connector",
    "get_hook_registry",
    "get_trigger_engine",
    "get_command_registry",
    "get_skill_registry",
    "get_mcp_server",
    "get_a2a_router",
    "get_audit_engine",
]
