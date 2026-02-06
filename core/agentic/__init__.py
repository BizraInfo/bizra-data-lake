"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA AGENTIC SYSTEM — Autonomous Decision-Making Infrastructure          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   Standing on the Shoulders of Giants:                                       ║
║   • Maturana & Varela (Autopoiesis)                                          ║
║   • Brooks (Subsumption Architecture)                                        ║
║   • Anthropic (Constitutional AI)                                            ║
║   • LangChain/AutoGPT (Agent Patterns)                                       ║
║   • OpenClaw/PAT (Personal AI Team)                                          ║
║                                                                              ║
║   Agentic Principles:                                                        ║
║   • Autonomy: Make decisions without constant supervision                    ║
║   • Proactivity: Anticipate needs and act preemptively                       ║
║   • Reactivity: Respond appropriately to changes                             ║
║   • Social: Coordinate with other agents                                     ║
║   • Constitutional: All actions bound by Ihsān constraints                   ║
║                                                                              ║
║   Created: 2026-02-02 | BIZRA Agentic Integration                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# Constants
AGENTIC_VERSION = "1.0.0"
MAX_CONCURRENT_AGENTS = 10
AGENT_TIMEOUT_SECONDS = 300
DEFAULT_PLANNING_DEPTH = 3

# Agent types
AGENT_TYPES = [
    "orchestrator",   # Coordinates other agents
    "researcher",     # Gathers information
    "analyzer",       # Processes data
    "executor",       # Takes actions
    "monitor",        # Watches for changes
    "healer",         # Fixes problems
]

# Lazy imports
def __getattr__(name: str):
    if name == "AgentOrchestrator":
        from .orchestrator import AgentOrchestrator
        return AgentOrchestrator
    elif name == "AutonomousAgent":
        from .agent import AutonomousAgent
        return AutonomousAgent
    elif name == "AgentTask":
        from .agent import AgentTask
        return AgentTask
    elif name == "SelfOptimizer":
        from .optimizer import SelfOptimizer
        return SelfOptimizer
    elif name == "ProactiveEngine":
        from .proactive import ProactiveEngine
        return ProactiveEngine
    raise AttributeError(f"module 'core.agentic' has no attribute '{name}'")

__all__ = [
    "AgentOrchestrator",
    "AutonomousAgent",
    "AgentTask",
    "SelfOptimizer",
    "ProactiveEngine",
    "AGENTIC_VERSION",
    "MAX_CONCURRENT_AGENTS",
    "AGENT_TYPES",
]
