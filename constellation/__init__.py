# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Islamic Masterminds Agentic Constellation
# ═══════════════════════════════════════════════════════════════════════════════
"""
BIZRA Constellation - Multi-Agent System for the Islamic Masterminds

This package implements a production-ready multi-agent orchestration system
featuring 27 historical Islamic masterminds + 2 meta-agents, with SNR-tier
routing and GoT/ToT/CoT reasoning controls.

Key Components:
- agents/roster.yaml: 29 agent definitions with SNR targets and output contracts
- teams/configurations.yaml: 8 cross-pollination team presets
- router/policy.yaml: SNR-tier routing and reasoning mode escalation
- evaluation/gates.yaml: Evidence gates and verification requirements
- prompts/system_prompts.yaml: LangGraph-ready system prompts for all agents
- orchestrator.py: Python orchestrator implementing full execution flow

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
"""

from pathlib import Path

# Package metadata
__version__ = "1.0.0"
__author__ = "BIZRA"
__description__ = "Islamic Masterminds Agentic Constellation"

# Constellation path
CONSTELLATION_PATH = Path(__file__).parent

# Lazy imports to avoid circular dependencies
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


# Convenience exports
__all__ = [
    "get_orchestrator",
    "get_loader",
    "CONSTELLATION_PATH",
    "__version__",
]
