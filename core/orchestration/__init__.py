"""
+==============================================================================+
|   BIZRA ORCHESTRATION -- Event Bus & Agent Coordination                       |
+==============================================================================+
|   Event-driven coordination, team planning, and background agent management. |
|                                                                              |
|   Components:                                                                |
|   - event_bus: Topic-based pub/sub with priority queues                      |
|   - team_planner: Multi-agent task decomposition                             |
|   - background_agents: Domain-specific autonomous plugins                    |
|   - opportunity_pipeline: Sensor-to-action nervous system                    |
|   - proactive_scheduler: Anticipatory task scheduling                        |
|                                                                              |
|   Constitutional Constraint: All actions must satisfy Ihsan >= 0.95          |
|                                                                              |
|   Standing on Giants:                                                        |
|   - Hewitt (1973): Actor Model                                               |
|   - Ousterhout (1996): Event-Driven Programming                              |
+==============================================================================+

Created: 2026-02-05 | SAPE Sovereign Module Decomposition
Migrated: 2026-02-05 | Files now in dedicated orchestration package
"""

# --------------------------------------------------------------------------
# PHASE 1: Safe imports (no cross-package dependencies)
# --------------------------------------------------------------------------
from .event_bus import (
    EventBus,
    Event,
    EventPriority,
)
from .team_planner import TeamPlanner
from .proactive_scheduler import ProactiveScheduler
from .muraqabah_engine import (
    MuraqabahEngine,
)
from .muraqabah_sensors import MuraqabahSensorHub
from .predictive_monitor import PredictiveMonitor

# --------------------------------------------------------------------------
# PHASE 2: Lazy imports for modules with cross-package dependencies.
# These are deferred to break circular import chains between
# orchestration <-> reasoning, orchestration <-> bridges, orchestration <-> governance.
# --------------------------------------------------------------------------
_LAZY_MODULES = {
    "EnhancedTeamPlanner": (".enhanced_team_planner", "EnhancedTeamPlanner"),
    "BackgroundAgent": (".background_agents", "BackgroundAgent"),
    "BackgroundAgentRegistry": (".background_agents", "BackgroundAgentRegistry"),
    "OpportunityPipeline": (".opportunity_pipeline", "OpportunityPipeline"),
    "PipelineStage": (".opportunity_pipeline", "PipelineStage"),
    "ProactiveSovereignEntity": (".proactive_integration", "ProactiveSovereignEntity"),
    "ProactiveTeam": (".proactive_team", "ProactiveTeam"),
}


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        module_path, attr_name = _LAZY_MODULES[name]
        import importlib
        mod = importlib.import_module(module_path, __name__)
        value = getattr(mod, attr_name)
        globals()[name] = value  # Cache for subsequent access
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Event Bus
    "EventBus",
    "Event",
    "EventPriority",
    # Team Planning
    "TeamPlanner",
    "EnhancedTeamPlanner",
    # Background Agents
    "BackgroundAgent",
    "BackgroundAgentRegistry",
    # Opportunity Pipeline
    "OpportunityPipeline",
    "PipelineStage",
    # Proactive
    "ProactiveScheduler",
    "ProactiveSovereignEntity",
    "ProactiveTeam",
    # Muraqabah (Monitoring)
    "MuraqabahEngine",
    "MuraqabahSensorHub",
    # Predictive
    "PredictiveMonitor",
]
