"""BIZRA Swarm Coordination — Unified execution engine for agent missions.

Implements ADR-004: V3 Unified Swarm Coordination Engine.

Modules:
    types       — Topology enums, AgentSpec, SwarmConfig, SwarmEvent
    engine      — SwarmEngine phased execution coordinator
    event_bridge — SwarmEvent → EventBus translation bridge
"""

from core.swarm.engine import (
    HierarchicalMeshStrategy,
    ParallelStrategy,
    SequentialStrategy,
    SwarmEngine,
)
from core.swarm.event_bridge import SwarmEventBridge, wire_swarm_to_bus
from core.swarm.types import (
    AgentRole,
    AgentSpec,
    SwarmConfig,
    SwarmEvent,
    SwarmEventKind,
    SwarmPhase,
    SwarmTopology,
)

__all__ = [
    # Types
    "AgentRole",
    "AgentSpec",
    "SwarmConfig",
    "SwarmEvent",
    "SwarmEventKind",
    "SwarmPhase",
    "SwarmTopology",
    # Engine
    "HierarchicalMeshStrategy",
    "ParallelStrategy",
    "SequentialStrategy",
    "SwarmEngine",
    # Event Bridge
    "SwarmEventBridge",
    "wire_swarm_to_bus",
]
