# Phase 1: Topology Types and Swarm Primitives

> ADR-004 | V3 Unified Swarm Coordination Engine
> Standing on Giants: Lamport (distributed state, 1978) . Burns (Kubernetes patterns, 2016) . Hamilton (operations, 2007)

## 1.1 Problem Statement

Three independent orchestration paths exist today:

| Path | File | What It Does | Talks To Others? |
|------|------|-------------|-----------------|
| A | `scripts/node0_activate.py` | 7 PAT agents, sequential, direct httpx | NO |
| B | `core/apex/swarm_orchestrator.py` | Deployment, scaling, health, self-healing | NO |
| C | `core/orchestration/event_bus.py` | Background agents, opportunity pipeline | NO |

Each path independently manages agent lifecycle, model routing, and health.
There is no unified topology, no shared event bus, no cross-path coordination.

The AutoModelRouter (just implemented) adds VRAM-aware pre-loading and
Equalizer command consumption, but only for Path A.

## 1.2 Requirements

- Define `SwarmTopology` enum: SEQUENTIAL, PARALLEL, HIERARCHICAL_MESH
- Define `AgentRole` and `AgentSpec` as unified agent descriptors
- Define `SwarmConfig` as the single configuration surface
- Define `SwarmEvent` for cross-component communication
- All types must be importable without side effects (no I/O in module scope)
- Compatible with existing `PAT_AGENTS` dict in node0_activate.py
- Compatible with existing `AgentConfig` in core/apex/swarm_orchestrator.py

## 1.3 Pseudocode: `core/swarm/types.py`

```
IMPORT dataclass, field FROM dataclasses
IMPORT Enum FROM enum
IMPORT Optional, Dict, List, Any FROM typing

# ── Topology ──

CLASS SwarmTopology(str, Enum):
    SEQUENTIAL        = "sequential"         # Current node0 behavior
    PARALLEL          = "parallel"           # All agents run concurrently
    HIERARCHICAL_MESH = "hierarchical_mesh"  # Queen -> worker clusters

CLASS AgentRole(str, Enum):
    STRATEGIST  = "strategist"
    RESEARCHER  = "researcher"
    ANALYST     = "analyst"
    CREATOR     = "creator"
    EXECUTOR    = "executor"
    GUARDIAN    = "guardian"
    COORDINATOR = "coordinator"

CLASS SwarmPhase(str, Enum):
    INITIALIZING = "initializing"
    PRELOADING   = "preloading"     # AutoModelRouter pre-load step
    EXECUTING    = "executing"
    SYNTHESIZING = "synthesizing"   # GoT convergence
    SCORING      = "scoring"        # SNR/Ihsan
    EQUALIZING   = "equalizing"     # Equalizer command processing
    COMPLETE     = "complete"
    FAILED       = "failed"

# ── Agent Spec ──

@dataclass(frozen=True)
CLASS AgentSpec:
    """Unified agent descriptor — bridges PAT_AGENTS and AgentConfig."""

    id: str                          # e.g. "strategist"
    role: AgentRole
    model_purpose: str               # e.g. "thinking", "reasoning"
    system_prompt: str
    max_tokens: int = 600
    timeout_seconds: float = 30.0
    is_thinking_model: bool = False  # Longer timeouts for R1/thinking models

    @staticmethod
    FUNCTION from_pat_agent(agent_id: str, pat_dict: dict) -> AgentSpec:
        """Convert PAT_AGENTS[agent_id] to AgentSpec."""
        role_map = {
            "strategist": AgentRole.STRATEGIST,
            "researcher": AgentRole.RESEARCHER,
            "analyst": AgentRole.ANALYST,
            "creator": AgentRole.CREATOR,
            "executor": AgentRole.EXECUTOR,
            "guardian": AgentRole.GUARDIAN,
            "coordinator": AgentRole.COORDINATOR,
        }
        purpose = pat_dict.get("model_purpose", "reasoning")
        is_thinking = purpose in ("thinking", "reasoning", "reasoning_large")
        RETURN AgentSpec(
            id=agent_id,
            role=role_map.get(agent_id, AgentRole.COORDINATOR),
            model_purpose=purpose,
            system_prompt=(
                f"You are the PAT {pat_dict['name']}. "
                f"Your role is {pat_dict['role']}.\n"
                f"Standing on Giants: {pat_dict['giants']}.\n"
                f"Be concise (2-3 paragraphs). Focus on actionable insights."
            ),
            max_tokens=1200 IF is_thinking ELSE 600,
            timeout_seconds=120.0 IF is_thinking ELSE 30.0,
            is_thinking_model=is_thinking,
        )

# ── Swarm Config ──

@dataclass
CLASS SwarmConfig:
    """Single configuration surface for all swarm operations."""

    topology: SwarmTopology = SwarmTopology.SEQUENTIAL
    max_concurrent: int = 3          # For PARALLEL topology
    preload_models: bool = True      # Use AutoModelRouter
    equalizer_enabled: bool = True   # Consume Equalizer commands
    got_synthesis: bool = True       # GoT convergence step
    ihsan_threshold: float = 0.95    # From constants.py
    agent_timeout: float = 120.0     # Per-agent max
    mission_timeout: float = 600.0   # Entire mission max

# ── Swarm Events ──

CLASS SwarmEventKind(str, Enum):
    SWARM_CREATED     = "swarm_created"
    AGENT_STARTED     = "agent_started"
    AGENT_COMPLETED   = "agent_completed"
    AGENT_FAILED      = "agent_failed"
    MODEL_PRELOADED   = "model_preloaded"
    EQUALIZER_ACTION  = "equalizer_action"
    PHASE_CHANGED     = "phase_changed"
    MISSION_COMPLETE  = "mission_complete"

@dataclass(frozen=True)
CLASS SwarmEvent:
    kind: SwarmEventKind
    swarm_id: str
    agent_id: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0          # time.monotonic()
```

## 1.4 TDD Anchors

```
# tests/core/swarm/test_types.py

TEST test_agent_spec_from_pat_agent():
    """AgentSpec.from_pat_agent converts PAT_AGENTS format correctly."""
    pat = {"name": "Strategist", "role": "Strategic planning",
           "giants": "Sun Tzu", "model_purpose": "thinking"}
    spec = AgentSpec.from_pat_agent("strategist", pat)
    ASSERT spec.role == AgentRole.STRATEGIST
    ASSERT spec.is_thinking_model IS True
    ASSERT spec.timeout_seconds == 120.0
    ASSERT spec.max_tokens == 1200

TEST test_agent_spec_from_pat_non_thinking():
    """Non-thinking models get shorter timeouts."""
    pat = {"name": "Creator", "role": "Content creation",
           "giants": "Da Vinci", "model_purpose": "creative"}
    spec = AgentSpec.from_pat_agent("creator", pat)
    ASSERT spec.is_thinking_model IS False
    ASSERT spec.timeout_seconds == 30.0

TEST test_swarm_config_defaults():
    cfg = SwarmConfig()
    ASSERT cfg.topology == SwarmTopology.SEQUENTIAL
    ASSERT cfg.preload_models IS True
    ASSERT cfg.ihsan_threshold == 0.95

TEST test_swarm_event_immutable():
    evt = SwarmEvent(kind=SwarmEventKind.AGENT_STARTED, swarm_id="s1")
    # frozen=True -> cannot mutate
    WITH RAISES(FrozenInstanceError):
        evt.swarm_id = "other"
```

## 1.5 Compatibility Notes

- `AgentSpec.from_pat_agent()` accepts the exact dict format from `PAT_AGENTS` in `node0_activate.py:205-248`
- `SwarmConfig.ihsan_threshold` defaults to `UNIFIED_IHSAN_THRESHOLD` from `core/integration/constants.py`
- `SwarmPhase.PRELOADING` maps to `AutoModelRouter.preload_mission_fleet()`
- `SwarmPhase.EQUALIZING` maps to `AutoModelRouter.check_equalizer()`
