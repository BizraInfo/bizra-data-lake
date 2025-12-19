# ADR-004: Agent Factory Implementation

**Status:** Accepted  
**Date:** 2025-01-27  
**Author:** BIZRA Genesis System  
**Finding:** F-ARCH-002 (Static Agents)

## Context

The SAPE Multi-Lens Audit identified that the BIZRA system lacked a factory pattern for agent instantiation:

> **F-ARCH-002:** No factory class exists to spawn PAT/SAT agents with persistent state/memory.

Agents were being created ad-hoc without:
- Consistent resource allocation
- Session memory for context persistence
- Unified lifecycle management
- URP/FATE integration

## Decision

Implement an **Agent Factory** pattern with:

1. **Singleton Factory** - Central registry for all agent instances
2. **Session Memory** - Persistent conversation context per agent
3. **URP Integration** - Resource allocation before spawn
4. **REST API** - `/v1/system/spawn` endpoint for external access

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                         CLIENT                                  │
│                            │                                    │
│                    POST /v1/system/spawn                        │
│                    {"agent_name": "MasterReasoner"}             │
│                            │                                    │
│                            ▼                                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    SPAWN API                             │   │
│  └───────────────────────────┬─────────────────────────────┘   │
│                              │                                  │
│                              ▼                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                   AGENT FACTORY                          │   │
│  │                                                          │   │
│  │   ┌──────────────┐    ┌──────────────┐                  │   │
│  │   │  URP Manager  │    │ FATE Engine  │                  │   │
│  │   │  (Resources)  │    │   (Ethics)   │                  │   │
│  │   └───────┬──────┘    └──────────────┘                  │   │
│  │           │                                              │   │
│  │           ▼                                              │   │
│  │   ┌──────────────────────────────────────────────────┐  │   │
│  │   │              AGENT INSTANCE                       │  │   │
│  │   │                                                   │  │   │
│  │   │   agent_id: "agent-a1b2c3d4"                     │  │   │
│  │   │   instance_id: "inst-e5f6g7h8"                   │  │   │
│  │   │   lease_id: "lease-x9y0z1"                       │  │   │
│  │   │                                                   │  │   │
│  │   │   ┌─────────────────────────────────────────┐    │  │   │
│  │   │   │           SESSION MEMORY                │    │  │   │
│  │   │   │                                         │    │  │   │
│  │   │   │  [system] You are MasterReasoner...    │    │  │   │
│  │   │   │  [user] Analyze this...                │    │  │   │
│  │   │   │  [assistant] Based on analysis...      │    │  │   │
│  │   │   │                                         │    │  │   │
│  │   │   └─────────────────────────────────────────┘    │  │   │
│  │   │                                                   │  │   │
│  │   └──────────────────────────────────────────────────┘  │   │
│  │                                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Agent Types

#### PAT (Personal Agentic Team)
Full LLM-powered agents with session memory:

| Agent | Model | Backend | VRAM | Role |
|-------|-------|---------|------|------|
| MasterReasoner | deepseek-r1:7b | Ollama | 4.5GB | Strategic thinking |
| MemoryArchitect | qwen2.5:7b | Ollama | 4.0GB | Knowledge management |
| CreativeSynthesizer | qwen2.5:7b | Ollama | 4.0GB | Creative ideation |
| DataAnalyzer | mistral:7b | Ollama | 4.0GB | Data analysis |
| Communicator | mistral:7b | Ollama | 4.0GB | External messaging |
| ExecutionPlanner | agentflow-7b | LM Studio | 4.0GB | Task planning |
| EthicsGuardian | qwen2.5:7b | Ollama | 4.0GB | Ihsān compliance |

#### SAT (System Agentic Team)
Rule-based micro-agents:

| Agent | VRAM | Role |
|-------|------|------|
| PoiVerifier | 0.1GB | Proof-of-Impact validation |
| ResourceAllocator | 0.1GB | Compute optimization |
| RiskGuardian | 0.1GB | Security monitoring |
| GovernanceEngine | 0.1GB | Policy enforcement |
| EvidenceEngine | 0.1GB | Audit trails |

### Session Memory

Each agent maintains conversation context:

```python
@dataclass
class SessionMemory:
    session_id: str
    agent_id: str
    turns: List[MemoryTurn]
    max_turns: int = 20
```

Features:
- Automatic trimming to prevent context overflow
- System prompt preserved during trim
- Serializable for persistence
- Resume capability via session_id

### API Endpoints

```
POST   /v1/system/spawn          - Spawn agent
GET    /v1/system/agents         - List all agents
GET    /v1/system/agents/{id}    - Get agent details
DELETE /v1/system/agents/{id}    - Terminate agent
GET    /v1/system/status         - Factory status
GET    /v1/system/specs          - Agent specifications
```

### Spawn Request

```json
{
    "agent_name": "MasterReasoner",
    "session_id": null  // Optional: resume existing session
}
```

### Spawn Response

```json
{
    "success": true,
    "agent": {
        "agent_id": "agent-a1b2c3d4",
        "instance_id": "inst-e5f6g7h8",
        "agent_type": "PAT",
        "name": "MasterReasoner",
        "status": "READY",
        "lease_id": "lease-x9y0z1",
        "session_id": "sess-abc123def456",
        "spawned_at": "2025-01-27T10:30:00Z"
    }
}
```

## Invariants

1. **I3: Agent Registry Consistency**  
   All agents must be tracked in factory registry

2. **I4: Session Memory Bound**  
   Session turns ≤ max_turns (default: 20)

3. **I5: Resource Coupling**  
   PAT spawn requires URP lease acquisition

## Consequences

### Positive
- Consistent agent lifecycle management
- Session memory enables context continuity
- URP integration prevents resource exhaustion
- Evidence trail for all spawn/terminate events
- REST API enables external orchestration

### Negative
- Singleton pattern limits horizontal scaling
- Memory sessions stored in-memory (persistence needed)
- No agent state checkpointing yet

### Future Work
- Session persistence to disk/Redis
- Agent state checkpointing
- Horizontal scaling via distributed registry
- WebSocket support for real-time updates

## Files

- `core/agent_factory.py` - Factory implementation
- `core/spawn_api.py` - REST API server
- `docs/evidence/agents/spawn_events.jsonl` - Audit log

## References

- ADR-002: URP Implementation
- BIZRA Unified Execution Blueprint v2.0
- F-ARCH-002: Static Agents Finding
