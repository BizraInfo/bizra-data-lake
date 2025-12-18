# ADR-002: Unified Resource Planner (URP) Implementation

**Status:** Accepted  
**Date:** 2025-12-19  
**Updated:** 2025-01-27  
**Context Finding:** F-PERF-001 (Resource Blindness)

## Context

Previous designs lacked awareness of actual Node0 hardware capabilities.
Risk of OOM crashes and underutilization of available resources.

## Node0 Hardware Profile

| Component | Specification | Usable |
|-----------|---------------|--------|
| **GPU** | NVIDIA RTX 4090 | 14GB VRAM (16GB - 2GB overhead) |
| **CPU** | Intel Core i9-14900 | 24 cores / 32 threads |
| **RAM** | DDR5 | 112GB (128GB - 16GB overhead) |
| **Storage** | NVMe SSD | 2.5TB (3TB - 500GB system) |

## Decision

Implement a **Lease-based Resource Manager** with:

1. **Dual-Mode Allocation**: GPU mode (VRAM) or CPU mode (RAM)
2. **Lease System**: Pre-allocate before spawn, auto-release on expiry
3. **Concurrency Limits**: 
   - GPU: 3 agents (VRAM-bound)
   - CPU: 10 agents (RAM-bound)
   - Total: 13 concurrent agents
4. **Evidence Logging**: All lease events recorded

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    URP LEASE MANAGER                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────────┐    ┌─────────────────┐                │
│   │    GPU POOL     │    │    RAM POOL     │                │
│   │   14GB VRAM     │    │    112GB RAM    │                │
│   │   3 agents max  │    │   10 agents max │                │
│   └────────┬────────┘    └────────┬────────┘                │
│            │                      │                          │
│            └──────────┬───────────┘                          │
│                       │                                      │
│   Request ──▶ [Mode Selection] ──▶ [Capacity Check]         │
│                                          │                   │
│                              ┌───────────┴───────────┐      │
│                              │                       │      │
│                        OverCapacityError        Lease Token │
│                                                      │      │
│   Agent Execution ◀── [Resource Bound] ──▶ [TTL Monitor]   │
│                                                │             │
│                                         [Auto Release]       │
└─────────────────────────────────────────────────────────────┘
```

## Resource Modes

| Mode | Resources | Use Case |
|------|-----------|----------|
| `GPU` | VRAM only | Fast inference (default) |
| `CPU` | RAM only | More concurrent agents |
| `HYBRID` | VRAM + RAM | Large models with offloading |

## Invariants

- **I1**: `Total_Allocated_VRAM <= 14GB`
- **I2**: `GPU_Agents <= 3`
- **I3**: `Total_Allocated_RAM <= 112GB`
- **I4**: `CPU_Agents <= 10`
- **I5**: `Lease_TTL <= 300s` (5 min max)

## Implementation

**File**: `core/urp/manager.py`

```python
from core.urp import URPManager, URPLease, ResourceRequest, ResourceMode

# GPU mode (default - fast inference)
with URPLease("MasterReasoner") as lease:
    response = call_agent(message)

# CPU mode (more concurrent agents)
request = ResourceRequest(
    agent_id="DataAnalyzer",
    mode=ResourceMode.CPU,
    ram_gb=14.0
)
lease = urp.acquire(request)

# Check capacity before spawn
if urp.can_allocate(vram_gb=4.0, mode=ResourceMode.GPU):
    agent = factory.spawn_pat("MasterReasoner")
```

## Agent Resource Requirements

### GPU Mode (VRAM)

| Agent | Model | VRAM (GB) |
|-------|-------|-----------|
| MasterReasoner | deepseek-r1:7b | 4.5 |
| MemoryArchitect | qwen2.5:7b | 4.0 |
| CreativeSynthesizer | qwen2.5:7b | 4.0 |
| DataAnalyzer | mistral:7b | 4.0 |
| Communicator | mistral:7b | 4.0 |
| ExecutionPlanner | agentflow-7b | 4.0 |
| EthicsGuardian | qwen2.5:7b | 4.0 |
| SAT Agents | rule-based | 0.1 |

### CPU Mode (RAM)

| Agent | RAM (GB) | Notes |
|-------|----------|-------|
| PAT Agents | 14-16 | CPU inference (slower) |
| SAT Agents | 0.5 | Rule-based |
| EmbeddingService | 2.0 | Sentence transformers |

## Consequences

### Positive
- Prevents OOM crashes
- Predictable resource utilization
- Evidence trail for auditing
- Graceful degradation (queue instead of crash)

### Negative
- Slight overhead from lease management
- Requires agent code to use URPLease context

### Risks Mitigated
- **VRAM OOM**: Now impossible if all agents use URP
- **Resource Starvation**: TTL ensures orphaned leases are reclaimed

## Evidence

- Implementation: `core/urp/manager.py`
- Tests: `python -m core.urp.manager --test`
- Logs: `docs/evidence/urp/lease_events.jsonl`

## Related

- ADR-003: FATE Recursive Correction
- F-PERF-001: Resource Blindness finding
