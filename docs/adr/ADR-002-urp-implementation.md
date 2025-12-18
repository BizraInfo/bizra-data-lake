# ADR-002: Unified Resource Planner (URP) Implementation

**Status:** Accepted  
**Date:** 2025-12-19  
**Context Finding:** F-PERF-001 (Resource Blindness)

## Context

Previous URP designs assumed 24GB VRAM; actual Node0 hardware (RTX 4090) has 16GB. 
Risk of OOM (Out of Memory) crashes during training and multi-agent inference.

## Decision

Implement a **Lease-based VRAM Manager** with:

1. **Hard Cap**: 14GB usable (2GB system overhead)
2. **Lease System**: Pre-allocate before spawn, auto-release on expiry
3. **Concurrency Limit**: Max 3 concurrent "Thinking" agents
4. **Evidence Logging**: All lease events recorded

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    URP LEASE MANAGER                         │
├─────────────────────────────────────────────────────────────┤
│   Request ──▶ [Capacity Check] ──▶ [Lease Grant/Reject]    │
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

## Invariants

- **I1**: `Total_Allocated_VRAM <= 14GB`
- **I2**: `Active_Agents <= 3`
- **I3**: `Lease_TTL <= 300s` (5 min max)

## Implementation

**File**: `core/urp/manager.py`

```python
from core.urp import URPManager, URPLease, ResourceRequest

# Acquire with context manager
with URPLease("MasterReasoner") as lease:
    response = call_agent(message)
# Auto-released

# Or manual control
urp = URPManager()
lease = urp.acquire(ResourceRequest(agent_id="DataAnalyzer", vram_gb=4.0))
try:
    # use resources
finally:
    urp.release(lease.lease_id)
```

## Agent VRAM Requirements

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
