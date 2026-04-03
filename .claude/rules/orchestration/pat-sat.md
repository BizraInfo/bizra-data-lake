---
paths:
  - "src/bridge.rs"
  - "src/pat*.rs"
  - "src/sat*.rs"
  - "src/a2a.rs"
  - "core/synapse.py"
  - "constellation/**/*.py"
---

# PAT-SAT Orchestration Rules

Rules for BIZRA's dual-agentic coordination system.

## Architecture Overview

### Request Flow
```
User Request
    ↓
SAT Pre-Validation (3/5 consensus)
    ↓
SAPE Probing (9 probes)
    ↓
Ihsān Gate (≥ 0.99)
    ↓
PAT Execution (specialized agents)
    ↓
SAT Post-Validation
    ↓
Receipt Emission
    ↓
Response
```

## PAT (Personal Agentic Team)

### 7 Specialized Agents

| Agent | Model | Purpose |
|-------|-------|---------|
| MasterReasoner | deepseek-r1:7b | Strategic thinking |
| MemoryArchitect | qwen2.5:7b | Knowledge organization |
| CreativeSynthesizer | qwen2.5:7b | Writing, ideation |
| DataAnalyzer | mistral:7b | Pattern recognition |
| Communicator | mistral:7b | External comms |
| ExecutionPlanner | agentflow-7b | Task planning |
| EthicsGuardian | qwen2.5:7b | Safety, bias detection |

### Agent Selection
```rust
fn select_pat_agents(task: &Task) -> Vec<AgentSpec> {
    let mut agents = vec![];

    // Always include reasoner for strategic tasks
    if task.requires_reasoning() {
        agents.push(AgentSpec::master_reasoner());
    }

    // Add specialized agents based on task type
    match task.category {
        TaskCategory::Analysis => agents.push(AgentSpec::data_analyzer()),
        TaskCategory::Creation => agents.push(AgentSpec::creative_synthesizer()),
        TaskCategory::Communication => agents.push(AgentSpec::communicator()),
        _ => {}
    }

    // Always include ethics guardian for parallel validation
    agents.push(AgentSpec::ethics_guardian());

    agents
}
```

## SAT (System Agentic Team)

### 5 Guardian Agents

| Agent | Purpose |
|-------|---------|
| PoiVerifier | Proof-of-Impact validation |
| ResourceAllocator | Compute/memory optimization |
| RiskGuardian | Security monitoring |
| GovernanceEngine | Policy enforcement |
| EvidenceEngine | Audit trail generation |

### Consensus Rules
- **Pre-validation**: 3/5 SAT agents must approve
- **Post-validation**: Majority confirmation
- **On failure**: FATE escalation + rejection receipt

```rust
async fn sat_consensus(agents: &[SatAgent], context: &ValidationContext) -> ConsensusResult {
    let votes = futures::future::join_all(
        agents.iter().map(|a| a.vote(context))
    ).await;

    let approvals = votes.iter().filter(|v| v.approved).count();
    let required = 3; // 3/5 consensus

    if approvals >= required {
        ConsensusResult::approved(votes)
    } else {
        // NEVER proceed without consensus
        let rejections: Vec<_> = votes.iter()
            .filter(|v| !v.approved)
            .map(|v| v.rejection_reason.clone())
            .collect();

        ConsensusResult::rejected(rejections)
    }
}
```

## A2A (Agent-to-Agent) Protocol

### Communication Channels
- `bizra:broadcast` → All agents
- `bizra:agent:{id}` → Specific agent
- `bizra:team:pat` → All PAT agents
- `bizra:team:sat` → All SAT agents
- `bizra:task:{id}` → Task coordination

### Message Types
```rust
pub enum A2AMessage {
    // Lifecycle
    AgentOnline { agent_id: String, capabilities: Vec<String> },
    AgentOffline { agent_id: String },
    AgentHeartbeat { agent_id: String },

    // Task coordination
    TaskAssigned { task_id: String, agent_id: String },
    TaskAccepted { task_id: String },
    TaskCompleted { task_id: String, result: TaskResult },
    TaskDelegated { task_id: String, to_agent: String },

    // Consensus
    ConsensusRequest { task_id: String, context: ValidationContext },
    ConsensusVote { task_id: String, approved: bool, reason: Option<String> },
    ConsensusResult { task_id: String, approved: bool },
}
```

### Delegation Rules
- Max delegation depth: 5 levels
- System agents cannot receive delegations
- Timeout: 60 seconds per delegation
- Always log delegation chains for audit

## Trinity Synapse (Redis Pub/Sub)

### Connection Requirements
- Use TLS: `rediss://` (not `redis://`)
- Authenticate with password
- Set presence TTL (30 seconds)

### State Management
```python
# Shared state
await redis.set(f"state:{key}", json.dumps(value))

# Distributed locks
lock = await redis.lock(f"locks:{resource}", timeout=30)
async with lock:
    # Critical section

# Agent presence
await redis.setex(f"presence:{agent_id}", 30, "online")
```

## Warm Pools

### Performance Optimization
Pre-spawn agents to reduce latency:

```python
# Configuration
BIZRA_WARM_POOL=true
BIZRA_POOL_MASTER_REASONER=2
BIZRA_POOL_MEMORY_ARCHITECT=1
BIZRA_POOL_ETHICS_GUARDIAN=1
```

### Pool Management
- Check pool first, cold spawn as fallback
- Async replenishment when pool drops
- Thread-safe dual-lock architecture

## Error Handling

### Consensus Failure
```rust
if !consensus.approved {
    // 1. Log failure with full context
    tracing::error!(
        "SAT consensus failed",
        task_id = %task.id,
        rejections = ?consensus.rejections
    );

    // 2. Escalate to FATE
    fate.escalate(EscalationLevel::High, &consensus.rejections).await?;

    // 3. Emit rejection receipt
    receipts.emit_rejection(&task, &consensus.rejection_codes).await?;

    // 4. Return error (fail-closed)
    return Err(BizraError::ConsensusFailure(consensus));
}
```

### Agent Failure
```rust
// Handle individual agent failure gracefully
match agent.execute(&task).await {
    Ok(result) => result,
    Err(e) => {
        tracing::warn!("Agent {} failed: {}", agent.id, e);

        // Try fallback agent if available
        if let Some(fallback) = get_fallback_agent(&agent.role) {
            fallback.execute(&task).await?
        } else {
            return Err(BizraError::AgentFailure(e));
        }
    }
}
```

## Testing

- Test PAT agent selection for various task types
- Test SAT consensus with different vote combinations
- Test A2A message routing
- Test delegation depth limits
- Test warm pool acquisition and replenishment
- Test failure escalation paths
- Mock LLM responses for deterministic tests
