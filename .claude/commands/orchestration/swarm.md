---
allowed-tools: Bash(python*:*), Bash(cargo:*), Read, Grep, Glob
description: Coordinate parallel agent execution via swarm intelligence
argument-hint: [task] [--mode independent|collaborative|hivemind] [--count N]
---

# Swarm - Multi-Agent Orchestration

## Overview

The Swarm command orchestrates **parallel agent execution** using swarm intelligence patterns. Multiple agents work together on a task, with coordination mode determining how they interact.

## Swarm Modes

### Independent Mode
```
    ┌─────────┐
    │  Task   │
    └────┬────┘
         │ broadcast
    ┌────┼────┬────┐
    ▼    ▼    ▼    ▼
  [A1] [A2] [A3] [A4]
    │    │    │    │
    ▼    ▼    ▼    ▼
  [R1] [R2] [R3] [R4]
    │    │    │    │
    └────┴────┴────┘
         │ aggregate
    ┌────▼────┐
    │ Results │
    └─────────┘
```

**Behavior**: Agents work autonomously, results aggregated at end
**Best For**: Embarrassingly parallel tasks, diverse perspectives
**Overhead**: Low

### Collaborative Mode
```
    ┌─────────┐
    │  Task   │
    └────┬────┘
         │ distribute
    ┌────┼────┬────┐
    ▼    ▼    ▼    ▼
  [A1]◄─►[A2]◄─►[A3]◄─►[A4]
    │ share │ share │ share │
    └────┴────┴────┴────┘
              │
    ┌─────────▼─────────┐
    │  Shared Workspace │
    └─────────┬─────────┘
              │ synthesize
    ┌─────────▼─────────┐
    │      Results      │
    └───────────────────┘
```

**Behavior**: Agents share intermediate results, build on each other
**Best For**: Complex tasks requiring coordination
**Overhead**: Medium

### HiveMind Mode
```
    ┌─────────┐
    │  Task   │
    └────┬────┘
         │
    ┌────▼────┐
    │  Hive   │ (collective state)
    │  Mind   │
    └────┬────┘
    ┌────┼────┬────┐
    ▼    ▼    ▼    ▼
  [A1] [A2] [A3] [A4]
    │    │    │    │
    └────┴────┴────┘
         │ vote
    ┌────▼────┐
    │Consensus│
    └────┬────┘
         │
    ┌────▼────┐
    │ Result  │
    └─────────┘
```

**Behavior**: Collective decision-making with consensus voting
**Best For**: High-stakes decisions, Byzantine fault tolerance
**Overhead**: High

## Current System Status

- Agent Factory: !`ls -lh core/agent_factory.py 2>/dev/null || echo "Not found"`
- Synapse (Redis): !`redis-cli -u ${SYNAPSE_URL} --tls --cacert config/redis/ca-cert.pem ping 2>/dev/null || echo "Not accessible"`
- Active Agents: !`curl -s http://localhost:8010/v1/system/agents 2>/dev/null | jq 'length // 0' || echo "0"`
- Warm Pool Status: !`curl -s http://localhost:8010/v1/system/status 2>/dev/null | jq -r '.warm_pool_enabled // false' || echo "N/A"`

## Your Task

### Phase 1: Task Analysis

Analyze the task for swarm suitability:

**Swarm Indicators** (good candidates):
- Task is decomposable into subtasks
- Multiple perspectives valuable
- Parallel execution beneficial
- Requires consensus or validation

**Anti-Patterns** (avoid swarm for):
- Sequential dependencies
- Single-threaded logic
- Trivial tasks (overhead > benefit)

### Phase 2: Swarm Configuration

```python
swarm_config = {
    "mode": "collaborative",  # independent | collaborative | hivemind
    "agent_count": 4,
    "agents": [
        {"type": "MasterReasoner", "role": "strategy"},
        {"type": "DataAnalyzer", "role": "analysis"},
        {"type": "CreativeSynthesizer", "role": "generation"},
        {"type": "EthicsGuardian", "role": "validation"}
    ],
    "coordination": {
        "shared_workspace": True,
        "sync_interval_ms": 1000,
        "timeout_ms": 30000
    },
    "consensus": {
        "required": False,  # True for HiveMind
        "threshold": 0.6    # 3/5 for critical
    }
}
```

### Phase 3: Agent Selection

**PAT Agents for Swarm**:

| Agent | Strength | Swarm Role |
|-------|----------|------------|
| MasterReasoner | Strategic thinking | Lead/Coordinator |
| DataAnalyzer | Pattern recognition | Analyst |
| CreativeSynthesizer | Generation | Generator |
| MemoryArchitect | Context management | Context keeper |
| Communicator | External comms | Reporter |
| ExecutionPlanner | Planning | Planner |
| EthicsGuardian | Validation | Validator |

**SAT Agents for Validation**:

| Agent | Role in Swarm |
|-------|---------------|
| PoiVerifier | Impact verification |
| RiskGuardian | Risk assessment |
| GovernanceEngine | Policy enforcement |

### Phase 4: Launch Swarm

**Via API**:

```bash
curl -X POST http://localhost:8080/enhanced \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${BIZRA_API_TOKEN}" \
  -d '{
    "enable_swarm": "Collaborative",
    "base": {
      "user_id": "swarm_user",
      "task": "{task_description}",
      "requirements": ["quality", "speed"],
      "target": "swarm_result"
    }
  }'
```

**Via Python**:

```python
from core.agent_factory import AgentFactory

async def launch_swarm(task, mode="collaborative", count=4):
    factory = AgentFactory()

    # Spawn agents
    agents = []
    for i in range(count):
        agent_type = select_agent_type(task, i)
        agent = await factory.spawn_pat(agent_type)
        agents.append(agent)

    # Configure coordination
    if mode == "collaborative":
        workspace = SharedWorkspace()
        for agent in agents:
            agent.attach_workspace(workspace)

    elif mode == "hivemind":
        hive = HiveMind(agents, consensus_threshold=0.6)
        return await hive.execute(task)

    # Execute
    results = await asyncio.gather(*[
        agent.execute(task) for agent in agents
    ])

    return aggregate_results(results, mode)
```

### Phase 5: Monitor & Aggregate

**Real-time Monitoring**:

```bash
# Watch agent status
watch -n 1 'curl -s http://localhost:8010/v1/system/agents | jq'

# Watch Synapse messages
redis-cli -u ${SYNAPSE_URL} --tls --cacert config/redis/ca-cert.pem \
  subscribe "bizra:broadcast"
```

**Result Aggregation**:

```python
def aggregate_results(results, mode):
    if mode == "independent":
        # Deduplicate and rank
        return rank_by_quality(deduplicate(results))

    elif mode == "collaborative":
        # Merge insights from shared workspace
        return merge_collaborative(results)

    elif mode == "hivemind":
        # Return consensus result
        return consensus_result(results)
```

## Swarm Template

### Task: [User's Task]

---

#### Swarm Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| Mode | [mode] | [why this mode] |
| Agent Count | [N] | [why this number] |
| Timeout | [ms] | [based on complexity] |

#### Agent Assignment

| # | Agent Type | Role | Subtask |
|---|------------|------|---------|
| 1 | MasterReasoner | Lead | [subtask] |
| 2 | DataAnalyzer | Analyst | [subtask] |
| 3 | CreativeSynthesizer | Generator | [subtask] |
| 4 | EthicsGuardian | Validator | [subtask] |

#### Coordination Strategy

**Mode**: [Independent/Collaborative/HiveMind]

**Shared Resources**:
- Workspace: [yes/no]
- Sync Interval: [ms]
- Consensus Required: [yes/no]

#### Execution

```
[Execution timeline or log]
```

#### Results

**Agent 1 (MasterReasoner)**:
- Output: [summary]
- Quality: [score]

**Agent 2 (DataAnalyzer)**:
- Output: [summary]
- Quality: [score]

...

#### Aggregated Result

**Final Output**: [synthesized result]

**Consensus Level**: [if HiveMind]

---

## Validation Checks

### Swarm Validity

- [ ] Mode appropriate for task
- [ ] Agent count justified
- [ ] Agent types match subtasks
- [ ] Coordination configured

### Execution Validity

- [ ] All agents spawned
- [ ] No agent failures
- [ ] Results collected
- [ ] Aggregation successful

## Evidence Generation

Generate Swarm receipt:

```json
{
  "receipt_id": "swarm-$(date +%s)",
  "timestamp": "$(date -Iseconds)",
  "task_summary": "[task]",
  "swarm_config": {
    "mode": "collaborative",
    "agent_count": 4,
    "agents": [],
    "timeout_ms": 30000
  },
  "execution": {
    "agents_spawned": 4,
    "agents_completed": 4,
    "agents_failed": 0,
    "total_time_ms": 0
  },
  "results": {
    "individual_results": [],
    "aggregated_result": "",
    "consensus_level": null
  },
  "integrity_hash": ""
}
```

## Report Format

```
## Swarm Execution Report

**Task**: [task]
**Mode**: [Independent/Collaborative/HiveMind]
**Timestamp**: [ISO timestamp]

### Swarm Composition

| Agent | Type | Role | Status |
|-------|------|------|--------|
| A1 | MasterReasoner | Lead | Complete |
| A2 | DataAnalyzer | Analyst | Complete |
| A3 | CreativeSynthesizer | Generator | Complete |
| A4 | EthicsGuardian | Validator | Complete |

### Execution Timeline

```
T+0ms    : Swarm initialized
T+100ms  : Agents spawned (4/4)
T+5000ms : A1 completed
T+6000ms : A2 completed
T+7000ms : A3 completed
T+8000ms : A4 completed
T+8500ms : Aggregation complete
```

### Results

**Individual Outputs**:
- A1: [summary]
- A2: [summary]
- A3: [summary]
- A4: [summary]

**Aggregated Result**:
[Final synthesized result]

### Metrics
- Total Time: Xms
- Agent Utilization: X%
- Consensus Level: X% (if HiveMind)

### Receipt
- ID: swarm-[timestamp]
- Location: docs/evidence/receipts/
```

---

**Swarm Philosophy**: "Many minds, one purpose. Independent for speed, collaborative for synergy, hivemind for consensus. Match coordination to complexity."
