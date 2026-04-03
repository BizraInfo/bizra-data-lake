---
name: agentic-swarm
description: Invoke agentic-flow's self-learning swarm for parallel agent execution
autoInvoke: false
---

# Agentic Swarm Skill

Provides access to agentic-flow's 66 self-learning agents through BIZRA's
orchestration framework with receipt-native evidence tracking.

## Topologies

| Topology | Max Agents | BIZRA Mode | Best For |
|----------|-----------|------------|----------|
| mesh | 15 | Independent | Parallel autonomous tasks |
| hierarchical-mesh | 15 | Collaborative | Complex coordinated work |
| hierarchical | 10 | — | Leader-follower delegation |
| star | 8 | HiveMind | Consensus decisions |

## SONA Features

- Flash Attention: 2.49x-7.47x speedup over standard attention
- GNN Query Refinement: +12.4% recall improvement
- LoRA Fine-Tuning: 99% parameter reduction for adaptation
- ReasoningBank: Cross-agent pattern sharing and learning

## API Usage

### Invoke Swarm
```
POST http://localhost:8010/v1/agentic-flow/swarm
Authorization: Bearer $BIZRA_API_TOKEN
Content-Type: application/json

{
  "task": "description of work",
  "topology": "hierarchical-mesh",
  "agent_count": 5,
  "timeout_ms": 30000
}
```

### Query ReasoningBank
```
POST http://localhost:8010/v1/agentic-flow/mcp/call
Authorization: Bearer $BIZRA_API_TOKEN
Content-Type: application/json

{
  "tool_name": "reasoning_bank_query",
  "arguments": {"query": "pattern to search", "limit": 10}
}
```

### Dispatch Worker
```
POST http://localhost:8010/v1/agentic-flow/worker
Authorization: Bearer $BIZRA_API_TOKEN
Content-Type: application/json

{
  "worker_type": "ultralearn",
  "directive": "what to learn"
}
```

## Key Files

- `core/agentic_flow_bridge.py` — Python bridge to agentic-flow
- `config/agentic_flow.yaml` — Topology mappings and SONA config
- `config/mcp_servers.json` — MCP server registration
- `vendor/agentic-flow/` — Source (git submodule)

## Evidence

All operations emit receipts to `docs/evidence/receipts/agentic_flow/operations.jsonl`
with SHA-256 integrity hashes following BIZRA's receipt schema.
