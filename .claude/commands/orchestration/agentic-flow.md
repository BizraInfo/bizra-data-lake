---
allowed-tools: Bash(npx:*), Bash(node:*), Bash(curl:*), Read, Grep, Glob
description: Invoke agentic-flow swarm intelligence and self-learning agents
argument-hint: [swarm|agents|tools|workers|reasoning|health] [args...]
---

# Agentic-Flow Swarm Intelligence

Integrates agentic-flow's 66 self-learning agents, 213 MCP tools, and SONA
neural architecture into BIZRA's orchestration layer.

## Service Status

Check health:
```bash
curl -sf http://localhost:3100/health 2>/dev/null | python3 -m json.tool || echo "agentic-flow: NOT RUNNING"
```

Via BIZRA kernel proxy:
```bash
curl -sf http://localhost:8010/v1/agentic-flow/health 2>/dev/null | python3 -m json.tool || echo "kernel proxy: unavailable"
```

## Commands

### swarm — Launch agent swarm

Topology options: `mesh` (independent), `hierarchical-mesh` (collaborative), `star` (hivemind), `hierarchical` (leader-follower).

```bash
curl -X POST http://localhost:8010/v1/agentic-flow/swarm \
  -H "Authorization: Bearer $BIZRA_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"task": "TASK_DESCRIPTION", "topology": "hierarchical-mesh", "agent_count": 5}'
```

### agents — List available agents
```bash
curl -sf http://localhost:8010/v1/agentic-flow/agents | python3 -m json.tool
```

### tools — List MCP tools
```bash
curl -sf http://localhost:8010/v1/agentic-flow/tools | python3 -m json.tool
```

### workers — Dispatch background worker

Types: `audit`, `optimize`, `consolidate`, `document`, `deepdive`, `ultralearn`.

```bash
curl -X POST http://localhost:8010/v1/agentic-flow/worker \
  -H "Authorization: Bearer $BIZRA_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"worker_type": "ultralearn", "directive": "DIRECTIVE"}'
```

### reasoning — Query ReasoningBank
```bash
curl -X POST http://localhost:8010/v1/agentic-flow/mcp/call \
  -H "Authorization: Bearer $BIZRA_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "reasoning_bank_query", "arguments": {"query": "PATTERN", "limit": 10}}'
```

## BIZRA Integration

All agentic-flow operations are:
1. Gated by Ihsan threshold (>= 0.95)
2. Receipt-audited in `docs/evidence/receipts/agentic_flow/operations.jsonl`
3. FATE escalation on failures
4. Proxied through BIZRA kernel (port 8010)

## Topology Mapping to BIZRA Modes

| BIZRA Mode | Agentic-Flow Topology | Max Agents |
|------------|----------------------|------------|
| Independent | mesh | 15 |
| Collaborative | hierarchical-mesh | 15 |
| HiveMind | star | 8 |

## Docker Management

```bash
docker compose up -d agentic-flow       # Start
docker compose logs -f agentic-flow     # Watch logs
docker compose restart agentic-flow     # Restart
```

## Your Task

Based on the argument "$1", execute the appropriate command above. If no argument given, show the service status.
