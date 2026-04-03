---
paths:
  - "core/agent_factory.py"
  - "core/spawn_api.py"
  - "src/pat.rs"
  - "src/sat.rs"
  - "src/types.rs"
---

# Agent Features & Capabilities

Detailed documentation for BIZRA's agent system.

## Memory Architecture

Multi-tier memory system (defined in `src/types.rs`):

| Tier | Description | Storage |
|------|-------------|---------|
| Working | Current context (20 turns) | In-memory |
| Episodic | Past conversations | Redis |
| Semantic | Facts from expertise files | PostgreSQL |
| Procedural | Learned skills/patterns | Expertise YAML |

## Agent Lifecycle

```python
# Spawn agent
POST /v1/system/spawn {"agent_name": "MasterReasoner"}

# List agents
GET /v1/system/agents

# Terminate
DELETE /v1/system/agents/{agent_id}
```

## PAT Agents (7)

| Agent | Model | VRAM | Role |
|-------|-------|------|------|
| MasterReasoner | deepseek-r1:7b | 4.5GB | Strategic thinking |
| MemoryArchitect | qwen2.5:7b | 4GB | Knowledge organization |
| CreativeSynthesizer | qwen2.5:7b | 4GB | Writing, ideation |
| DataAnalyzer | mistral:7b | 4GB | Pattern recognition |
| Communicator | mistral:7b | 4GB | External comms |
| ExecutionPlanner | agentflow-7b | 4GB | Task planning |
| EthicsGuardian | qwen2.5:7b | 4GB | Safety, bias |

## SAT Agents (5)

Rule-based validators with minimal resources:
- **PoiVerifier**: Proof-of-Impact validation
- **ResourceAllocator**: Compute/memory optimization
- **RiskGuardian**: Security monitoring
- **GovernanceEngine**: Policy enforcement
- **EvidenceEngine**: Audit trail generation

## Warm Pools

Performance optimization: **5000ms → 500ms** (90% reduction)

```bash
BIZRA_WARM_POOL=true
BIZRA_POOL_MASTER_REASONER=2
BIZRA_POOL_MEMORY_ARCHITECT=1
```

## MCP Tool System

Secure JSON-RPC 2.0 tool execution:
- Blocklist: shell_exec, eval, file_delete
- Default allowlist: filesystem_read, web_search, code_analysis
- SAPE gating on all tool calls
- 30s timeout, 1MB output limit

## A2A Protocol

Agent-to-agent communication via Trinity Synapse:
- Channels: `bizra:broadcast`, `bizra:agent:{id}`, `bizra:team:{pat|sat}`
- Message types: TASK_ASSIGNED, CONSENSUS_REQUEST, KNOWLEDGE_QUERY
- Redis pub/sub with TLS (rediss://)

## Slash Commands

Built-in: `/help`, `/agent`, `/team`, `/recall`, `/verify`, `/status`, `/list`

Enhanced: `/reason`, `/spawn`, `/swarm`, `/memory`, `/hook`, `/tools`, `/delegate`

## Key Files

- `core/agent_factory.py` - Agent spawning, warm pools
- `core/spawn_api.py` - REST API endpoints
- `core/synapse.py` - Trinity Synapse messaging
- `src/pat.rs` - PAT implementation
- `src/sat.rs` - SAT implementation
- `src/mcp.rs` - MCP tool system
- `src/a2a.rs` - A2A protocol
