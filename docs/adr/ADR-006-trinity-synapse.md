# ADR-006: Trinity Synapse Implementation

**Status:** Accepted  
**Date:** 2025-01-27  
**Author:** BIZRA Genesis System  
**Finding:** F-ARCH-001 (Trinity Disconnect)

## Context

The SAPE Multi-Lens Audit identified a critical architectural gap:

> **F-ARCH-001:** No shared runtime memory between the Trinity (MasterReasoner, MemoryArchitect, EthicsGuardian); each agent runs in isolation.

Agents were operating independently without:
- Real-time communication between agents
- Shared state for coordination
- Agent discovery and presence
- Event sourcing for audit trails

## Decision

Implement **Trinity Synapse** - a Redis-based Agent-to-Agent (A2A) communication layer:

1. **Pub/Sub Channels** - Real-time messaging between agents
2. **Shared State Store** - Coordination data via Redis keys
3. **Presence Tracking** - Agent discovery with TTL heartbeats
4. **Event Sourcing** - Audit trail via Redis Streams
5. **Factory Integration** - Auto-connect agents on spawn

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRINITY SYNAPSE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│   │  PAT Agent  │     │  PAT Agent  │     │  SAT Agent  │                   │
│   │  (Reasoner) │     │  (Memory)   │     │  (Evidence) │                   │
│   └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                   │
│          │                   │                   │                          │
│          │                   │                   │                          │
│          └───────────────────┼───────────────────┘                          │
│                              │                                              │
│                              ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐ │
│   │                        SYNAPSE BUS                                    │ │
│   │                                                                       │ │
│   │  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐          │ │
│   │  │    CHANNELS    │  │     STATE      │  │    EVENTS      │          │ │
│   │  │                │  │                │  │                │          │ │
│   │  │ • broadcast    │  │ • state:*      │  │ • bizra:events │          │ │
│   │  │ • agent:{id}   │  │ • lock:*       │  │   (Stream)     │          │ │
│   │  │ • team:pat     │  │ • presence:*   │  │                │          │ │
│   │  │ • team:sat     │  │                │  │                │          │ │
│   │  │ • task:{id}    │  │                │  │                │          │ │
│   │  │                │  │                │  │                │          │ │
│   │  └────────────────┘  └────────────────┘  └────────────────┘          │ │
│   │                                                                       │ │
│   └──────────────────────────────────────────────────────────────────────┘ │
│                              │                                              │
│                              ▼                                              │
│   ┌──────────────────────────────────────────────────────────────────────┐ │
│   │                    REDIS (synapse:6379)                               │ │
│   └──────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Channel Hierarchy

| Channel | Purpose | Subscribers |
|---------|---------|-------------|
| `bizra:channel:broadcast` | System announcements | All agents |
| `bizra:channel:agent:{id}` | Direct messaging | Single agent |
| `bizra:channel:team:pat` | PAT team coordination | PAT agents |
| `bizra:channel:team:sat` | SAT team coordination | SAT agents |
| `bizra:channel:task:{id}` | Task-specific coordination | Task participants |

### Message Types

```python
class MessageType(Enum):
    # Agent lifecycle
    AGENT_ONLINE = "agent.online"
    AGENT_OFFLINE = "agent.offline"
    AGENT_HEARTBEAT = "agent.heartbeat"
    
    # Task coordination
    TASK_ASSIGNED = "task.assigned"
    TASK_ACCEPTED = "task.accepted"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_DELEGATED = "task.delegated"
    
    # Knowledge sharing
    KNOWLEDGE_QUERY = "knowledge.query"
    KNOWLEDGE_RESPONSE = "knowledge.response"
    KNOWLEDGE_UPDATE = "knowledge.update"
    
    # Consensus
    CONSENSUS_REQUEST = "consensus.request"
    CONSENSUS_VOTE = "consensus.vote"
    CONSENSUS_RESULT = "consensus.result"
    
    # Direct communication
    DIRECT_MESSAGE = "direct.message"
    DIRECT_REQUEST = "direct.request"
    DIRECT_RESPONSE = "direct.response"
```

### Message Format

```json
{
    "id": "msg-abc123def456",
    "type": "task.assigned",
    "sender_id": "agent-a1b2c3d4",
    "sender_name": "MasterReasoner",
    "payload": {
        "task_id": "task-xyz789",
        "description": "Analyze data patterns",
        "priority": "high"
    },
    "timestamp": "2025-01-27T10:30:00Z",
    "correlation_id": null,
    "ttl_ms": null
}
```

### Presence System

Agents register presence with TTL (30 seconds default):

```json
{
    "agent_id": "agent-a1b2c3d4",
    "agent_name": "MasterReasoner",
    "agent_type": "PAT",
    "status": "online",
    "capabilities": ["reasoning", "analysis"],
    "last_seen": "2025-01-27T10:30:00Z",
    "metadata": {}
}
```

Features:
- Automatic expiry if heartbeat stops
- Agent discovery by type or capability
- Status tracking (online, busy, away)

### Shared State

Key-value storage with optional TTL:

```python
# Set state
bus.set_state("task:current", {"id": "task-123", "status": "in_progress"})

# Get state
state = bus.get_state("task:current")

# Distributed locks
if bus.acquire_lock("resource:model", ttl=30):
    # Critical section
    bus.release_lock("resource:model")
```

### Event Sourcing

All broadcast events recorded to Redis Stream:

```python
# Get recent events
events = bus.get_recent_events(count=100)
```

Stream capped at 10,000 events (configurable via `SYNAPSE_EVENT_MAXLEN`).

### Factory Integration

Agents auto-connect to Synapse on spawn:

```python
# In AgentFactory.spawn_pat()
if self._synapse_enabled:
    bus = self._synapse_factory(agent_id, name, "PAT")
    bus.connect(capabilities=[spec["role"]])
    self._agent_buses[agent_id] = bus

# Get synapse bus for an agent
bus = factory.get_synapse_bus(agent_id)
bus.publish_to_agent(other_agent_id, MessageType.DIRECT_REQUEST, {...})
```

### Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SYNAPSE_URL` | `redis://127.0.0.1:6379` | Redis connection URL |
| `SYNAPSE_PREFIX` | `bizra` | Key namespace prefix |
| `SYNAPSE_PRESENCE_TTL` | `30` | Presence expiry (seconds) |
| `SYNAPSE_EVENT_MAXLEN` | `10000` | Event stream max length |

## Invariants

1. **I8: Presence Consistency**  
   All online agents have valid presence records

2. **I9: Channel Isolation**  
   Direct messages only delivered to target agent

3. **I10: Event Immutability**  
   Recorded events cannot be modified

## Consequences

### Positive
- Real-time A2A communication via Pub/Sub
- Agent discovery through presence system
- Shared state enables coordination
- Event stream provides audit trail
- Docker-ready (uses existing synapse service)
- Automatic factory integration

### Negative
- Redis dependency required for A2A
- Single Redis instance (no HA yet)
- Message delivery is fire-and-forget

### Future Work
- Redis Cluster for high availability
- Request-response patterns with timeouts
- Message persistence for offline agents
- Rate limiting per agent
- Encryption for sensitive payloads

## Files

- `core/synapse.py` - Trinity Synapse implementation
- `core/agent_factory.py` - Factory integration added
- `docker-compose.yml` - Uses existing synapse service

## Usage Examples

### Agent-to-Agent Direct Message

```python
# Get factory and synapse bus
factory = get_factory()
reasoner = factory.spawn_pat("MasterReasoner")
bus = factory.get_synapse_bus(reasoner.agent_id)

# Send to another agent
memory_agent = factory.get_agent_by_name("MemoryArchitect")
if memory_agent:
    bus.publish_to_agent(
        memory_agent.agent_id,
        MessageType.KNOWLEDGE_QUERY,
        {"query": "Retrieve context from last session"}
    )
```

### Task Coordination

```python
# Assign task to team
bus.publish_to_team("PAT", MessageType.TASK_ASSIGNED, {
    "task_id": "task-123",
    "description": "Analyze document",
    "deadline": "2025-01-28T00:00:00Z"
})

# Agent accepts task
bus.publish_to_task("task-123", MessageType.TASK_ACCEPTED, {
    "agent_id": bus.agent_id,
    "eta_seconds": 300
})
```

### Agent Discovery

```python
# Find agent with specific capability
agent = bus.find_agent_by_capability("ethics")
if agent:
    bus.publish_to_agent(agent.agent_id, MessageType.DIRECT_REQUEST, {
        "action": "review",
        "content": "Proposed action..."
    })
```

## References

- ADR-004: Agent Factory
- BIZRA Unified Execution Blueprint v2.0
- F-ARCH-001: Trinity Disconnect Finding
- Redis Pub/Sub Documentation
- Redis Streams Documentation
