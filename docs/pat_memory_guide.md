# PAT Memory - Persistent Memory System

**Location**: `/mnt/c/BIZRA-Dual-Agentic-system--main/core/pat_memory.py`

## Overview

PAT Memory provides dual-layer persistent storage for BIZRA's Personal Agentic Team (PAT), surviving full system restarts with automatic restoration.

### Dual-Layer Architecture

```
┌───────────────────────────────────────────────────────────┐
│                    PAT MEMORY STORE                        │
├───────────────────────────────────────────────────────────┤
│                                                            │
│   HOT LAYER (Redis)          COLD LAYER (JSON)            │
│   ┌─────────────────┐        ┌─────────────────┐          │
│   │ user_preferences│        │ .bizra/          │          │
│   │ session_history │   ←→   │ pat_memory.json │          │
│   │ learned_patterns│        │                 │          │
│   │ model_routing   │        │ (persistent)    │          │
│   │ system_config   │        │                 │          │
│   └─────────────────┘        └─────────────────┘          │
│          ↕                            ↕                    │
│   bizra:pat:memory:*        ~/.bizra/ (Linux/WSL)         │
│   (fast, session)            C:/Users/{user}/.bizra (Win) │
│                                                            │
└───────────────────────────────────────────────────────────┘
```

## Features

- **Dual Persistence**: Redis hot layer + JSON cold layer
- **Automatic Restoration**: Loads cold storage into Redis on startup
- **Graceful Degradation**: Works without Redis (cold storage only)
- **System Detection**: Auto-detects GPU, RAM, Ollama/LM Studio models
- **Receipt Native**: All operations emit JSONL receipts
- **5 Memory Categories**: Organized storage for different data types

## Memory Categories

| Category | Purpose | Example Keys |
|----------|---------|--------------|
| `user_preferences` | UI settings, favorite models, display prefs | `theme`, `language`, `favorite_model` |
| `session_history` | Recent session records (last 50) | `sess-{id}` |
| `learned_patterns` | User workflow patterns PAT has learned | `privacy_preference`, `code_style` |
| `model_routing` | Which models work best for which tasks | `code_generation`, `reasoning` |
| `system_config` | Auto-detected system capabilities | `detected` |

## Installation

```bash
# Required
pip install redis

# Optional (for system detection)
pip install psutil      # RAM detection
pip install httpx       # LLM backend probing
```

## Configuration

Environment variables:

```bash
# Redis connection (hot layer)
SYNAPSE_URL="redis://:password@localhost:6380"

# Optional: Use TLS
SYNAPSE_URL="rediss://:password@localhost:6379"
```

Cold storage location:
- **Linux/WSL**: `~/.bizra/pat_memory.json`
- **Windows**: `C:\Users\{username}\.bizra\pat_memory.json`

## Usage

### Basic Operations

```python
from core.pat_memory import get_pat_memory

# Get singleton instance
memory = await get_pat_memory()

# Store a value
await memory.store("user_preferences", "theme", "dark")

# Retrieve a value
theme = await memory.retrieve("user_preferences", "theme", default="light")

# Retrieve all in category
prefs = await memory.retrieve_all("user_preferences")

# Store with TTL (Redis only, not persisted)
await memory.store("session_history", "temp", data, ttl=3600)
```

### Session Tracking

```python
session = {
    "session_id": "sess-001",
    "timestamp": "2026-02-14T07:00:00Z",
    "task": "Implement SAPE validation",
    "outcome": "success",
    "duration_seconds": 120,
    "model_used": "deepseek-r1:14b",
}

await memory.store("session_history", session["session_id"], session)
```

### Pattern Learning

```python
pattern = {
    "pattern": "User prefers reasoning models for technical tasks",
    "confidence": 0.85,
    "observations": 12,
}

await memory.learn_pattern("technical_model_preference", pattern)
```

### Model Performance Tracking

```python
routing = {
    "task_type": "code_generation",
    "best_model": "deepseek-r1:14b",
    "avg_latency_ms": 850,
    "success_rate": 0.95,
    "trials": 20,
}

await memory.store("model_routing", "code_generation", routing)
```

### LLM Context Injection

```python
# Get full user context for system prompts
context = await memory.get_user_context()

system_prompt = f"""
You are PAT for this user.

Preferences:
- Theme: {context['user_preferences'].get('theme')}
- Language: {context['user_preferences'].get('language')}
- Favorite Model: {context['user_preferences'].get('favorite_model')}

System:
- GPU: {context['system_config']['detected']['gpu']['name']}
- RAM: {context['system_config']['detected']['ram']['total_gb']} GB

Adapt your responses to this user's preferences.
"""
```

### System Detection

```python
# Auto-detect GPU, RAM, available models
system_info = await memory.detect_system()

print(f"GPU: {system_info['gpu']['name']}")
print(f"RAM: {system_info['ram']['total_gb']} GB")
print(f"Ollama Models: {len(system_info['ollama_models'])}")
```

### Manual Sync

```python
# Flush hot layer to disk (automatic on close)
await memory.sync_to_disk()

# Load from disk (automatic on initialize)
await memory.load_from_disk()

# Close (syncs automatically)
await memory.close()
```

## CLI Usage

```bash
# Detect system capabilities
python3 core/pat_memory.py --detect

# Show user context
python3 core/pat_memory.py --context

# Sync to disk
python3 core/pat_memory.py --sync

# Load from disk
python3 core/pat_memory.py --load

# Run test scenario
python3 core/pat_memory.py --test
```

## Receipt Schema

All operations emit receipts to `docs/evidence/receipts/pat_memory/operations.jsonl`:

```json
{
  "schema": "bizra.pat_memory.v1",
  "receipt_type": "MemoryOperation",
  "receipt_id": "MEM-20260214070000-a1b2c3d4",
  "timestamp": "2026-02-14T07:00:00.123456+00:00",
  "operation": "store",
  "category": "user_preferences",
  "key": "theme",
  "value_hash": "sha256...",
  "success": true,
  "error": null,
  "integrity_hash": "sha256..."
}
```

## Integration with BIZRA Kernel

### FastAPI Endpoint

```python
from fastapi import FastAPI
from core.pat_memory import get_pat_memory

app = FastAPI()

@app.on_event("startup")
async def startup():
    memory = await get_pat_memory()
    await memory.detect_system()

@app.get("/context")
async def get_context():
    memory = await get_pat_memory()
    return await memory.get_user_context()
```

### PAT Agent Context

```python
from core.pat_memory import get_pat_memory

async def execute_task(task: str):
    memory = await get_pat_memory()
    context = await memory.get_user_context()

    # Inject context into PAT agent
    system_prompt = build_prompt(context)
    response = await llm.generate(task, system_prompt=system_prompt)

    # Record session
    session = {
        "session_id": generate_id(),
        "timestamp": utc_now_iso(),
        "task": task,
        "outcome": "success",
    }
    await memory.store("session_history", session["session_id"], session)

    return response
```

## Best Practices

1. **Always use the singleton**: Call `get_pat_memory()` instead of instantiating directly
2. **Close on shutdown**: Ensure `await memory.close()` runs to sync final state
3. **Use TTL for ephemeral data**: Session-only data should use TTL to avoid cold storage
4. **Respect category boundaries**: Don't mix data types across categories
5. **Log pattern learning**: Use `learn_pattern()` to capture behavioral insights
6. **Trust auto-sync**: Cold storage syncs automatically on writes (no TTL) and close

## Troubleshooting

### Redis Not Available

```
Redis unavailable: Timeout connecting to server. Using cold storage only.
```

**Solution**: System operates normally with cold storage only. Redis provides speed but is optional.

### Deprecation Warning (redis-py 5.x)

```
DeprecationWarning: Call to deprecated close. (Use aclose() instead)
```

**Solution**: Already handled with fallback. Update redis-py to 5.0.1+ for clean output.

### Cold Storage Not Found

```
No cold storage found. Starting fresh.
```

**Solution**: Normal on first run. Cold storage created automatically on first write.

## Performance

| Operation | Hot (Redis) | Cold (JSON) |
|-----------|-------------|-------------|
| Store | ~1-2ms | ~5-10ms |
| Retrieve | ~1ms | ~3-5ms |
| Retrieve All | ~5ms | ~3-5ms |
| Sync to Disk | N/A | ~10-20ms |
| System Detect | N/A | ~200-500ms |

## Security

- **No secrets**: Never store API keys, passwords, or tokens
- **Privacy-first**: User data stays local (never sent to cloud)
- **Receipt audit**: All operations logged with SHA-256 integrity hashes
- **Cold storage security**: File permissions should be user-only (600)

## Future Enhancements

- [ ] Encryption for cold storage (AES-256)
- [ ] Compression for session history (gzip)
- [ ] Multi-user support (separate cold storage per user)
- [ ] Redis cluster support (multi-node hot layer)
- [ ] GraphQL API for memory queries
- [ ] Web dashboard for memory visualization

## Related Documentation

- [Trinity Synapse](../core/synapse.py) - Redis communication layer
- [Unified Memory](../core/unified_memory.py) - Multi-tier memory system
- [FATE Engine](../core/fate.py) - Receipt emission patterns
- [Receipt Schema](../src/receipts.rs) - Rust receipt validation

## Example: Full Integration

See [`examples/pat_memory_integration.py`](/mnt/c/BIZRA-Dual-Agentic-system--main/examples/pat_memory_integration.py) for complete working examples.

---

**Last Updated**: 2026-02-14
**Author**: BIZRA Core Team
**Version**: 1.0.0
