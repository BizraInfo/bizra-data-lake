# BIZRA Auto-Configuration Module

## Quick Start

```python
from core.autoconfig import auto_configure
import asyncio

config = asyncio.run(auto_configure())
print(f"Mode: {config['mode']}")
print(f"Capabilities: {config['capabilities']}")
print(f"Routing: {config['model_routing']}")
```

## What It Does

1. **Probes all backend services** (Ollama, LM Studio, Redis, Neo4j, ChromaDB, Postgres)
2. **Detects available LLM models** from Ollama and LM Studio
3. **Generates optimal model routing** based on canonical config + actual availability
4. **Auto-heals** - sets mode to real/degraded/simulated based on service health
5. **Persists config** to `~/.bizra/autoconfig.json`

## Files

- `/core/autoconfig.py` - Main module (284 lines)
- `/.claude/hooks/autoconfig-hook.sh` - SessionStart hook (76 lines)
- `/docs/AUTOCONFIG_SYSTEM.md` - Full documentation

## Test Results

Tested on 2026-02-14 07:46:18Z:

```json
{
  "mode": "degraded",
  "capabilities": ["reasoning", "embeddings"],
  "model_routing": {
    "cold_core": {"model": "mistral:latest", "provider": "ollama"},
    "warm_surface": {"model": "mistral:latest", "provider": "ollama"},
    "embeddings": {"model": "nomic-embed-text:latest", "provider": "ollama"}
  },
  "services": {
    "ollama": {"reachable": true, "models": ["phi3:mini", "mistral:latest", "nomic-embed-text:latest", "llama3.1:8b", "deepseek-r1:14b"]},
    "lmstudio": {"reachable": false},
    "redis": {"reachable": false, "error": "timeout"},
    "neo4j": {"reachable": false, "error": "neo4j driver not installed"}
  }
}
```

## Integration

### Add Hook to `.claude/settings.json`

```json
{
  "hooks": {
    "SessionStart": [
      ".claude/hooks/session-start.sh",
      ".claude/hooks/autoconfig-hook.sh"
    ]
  }
}
```

### Use in Kernel Startup

```python
from core.autoconfig import get_autoconfigurator

@app.on_event("startup")
async def startup_event():
    configurator = get_autoconfigurator()
    config = await configurator.auto_configure()

    if config["mode"] == "simulated":
        os.environ["BIZRA_ADAPTER_MODE"] = "simulated"

    print(f"[BIZRA] Kernel started in {config['mode']} mode")
    print(f"[BIZRA] Capabilities: {', '.join(config['capabilities'])}")
```

## Environment Variables Set by Hook

```bash
BIZRA_ACTIVE_MODELS="phi3:mini,mistral:latest,nomic-embed-text:latest,llama3.1:8b,deepseek-r1:14b"
BIZRA_PRIMARY_REASONING="mistral:latest"
BIZRA_ADAPTER_MODE="degraded"
BIZRA_SERVICES_ONLINE="ollama"
```

## Dependencies

```bash
pip install aiohttp pyyaml redis neo4j asyncpg
```

All optional - missing dependencies cause graceful degradation.

## Receipt Schema (Future)

```json
{
  "schema": "bizra_autoconfig_receipt_v1",
  "timestamp": "2026-02-14T07:46:18Z",
  "node_id": "node0-genesis",
  "mode": "degraded",
  "services_online": ["ollama"],
  "model_count": 5,
  "capabilities": ["reasoning", "embeddings"],
  "config_hash": "sha256:..."
}
```
