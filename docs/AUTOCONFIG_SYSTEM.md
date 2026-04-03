# BIZRA Kernel Auto-Configuration System

## Overview

The auto-configuration system probes all backend services on startup, detects available LLM models, and generates runtime configuration. It handles graceful degradation when services are unavailable.

## Files Created

### 1. `/core/autoconfig.py`

**Purpose**: Core auto-configuration module that probes services and generates runtime config.

**Key Components**:

- `AutoConfigurator` class - Main configuration engine
- `ServiceProbeResult` dataclass - Service probe results
- `ModelInfo` dataclass - Model metadata
- `get_autoconfigurator()` - Singleton accessor
- `auto_configure()` - Entry point function

**Service Probing**:

```python
async def probe_services() -> Dict[str, ServiceProbeResult]:
    """Probe all backend services concurrently."""
```

Probes:
- **Ollama**: `http://localhost:11434/api/tags` - List available models
- **LM Studio**: `http://localhost:1234/v1/models` - List available models
- **Redis/Synapse**: `localhost:6380` with password `bizra_synapse_secure`
- **Neo4j/Wisdom**: `bolt://localhost:7474` with auth from `NEO4J_AUTH`
- **ChromaDB**: `http://localhost:8001/api/v1/heartbeat`
- **PostgreSQL**: Connection test via `asyncpg`

**Model Routing**:

```python
def configure_model_routing(services: Dict[str, ServiceProbeResult]) -> Dict[str, Dict[str, str]]:
    """Generate model routing table based on available models."""
```

Reads canonical routing from `model-family-genesis-v1-SEALED.yaml` and cross-references with actually available models from Ollama/LM Studio.

**Fallback Chain**:
1. Try primary model from YAML
2. Try fallback model from YAML
3. Try any allowed model that's available
4. Skip slot if no models available

**Auto-Healing**:

```python
def auto_heal(services, routing) -> str:
    """Determine operational mode based on service availability."""
    # Returns: "real", "degraded", or "simulated"
```

- **real**: All services online, LLM available, routing configured
- **degraded**: LLM available but Redis/Neo4j down, or partial routing
- **simulated**: No LLM backend available

**Config Output** (saved to `~/.bizra/autoconfig.json`):

```json
{
  "timestamp": "2026-02-14T07:46:18Z",
  "node_id": "node0-genesis",
  "services": {
    "ollama": {
      "reachable": true,
      "latency_ms": 103.08,
      "info": {"models": ["phi3:mini", "mistral:latest", "nomic-embed-text:latest", "llama3.1:8b", "deepseek-r1:14b"]},
      "error": null
    },
    ...
  },
  "model_routing": {
    "cold_core": {"model": "deepseek-r1:14b", "provider": "ollama"},
    "warm_surface": {"model": "mistral:latest", "provider": "ollama"},
    "embeddings": {"model": "nomic-embed-text:latest", "provider": "ollama"}
  },
  "mode": "real",
  "capabilities": ["reasoning", "embeddings", "graph_memory"]
}
```

### 2. `/.claude/hooks/autoconfig-hook.sh`

**Purpose**: Bash hook that runs on SessionStart and injects runtime config as context.

**Execution Flow**:

1. Run `python3 -c "from core.autoconfig import auto_configure; asyncio.run(auto_configure())"`
2. Parse JSON output with `jq`
3. Extract key metrics (mode, capabilities, services, models)
4. Output human-readable context
5. Set environment variables in `$CLAUDE_ENV_FILE`

**Environment Variables Set**:

```bash
export BIZRA_ACTIVE_MODELS="phi3:mini,mistral:latest,nomic-embed-text:latest,llama3.1:8b,deepseek-r1:14b"
export BIZRA_PRIMARY_REASONING="deepseek-r1:14b"
export BIZRA_ADAPTER_MODE="real"  # or "degraded" or "simulated"
export BIZRA_SERVICES_ONLINE="ollama,redis,neo4j"
```

**Output Example**:

```
--- BIZRA Auto-Configuration ---
Mode: real
Capabilities: reasoning, embeddings, graph_memory

Backend Services:
  - Ollama: true (phi3:mini, mistral:latest, nomic-embed-text:latest, llama3.1:8b, deepseek-r1:14b)
  - LM Studio: false (no models)
  - Redis/Synapse: true
  - Neo4j/Wisdom: true

Primary Reasoning Model: deepseek-r1:14b
Online Services: ollama,redis,neo4j

Full config saved to: ~/.bizra/autoconfig.json
---
```

## Testing

### Test Auto-Configuration Module

```bash
cd /mnt/c/BIZRA-Dual-Agentic-system--main
python3 -c "
import asyncio
import json
from core.autoconfig import auto_configure

config = asyncio.run(auto_configure())
print(json.dumps(config, indent=2))
"
```

### Test Hook Execution

```bash
cd /mnt/c/BIZRA-Dual-Agentic-system--main
export CLAUDE_PROJECT_DIR=$(pwd)
echo '{"source":"test"}' | ./.claude/hooks/autoconfig-hook.sh
```

### Verify Saved Config

```bash
cat ~/.bizra/autoconfig.json | jq .
```

## Integration with Existing System

### Session Start Hook Chain

The autoconfig hook should run **after** the existing `session-start.sh`. Update `.claude/settings.json`:

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

### Usage in FastAPI Kernel

The kernel can load the config on startup:

```python
from core.autoconfig import get_autoconfigurator

@app.on_event("startup")
async def startup_event():
    configurator = get_autoconfigurator()
    config = await configurator.auto_configure()

    if config["mode"] == "simulated":
        logger.warning("No LLM backend available - running in simulated mode")
        os.environ["BIZRA_ADAPTER_MODE"] = "simulated"
    elif config["mode"] == "degraded":
        logger.warning("Degraded mode - some services unavailable")
```

## Known Models on This System

Based on the probe, these Ollama models are available:

- `phi3:mini` - Small reasoning model
- `mistral:latest` - Primary warm surface model
- `nomic-embed-text:latest` - Embeddings
- `llama3.1:8b` - Alternative reasoning
- `deepseek-r1:14b` - Deep reasoning (recommended for cold_core)

## Fallback Behavior

### No LLM Backend

```json
{
  "mode": "simulated",
  "model_routing": {},
  "capabilities": []
}
```

Sets `BIZRA_ADAPTER_MODE=simulated` to use mock responses.

### Partial Availability

```json
{
  "mode": "degraded",
  "model_routing": {
    "cold_core": {"model": "mistral:latest", "provider": "ollama"}
  },
  "capabilities": ["reasoning"]
}
```

Uses available models, logs warnings for missing services.

### Full Availability

```json
{
  "mode": "real",
  "model_routing": {
    "cold_core": {"model": "deepseek-r1:14b", "provider": "ollama"},
    "warm_surface": {"model": "mistral:latest", "provider": "ollama"},
    "embeddings": {"model": "nomic-embed-text:latest", "provider": "ollama"}
  },
  "capabilities": ["reasoning", "embeddings", "graph_memory"]
}
```

All systems operational.

## Error Handling

- **aiohttp not installed**: HTTP probes return `ServiceProbeResult(False, error="aiohttp not installed")`
- **redis.asyncio not installed**: Redis probe fails gracefully
- **neo4j driver not installed**: Neo4j probe skipped
- **Timeout (2s)**: Service marked unreachable with latency recorded
- **Connection refused**: Service marked unreachable with error message

All errors are fail-closed - degraded mode is preferred over silent failure.

## Performance

- **Concurrent probing**: All services probed in parallel via `asyncio.gather()`
- **Timeout budget**: 2 seconds per service (configurable)
- **Total startup overhead**: ~2-3 seconds for full probe
- **Config persistence**: Saved to `~/.bizra/autoconfig.json` for inspection

## Security

- **No secrets in config**: Passwords not stored in output
- **Safe defaults**: Fail-closed error handling
- **Environment variables**: Only service URLs from env vars
- **TLS support**: Redis probes support `rediss://` URLs

## Receipt Emission

Future enhancement: Emit a receipt for each auto-configuration run:

```json
{
  "schema": "bizra_autoconfig_receipt_v1",
  "timestamp": "2026-02-14T07:46:18Z",
  "node_id": "node0-genesis",
  "mode": "real",
  "services_online": ["ollama", "redis", "neo4j"],
  "model_count": 5,
  "capabilities": ["reasoning", "embeddings", "graph_memory"],
  "config_hash": "sha256:..."
}
```

## Next Steps

1. **Hook Registration**: Add `autoconfig-hook.sh` to `.claude/settings.json`
2. **Kernel Integration**: Load config in `core/main.py` startup event
3. **Dashboard Display**: Show config in React dashboard
4. **Health Endpoint**: Expose config via `/v1/autoconfig/status`
5. **Receipt Emission**: Emit config receipts for audit trail

## Dependencies

- Python 3.11+
- `aiohttp` - HTTP client for API probing
- `pyyaml` - YAML parsing for canonical routing
- `redis.asyncio` (optional) - Redis probing
- `neo4j` (optional) - Neo4j probing
- `asyncpg` (optional) - PostgreSQL probing
- `jq` - JSON parsing in bash hook

Install missing dependencies:

```bash
pip install aiohttp pyyaml redis neo4j asyncpg
```
