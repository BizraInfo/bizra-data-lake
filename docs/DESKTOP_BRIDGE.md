# Desktop Bridge — Sovereign Command Surface

> **Protocol:** Newline-delimited JSON-RPC 2.0 over TCP
> **Endpoint:** `127.0.0.1:9742` (localhost only)
> **Commands:** `ping`, `status`, `sovereign_query`, `invoke_skill`, `list_skills`, `get_receipt`, `actuator_execute`, `get_context`
> **Version:** 2.2.0

## Overview

The Desktop Bridge connects Windows desktop automation (AutoHotkey v2) to the BIZRA Sovereign Runtime running in WSL2. It exposes 8 JSON-RPC methods over a raw TCP socket — no HTTP, no WebSocket, zero new Python dependencies.

```
AutoHotkey v2 (Windows)               Python asyncio TCP (WSL2)
┌──────────────────────┐              ┌──────────────────────────────┐
│  Win+B    → ping     │──TCP:9742──→│  DesktopBridge               │
│  Ctrl+B,Q → query    │  JSON-RPC   │  ├─ HookPhase.DESKTOP_INVOKE │
│  Ctrl+B,S → status   │  \n delim   │  ├─ FATE Gate + Ihsan Check  │
└──────────────────────┘              │  ├─ SkillRouter (43 skills)  │
                                      │  ├─ RDVE Research Engine     │
                                      │  ├─ Rust GateChain (PyO3)    │
                                      │  ├─ InferenceGateway         │
                                      │  ├─ BridgeReceiptEngine      │
                                      │  └─ BLAKE3 content hashing   │
                                      └──────────────────────────────┘
```

### Why TCP, Not WebSocket or Named Pipe

WSL2 runs in a Hyper-V VM. Windows Named Pipes (`\\.\pipe\*`) are not accessible from WSL2 Python. Raw TCP on `127.0.0.1` is the lowest-overhead cross-boundary IPC:

- No HTTP upgrade handshake (unlike WebSocket)
- No binary framing
- Zero new Python dependencies (`asyncio` TCP is stdlib)
- Bind to `127.0.0.1` only — equivalent security to Named Pipes

---

## Quick Start

### 1. Start the Bridge (WSL2)

```bash
# Standalone
PYTHONPATH=/mnt/c/BIZRA-DATA-LAKE python -m core.bridges.desktop_bridge

# Via Sovereign CLI
PYTHONPATH=/mnt/c/BIZRA-DATA-LAKE python -m core.sovereign bridge start

# Via Sovereign Launcher (starts with full stack)
PYTHONPATH=/mnt/c/BIZRA-DATA-LAKE python -m core.sovereign.launch
```

### 2. Test from Python (no AHK needed)

```python
import os, socket, json, time, uuid
s = socket.socket(); s.connect(("127.0.0.1", 9742))
token = os.environ["BIZRA_BRIDGE_TOKEN"]
payload = {
  "jsonrpc": "2.0",
  "method": "ping",
  "id": 1,
  "headers": {
    "X-BIZRA-TOKEN": token,
    "X-BIZRA-TS": int(time.time() * 1000),
    "X-BIZRA-NONCE": uuid.uuid4().hex
  }
}
s.sendall(json.dumps(payload).encode() + b"\n")
print(s.recv(4096).decode()); s.close()
```

### 3. Start AHK Client (Windows)

Double-click `bin/bizra_bridge.ahk`. Requires [AutoHotkey v2](https://www.autohotkey.com/).

---

## JSON-RPC Protocol

Every message is a single JSON object terminated by `\n` (newline). No binary framing.

### Request Format

```json
{
  "jsonrpc": "2.0",
  "method": "ping",
  "id": 1,
  "headers": {
    "X-BIZRA-TOKEN": "<env:BIZRA_BRIDGE_TOKEN>",
    "X-BIZRA-TS": 1739420000000,
    "X-BIZRA-NONCE": "5d2a4a1f0d8b4f0fb2d9ef1de8c4aa93"
  }
}
```

All methods require `headers`. Missing/invalid auth is fail-closed.

### Response Format (success)

```json
{"jsonrpc": "2.0", "result": {"status": "alive", "uptime_s": 42.5}, "id": 1}
```

### Response Format (error)

```json
{"jsonrpc": "2.0", "error": {"code": -32601, "message": "Method not found: foo"}, "id": 1}
```

### Error Codes

| Code | Meaning |
|------|---------|
| `-32700` | Parse error (malformed JSON) |
| `-32600` | Invalid request (missing `jsonrpc: "2.0"` or `method`) |
| `-32601` | Method not found |
| `-32603` | Internal error (handler exception) |
| `-32000` | Rate limit exceeded (max 20 req/s) |
| `-32001` | Auth headers missing/invalid |
| `-32002` | Auth token invalid |
| `-32003` | Auth timestamp invalid/stale |
| `-32004` | Auth nonce replay detected |

---

## Commands

All command requests require the `headers` auth block shown in the JSON-RPC request format.

### `ping`

Liveness check. Returns immediately.

**Request:**
```json
{"jsonrpc": "2.0", "method": "ping", "id": 1}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "status": "alive",
    "uptime_s": 127.45,
    "request_count": 42,
    "rust_available": true
  },
  "id": 1
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Always `"alive"` |
| `uptime_s` | float | Seconds since bridge started |
| `request_count` | int | Total requests processed |
| `rust_available` | bool | Whether Rust PyO3 bindings are compiled |

### `status`

Full sovereign node health snapshot.

**Request:**
```json
{"jsonrpc": "2.0", "method": "status", "id": 1}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "node": "node0",
    "origin": {
      "designation": "node0",
      "genesis_node": true,
      "genesis_block": true,
      "block_id": "block0",
      "home_base_device": true,
      "node_id": "node0_ce5af35c848ce889",
      "node_name": "MoMo (\u0645\u062d\u0645\u062f)",
      "authority_source": "genesis_files",
      "hash_validated": true
    },
    "bridge_uptime_s": 300.12,
    "request_count": 85,
    "fate_gate": {
      "passed": true,
      "overall": 0.98,
      "fidelity": 1.0,
      "accountability": 0.95,
      "transparency": 1.0,
      "ethics": 0.97
    },
    "inference": {
      "available": true,
      "status": "ready"
    },
    "rust": {
      "available": true,
      "constitution_version": "1.0.0",
      "ihsan_threshold": 0.95,
      "snr_threshold": 0.85
    }
  },
  "id": 1
}
```

| Field | Type | Description |
|-------|------|-------------|
| `node` | string | `"node0"` in Node0 role, otherwise `"node"` |
| `origin` | object | Canonical origin projection (`designation`, `genesis_node`, `genesis_block`, `authority_source`, `hash_validated`) |
| `fate_gate` | object | FATE gate validation result |
| `inference` | object | InferenceGateway health |
| `rust` | object | Rust PyO3 availability and constitutional thresholds |

### `sovereign_query`

Route a query through the FATE gate, Rust GateChain, and InferenceGateway.

**Request:**
```json
{"jsonrpc": "2.0", "method": "sovereign_query", "params": {"query": "What is sovereignty?"}, "id": 1}
```

**Response (success):**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "content": "Sovereignty is the full right and power of a governing body...",
    "model": "qwen2.5-coder-7b",
    "backend": "lmstudio",
    "latency_ms": 1250.5,
    "tokens_generated": 128,
    "fate": {"passed": true, "overall": 0.98},
    "rust_gates": {
      "passed": true,
      "engine": "rust",
      "gates": {
        "schema": {"passed": true, "code": "PASS"},
        "ihsan": {"passed": true, "code": "PASS"},
        "snr": {"passed": true, "code": "PASS"}
      }
    },
    "content_hash": "a3b8f4c2e1d..."
  },
  "id": 1
}
```

**Response (gate blocked):**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "error": "Rust gate chain blocked query",
    "fate": {"passed": true, "overall": 0.97},
    "rust_gates": {
      "passed": false,
      "engine": "rust",
      "gates": {
        "schema": {"passed": true, "code": "PASS"},
        "ihsan": {"passed": false, "code": "BELOW_THRESHOLD"}
      }
    }
  },
  "id": 1
}
```

#### Gate Execution Order

1. **Python FATE Gate** — Fidelity, Accountability, Transparency, Ethics (weighted geometric mean, threshold 0.95)
2. **Rust GateChain** — Schema, Ihsan, SNR (fail-fast, PyO3 bindings) — skipped if Rust not compiled
3. **InferenceGateway** — LM Studio / Ollama / Cloud fallback
4. **BLAKE3 Hash** — `content_hash` field via Rust `domain_separated_digest`

### `invoke_skill`

Route a request to any registered skill via the SkillRouter. Includes RDVE research engine and all 42+ SKILL.md-defined skills.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "invoke_skill",
  "params": {
    "skill": "rdve_research",
    "inputs": {"operation": "statistics"}
  },
  "id": 1
}
```

**Response (success):**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "skill": "rdve_research",
    "output": {
      "operation": "statistics",
      "rdve": {"version": "1.0.0", "skill_name": "rdve_research", "invocation_count": 5},
      "orchestrator": {"state_dir": "/app/state", "mission_count": 12},
      "mission_history": []
    },
    "receipt_id": "br-20260213-a1b2c3d4",
    "elapsed_ms": 45.2
  },
  "id": 1
}
```

**Response (skill not found):**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "error": "Skill 'nonexistent' not found in registry",
    "available_count": 43
  },
  "id": 1
}
```

| Field | Type | Description |
|-------|------|-------------|
| `skill` | string | Name of the invoked skill |
| `output` | object | Skill handler's response |
| `receipt_id` | string | PCI receipt ID (retrievable via `get_receipt`) |
| `elapsed_ms` | float | Skill execution time |

#### RDVE Operations via `invoke_skill`

The `rdve_research` skill supports 4 operations. See [SPEARPOINT.md](SPEARPOINT.md) for full details.

| Operation | Required Inputs | Description |
|-----------|----------------|-------------|
| `research_pattern` | `pattern_id` (P01-P15) | Research using a specific Sci-Reasoning thinking pattern |
| `reproduce` | `claim` (string) | Attempt to falsify a claim via AutoEvaluator |
| `improve` | (optional: `observation`, `top_k`) | Auto-optimize via AutoResearcher |
| `statistics` | (none) | RDVE metadata and orchestrator state |

**Example: Research Pattern P01 (Prior Work Extraction)**
```json
{
  "jsonrpc": "2.0",
  "method": "invoke_skill",
  "params": {
    "skill": "rdve_research",
    "inputs": {
      "operation": "research_pattern",
      "pattern_id": "P01",
      "claim_context": "optimize attention mechanism",
      "top_k": 3
    }
  },
  "id": 2
}
```

**Example: Reproduce a Claim**
```json
{
  "jsonrpc": "2.0",
  "method": "invoke_skill",
  "params": {
    "skill": "rdve_research",
    "inputs": {
      "operation": "reproduce",
      "claim": "BIZRA latency < 200ms P99 under load"
    }
  },
  "id": 3
}
```

### `list_skills`

List all registered skills from the SkillRegistry.

**Request:**
```json
{"jsonrpc": "2.0", "method": "list_skills", "id": 1}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "skills": [
      {
        "name": "rdve_research",
        "description": "Recursive Discovery & Verification Engine...",
        "status": "available",
        "agent": "rdve-researcher",
        "tags": ["research", "spearpoint", "rdve", "sci-reasoning", "got"]
      }
    ],
    "count": 43
  },
  "id": 1
}
```

**Filter by status:**
```json
{"jsonrpc": "2.0", "method": "list_skills", "params": {"filter": "available"}, "id": 1}
```

### `get_receipt`

Retrieve a PCI receipt by ID. Signed receipts are generated for all critical methods and rejected requests.

**Request:**
```json
{"jsonrpc": "2.0", "method": "get_receipt", "params": {"receipt_id": "br-20260213-a1b2c3d4"}, "id": 1}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "receipt_id": "br-20260213-a1b2c3d4",
    "status": "accepted",
    "method": "invoke_skill",
    "origin": {
      "designation": "node0",
      "genesis_node": true,
      "genesis_block": true,
      "block_id": "block0",
      "home_base_device": true,
      "authority_source": "genesis_files",
      "hash_validated": true
    },
    "origin_digest": "<blake3-hex>",
    "fate_score": 0.97,
    "snr_score": 0.95,
    "signature": "<hex>",
    "signer_pubkey": "<hex>",
    "request_receipt": {
      "receipt_id": "br-20260213-a1b2c3d5",
      "status": "accepted"
    }
  },
  "id": 1
}
```

### `actuator_execute`

Execute an AHK desktop instruction through the 3-gate pipeline: FATE Gate, Shannon Entropy Gate, and Rust GateChain. Returns a SEALED envelope if all gates pass.

**Request:**
```json
{
  "jsonrpc": "2.0",
  "method": "actuator_execute",
  "params": {
    "code": "Send('^s'); WinWait('Save Dialog'); Click(200, 150)",
    "intent": "save_file",
    "target_app": "Code.exe"
  },
  "id": 1
}
```

**Response (SEALED):**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "status": "SEALED",
    "intent": "save_file",
    "target_app": "Code.exe",
    "instruction_length": 52,
    "entropy": {
      "passed": true,
      "entropy": 4.1523,
      "normalized": 0.6428,
      "threshold": 3.5,
      "unique_chars": 28
    },
    "fate": {"passed": true, "overall": 0.97}
  },
  "id": 1
}
```

| Field | Type | Description |
|-------|------|-------------|
| `code` | string | AHK v2 instruction to execute (required) |
| `intent` | string | Human-readable intent label (optional) |
| `target_app` | string | Target application process name (optional) |

#### Gate Execution Order (Actuator)

1. **Python FATE Gate** — Ihsan >= 0.95 constitutional check
2. **Shannon Entropy Gate** — H(X) >= 3.5 bits/char to block low-signal noise
3. **Rust GateChain** — Schema, Ihsan, SNR fail-fast (if compiled)

### `get_context`

Request the UIA (UI Automation) schema describing the expected desktop context fields.

**Request:**
```json
{"jsonrpc": "2.0", "method": "get_context", "id": 1}
```

**Response:**
```json
{
  "jsonrpc": "2.0",
  "result": {
    "schema_version": "1.0",
    "expected_fields": {
      "title": "Active window title",
      "class": "Window class name",
      "process": "Process name (e.g., Code.exe)",
      "structure": "Serialized UIA tree snapshot",
      "hash": "BLAKE3 hash of structure"
    }
  },
  "id": 1
}
```

---

## Architecture

### Gate Pipeline

Every `sovereign_query` flows through: **Python FATE Gate** (Fidelity, Accountability, Transparency, Ethics) → **Rust GateChain** (Schema, Ihsan >=0.95, SNR >=0.85) → **InferenceGateway** → **BLAKE3 Hash**. Every `invoke_skill` flows through: **FATE Gate** → **Ihsan Check** → **SkillRouter** → **PCI Receipt**. Every `actuator_execute` flows through: **FATE Gate** → **Shannon Entropy Gate** (H >= 3.5 bits/char) → **Rust GateChain** → **SEALED envelope**.

### Rust Integration (PyO3)

Compile via `cd bizra-omega/bizra-python && maturin develop --release`. The bridge uses `GateChain` (fail-fast gates), `Constitution` (thresholds in `status`), and `domain_separated_digest` (BLAKE3 hashing). If Rust is not compiled, all Rust features degrade gracefully.

### Security & Rate Limiting

- **Localhost only**: Rejects non-`127.0.0.1` bind addresses
- **Rate limiter**: 20 req/s token bucket, burst 30 (error code `-32000`)
- **Auth envelope**: `X-BIZRA-TOKEN`, `X-BIZRA-TS`, `X-BIZRA-NONCE` required on every request
- **Hard startup checks**: bridge refuses startup without `BIZRA_BRIDGE_TOKEN` and `BIZRA_RECEIPT_PRIVATE_KEY_HEX`
- **Node0 fail-closed**: when `BIZRA_NODE_ROLE=node0`, startup also requires valid `sovereign_state/node0_genesis.json` + `sovereign_state/genesis_hash.txt`
- **Hook audit trail**: Every command fires `HookPhase.DESKTOP_INVOKE`
- **Signed origin-bound receipts**: Critical success/rejection paths emit signed receipts containing `origin` + `origin_digest`

---

## AutoHotkey v2 Client

**File:** `bin/bizra_bridge.ahk`

### Hotkeys

| Hotkey | Command | Action |
|--------|---------|--------|
| `Win+B` | `ping` | Tooltip: "BIZRA: alive \| Uptime: Xs" |
| `Ctrl+B`, then `Q` | `sovereign_query` | Input dialog, result in MsgBox |
| `Ctrl+B`, then `S` | `status` | Tooltip: Node0 status summary |

### Connection Behavior

- Auto-reconnect on failure with exponential backoff (1s → 2s → 4s → ... → 30s max)
- System tray icon shows connection state (green shell = connected, red = disconnected)
- Uses Winsock2 (`ws2_32.dll`) via `DllCall` — no AHK dependencies

### Requirements

- [AutoHotkey v2.0+](https://www.autohotkey.com/)
- Bridge server running in WSL2

---

## Launcher Integration

The bridge is step 6/6 in `SovereignLauncher`. Shutdown is LIFO.

```bash
python -m core.sovereign.launch                    # Full stack (bridge on 9742)
python -m core.sovereign.launch --no-bridge        # Disable desktop bridge
python -m core.sovereign.launch --bridge-port 9999 # Custom port
```

---

## Testing

```bash
# Bridge tests (22 tests — lifecycle, commands, security, rate limiter)
PYTHONPATH=/mnt/c/BIZRA-DATA-LAKE pytest tests/core/bridges/test_desktop_bridge.py -v

# Actuator layer tests (26 tests — entropy gate, actuator handlers, skill ledger)
PYTHONPATH=/mnt/c/BIZRA-DATA-LAKE pytest tests/core/bridges/test_actuator_layer.py -v

# RDVE skill tests (28 tests — operations, registration, bridge integration)
PYTHONPATH=/mnt/c/BIZRA-DATA-LAKE pytest tests/core/spearpoint/test_rdve_skill.py -v

# All bridge + RDVE + actuator tests
PYTHONPATH=/mnt/c/BIZRA-DATA-LAKE pytest tests/core/bridges/ tests/core/spearpoint/ -v
```

---

## Docker Compose

```bash
# Start bridge with full stack
docker compose up desktop-bridge

# Bridge with monitoring
docker compose --profile monitoring up -d
```

The `desktop-bridge` service in `docker-compose.yml` mounts:
- `./core:/app/core:ro` — application code
- `./data:/app/data:ro` — Sci-Reasoning patterns for RDVE
- `bridge-receipts:/app/sovereign_state/bridge_receipts` — PCI receipt storage

---

## File Map

```
core/bridges/desktop_bridge.py          # TCP server + 8 commands (~1050 lines)
core/spearpoint/rdve_skill.py           # RDVE skill handler (347 lines)
core/spearpoint/actuator_skills.py      # AHK skill ledger (123 lines)
core/skills/router.py                   # SkillRouter + FATE gate
core/skills/registry.py                 # SkillRegistry + 42 SKILL.md skills
core/sovereign/launch.py                # Launcher integration (step 6/6)
core/sovereign/__main__.py              # CLI: bridge start/ping/status
core/elite/hooks.py                     # HookPhase.DESKTOP_INVOKE
bin/bizra_bridge.ahk                    # AutoHotkey v2 client
tests/core/bridges/test_desktop_bridge.py         # Bridge tests (22)
tests/core/bridges/test_actuator_layer.py         # Actuator layer tests (26)
tests/core/spearpoint/test_rdve_skill.py          # RDVE skill tests (28)
```

---

## Standing on Giants

| Thinker | Principle Applied |
|---------|------------------|
| **Boyd** (OODA) | Fast feedback loop — prove 3 commands work before scaling |
| **Shannon** (SNR) | Increase signal before adding channel bandwidth |
| **Lamport** | Avoid new network surface unless necessary |
| **Al-Ghazali** (Ihsan) | Excellence over ego-driven expansion |
| **Deming** (PDCA) | Plan small, test, iterate |
