# Phase 57.02: Port Architecture

## The Port Collision Problem

Three components currently target port 9742:

1. `core/bridges/desktop_bridge.py` — Python TCP server (binds 9742)
2. `filedfs/ahk_bridge.ahk` — AHK TCP server (binds 9742)
3. `bin/bizra_bridge.ahk` — AHK client (connects to 9742)

Only one server can bind a port. We need a clear ownership model.

## Resolution: Two-Server Architecture

```
                    ┌─────────────┐
                    │  AHK Client │
                    │ (no port)   │
                    └──────┬──────┘
                           │ TCP connect
                           ▼
                    ┌─────────────────────┐
Port 9742 ────────→│  Python Bridge      │ ◄── COMMAND SURFACE
                    │  desktop_bridge.py  │     (receives user intents)
                    └──────┬──────────────┘
                           │ TCP connect (when desktop action needed)
                           ▼
                    ┌─────────────────────┐
Port 9743 ────────→│  AHK HDA Server     │ ◄── ACTUATOR
                    │  ahk_bridge.ahk     │     (executes desktop actions)
                    └─────────────────────┘
```

### Port Assignments

| Port | Owner | Protocol | Direction |
|---|---|---|---|
| 9742 | `desktop_bridge.py` (Python) | JSON-RPC over TCP | AHK Client → Python |
| 9743 | `ahk_bridge.ahk` (AHK HDA) | JSON-RPC over TCP | Python → AHK Server |
| 1234 | LM Studio | HTTP REST | Python → LM Studio |
| 11434 | Ollama | HTTP REST | Python → Ollama |
| 6379 | Redis (bizra) | Redis protocol | Python → Redis |
| 6380 | Redis (synapse) | Redis protocol | Python → Redis |

### Changes Required

1. **`filedfs/ahk_bridge.ahk`**: Change bind port from `9742` to `9743`
   ```autohotkey
   ; OLD
   BizraBridge.Port := 9742
   ; NEW
   BizraBridge.Port := 9743
   ```

2. **`core/bridges/desktop_bridge.py`**: Add HDA client that connects to `9743`
   ```python
   # When desktop action needed, Python connects to AHK HDA on 9743
   HDA_PORT = int(os.environ.get("BIZRA_HDA_PORT", "9743"))
   ```

3. **`bin/bizra_bridge.ahk`**: No change — already connects to 9742 (Python)

### Connection Flow for a Mission

```
Step 1: User presses Win+Shift+B
Step 2: AHK Client → TCP 9742 → Python Bridge (execute_mission)
Step 3: Python Bridge → MissionOrchestrator.run()
Step 4: MissionOrchestrator → ChannelDispatcher.decompose()
Step 5a: BROWSER channel → BrowserMCPClient.research() (in-process)
Step 5b: DESKTOP channel → TCP 9743 → AHK HDA (file_open, type_text)
Step 6: Results merge → Synthesis → Gate → Evidence → Return
Step 7: Python Bridge → TCP 9742 → AHK Client (response)
Step 8: AHK Client → Toast notification
```

### Security: Auth Token Propagation

Both TCP connections use the same auth model:
- `BIZRA_BRIDGE_TOKEN` env var (HMAC token)
- Timestamp header (120s skew tolerance)
- Nonce replay protection

The Python bridge generates a fresh nonce for its outbound call to the HDA,
preventing replay attacks on the desktop actuator.

### Env Vars

```bash
# .env additions for Phase 57
BIZRA_BRIDGE_PORT=9742     # Python bridge listens here
BIZRA_HDA_PORT=9743        # AHK HDA server listens here
BIZRA_BRIDGE_TOKEN=<hmac>  # Shared auth token
```
