# BIZRA TUI Command Reference

The sovereign terminal (`scripts/bizra`) is the primary interface to BIZRA.

## Quick Start

```bash
# Interactive mode
./scripts/bizra

# Direct command
./scripts/bizra mission "What is BIZRA?"
```

## Commands

### mission / m

Run a mission through the full 9-stage pipeline.

```bash
bizra mission "your task here"
bizra m "explain the Ihsan principle"
```

**Output:** LLM response + receipt card (model, Ihsan score, BLAKE3 receipt, SEED earned, duration) + live stage streaming.

**Pipeline:** FAISS → Amplify → Inference → Skill → SEED → Memory → EventBus → Notify → Watcher

### organize / o

Smart file management with P4 Guardian approval.

```bash
bizra organize ~/Downloads
bizra o ~/Desktop
```

**Flow:** Scan → Generate plan → Show preview → Await approval → Execute via MissionExecutor (skill=file_management) → Receipt + SEED

### browse / web

Search and extract web content.

```bash
bizra browse "latest AI safety research"
bizra web "blockchain consensus algorithms"
```

### agents / a

Display the 12-agent sovereign team (7 PAT + 5 SAT).

```bash
bizra agents
```

### wallet / w

SEED balance, tier, streak, Ihsan trend, reflex compilation status.

```bash
bizra wallet
```

### status / s

System health: kernel, Ollama, binary, state directory.

```bash
bizra status
```

### briefing / b

Morning briefing with FAISS-powered context from your sovereign data.

```bash
bizra briefing
```

### ghost / g

Ghost Panel — proactive agent intelligence. Filesystem watcher + git status + FAISS suggestions + time-aware context.

```bash
bizra ghost
```

### teach / t

Teach your agents a new fact.

```bash
bizra teach "my name is Mumo"
bizra t "BIZRA was founded in Ramadan 2023"
```

### receipts / r

View the BLAKE3 + Ed25519 receipt chain.

```bash
bizra receipts
```

### scan

Home scan — hardware inventory + sovereign data catalog.

```bash
bizra scan
```

### skills

List available sovereign skills.

```bash
bizra skills
```

### desktop / dx

Desktop nervous system status (AHK bridge, hotkeys).

```bash
bizra desktop
bizra desktop-exec "open browser"
```

### start

Start the sovereign stack (kernel daemon + desktop bridge + federation if enabled).

```bash
bizra start
```

### stop

Stop the kernel daemon.

```bash
bizra stop
```

### proactive

Run pre-approved proactive actions (e.g., auto-organize Downloads if > 30 files).

```bash
bizra proactive
```

### frontend

Launch the React frontend connected to the kernel.

```bash
bizra frontend
```

### install-skill

Install a sovereign skill from URL or path.

```bash
bizra install-skill https://example.com/skill.py
bizra install-skill /path/to/local/skill.py
```

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `BIZRA_MODEL` | `qwen2.5:3b` | Ollama model for inference |
| `BIZRA_FEDERATION_ENABLED` | `0` | Enable federation gossip |
| `BIZRA_GOSSIP_PORT` | `9750` | Federation gossip port |
| `BIZRA_SEED_NODES` | (none) | Comma-separated seed node addresses |
| `BIZRA_AUTOPOIESIS_ENABLED` | `false` | Enable governed self-improvement |
| `BIZRA_AUTOPOIESIS_CYCLE_SECONDS` | `60` | Self-improvement cycle interval |

## Kernel API (port 9740)

When running via `bizra start`:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/health` | GET | Liveness + version + uptime |
| `/api/knowledge` | GET | GOLD corpus stats + FAISS search |
| `/api/mission` | POST | Knowledge-enriched mission execution |
| `/api/briefing` | GET | Daily sovereign morning briefing |

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Mission failed (constitutional rejection) |
| 124 | Timeout (inference exceeded budget) |
