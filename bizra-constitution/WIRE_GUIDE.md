# BIZRA Genesis Engine → Node0 Integration Guide
# ════════════════════════════════════════════════
#
# This guide closes the gap identified in the Phase 2 diagnostic:
#   "Genesis Engine v5 deployed as standalone constitutional package,
#    bridged into core/ but not yet wired into the live Node0 mission pipeline."
#
# Time to complete: ~30 minutes
# Risk: Zero — adapter pattern with automatic fallback
# Rollback: Set BIZRA_GENESIS_WIRE=false

## Step 1: Copy Genesis Engine into Node0 workspace

```bash
# From C:\BIZRA-DATA-LAKE (your workspace root):
mkdir -p bizra-constitution
# Extract bizra-node0-v6.zip contents into bizra-constitution/
```

## Step 2: Install dependencies

```bash
pip install pynacl  # Real Ed25519 (optional — HMAC fallback works)
# Ollama should already be installed and running
```

## Step 3: Wire into MissionOrchestrator

In `core/mission_orchestrator.py`, add at the top:

```python
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "bizra-constitution"))
from node0_wire import wire_genesis_engine
```

In your orchestrator's `__init__`:

```python
# After existing initialization
self.genesis_wire = wire_genesis_engine(
    data_dir=Path("sovereign_state/genesis"),
    ollama_url="http://localhost:11434",
)
```

In your orchestrator's mission handling method:

```python
def handle_mission(self, user_input: str) -> dict:
    # Try constitutional pipeline first
    if self.genesis_wire:
        result = self.genesis_wire.execute(user_input)
        if result:
            # Publish to event bus
            payload = result.to_event_bus_payload()
            self.event_bus.publish("mission_complete", payload)
            return payload

    # Fallback: existing pipeline
    return self._legacy_execute(user_input)
```

## Step 4: Wire into Node0 status

In your status endpoint, add genesis health:

```python
def get_status(self) -> dict:
    status = self._existing_status()
    if self.genesis_wire:
        status["genesis_engine"] = self.genesis_wire.health()
    return status
```

## Step 5: Verify

```bash
# Run the test suite
cd bizra-constitution
python -m pytest tests/ generated/ -q

# Expected: 328 passed, 4 skipped

# Test the wire specifically
python -c "
from node0_wire import wire_genesis_engine
wire = wire_genesis_engine()
result = wire.execute('Hello from Node0')
print(f'Output: {result.output[:80]}...')
print(f'Ihsan:  {result.ihsan_composite:.4f}')
print(f'Signed: {result.signed}')
print(f'Tier:   {result.tier}')
print(f'Node:   {result.node_id[:16]}...')
payload = result.to_event_bus_payload()
print(f'Event:  {payload[\"type\"]}')
"
```

## Step 6: Cleanup blockers from diagnostic

```bash
# 1. Remove stale PID file
rm sovereign_state/proactive.pid

# 2. Create log directory
mkdir -p logs/proactive
```

## Architecture after wiring

```
User Input
  ↓
MissionOrchestrator
  ↓
GenesisWire.execute()          ← NEW: adapter
  ↓
ProductionPipeline             ← Genesis Engine v5
  ├── Identity (Ed25519)
  ├── HHMM Router (classify)
  ├── Reflex Cache (check)
  ├── Ollama Provider (infer)
  ├── PAT Pipeline (7 agents)
  ├── Ihsan Gate (6-dim)
  ├── SNR Measurement
  └── Evidence Receipt (signed)
  ↓
WireResult
  ↓
.to_event_bus_payload()        ← Format for Rust bus
  ↓
Event Bus (12 subscribers)     ← Existing infrastructure
```

## Environment Variables

- `BIZRA_GENESIS_WIRE=false` — Disable genesis wiring (instant rollback)
- `BIZRA_CONSTITUTION_PATH` — Path to constitution.toml
- `OLLAMA_URL` — Ollama server (default: http://localhost:11434)

## File inventory (v6)

```
bizra-constitution/
├── constitution.toml           (462 lines, §1-§13)
├── bizra_constitution.py       (697 lines, parser)
├── generate_from_constitution.py (419 lines, codegen)
├── ihsan_gate.py               (463 lines, 6-dim gate)
├── snr.py                      (217 lines, SNR)
├── evidence_receipt.py         (280 lines, hash-chained)
├── reflex_cache.py             (432 lines, O(1))
├── hhmm_router.py              (450 lines, 4-tier)
├── mission_pipeline.py         (526 lines, PAT)
├── identity_genesis.py         (348 lines, Ed25519)
├── ollama_provider.py          (486 lines, circuit breaker)
├── production_pipeline.py      (221 lines, signed evidence)
├── genesis_engine.py           (297 lines, heartbeat)
├── node0_server.py             (419 lines, FastAPI)
├── node0_wire.py               (301 lines, integration adapter)  ← NEW
├── generated/
│   ├── generated_constants.py  (130 lines, 67 constants)
│   └── test_constitutional_conformance.py (215 lines, 35 tests)
├── tests/
│   ├── test_evidence_receipt.py     (15 tests)
│   ├── test_hhmm_router.py          (30 tests)
│   ├── test_ihsan_gate_and_snr.py   (43 tests)
│   ├── test_mission_pipeline.py     (35 tests)
│   ├── test_reflex_cache.py         (29 tests)
│   ├── test_identity_genesis.py     (35 tests)
│   ├── test_ollama_provider.py      (23 tests)
│   ├── test_production_pipeline.py  (20 tests)
│   ├── test_node0_server.py         (29 tests)
│   └── test_node0_wire.py           (29 tests)  ← NEW
├── verify_all.py               (153 lines)
├── poi.proto                   (193 lines)
├── MIGRATION.md                (integration guide)
└── WIRE_GUIDE.md               (this file)
```

## Test counts

| Suite | Tests | Status |
|-------|-------|--------|
| Constitutional conformance | 35 | ✅ |
| Evidence receipt | 15 | ✅ |
| HHMM router | 30 | ✅ |
| Ihsan gate + SNR | 43 | ✅ |
| Mission pipeline | 35 | ✅ |
| Reflex cache | 29 | ✅ |
| Identity genesis | 35 (4 skip) | ✅ |
| Ollama provider | 23 | ✅ |
| Production pipeline | 20 | ✅ |
| Node0 server | 29 | ✅ |
| Node0 wire | 29 | ✅ |
| **Total** | **328** | **✅** |
