# Phase 87 — The 9 Wires: Specification Manifest

## Standing on Giants
- **Hebb (1949)**: Neurons that fire together wire together → subscriber wiring
- **Lamport (1978)**: Distributed event ordering → EventBus monotonic IDs
- **Shannon (1948)**: Channel capacity is finite → context budget
- **Al-Ghazali**: إحسان as non-negotiable → fail-closed halt
- **Satoshi (2008)**: Hash-linked chain → receipt integrity
- **General Magic (1990s)**: Mobile agent execution → TeleScript

## The Gap

After reading every crate, here is the precise surgical diagnosis:

```
DONE  Wire 1: DEMA stdin → Node.handle_command()         ← node.rs:156, run() loop
DONE  Wire 2: Node → AgentRuntime.receive()               ← handler.rs RECEIVE verb
═══>  Wire 3: AgentRuntime → InferenceGateway.generate()  ← THE GAP (sync→async bridge)
DONE  Wire 4: InferenceGateway → Ollama/LM Studio         ← bizra-inference backends
DONE  Wire 5: Response → Guardian/Ihsān scoring            ← orchestrator.rs guardian_approved
DONE  Wire 6: Guardian → Constitutional halt               ← subscriber #9 HookResult::Halt
DONE  Wire 7: Pass → Mission state machine + receipt       ← bizra-mission 14-state FSM
STUB  Wire 8: Receipt → HHMM memory reinforcement          ← subscribers #1,#2 are HookResult::Continue stubs
STUB  Wire 9: Memory → Autopoiesis feedback                ← subscriber #7 is stub
```

## Reality: Not 9 Missing Wires — 1 Gap + 2 Stubs

| Work Item | Crate | File | LOC Estimate |
|-----------|-------|------|-------------|
| Wire 3: sync→async inference bridge | bizra-agent | runtime.rs | ~80 |
| Wire 8: flesh subscriber #1 (reinforce) + #2 (promote) | bizra-hooks | subscribers.rs | ~60 |
| Wire 9: flesh subscriber #7 (session compile) | bizra-hooks | subscribers.rs | ~40 |
| Integration test: 10-mission chain | bizra-node | tests/ | ~120 |
| **TOTAL** | | | **~300 LOC** |

## Execution Order

```
Phase 87-A: Wire 3 (inference bridge)         → "the node speaks"
Phase 87-B: Wire 8 (memory reinforcement)     → "the node learns"
Phase 87-C: Wire 9 (session compile)          → "the node improves"
Phase 87-D: Integration test (10 missions)    → "Block 0 proof"
```

## Test Matrix (6 SAPE Probes)

| # | Probe | Wire | Pass Condition |
|---|-------|------|---------------|
| 1 | Real inference through Guardian | 3+5 | Ollama returns response, Ihsān ≥ 0.95 |
| 2 | SAT rejection of low-quality | 5+6 | Response with Ihsān < 0.90 → Halt |
| 3 | Receipt chain across 10 missions | 7 | 10 receipts, prev_hash chain valid, Ed25519 verified |
| 4 | HHMM memory across restart | 8 | Teach 5 facts → shutdown → restart → knows_me > 0 |
| 5 | File Management through Guardian | 3+5+7 | Organize task → receipt with file paths |
| 6 | Browser Control through Guardian | 3+5+7 | Web extraction → receipt with content |

## Files in This Spec

| File | Content |
|------|---------|
| `00_wire_manifest.md` | This file — the map |
| `01_wire3_inference_bridge.md` | Sync→async LLM bridge pseudocode |
| `02_wire8_memory_reinforcement.md` | Subscriber #1 + #2 flesh-out |
| `03_wire9_session_compile.md` | Subscriber #7 flesh-out |
| `04_integration_test.md` | 10-mission chain test spec |
