# Phase 55: Unified Message Backbone (UMB)

## Overview

The Unified Message Backbone (`bizra-umb`) is the central nervous system of the BIZRA
ecosystem — a single Rust crate that implements Action Bus + Event Bus + Hook System +
Typestate State Machine + Saga Coordinator as one cohesive infrastructure layer.

## Theoretical Foundation

Five independent discoveries converge on the same insight: **the connective tissue
between components is more important than the components themselves.**

| Theorist | Contribution | UMB Application |
|----------|-------------|-----------------|
| Carl Hewitt (1973) | Actor Model — everything is an actor, communication is the primitive | 26 components = 26 actors, messages = the system |
| Joe Armstrong | Erlang/OTP — "Let it crash", supervision trees | Circuit breakers, isolated failure domains |
| Greg Young | Event Sourcing — store events, derive state | BlockGraph as source of truth, replay capability |
| Hector Garcia-Molina | Saga Pattern — local transactions with compensating actions | Multi-agent tasks as distributed sagas |
| Ilya Prigogine | Dissipative Structures — throughput prevents entropy | UMB = information throughput channel |

## Architecture Summary

```
bizra-umb/
├── src/
│   ├── lib.rs              # Public API surface
│   ├── envelope.rs         # Envelope<T>, MessageId, TraceId, ActorId, Priority
│   ├── action.rs           # Action enum (mpsc dispatch, imperative commands)
│   ├── event.rs            # Event enum (broadcast, declarative notifications)
│   ├── action_bus.rs       # mpsc-based action routing (one handler per action)
│   ├── event_bus.rs        # broadcast-based event fanout (many subscribers)
│   ├── hook.rs             # HookPoint, HookResult, Hook trait, HookRegistry
│   ├── hooks/              # Built-in hook implementations
│   │   ├── mod.rs
│   │   ├── budget.rs       # BudgetEnforcementHook (Component 20)
│   │   ├── auth.rs         # AuthorizationHook (Component 24)
│   │   ├── tracing.rs      # TracingHook (Component 25)
│   │   ├── constitutional.rs # ConstitutionalPreCheckHook (Component 23/ASPH)
│   │   ├── attestation.rs  # BlockGraphAttestationHook (Component 22)
│   │   ├── circuit.rs      # CircuitBreakerHook (Component 26)
│   │   └── ihsan.rs        # IhsanScoringHook (quality enforcement)
│   ├── state.rs            # Typestate state machine (phantom types)
│   ├── saga.rs             # SagaCoordinator, SagaStep, compensating actions
│   └── backbone.rs         # UnifiedMessageBackbone (entry point)
├── tests/
│   ├── envelope_tests.rs
│   ├── bus_tests.rs
│   ├── hook_tests.rs
│   ├── state_tests.rs
│   ├── saga_tests.rs
│   └── integration_tests.rs
├── Cargo.toml
└── README.md
```

## Why N Connections Instead of N²

Without UMB: 26 components × 25 potential connections = **650 integration points**.
With UMB: 26 components × 1 backbone connection = **26 integration points**.

Every component connects to ONE thing — the backbone. The backbone routes, intercepts,
validates, traces, and attests.

## Implementation Timeline

| Week | Deliverable | Spec File |
|------|-------------|-----------|
| 1 | Core types, Envelope, Action/Event enums | `phase_55_01_core_types.md` |
| 2 | Action Bus (mpsc) + Event Bus (broadcast) | `phase_55_02_bus_system.md` |
| 3 | Hook registry + 7 built-in hooks | `phase_55_03_hook_system.md` |
| 4 | Typestate state machine (phantom types) | `phase_55_04_state_machine.md` |
| 5a | Saga coordinator (multi-agent transactions) | `phase_55_05_saga.md` |
| 5b | Unified backbone (full pipeline wiring) | `phase_55_06_backbone.md` |
| -- | Golden gems: 6 architectural refinements | `phase_55_07_golden_gems.md` |

**Week 5 = Phase 7 complete.** First heartbeat through the UMB. Genesis block attested.

## Dependencies

```toml
[dependencies]
tokio = { version = "1", features = ["full"] }
ed25519-dalek = "2"
uuid = { version = "1", features = ["v4"] }
async-trait = "0.1"
blake3 = "1"
serde = { version = "1", features = ["derive"] }
tracing = "0.1"

[dev-dependencies]
tokio-test = "0.4"
```

## Integration Points

The UMB connects to existing BIZRA components:

| Component | Connection |
|-----------|-----------|
| `bizra-hooks` (existing) | Event Bus subscriber taxonomy migrates to UMB Event enum |
| `bizra-core` | Constitutional gates become Saga steps |
| `bizra-agent` | OmniKernel dispatches Actions, receives Events |
| `bizra-python` (PyO3) | PyUMB bridge for Python-side event emission |
| `core/sovereign/event_bus.py` | Python Event Bus becomes thin client to UMB |
| `core/proof_engine/` | BlockGraph attestation via AttestationHook |

## Success Criteria

- [ ] All 26 component interactions flow through UMB
- [ ] Invalid state transitions fail at compile time (not runtime)
- [ ] Circuit breakers prevent cascade failures
- [ ] Full distributed tracing for every request
- [ ] Saga rollback works for multi-agent task failures
- [ ] Ihsan score computed at every lifecycle point via hooks
- [ ] BlockGraph attestation for completed requests
- [ ] Zero `unsafe` blocks in UMB crate
