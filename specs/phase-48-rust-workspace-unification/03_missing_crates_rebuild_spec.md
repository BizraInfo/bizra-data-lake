# Phase 48 Spec — Part 3: Missing Crates Rebuild Specification

> Standing on Giants: Fowler (hexagonal architecture) · Besta (GoT reasoning) · Maturana (autopoiesis)

## Context

A prior Claude.ai session built 4 Rust crates. Only 2 were persisted to the repo:
- `bizra-hooks` (3,147 lines, 44 tests) — persisted
- `bizra-memory` (3,025 lines, 41 tests) — persisted
- `bizra-agent` (~2,000 lines) — **lost** (sandbox-only)
- `bizra-node` (~2,000 lines) — **lost** (sandbox-only)

This spec defines what `bizra-agent` and `bizra-node` must contain for rebuild.

---

## Crate 1: `bizra-agent` — The Agent Runtime

### Purpose
Event-driven agent runtime that processes hooks, manages tool dispatch, and coordinates with `bizra-memory` for contextual understanding.

### Location
`native/bizra-agent/`

### Dependencies
```toml
[dependencies]
bizra-hooks = { path = "../bizra-hooks" }
bizra-memory = { path = "../bizra-memory" }
serde = { workspace = true }
serde_json = { workspace = true }
tokio = { workspace = true }
```

### Module Structure

```
bizra-agent/src/
├── lib.rs              # Re-exports, BizraAgent facade
├── runtime.rs          # Event loop: poll hooks → dispatch → report
├── capability.rs       # Capability matching: tool requirements vs agent abilities
├── dispatch.rs         # Tool dispatch: route actions to registered tool handlers
├── context.rs          # Context builder: pull relevant memory for current task
└── health.rs           # Agent health: heartbeat, resource usage, error budget
```

### Key Types

```
PSEUDOCODE — bizra-agent types

struct BizraAgent {
    id: AgentId,
    system: BizraSystem,          // from bizra-hooks
    memory: BizraMemory,          // from bizra-memory
    capabilities: Vec<Capability>,
    tools: ToolRegistry,
    state: AgentState,
    config: AgentConfig,
}

enum AgentState {
    Idle,
    Processing(TaskId),
    WaitingForTool(ToolId),
    Reporting,
    ShuttingDown,
}

struct Capability {
    name: String,
    version: SemVer,
    requirements: Vec<Requirement>,
}

struct ToolRegistry {
    tools: HashMap<ToolId, Box<dyn ToolHandler>>,
}

trait ToolHandler: Send + Sync {
    fn name(&self) -> &str;
    fn capabilities(&self) -> &[Capability];
    fn execute(&self, input: ToolInput) -> Result<ToolOutput>;
}
```

### Event Loop Pseudocode

```
FUNCTION agent_run(agent, shutdown_signal):
    agent.system.register(agent.component_id)
    agent.system.subscribe("task.*")
    agent.system.subscribe("tool.*")

    LOOP:
        IF shutdown_signal.is_set():
            agent.state = ShuttingDown
            BREAK

        events = agent.system.flush(timeout=100ms)
        FOR event IN events:
            MATCH event.topic:
                "task.assigned" =>
                    task = deserialize(event.payload)
                    IF agent.can_handle(task):
                        context = agent.memory.build_context(task)
                        result = agent.dispatch(task, context)
                        agent.report(task, result)
                    ELSE:
                        agent.decline(task)
                "tool.result" =>
                    agent.handle_tool_result(event)
                _ =>
                    // Forward to memory for observation
                    agent.memory.observe(event)

    agent.system.unregister(agent.component_id)
```

### TDD Anchors

```rust
#[test]
fn agent_registers_with_system() { /* component shows up in registry */ }

#[test]
fn agent_dispatches_matching_task() { /* capability match → tool handler called */ }

#[test]
fn agent_declines_unmatched_task() { /* no capability → decline event emitted */ }

#[test]
fn agent_builds_context_from_memory() { /* memory queried before dispatch */ }

#[test]
fn agent_reports_result() { /* result event emitted after tool execution */ }

#[test]
fn agent_graceful_shutdown() { /* shutdown signal → clean unregister */ }

#[test]
fn agent_error_budget_tracks_failures() { /* consecutive failures → health degraded */ }
```

### Estimated: ~1,500-2,000 lines, ~25 tests

---

## Crate 2: `bizra-node` — The Node Binary

### Purpose
The desktop-facing binary that ties together hooks, memory, agent, and FATE into a running Node0 process. Optionally includes Tauri for desktop UI.

### Location
`native/bizra-node/`

### Dependencies
```toml
[dependencies]
bizra-hooks = { path = "../bizra-hooks" }
bizra-memory = { path = "../bizra-memory" }
bizra-agent = { path = "../bizra-agent" }
fate-binding = { path = "../fate-binding" }
serde = { workspace = true }
serde_json = { workspace = true }
tokio = { workspace = true }
```

### Module Structure

```
bizra-node/src/
├── main.rs             # Binary entry point: parse args, init system, run
├── lib.rs              # Node0 facade for library consumers
├── bootstrap.rs        # System bootstrap: hooks → memory → agent → FATE
├── continuity.rs       # Session persistence: save/restore state across restarts
├── bridge.rs           # Python bridge: expose Node0 state to Python core/ via FFI
└── config.rs           # Configuration: CLI args, env vars, config file
```

### Bootstrap Sequence Pseudocode

```
FUNCTION bootstrap_node0(config):
    // Phase 1: Hooks (nervous system)
    system = BizraSystem::new()
    ihsan_gate = IhsanGate::new(config.ihsan_floor)
    system.attach_gate(ihsan_gate)

    // Phase 2: Memory (cognitive layer)
    memory_config = PipelineConfig::from_env()
    memory = BizraMemory::with_config(memory_config)
    system.register_component(memory.component_id())

    // Phase 3: Agent (executive function)
    agent = BizraAgent::new(system.clone(), memory.clone())
    agent.register_tools(builtin_tools())

    // Phase 4: FATE (constitutional gate)
    fate_chain = FateGateChain::default()

    // Phase 5: Continuity (persistence)
    IF config.state_file.exists():
        restore_state(memory, config.state_file)

    // Phase 6: Run
    node = Node0 { system, memory, agent, fate_chain, config }
    node.run_until_shutdown()

    // Phase 7: Persist
    save_state(memory, config.state_file)
```

### Continuity Pseudocode

```
FUNCTION save_state(memory, path):
    snapshot = {
        "version": "1.0.0",
        "timestamp": now(),
        "health": memory.health(),
        "knowledge": memory.knowledge_summary(),
        "atoms": memory.export_atoms(),
        "profile": memory.who_is_the_user(),
    }
    write_json(path, snapshot)
    // FATE-sign the snapshot for tamper detection
    receipt = fate_sign(snapshot)
    write_json(path + ".receipt", receipt)

FUNCTION restore_state(memory, path):
    snapshot = read_json(path)
    receipt = read_json(path + ".receipt")
    IF NOT fate_verify(snapshot, receipt):
        WARN "State file tampered — starting fresh"
        RETURN
    memory.import_atoms(snapshot.atoms)
    memory.restore_profile(snapshot.profile)
```

### TDD Anchors

```rust
#[test]
fn bootstrap_sequence_completes() { /* all 6 phases without error */ }

#[test]
fn state_persistence_roundtrip() { /* save → restore → knowledge intact */ }

#[test]
fn tampered_state_rejected() { /* modified state file → FATE verify fails */ }

#[test]
fn config_from_env() { /* env vars override defaults */ }

#[test]
fn graceful_shutdown_persists() { /* SIGTERM → state saved before exit */ }

#[test]
fn python_bridge_exposes_health() { /* FFI call returns valid health struct */ }
```

### Estimated: ~1,500-2,000 lines, ~20 tests

---

## Build Order

```
1. bizra-agent (depends on: bizra-hooks, bizra-memory)
2. bizra-node  (depends on: bizra-hooks, bizra-memory, bizra-agent, fate-binding)
```

After both are built, update `native/Cargo.toml`:

```toml
members = [
    "bizra-hooks",
    "bizra-memory",
    "bizra-agent",    # NEW
    "bizra-node",     # NEW
    "fate-binding",
    "iceoryx-bridge",
]
```

---

## Acceptance Criteria

1. `cargo test --workspace` in `native/` passes with all 6 crates
2. `bizra-node` binary boots, processes 3 conversation turns, persists state, restores it
3. `the_four_word_test` passes through the full stack (node → agent → memory)
4. FATE gate chain rejects tampered persistence files
5. No new dependencies beyond what's already in the workspace
6. CI pipeline (`native-ci.yml`) passes without modification (it already covers `--workspace`)
