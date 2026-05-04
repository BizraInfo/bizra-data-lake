# Dema Node Console v0.1

**Schema:** `DEMA_NODE_CONSOLE.v0.1`
**Status:** ACTIVE
**Type:** PRODUCT_CONTRACT
**Truth label:** WIRED
**Maintainer:** Node0 — First Architect

---

Dema Node Console v0.1 is the safe operator surface for Node0 readiness. It
turns dependency state into plain language without starting a daemon, dispatching
a mission, loading a model, activating Node1, routing to external providers, or
making economic/token claims.

## Purpose

The console answers one bounded question:

```text
What is observable about Node0 right now, and what remains gated?
```

It does not answer:

```text
Has the runtime pulse fired?
Has materialization been reached?
Is Node1 active?
Is reward or token value available?
```

Those answers require receipts the console must not invent.

## Dependency panels

The v0.1 console exposes seven dependency rows:

| Panel | Requirement | Runtime authority |
| --- | --- | --- |
| Python venv | Repo virtualenv active for Python gates | Process state only |
| PyO3 bridge | `bizra.PyEventBridge` importable | Import visibility only |
| Rust Bus | Rust event bus binding visible | No subscriber wiring in status |
| Model backend | Local LM Studio reachable with loaded model | Local read-only probe |
| Token visibility | Token visible in current process before activation | Current process only |
| Daemon state | Stopped before first bounded diagnostic pulse | PID/lock status only |
| Evidence ledger | Receipt surface observable | File existence only |

## Activation gate

The console always reports:

```text
activation_gate = EXPLICIT_GO_REQUIRED
```

Ready dependencies do not grant runtime activation. A bounded diagnostic mission
still requires the exact operator authorization phrase:

```text
GO: Node0 bounded diagnostic activation only
```

## Forbidden actions

The console forbids these actions in v0.1:

```text
daemon_start
mission_dispatch
node1_activation
public_demo
external_provider_routing
economic_token_claim
```

## Integration points

The implementation lives in:

```text
core/dema/node0_status.py
```

It is surfaced through:

```text
bizra node0 status
bizra dema status
```

JSON consumers read:

```text
dema_node_console.kind = dema_node_console_status
dema_node_console.schema_version = 0.1.0
dema_node_console.truth_label = MEASURED
dema_node_console.activation_gate = EXPLICIT_GO_REQUIRED
```

## Claim discipline

Allowed language:

```text
Node Console dependencies are observable.
Activation remains gated.
Token visibility is process-local.
The daemon is stopped/running.
Evidence ledger is observable/not observed.
```

Forbidden language:

```text
Runtime pulse fired.
Materialization threshold reached.
Node1 is active.
The token economy is live.
The console proves impact by itself.
```

The console is a window. The receipt is the truth.
