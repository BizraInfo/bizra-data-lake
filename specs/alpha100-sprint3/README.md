# Alpha-100 Sprint 3: Action Infrastructure — Desktop + Transport + Audit + Vault

> Standing on Giants: General Magic (Telescript, 1994) · Lamport (distributed state, 1978) · Shannon (channel capacity, 1948) · Cerf & Kahn (TCP/IP, 1974) · Al-Ghazali (Ihsan, 1095) · Diffie & Hellman (public key, 1976)

---

## Sprint 2 Recap (Delivered)

| Deliverable | Status | Tests |
|-------------|--------|-------|
| Action Bus (`action_bus.rs`, `action_types.rs`) | Complete | 845 total |
| Protocol commands (INTENT_CLASSIFY, ACTION_DISPATCH) | Complete | Clippy clean |
| Smart model routing (`llm_bridge.js`) | Complete | Integration verified |
| PWA support (manifest.json, sw.js) | Complete | Vite build passes |

**Baseline:** 845 tests, 0 failures, clippy `-D warnings` clean, `cargo fmt` clean.

---

## Sprint 3 Overview

Sprint 2 gave the node a **voice** (protocol commands) and a **brain** (action bus). Sprint 3 gives it **hands** (AHK bridge), **ears** (MCP transport), a **memory** (audit trail), and a **wallet** (key vault).

### 4 Phases

| Phase | Deliverable | ~Lines | Dependencies |
|-------|-------------|--------|--------------|
| [Phase 1](phase_1_ahk_bridge.md) | AHK Bridge Server | ~330 | None (standalone AHK script) |
| [Phase 2](phase_2_mcp_transport.md) | MCP JSON-RPC Transport | ~510 | `protocol.rs`, `handler.rs` |
| [Phase 3](phase_3_eventbus_integration.md) | EventBus Audit Trail | ~260 | `event_bus.rs`, `action_executor.rs` |
| [Phase 4](phase_4_key_vault.md) | Multi-Provider Key Vault | ~560 | `action_executor.rs` |

**Total: ~1,660 lines across 12 new files + 10 modified files.**

### Execution Order

```
Phase 1 (AHK)          Phase 4 (Vault)
    │                       │
    └──── independent ──────┘
              │
    Phase 3 (EventBus)  -- depends on action_executor changes from Phase 4
              │
    Phase 2 (MCP Transport) -- depends on nothing, fully parallel
```

- **Phases 1 + 4:** Fully parallel. Phase 1 is AHK (Windows scripting), Phase 4 is Rust.
- **Phase 2:** Independent, can run in parallel with 1+4.
- **Phase 3:** Best done after Phase 4 (both modify `action_executor.rs`), or merge changes.
- **Recommended:** 1 ∥ 2 ∥ 4, then 3.

---

## File Inventory (All Phases)

### New Files (12)

| File | Phase | ~Lines |
|------|-------|--------|
| `filedfs/ahk_bridge.ahk` | 1 | 300 |
| `filedfs/bridge_config.ini` | 1 | 20 |
| `filedfs/skills/hello_world.ahk` | 1 | 10 |
| `bizra-omega/bizra-node/src/mcp_transport.rs` | 2 | 280 |
| `bizra-omega/bizra-node/tests/mcp_transport_tests.rs` | 2 | 200 |
| `bizra-omega/bizra-node/src/audit_hook.rs` | 3 | 80 |
| `bizra-omega/bizra-node/tests/eventbus_integration_tests.rs` | 3 | 120 |
| `bizra-omega/bizra-agent/src/key_vault.rs` | 4 | 250 |
| `bizra-omega/bizra-agent/src/vault_env.rs` | 4 | 40 |
| `bizra-omega/bizra-agent/src/vault_file.rs` | 4 | 70 |
| `bizra-omega/bizra-agent/src/vault_toml.rs` | 4 | 50 |
| `bizra-omega/bizra-agent/tests/key_vault_tests.rs` | 4 | 150 |

### Modified Files (10)

| File | Phase | ~Lines Added |
|------|-------|-------------|
| `bizra-omega/bizra-node/src/lib.rs` | 2, 3 | +2 |
| `bizra-omega/bizra-node/src/main.rs` | 2 | +30 |
| `bizra-omega/bizra-node/src/node.rs` | 2, 3 | +25 |
| `bizra-omega/bizra-node/src/action_executor.rs` | 3, 4 | +50 |
| `bizra-omega/bizra-node/src/handler.rs` | 3 | +5 |
| `bizra-omega/bizra-agent/src/lib.rs` | 4 | +4 |
| `bizra-omega/bizra-agent/Cargo.toml` | 4 | +1 |
| `bizra-omega/bizra-hooks/src/lib.rs` | 3 | +2 |
| `filedfs/llm_bridge.js` | 4 (optional) | +5 |
| `filedfs/skills/` (directory) | 1 | — |

---

## Test Targets

| Phase | New Tests | Target Total |
|-------|-----------|-------------|
| Phase 1 | 10 (AHK — manual/script) | N/A (Windows-only) |
| Phase 2 | 14 (Rust) | 845 + 14 = 859+ |
| Phase 3 | 11 (Rust) | 859 + 11 = 870+ |
| Phase 4 | 15 (Rust) | 870 + 15 = 885+ |

**Sprint 3 target: 885+ Rust tests, 0 failures, clippy clean.**

---

## Verification Plan

```bash
# Phase 2: MCP Transport
cd bizra-omega && cargo test --workspace --release
echo '{"jsonrpc":"2.0","method":"ping","id":1}' | nc 127.0.0.1 9741

# Phase 3: EventBus Integration
cargo test -p bizra-node --test eventbus_integration_tests
ls -la data/audit/action_receipts.jsonl

# Phase 4: Key Vault
cargo test -p bizra-agent --test key_vault_tests
# Verify no plaintext in debug output:
cargo test -p bizra-agent 2>&1 | grep -i "hunter2\|sk-\|secret_value" && echo "LEAK!" || echo "CLEAN"

# Full workspace
cd bizra-omega && cargo test --workspace --release
cd bizra-omega && cargo clippy --workspace --all-targets -- -D warnings
cd bizra-omega && cargo fmt --all -- --check
```

---

## Gem Traceability (Sprint 2 → Sprint 3)

| Gem | Sprint 2 Application | Sprint 3 Extension |
|-----|---------------------|-------------------|
| Gem 1: Lyapunov IhsanScore | Gate 2 in ActionBus | EventBus ihsan propagation (Phase 3 FR-5) |
| Gem 2: PostDeliver audit | Noted as "Sprint 3" | Fully implemented (Phase 3) |
| Gem 3: Hash chain | ActionReceipt.seal() | Receipt hash in EventBus payload (Phase 3) |
| Gem 4: Fail-closed | PolicyMissing gate | Vault NotFound → ActionError propagation (Phase 4) |
| Gem 7: Double-safety | Permit + FATE gates | Vault + EventBus both independently gated |
| Gem 11: Quarantine-not-evict | Test verified | Vault cache invalidation (refresh, not delete) |

---

## Not In This Sprint (Sprint 4+)

- TLS on MCP transport
- AES-256-GCM encrypted-at-rest secrets
- HSM / cloud vault backends
- Async EventBus emission
- Event replay from audit log
- Cross-node event gossip
- Multi-monitor AHK awareness
- HTTP/SSE MCP transport variants
- Prometheus metrics from receipt events
