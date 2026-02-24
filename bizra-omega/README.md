# bizra-omega — Unified Rust Workspace

> Standing on Giants: Shannon (information theory) . Lamport (distributed systems) . Besta (Graph-of-Thoughts) . Maturana (autopoiesis) . Al-Ghazali (Ihsan ethics)

**Version:** 2.1.0 | **Edition:** 2021 | **Crates:** 20 | **Tests:** 944 | **Clippy:** 0 warnings

The single Rust workspace for the BIZRA ecosystem. Contains the **platform layer** (decentralized infrastructure), the **cognitive layer** (Node0 memory, hooks, FATE gates, IPC), and the **desktop node layer** (sovereign agent runtime + binary).

## Quick Start

```bash
cd bizra-omega

# Build everything
cargo build --workspace

# Run all 944 tests
cargo test --workspace

# Lint (zero warnings enforced)
cargo clippy --workspace --all-targets -- -D warnings

# Maximum optimization (AVX-512)
RUSTFLAGS="-C target-cpu=native" cargo build --profile omega
```

### Prerequisites

- Rust stable (1.88+)
- Z3 solver (for fate-binding): `sudo apt install libz3-dev`
- Set `Z3_SYS_Z3_HEADER=/usr/include/z3.h` if Z3 header is not auto-detected

## Architecture

```
bizra-omega/
│
├── Platform Layer ─────────────────────────────────────────────
│   ├── bizra-core/          Sovereign kernel: constitution, PCI,
│   │                        identity, Islamic finance, SIMD, GoT
│   ├── bizra-inference/     LLM gateway: tiered model selection
│   ├── bizra-federation/    P2P: gossip protocol, signed messages
│   ├── bizra-autopoiesis/   Self-healing and adaptation
│   ├── bizra-api/           REST API server (Axum)
│   ├── bizra-cli/           Terminal dashboard ("bizra" binary)
│   ├── bizra-hunter/        Bounty system + SNR pipeline
│   ├── bizra-proofspace/    Proof verification + validator binary
│   ├── bizra-resourcepool/  Compute allocation + genesis node
│   ├── bizra-telescript/    Mobile agent scripts
│   ├── bizra-hypergraph/    Hypergraph data structure
│   ├── bizra-installer/     One-command setup binary
│   ├── bizra-python/        PyO3 bridge to Python core/
│   └── bizra-tests/         E2E + integration + property tests
│
├── Cognitive Layer ────────────────────────────────────────────
│   ├── bizra-hooks/         Nervous system: event bus, hook
│   │                        pipeline, component registry, Ihsan gate
│   ├── bizra-memory/        Memory synthesis: atoms, insights,
│   │                        profile snapshots, Python bridge traits
│   ├── fate-binding/        FATE gates: Z3 formal verification,
│   │                        Dilithium post-quantum crypto, capability cards
│   └── iceoryx-bridge/      IPC: zero-copy shared memory (iceoryx2)
│
└── Desktop Node Layer ─────────────────────────────────────────
    ├── bizra-agent/         Sovereign agent: PAT team, reflex
    │                        compiler, action bus, key vault, hash
    │                        namespace, intent classification
    └── bizra-node/          Desktop node binary: protocol handler,
                             MCP JSON-RPC transport, action executor,
                             audit trail, persistence
```

## Crate Reference

### Platform Layer (14 crates)

| Crate | Lines | Tests | Binaries | Key Deps |
|-------|-------|-------|----------|----------|
| bizra-core | 13,189 | 147 | - | ed25519-dalek, blake3, rayon |
| bizra-resourcepool | 4,012 | 25 | `resourcepool-node`, `node0-genesis` | core, proofspace, telescript, federation |
| bizra-cli | 3,208 | 23 | `bizra` | - |
| bizra-hunter | 2,358 | 23 | `bizra-hunter-snr` | - |
| bizra-proofspace | 1,981 | 19 | `proofspace-validator` | core |
| bizra-inference | 1,869 | 21 | - | core, reqwest |
| bizra-api | 1,562 | 11 | `bizra-api` | core, inference, federation, autopoiesis |
| bizra-telescript | 1,562 | 22 | `telescript-demo` | core |
| bizra-federation | 1,149 | 26 | - | core |
| bizra-python | 924 | 2 | - (cdylib) | core, inference, pyo3 |
| bizra-installer | 839 | 7 | `bizra-install` | core, inference |
| bizra-hypergraph | 645 | 17 | - | - |
| bizra-autopoiesis | 290 | 4 | - | core |
| bizra-tests | harness | 29 | - | core, inference, federation, autopoiesis |

### Cognitive Layer (4 crates)

| Crate | Lines | Tests | Key Feature |
|-------|-------|-------|-------------|
| bizra-hooks | 3,147 | 43 | Zero-dependency nervous system |
| bizra-memory | 3,025 | 40 | Depends only on bizra-hooks |
| fate-binding | 1,319 | 19 | Z3 + Dilithium post-quantum |
| iceoryx-bridge | 1,277 | 5 | 250ns IPC target |

### Desktop Node Layer (2 crates)

| Crate | Lines | Tests | Binaries | Key Feature |
|-------|-------|-------|----------|-------------|
| bizra-agent | 8,146 | 118 | - | PAT team, reflex compiler, action bus, key vault |
| bizra-node | 4,427 | 72 | `bizra-node` | Protocol handler, MCP transport, audit trail |

## Dependency Graph

```
bizra-core ──────────────────────────────────────────────┐
  ├── bizra-inference ──┬── bizra-api                    │
  │                     ├── bizra-installer              │
  │                     └── bizra-python (PyO3)          │
  ├── bizra-federation ─┬── bizra-api                    │
  │                     └── bizra-resourcepool           │
  ├── bizra-autopoiesis ─── bizra-api                    │
  ├── bizra-proofspace ──── bizra-resourcepool           │
  ├── bizra-telescript ──── bizra-resourcepool           │
  └── bizra-tests (dev)                                  │
                                                         │
bizra-hooks ─── bizra-memory    (cognitive, standalone)  │
                 └── bizra-agent ─── bizra-node          │
                                                         │
fate-binding                    (standalone, Z3 + napi)  │
iceoryx-bridge                  (standalone, IPC + napi) │
bizra-hypergraph                (standalone)             │
bizra-hunter                    (standalone)             │
bizra-cli                       (standalone)             │
```

## Build Profiles

| Profile | Command | Use Case |
|---------|---------|----------|
| `dev` | `cargo build` | Development (opt-level 1) |
| `release` | `cargo build --release` | Production (fat LTO, strip, panic=abort) |
| `omega` | `RUSTFLAGS="-C target-cpu=native" cargo build --profile omega` | Maximum (AVX-512, native CPU) |

## Testing

```bash
# All 944 tests
cargo test --workspace

# Single crate
cargo test -p bizra-agent
cargo test -p bizra-node

# Doc tests only
cargo test --doc --workspace

# With coverage
cargo llvm-cov --workspace --lcov --output-path lcov.info
```

### Test Distribution

| Layer | Crates | Tests |
|-------|--------|-------|
| Platform | 14 | 376 |
| Cognitive | 4 | 107 |
| Desktop Node | 2 | 190 |
| Integration (bizra-tests) | 1 | 29 |
| Doc tests | - | 242 |
| **Total** | **20** | **944** |

## Binaries

| Binary | Build | Purpose |
|--------|-------|---------|
| `bizra-node` | `cargo build -p bizra-node` | Desktop sovereign node |
| `bizra` | `cargo build -p bizra-cli` | Terminal dashboard |
| `bizra-api` | `cargo build -p bizra-api` | REST API server |
| `bizra-install` | `cargo build -p bizra-installer` | Setup wizard |
| `bizra-hunter-snr` | `cargo build -p bizra-hunter` | SNR pipeline CLI |
| `proofspace-validator` | `cargo build -p bizra-proofspace` | Proof verification |
| `resourcepool-node` | `cargo build -p bizra-resourcepool` | Resource pool |
| `node0-genesis` | `cargo build -p bizra-resourcepool` | Genesis seed |
| `telescript-demo` | `cargo build -p bizra-telescript` | Telescript runner |

## Python Bridge

The `bizra-python` crate provides PyO3 bindings:

```bash
cd bizra-python
pip install maturin
maturin develop --release

# Then in Python:
# import bizra
```

Currently exposes: core identity, PCI envelopes, inference, federation, autopoiesis. The cognitive layer crates (hooks, memory) can be added to the bridge as path dependencies now that they share the workspace.

## Security

The workspace enforces security-first development:

- **Bounded reads**: All network I/O enforces `MAX_LINE_LENGTH` (64 KB) to prevent OOM
- **Constant-time comparison**: `constant_time_eq()` for secret comparison (timing-safe)
- **Zeroize-on-drop**: `SecretString` uses volatile writes + compiler fences
- **Symlink rejection**: File vault rejects symlinked secret files
- **Key validation**: Environment vault validates `[a-zA-Z0-9_]` only, max 128 chars
- **Atomic ordering**: `Acquire`/`AcqRel` on shared counters (not `Relaxed`)
- **UTF-8 guards**: `debug_assert!` before all `from_utf8_unchecked` call sites
- **Batch limits**: JSON-RPC batch requests capped at 100 items

```bash
# Dependency audit
cargo audit

# Pinned security fixes in workspace Cargo.toml:
# bytes >= 1.11.1 (RUSTSEC-2026-0007)
```

## CI Pipeline

Two workflows cover this workspace:

1. **`ci.yml`** (main pipeline): lint-rust + test-rust + pyo3-bindings
2. **`alpha100-release-binaries.yml`**: cross-compile + release binaries

Both install Z3 and run `cargo test --workspace` covering all 20 crates.

## History

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2026-01-30 | Initial 14-crate platform workspace |
| 2.0.0 | 2026-02-20 | Unified: merged 4 cognitive crates from native/ |
| 2.1.0 | 2026-02-21 | Alpha-100: desktop node layer (bizra-agent + bizra-node) + security hardening |
