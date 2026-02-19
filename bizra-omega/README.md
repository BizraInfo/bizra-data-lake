# bizra-omega — Unified Rust Workspace

> Standing on Giants: Shannon (information theory) . Lamport (distributed systems) . Besta (Graph-of-Thoughts) . Maturana (autopoiesis)

**Version:** 2.0.0 | **Edition:** 2021 | **Crates:** 18 | **Tests:** 610+

The single Rust workspace for the BIZRA ecosystem. Contains both the **platform layer** (decentralized infrastructure) and the **cognitive layer** (Node0 memory, hooks, FATE gates, IPC).

## Quick Start

```bash
cd bizra-omega

# Build everything
cargo build --workspace

# Run all tests
cargo test --workspace

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
└── Cognitive Layer ────────────────────────────────────────────
    ├── bizra-hooks/         Nervous system: event bus, hook
    │                        pipeline, component registry, Ihsan gate
    ├── bizra-memory/        Memory synthesis: atoms, insights,
    │                        profile snapshots, Python bridge traits
    ├── fate-binding/        FATE gates: Z3 formal verification,
    │                        Dilithium post-quantum crypto, capability cards
    └── iceoryx-bridge/      IPC: zero-copy shared memory (iceoryx2)
```

## Crate Reference

### Platform Layer

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

### Cognitive Layer

| Crate | Lines | Tests | Key Feature |
|-------|-------|-------|-------------|
| bizra-hooks | 3,147 | 44 | Zero-dependency nervous system |
| bizra-memory | 3,025 | 41 | Depends only on bizra-hooks |
| fate-binding | 1,319 | 19 | Z3 + Dilithium post-quantum |
| iceoryx-bridge | 1,277 | 5 | 250ns IPC target |

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
# All 610+ tests
cargo test --workspace

# Single crate
cargo test -p bizra-core
cargo test -p bizra-memory

# Doc tests only
cargo test --doc --workspace

# With coverage
cargo llvm-cov --workspace --lcov --output-path lcov.info
```

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

## Binaries

| Binary | Build | Purpose |
|--------|-------|---------|
| `bizra` | `cargo build -p bizra-cli` | Terminal dashboard |
| `bizra-api` | `cargo build -p bizra-api` | REST API server |
| `bizra-install` | `cargo build -p bizra-installer` | Setup wizard |
| `bizra-hunter-snr` | `cargo build -p bizra-hunter` | SNR pipeline CLI |
| `proofspace-validator` | `cargo build -p bizra-proofspace` | Proof verification |
| `resourcepool-node` | `cargo build -p bizra-resourcepool` | Resource pool |
| `node0-genesis` | `cargo build -p bizra-resourcepool` | Genesis seed |
| `telescript-demo` | `cargo build -p bizra-telescript` | Telescript runner |

## CI Pipeline

Two workflows cover this workspace:

1. **`ci.yml`** (main pipeline): lint-rust + test-rust + pyo3-bindings
2. **`native-ci.yml`** (extended): lint + test + security audit + benchmarks

Both install Z3 and run `cargo test --workspace` covering all 18 crates.

## Security

```bash
# Dependency audit
cargo audit

# Pinned security fixes in workspace Cargo.toml:
# bytes >= 1.11.1 (RUSTSEC-2026-0007)
```

## History

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2026-01-30 | Initial 14-crate platform workspace |
| 2.0.0 | 2026-02-20 | Unified: merged 4 cognitive crates from native/ |
