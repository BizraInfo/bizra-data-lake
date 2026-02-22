# Phase 48 Spec — Part 1: Current State Inventory

> Standing on Giants: Shannon (information theory) · Lamport (distributed systems) · Besta (GoT reasoning)

## Purpose

Before building anything new, document exactly what exists across all three codebases (Python `core/`, Rust `bizra-omega/`, Rust `native/`) so we know what overlaps, what's missing, and what bridges are needed.

## Audit Date: 2026-02-20

---

## 1. Python Core (`core/`)

| Metric | Value |
|--------|-------|
| Modules | 54 directories |
| Files | 420 `.py` files |
| Lines | 171,679 |
| Test files | 296 |
| Tests collected | 7,278 |
| Version | Implicit (no `__version__`) |

### Key Modules (mapped to Rust counterparts)

| Python Module | Lines | Rust Counterpart (omega) | Rust Counterpart (native) |
|---------------|-------|--------------------------|--------------------------|
| `core/sovereign/` | ~largest | `bizra-core/sovereign` | - |
| `core/pci/` | - | `bizra-core/pci` | - |
| `core/governance/` | - | `bizra-core/constitution` | - |
| `core/federation/` | - | `bizra-federation/` | - |
| `core/inference/` | - | `bizra-inference/` | - |
| `core/autopoiesis/` | - | `bizra-autopoiesis/` | - |
| `core/reasoning/` (GoT) | - | `bizra-core/sovereign` | - |
| `core/living_memory/` | - | - | `bizra-memory/` |
| `core/memory/` | - | - | `bizra-memory/` |
| `core/rollout/` | - | - | - |
| `core/search/` | - | - | - |
| `core/prediction/` | - | - | - |
| `core/resonance.py` | - | - | - |
| `core/iaas/` (SNR) | - | `bizra-core/sovereign` | - |

---

## 2. Rust Workspace: `bizra-omega/` (The Platform Layer)

| Metric | Value |
|--------|-------|
| Crates | 14 |
| Total lines | 40,570 `.rs` |
| Tests | 501 (all pass) |
| Workspace version | 1.0.0 |
| Edition | 2021 |
| Profile | Fat LTO + `panic=abort` + AVX-512 omega profile |

### Crate Inventory

| Crate | Lines | Tests | Purpose | Depends On |
|-------|-------|-------|---------|------------|
| `bizra-core` | 13,189 | 147 | Sovereign kernel: constitution, PCI, identity, Islamic finance, SIMD, GoT, omega | - |
| `bizra-resourcepool` | 4,012 | ~25 | Compute allocation, node0-genesis binary | core, proofspace, telescript, federation |
| `bizra-cli` | 3,208 | ~23 | Terminal UI dashboard, `bizra` binary | - |
| `bizra-hunter` | 2,358 | ~23 | Bounty system, SNR pipeline binary | - |
| `bizra-proofspace` | 1,981 | ~19 | Proof verification, proofspace-validator binary | core |
| `bizra-inference` | 1,869 | ~21 | Inference backends | core |
| `bizra-api` | 1,562 | ~11 | REST API server binary | core, inference, federation, autopoiesis |
| `bizra-telescript` | 1,562 | ~22 | Mobile agent scripts, benchmark | core |
| `bizra-federation` | 1,149 | ~26 | P2P gossip, signed messages | core |
| `bizra-python` | 924 | ~2 | PyO3 bindings | core, inference, autopoiesis, federation |
| `bizra-installer` | 839 | ~7 | Installer binary | core, inference |
| `bizra-hypergraph` | 645 | ~17 | Hypergraph data structure | - |
| `bizra-autopoiesis` | 290 | ~4 | Self-healing | core |
| `bizra-tests` | 0 (harness) | ~29 | E2E + integration + property tests | core, inference, federation, autopoiesis |

### Dependency Graph (simplified)

```
bizra-core ─────────────────────────────────────────────────────┐
  ├── bizra-inference                                           │
  │     ├── bizra-api ──→ (+ federation, autopoiesis)          │
  │     ├── bizra-installer                                    │
  │     └── bizra-python ──→ (+ autopoiesis, federation)       │
  ├── bizra-federation ──→ bizra-resourcepool                  │
  ├── bizra-autopoiesis                                        │
  ├── bizra-proofspace ──→ bizra-resourcepool                  │
  ├── bizra-telescript ──→ bizra-resourcepool                  │
  └── bizra-tests (dev)                                        │
                                                               │
bizra-hypergraph (standalone)                                  │
bizra-hunter (standalone)                                      │
bizra-cli (standalone, reads bizra-core at runtime)            │
```

### Binaries Produced

| Binary | Crate | Purpose |
|--------|-------|---------|
| `bizra` | bizra-cli | Terminal dashboard |
| `bizra-api` | bizra-api | REST API server |
| `bizra-install` | bizra-installer | One-command setup |
| `bizra-hunter-snr` | bizra-hunter | SNR pipeline CLI |
| `proofspace-validator` | bizra-proofspace | Proof verification |
| `resourcepool-node` | bizra-resourcepool | Resource pool node |
| `node0-genesis` | bizra-resourcepool | Genesis seed node |
| `telescript-demo` | bizra-telescript | Telescript runner |

---

## 3. Rust Workspace: `native/` (The Node0 Cognitive Layer)

| Metric | Value |
|--------|-------|
| Crates | 4 |
| Total lines | 8,768 `.rs` |
| Tests | 109 (107 unit + 2 doctests, all pass) |
| Workspace version | 2.2.0 |
| Edition | 2021 |
| Profile | LTO + strip |

### Crate Inventory

| Crate | Lines | Tests | Purpose | Depends On |
|-------|-------|-------|---------|------------|
| `bizra-hooks` | 3,147 | 44 | Nervous system: event bus, hook pipeline, component registry, Ihsan gate | - |
| `bizra-memory` | 3,025 | 41 | Cognitive layer: memory atoms, synthesis engine, profile snapshots, bridge FFI | bizra-hooks |
| `fate-binding` | 1,319 | 19 | FATE gates: Z3 formal verification, Dilithium post-quantum crypto, capability cards | - |
| `iceoryx-bridge` | 1,277 | 5 | IPC: zero-copy shared memory via iceoryx2 | - |

### Dependency Graph

```
bizra-hooks ──→ bizra-memory
fate-binding (standalone — Z3 + Dilithium)
iceoryx-bridge (standalone — IPC)
```

### Key Types (native/)

- `BizraSystem` — component registry + event bus
- `IhsanGate` — quality gate with throttle policy
- `BizraMemory` — facade: ingest → extract → synthesize → query
- `AtomKind` — Fact, Preference, Pattern, Goal, Negation, Principle, Context
- `FateGateChain` — Z3 + Ihsan + SNR + Schema gates
- `IpcRouter` — zero-copy message routing

---

## 4. Cross-Workspace Analysis

### Zero Cross-References

The two Rust workspaces (`bizra-omega/` and `native/`) have **no compile-time dependencies** on each other. They share no crate references.

### Conceptual Overlap

| Concept | bizra-omega | native | Status |
|---------|------------|--------|--------|
| Ihsan threshold | `bizra-core::constitution::IhsanThreshold` (f64) | `bizra-hooks::IhsanScore` (f32, [0.0,1.0]) | **Duplicate — different types** |
| Constitutional gate | `bizra-core::constitution::Constitution` | `fate-binding::FateGateChain` | **Parallel — different approaches** |
| SNR scoring | `bizra-core::sovereign` + `bizra-hunter` | - | omega only |
| Memory / synthesis | - | `bizra-memory::BizraMemory` | native only |
| Event system | - | `bizra-hooks::EventBus` | native only |
| Post-quantum crypto | - | `fate-binding::Dilithium` | native only |
| IPC | - | `iceoryx-bridge::IpcRouter` | native only |
| Federation / gossip | `bizra-federation` | - | omega only |
| Inference gateway | `bizra-inference` | - | omega only |
| PyO3 bindings | `bizra-python` | - | omega only |
| API server | `bizra-api` | - | omega only |

### Missing Crates (from transcript session, never persisted)

| Crate | Planned Lines | Purpose | Status |
|-------|---------------|---------|--------|
| `bizra-agent` | ~2,000 | Agent runtime: event loop, capability matching, tool dispatch | **NOT IN REPO** |
| `bizra-node` | ~2,000 | Node binary: Tauri desktop shell, continuity bridge, sacred geometry | **NOT IN REPO** |

---

## 5. CI Coverage

| Pipeline | File | Covers |
|----------|------|--------|
| Python CI | `.github/workflows/ci.yml` | ruff, black, isort, mypy, pytest, coverage, bandit, pip-audit, Docker |
| Native CI | `.github/workflows/native-ci.yml` | cargo fmt, clippy, test, llvm-cov, cargo-audit, benchmarks |
| bizra-omega CI | `.github/workflows/ci.yml` (cargo sections) | cargo fmt, clippy, test |
| Performance | `.github/workflows/performance.yml` | Benchmarks |
| Docs quality | `.github/workflows/docs-quality.yml` | Documentation linting |

### Gap: No unified cross-workspace CI job exists.

---

## 6. Summary Totals

| Layer | Lines | Tests | Status |
|-------|-------|-------|--------|
| Python `core/` | 171,679 | 7,278 | Active, 2 collection errors |
| Rust `bizra-omega/` | 40,570 | 501 | All pass |
| Rust `native/` | 8,768 | 109 | All pass (fixed this session) |
| **Grand Total** | **221,017** | **7,888** | |

### Health Verdict

- **bizra-omega**: Mature, 14-crate workspace with 501 passing tests, production binaries, PyO3 bridge
- **native**: Newer, 4-crate workspace with 109 passing tests, focused on Node0 cognitive layer
- **Python core**: Largest codebase, 7,278 tests, comprehensive but some collection errors
- **Cross-workspace integration**: None — the two Rust workspaces are isolated islands
