# BIZRA — Production Handover Document v0.1

> **STATUS:** DRAFT — founding draft preserved for historical record. Superseded by `docs/BIZRA-Handover-v1.md`.
>
> **Why preserved:** Per manifesto amendment protocol, founding drafts are not deleted when a new operative version lands. v0.1 is the author's (Mumo's) first professional handover draft, authored 2026-04-17. v1 extends and corrects it per the integrator audit recorded in conversation logs.
>
> **Lineage:** v0.1 (this file) → v1 (`docs/BIZRA-Handover-v1.md`). See integrator audit findings in Cycle-5 session logs.

---

بسم الله الرحمن الرحيم

**Author:** Mohamed Beshr (Mumo), BIZRA Foundation
**Date:** 2026-04-17
**Status:** DRAFT — first professional handover document for BIZRA NODE0

---

## 1. What is BIZRA?

BIZRA is a constitutional trust compiler for sovereign intelligence. It turns human intent into lawful, receipted, replayable action through a five-gate admissibility chain and a cryptographic receipt chain.

**One sentence:** Every action receipted. Every claim verifiable. Every mission lawful.

**What BIZRA is NOT:** a chatbot, an agent framework, a cloud service, or a model wrapper.

---

## 2. Repository Map

| Repo | Purpose | Language | Location on NODE0 |
|---|---|---|---|
| `bizra-data-lake` | Kernel, runtime, gateway, CLI, docs | Rust + Python | `/data/bizra/repos/bizra-data-lake` |
| `award-winner-design` | Frontend (Dema web UI) | TypeScript/Next.js | `/data/bizra/repos/award-winner-design` |

### bizra-data-lake structure

```
bizra-omega/                    # Rust workspace (28 crates)
├── bizra-cognition/            # Core: runtime, receipts, admissibility, missions
├── bizra-cognition-gateway/    # HTTP projection + dema CLI
├── bizra-hooks/                # no_std constitutional hooks
├── bizra-protocol/             # Mint, identity, consensus
└── [24 other crates]

cycle-4/                        # Retrospective (sealed)
cycle-5/                        # D5, G2, G2-hardening, G3 notes (sealed)
docs/
├── dema-cli-manifesto-v0.md    # Product manifesto
└── why-dema-wins.md            # Product thesis
```

---

## 3. System Requirements

| Component | Minimum | NODE0 actual |
|---|---|---|
| OS | Ubuntu 24.04 | Ubuntu 24.04 |
| CPU | 4 cores | i9-14900HX (24 cores) |
| RAM | 8 GB | 128 GB DDR5 |
| Disk | 50 GB free | 3.8 TB RAID 0 |
| GPU | Not required (Ollama optional) | RTX 4090 |
| Rust | 1.80+ | 1.94.0 |
| Node.js | 20+ | 24.5.0 |
| Git | 2.40+ | 2.53.0 |

*(Note — per integrator audit, Node.js actual is v22.22.2 and Git actual is 2.43.0. v1 carries corrected values.)*

---

## 4. Installation

```bash
# Option A: Unified installer
chmod +x install.sh && ./install.sh

# Option B: Manual
git clone git@github.com:BizraInfo/bizra-data-lake.git
git clone git@github.com:BizraInfo/award-winner-design.git
cd bizra-data-lake/bizra-omega && cargo build --release
cd ../../award-winner-design && pnpm install && pnpm build
```

---

## 5. Running

### Gateway (must be running for CLI and web UI)
```bash
./target/release/bizra-cognition-gateway
# Binds to 127.0.0.1:7421 (localhost only, no external exposure)
# Health check: curl http://127.0.0.1:7421/health
```

### Dema CLI
```bash
dema                # Node status
dema health         # Gateway liveness
dema chain          # Receipt chain state
dema activate       # Principal activation (first mission)
dema submit "task"  # Submit a mission
dema receipt <id>   # Inspect a receipt
```

### Dema Web UI (optional)
```bash
cd award-winner-design && pnpm dev
# Opens on http://localhost:3002/dema
```

---

## 6. Testing

```bash
# Rust (kernel + gateway)
cd bizra-data-lake/bizra-omega
cargo test -p bizra-cognition --lib        # 64 tests
cargo test -p bizra-cognition-gateway      # 7 tests
cargo test --workspace                     # Full workspace

# Frontend
cd award-winner-design
pnpm typecheck                             # TypeScript
pnpm test                                  # Vitest (135 tests)
pnpm lint                                  # ESLint
```

---

## 7. Architecture (Four Planes)

| Plane | Function | Enforces |
|---|---|---|
| **Kernel** (L1) | Constitutional gates, crypto, admissibility | Law — immutable |
| **Graph** (L2) | Cognition, memory, GoT, mission decomposition | Context — never becomes law |
| **Proof** (L3) | Receipts, chain, replay, manifests | Integrity — never invents truth |
| **Face** (L4) | Dema (CLI + web), trust surfaces | Visibility — never simulates law |

### The Lawful Loop (§6)

```
Intent → Mission → Claim → Admissibility → Execution → Receipt → Canonicalization → Replay
```

No bypasses. No side channels. No UI-only state mutation.

---

## 8. Key APIs

### Gateway HTTP

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + domain tag |
| GET | `/chain` | Receipt chain state |
| GET | `/chain/:hash` | Single receipt by hash |
| POST | `/mission` | Submit mission |

---

## 9-15. Additional sections (see v1 for complete and corrected content)

Original v0.1 included sections on current state, known gaps, governing documents, commit protocol, emergency procedures, cycle history, and contact. v1 extends and corrects all of these with empirically-verified numbers, the full founding-covenant reference, and the complete doctrine canon inventory.

---

*This is v0.1. For the current operative handover, see `docs/BIZRA-Handover-v1.md`.*

*Close it. Prove it. Reveal it.*
