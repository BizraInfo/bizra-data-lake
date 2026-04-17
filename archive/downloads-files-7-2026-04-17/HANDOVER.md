# BIZRA — Production Handover Document v0.1

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
│   └── src/
│       ├── runtime.rs          # CognitionRuntime: submit_mission, rehydrate
│       ├── receipts.rs         # ReceiptChain, ReceiptPayload, BLAKE3
│       ├── receipt_freeze_v1.rs    # §7 ReceiptArtifact contract
│       ├── admissibility_freeze_v1.rs  # §7 5-gate admissibility chain
│       ├── mission_freeze_v1.rs    # §7 MissionEnvelope + FourStateModel
│       ├── manifest_artifact.rs    # §7 ManifestArtifact
│       ├── canonical_hasher.rs     # 11 domain-separated BLAKE3 subsystems
│       ├── thought_graph.rs        # Graph of Thoughts + myelination
│       ├── configure_cognition.rs  # PAT-7/SAT-5 boot compositor
│       └── lib.rs
├── bizra-cognition-gateway/    # HTTP projection + dema CLI
│   └── src/
│       ├── main.rs             # Axum server: /health, /chain, /mission
│       └── bin/dema.rs         # Dema CLI binary (441 lines)
├── bizra-hooks/                # no_std constitutional hooks
├── bizra-protocol/             # Mint, identity, consensus
└── [24 other crates]

cycle-4/                        # Retrospective (sealed)
cycle-5/                        # D5, G2, G2-hardening, G3 notes (sealed)
docs/
├── dema-cli-manifesto-v0.md    # Product manifesto
└── why-dema-wins.md            # Product thesis
```

### award-winner-design structure

```
app/
├── dema/page.tsx               # Dema web console
├── api/chain/route.ts          # Proxy to gateway /chain
├── api/missions/route.ts       # Proxy to gateway /mission
└── api/auth/session/route.ts   # Auth session
components/dema/
├── receipt-explorer.tsx        # Chain visualization
├── daily-manifest.tsx          # Manifest panel
├── gate-viewer.tsx             # Admissibility visualization
├── intent-entry.tsx            # Mission submission
└── status-panel.tsx            # Node status
lib/dema/
└── gateway-client.ts           # Gateway HTTP client
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

# Clippy (Rust linting)
cargo clippy -p bizra-cognition -p bizra-cognition-gateway --no-deps
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

### Five Invariants (§3)

| Gate | Rule | Code location |
|---|---|---|
| IHSAN_FLOOR | Quality ≥ 0.95 | admissibility_freeze_v1.rs:534 |
| ZANN_ZERO | Evidence required | admissibility_freeze_v1.rs:540 |
| RIBA_ZERO | Non-extractive | admissibility_freeze_v1.rs:546 |
| CLAIM_MUST_BIND | Evidence hash ≠ 0 | admissibility_freeze_v1.rs:552 |
| NO_SHADOW_STATE | UI ≡ kernel state | admissibility_freeze_v1.rs:558 |

---

## 8. Key APIs

### Gateway HTTP

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness + domain tag |
| GET | `/chain` | Receipt chain state (head, length, records) |
| GET | `/chain/:hash` | Single receipt by hash |
| POST | `/mission` | Submit mission (intent → admissibility → receipt) |

### Rust Internal

| Function | Purpose |
|---|---|
| `CognitionRuntime::submit_mission(envelope, claim)` | Core lawful loop |
| `CognitionRuntime::mission_by_id(id)` | Registry lookup |
| `CognitionRuntime::rehydrate_mission(id)` | Replay verification |
| `AdmissibilityChain::canonical().evaluate(claim)` | 5-gate evaluation |
| `ReceiptChain::append_artifact(receipt)` | Chain append |
| `ManifestArtifact::from_window(start, end, refs, head)` | Daily manifest |

---

## 9. Current State (April 2026)

| Dimension | Status |
|---|---|
| Constitutional contracts | 5/5 frozen in Rust |
| Mission runtime | Operational (submit + rehydrate) |
| Receipt chain | In-memory (sled-store feature ready, not enabled) |
| Gateway | v0.2 with read + write paths |
| Dema CLI | 7 subcommands, exit-code discipline |
| Dema Web | D5 passed, honest empty state |
| CI | 10 workflows on push (GitHub Actions) |
| Tests | 71 Rust + 135 frontend = 206+ |
| Ihsan score | 0.964 (above 0.95 floor) |

---

## 10. Known Gaps

| Gap | Impact | Fix |
|---|---|---|
| In-memory payload store | Receipts lost on restart | Enable sled-store feature flag |
| MissionEnvelope no decode | Can't round-trip from chain | Add ReceiptPayloadDecode impl (~40 lines) |
| ReceiptKind::GovernanceDecision alias | Missions use wrong kind | Add ReceiptKind::MissionCreated = 0x70 |
| Cross-lang Ihsan drift | Rust=0.95, Python SSOT=4 tiers | Sync Rust to reference constants.py |
| 8 stub execution channels | channels/mod.rs has STUB markers | Wire real executors per channel type |
| No tool execution | Dema can't perform real actions yet | Cycle-6: MCP tool transport |
| No LLM integration | No model inference in pipeline | Cycle-7: Ollama wiring |

---

## 11. Governing Documents

| Document | Authority level | Location |
|---|---|---|
| Quran / Hadith | Supreme | External |
| البذرة (The Seed) | Founding covenant | `/mnt/user-data/uploads/البذرة_1.pdf` |
| الرسالة (The Letter) | Founding companion | `/mnt/user-data/uploads/the_massage.pdf` |
| Manifest v0.2 | Constitutional law | Canonical PDF |
| Enforceable Spine v1.0 | Operational law | `docs/` |
| Dema CLI Manifesto v0 | Product law | `docs/dema-cli-manifesto-v0.md` |
| CLAUDE.md | Dev conventions | Root of each repo |

---

## 12. Commit & Push Protocol

1. Test locally: `cargo test -p bizra-cognition -p bizra-cognition-gateway`
2. Clippy: `cargo clippy --no-deps` (zero warnings in session crates)
3. Frontend: `pnpm typecheck && pnpm test && pnpm lint`
4. Commit with conventional prefix: `feat(cognition):`, `fix(cognition):`, `docs(cycle-N):`
5. Push only on explicit authorization (CLAUDE.md rule)
6. CI fires 10 workflows on push — verify all green

---

## 13. Emergency Procedures

### Gateway crash
```bash
# Restart (receipts are lost with in-memory store)
bizra-gateway
# Or: sudo systemctl restart bizra-gateway (if systemd installed)
```

### Test failure after code change
```bash
# Revert last commit
git revert HEAD
cargo test -p bizra-cognition --lib  # verify clean
```

### Chain corruption (theoretical — in-memory store resets on restart)
```bash
# Kill gateway, restart fresh
# All chain state rebuilds from zero (no persistence yet)
# This is a known gap, not a bug
```

---

## 14. Cycle History

| Cycle | Date | Focus | Key commit |
|---|---|---|---|
| Genesis | 2026-03-27 | Block 0 minted | 350d642099bde68b |
| 1 | 2026-04-15 | Audit + kernel files | a4e97dc20ac2e10d |
| 2 | 2026-04-16 | Peak synthesis | 48e5395471d3ca77 |
| 3 | 2026-04-16 | Cross-lang constants drift | NODE0 local |
| 4 | 2026-04-17 | §17 build order Steps 2-7 | afe9cc30 |
| 5 | 2026-04-17 | Principal activation | 8b16762a |
| 6 | NEXT | First impact receipt (Downloads) | — |

---

## 15. Contact

**Founder:** Mohamed Beshr (Mumo)
**Entity:** BIZRA Foundation, Dubai
**Email:** m.beshr@bizra.info
**GitHub:** github.com/BizraInfo

---

*Close it. Prove it. Reveal it.*
