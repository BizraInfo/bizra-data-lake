# Architecture Audit — BIZRA v0.1

**Scope:** Node0 doctrine, DEMA, PAT×7, SAT×5, URP, P2/receipts cockpit, local-first runtime, cloud-optional design, data-lake / Foundry / canon-pack separation, no-runtime-canon-contamination invariant.

---

## 1. Node0 doctrine

**Doctrine source:** `memory/project_node0_sovereign_origin_sealed.md` + `bizra-omega/bizra-core/src/genesis_seal.rs` + `bizra-omega/bizra-core/src/canonical_receipt.rs`.

**Sealed 2026-04-21:** BIZRA-641A1D00 / Mumo / genesis `369319cd…676a`. Canonical Node0 reference.

**Assessment:** ✅ **Architecturally clean.** Node0 is a *sovereign identity* bound to a genesis seal, not an authority server. Every visible effect chains to that genesis via BLAKE3-chained, Ed25519-signed canonical receipts. This is the architectural invariant that keeps BIZRA from becoming the very centralized thing it opposes.

**Gotcha flagged by audit:** 806 Rust `.unwrap()` occurrences across `bizra-omega/`. The receipt-on-every-effect invariant is **contingent** on no panics along the path. Hot-path unwrap audit is required before Tier-D claims are public.

## 2. DEMA — single visible face

**Doctrine:** DEMA is the only consumer-visible surface. PAT / SAT names and internal team topology must never leak to consumers. Enforced by brand canon §15.

**Evidence:**
- `docs/brand/` brand canon + media kit — DEMA as "trusted companion, disciplined guide."
- `memory/project_pat_sat_canonical_topology.md` — explicit PAT/SAT must stay internal.
- Claim scanner found **20 PROHIBITED-class claim patterns** across docs — not all PAT/SAT leaks but the pattern is active.

**Assessment:** ✅ **Doctrine clean. Enforcement partial.** The canon is right; the discipline in source docs needs a sweep (75 "production-readiness" matches suggest drift in internal docs that could later leak to consumer surfaces).

## 3. PAT×7 + SAT×5 canonical topology

**Source of truth:** `bizra-omega/bizra-core/src/topology_canon.rs` (explicit canonical names).
**Mint functions:** `bizra-omega/bizra-resourcepool/src/genesis.rs`.
**Gateway wiring:** FLAGGED as partially wired per memory `project_pat_sat_canonical_topology.md`.

**Assessment:** ⚠️ **Watchlist.** Canon is defined; mint functions exist; gateway wiring is partial. This is tracked in Node0 closure scoreboard, not a new finding — but it is a visible WIRED_PARTIAL row that blocks Tier-D.

## 4. URP — shared constitutional / world layer

**Architecture:** URP = Universal Resource Pool — the shared constitutional resource substrate across nodes. Offline reconciliation: `AwaitingReconciliation → UrpValidating → Complete` in `bizra-mission` state machine.

**Assessment:** ✅ **Present in code, minimal in docs.** Receipt-state-machine ladder is implemented. URP-as-world-layer is more fully articulated in strategy docs than in consumer docs — appropriate given brand canon §15 claim discipline.

## 5. P2 / Receipts cockpit

**Architecture:** P2 = the Node0-visible cockpit where receipts, trust, and chain head are surfaced. `services/node_gateway/app/routers.py` + `/v1/chain` endpoint + Dema trust-surface binding.

**Evidence:**
- `memory/project_node0_closure_row6_trust_surface.md` — Dema web face now reads authoritative chain head via proxy at `/v1/chain`.
- `memory/project_mission_receipt_full_payload_signature_2026_04_23.md` — PR #50 closes weak-signature vuln (Ed25519 signs full body).

**Assessment:** ✅ **Wired.** Trust surface reads live chain head; no shadow state.

## 6. Local-first runtime

**Evidence:**
- `bizra-omega/bizra-node/src/substrate/` — cross-platform resource discovery, local LLM runtime detection (Ollama, LM Studio, HuggingFace, standalone GGUF).
- `bizra_config.py` — tiered LLM inference: LM Studio → Ollama → cloud fallback.

**Assessment:** ✅ **Architecturally intact.** Local-first is a real design primitive, not marketing.

**Tension with public "no cloud" claim:** 12 POSTGRES_URL_WITH_PASSWORD findings in `deploy/`, `runtime/`, `tools/` show Postgres is contemplated for multi-node / cloud-optional deployments. The correct framing is "**your node runs locally; you choose what (if anything) syncs upstream**" — not "never any cloud."

## 7. Cloud-optional design

**Architecture:** `services/` contains the multi-node / gateway / API components. Optional — core Node0 binary (`bizra-omega/bizra-node/`) does not require them.

**Assessment:** ✅ **Design-clean, consumer-copy-drift.** Architecture is right. Public copy overclaims the "no cloud" absoluteness. See `WEBSITE_PUBLIC_CLAIMS_AUDIT.md §C1`.

## 8. Data-lake / Foundry / canon-pack separation

**Doctrine:** Three separated stores — data-lake corpus (`00_INTAKE` → `04_GOLD`), Cognitive Foundry staging (review workbooks + canon packs), BIZRA runtime canon (`MEMORY.md`, `constants.py`, `topology_canon.rs`). **No automatic path between them.** Canon Store Ingestion Gate is the explicitly required human-gated tool that does not yet exist.

**Evidence:**
- `tools/cognitive_foundry/claude_lane/canon_packs/README.md` — "none of these packs are BIZRA canon. They are *candidates for* canon, pending a separate human-gated ingestion step that does not yet exist."
- `tools/cognitive_foundry/claude_lane/REVIEW_HANDOFF.md` — 27-entry preferred pack, explicitly labeled `non_promotion_tool` + `human_gated`.

**Assessment:** ✅ **Cleanest separation in the architecture.** This is a golden gem. Preserve it.

## 9. No-runtime-canon-contamination invariant

**Enforcement:**
- `promote.py` writes to `canon_packs/`; no code path writes to `MEMORY.md` or `constants.py` or `topology_canon.rs`.
- Preferred pack manifest says `non_promotion_tool: true`.
- Canon pack disposition docs reiterate "Canon Store Ingestion Gate is required boundary before runtime canon."

**Assessment:** ✅ **Invariant intact.** No auto-ingestion path exists. The ingestion gate is the explicit single-point human gate before runtime contact.

**Watchlist:** As the repo grows, someone might accidentally build an ingestion tool without naming it the gate. Recommendation: pre-register the gate's expected location (e.g., `tools/canon_store/ingestion_gate.py`) and add a CI check that no other file writes to `MEMORY.md` / `constants.py` / `topology_canon.rs` without specific sign-off.

---

## Architecture-level summary

| Element | State | Evidence confidence |
|---|---|---|
| Node0 doctrine | ✅ | HIGH |
| DEMA | ✅ | HIGH |
| PAT×7 + SAT×5 | ⚠️ gateway wiring partial | MEDIUM (scoreboard tracks) |
| URP | ✅ | MEDIUM |
| P2 cockpit | ✅ | HIGH |
| Local-first runtime | ✅ | HIGH |
| Cloud-optional | ✅ | HIGH (but consumer copy drifts) |
| Foundry ↔ runtime separation | ✅ golden gem | HIGH |
| No-runtime-canon-contamination | ✅ | HIGH |

## Architecture debts (actionable)

| # | Debt | Severity | Action | Owner |
|---|---|---|---|---|
| AD1 | `.unwrap()` hot-path audit in receipt/mission crates | MEDIUM | Targeted review + graceful degradation policy | runtime lead |
| AD2 | PAT/SAT gateway wiring — row in Node0 scoreboard | MEDIUM | Complete wiring + test | runtime lead |
| AD3 | Consumer "no cloud" copy must reflect cloud-optional architecture | MEDIUM | Site rewrite per `CLAIM_SAFE_LAUNCH_COPY.md` | operator |
| AD4 | Canon Store Ingestion Gate spec does not exist | MEDIUM | Spec-first design (separate lane, typed-auth) | architecture lead |
| AD5 | CI check that MEMORY.md / constants.py / topology_canon.rs cannot be written to without sign-off | LOW | Add CODEOWNERS + branch-protection review | repo-ops |
