# BIZRA Pinnacle Roadmap — Agent Runbook v3.0
**Generated:** 2025-12-19 06:06 (Dubai, UTC+4)  
**Goal:** move from sealed single-node sovereignty → dual-agentic autonomy (PAT/SAT) → trusted “Glass Cockpit” → data refinery flywheel → scalable federation.

## 0) Non‑negotiable Axioms (Ihsān hard constraints)
- **No unsealed code paths:** anything running must be tracked + taggable.
- **Fail‑closed governance:** SAT must be able to **reject** and quarantine (no “always approve”).
- **Receipts everywhere:** every meaningful action produces an auditable evidence receipt.
- **Targets vs reality:** performance numbers are labeled **measured** or **design target** (never mixed).

---

## 1) Where you stand (trust snapshot)
### Sealed baseline
- **v1.0.1-genesis-citadel** exists and includes Citadel + Kernel + SAPE artifacts (clean-room sealable).

### Stabilization branch status (expected in Dual-Agentic repo)
- SAT validators implemented (VETO model), negative tests exist, MCP/A2A allowlists+timeouts, CI performance gates, ADR-007.

---

## 2) Architecture Map → Execution Map (what’s real vs planned)
| Layer | Mind-map claim | Reality now (evidence-backed) | Action to make it “real” |
|---|---|---|---|
| L0 Seed/Ihsān | Ethical constitution & invariants | Present in docs + Ihsān vector module | Enforce Ihsān gate at runtime + CI |
| L1 Knowledge | Hypergraph/Neo4j/RAG | Partial (artifacts + plan) | Turn on Neo4j + ingest + retrieval tests |
| L2 Compute (URP) | Auto-scaling resource pool | Partial | Implement local URP leases + GPU awareness |
| L3 PoI/Consensus | BlockGraph + finality | Mostly planned | Keep single-node receipts; defer multi-node |
| L4 Resilience Mesh | SLOs, recovery, chaos | Partial | Add chaos tests + SLO burn rates in cockpit |
| L5 PAT/SAT | Dual agentic orchestration | PAT local models exist; SAT core now meaningful | Wire request flow + veto + receipts |
| L6 Economy/Gov | Dual token + governance | Anchors exist (genesis/tokenomics docs) | Keep as spec; implement minimal PoI ledger first |
| L7 Observability | OTel dashboards / cockpit | Partial | Build Glass Cockpit (local-first) |

---

## 3) Step-by-step action chain (local agent should follow)
Each step is **deterministic** with acceptance criteria.

### Phase A — Stabilize & Seal (P0: safety before features)
**A1 — Merge stabilization to main**
- `git checkout feature/critical-stabilization`
- Run full CI locally (or GH Actions), then open PR → merge.
- **Accept:** SAT rejection tests pass; MCP/A2A timeouts enforced; CI evidence receipts produced.

**A2 — Tag stabilization**
- `git tag -a v1.2.0-stabilization -m "SAT veto + negative tests + adapters + CI receipts"`
- **Accept:** tag points to merged commit; `git verify-tag` (if GPG used) passes.

**A3 — Confirm clean-room seal replay**
- Checkout `v1.0.1-genesis-citadel`
- `pwsh .\scripts\genesis_ignite_all.ps1`
- **Accept:** `/healthz` + `/livez` return OK; receipt file minted.

---

### Phase B — Dual Agentic Activation (PAT/SAT becomes operational)
**B1 — PAT-7 runtime contract**
- Implement a single **Agent Router** process with:
  - agent registry (7 agents)
  - tool adapter interface (MCP/A2A behind allowlists)
  - session memory (local)
- **Accept:** `pat bench` shows p95 latency < 500ms on canned prompts (local-only).

**B2 — SAT-5 runtime contract**
- SAT must expose:
  - `validate(request) -> {allow|reject, reasons, score}`
  - `quarantine(event)` wired to FATE
  - `receipt(action)` always-on
- **Accept:** introduce a known-bad prompt → SAT rejects + quarantine receipt exists.

**B3 — Wire PAT ↔ SAT**
- All PAT responses must pass SAT veto *before* returning to user.
- **Accept:** same request yields different outcomes when SAT policy changes, and every run produces receipts.

---

### Phase C — Glass Cockpit (trust layer becomes visible)
**C1 — Metrics spine**
- Emit structured events:
  - `ihsan_score`, `sat_decision`, `tool_timeout`, `urp_lease`, `rag_hit`
- **Accept:** events visible in logs + Prometheus scrape works.

**C2 — Cockpit UI**
- Minimal panels:
  - Ihsān score trend
  - SAT allow/reject rate
  - Tool timeout rate
  - URP resource utilization (GPU/CPU/RAM)
  - SLO burn (p95 latency)
- **Accept:** local dashboard loads and reflects live signals.

---

### Phase D — Data Refinery Flywheel (toward BIZRA family models)
**D1 — Inventory & dedupe**
- Create a unified **manifest**: size, hash, location, duplicates.
- **Accept:** duplicates report + “single source of truth” folder.

**D2 — Index & Lexicon Ledger**
- Extract terms, concepts, decisions, ADRs into Lexicon Ledger.
- **Accept:** lexicon export + concept graph edges validated.

**D3 — Instruction synthesis**
- Generate QA pairs from curated chunks.
- **Accept:** 10k+ pairs, 500 human-reviewed sample, Ihsān-labeled subset.

**D4 — LoRA v0.1**
- Train `bizra-reasoner-7b` adapter locally.
- **Accept:** eval shows improvement on your internal regression set.

---

## 4) Hardware policy module (where your TS snippet fits)
- Treat regulatory thresholds as **configurable policies**, not hard-coded truth.
- Rename for clarity:
  - `TRAINING_TOTAL_TFLOP_THRESHOLD_*` (total compute) vs `tflops` (rate).
- Add dtype fields: `tflops_fp32`, `tflops_fp16_tensor`, `tflops_int8` (inference is not training).
- Add your real Node0 SKU explicitly: **RTX 4090 Laptop (16GB)**.

---

## 5) Definition of “PAT/SAT ACTIVE”
PAT/SAT is “active” only when all are true:
1) PAT router running + all 7 agents reachable (local).  
2) SAT can reject and quarantine (negative tests + live test).  
3) Ihsān gate enforced (block on fail).  
4) Evidence receipts minted for each request.  
5) Glass Cockpit shows live signals for (2–4).  

---

## 6) Immediate next TODO (single-day)
1) Merge stabilization branch → main.  
2) Tag `v1.2.0-stabilization`.  
3) Add the SAT→FATE quarantine wire in runtime (not just tests).  
4) Ship Cockpit **MVP** (Ihsān trend + SAT veto rate + p95 latency).  
