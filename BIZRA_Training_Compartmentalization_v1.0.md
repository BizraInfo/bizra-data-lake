# BIZRA Peak Training & Fine‑Tuning Architectural Compartmentalization (v1.0)

**Scope:** Local fine‑tuning for `bizra-reasoner-7b` + `bizra-planner-7b` on Node0 (RTX 4090 16GB, 128GB RAM).  
**Hard Constraint:** *Sealable, sovereign, reversible.*  
**Ihsān Rule:** *No hidden assumptions; every training run produces an evidence receipt.*

---

## T0–T9 Compartment Grid

| ID | Compartment | What runs here | Trust | Allowed I/O | Key gates |
|---:|---|---|---|---|---|
| **T0** | Sovereign Host | OS, secrets vault, disks | **Highest** | local only | disk encryption, least‑privilege users |
| **T1** | Data Vault | raw corpus snapshots | High | read‑mostly | content hashing, immutability tags |
| **T2** | Data Refinery | parsing, chunking, dedup | Medium | T1→T3 | PII redaction, license filters, “deny unknown” |
| **T3** | Instruction Foundry | QA pairs / ShareGPT/Alpaca JSONL | Medium | T2→T4 | quality scoring + human spot checks |
| **T4** | Training Sandbox | Unsloth/Axolotl, LoRA/QLoRA runs | Low | T3→T5 | resource leases (URP), timeouts, deterministic seeds |
| **T5** | Eval Arena | benchmarks + adversarial suites | Medium | T4→T6 | negative tests mandatory; regression checks |
| **T6** | Model Registry | GGUF/adapters, versioning | High | T5→T7 | signed artifacts; provenance manifest |
| **T7** | Inference Runtime | Ollama/LM Studio models used by PAT | Medium | T6→PAT | SAT veto on tool‑calls; FATE risk routing |
| **T8** | Governance & Evidence | receipts, ADRs, policy versions | High | everywhere→T8 | signatures, append‑only log |
| **T9** | Glass Cockpit | dashboards + alerts | Medium | read‑only | SLO + drift alarms; audit views |

**Rule of thumb:** data moves **rightward** (T1→…→T7). Anything moving **left** must be explicitly justified and logged.

---

## Minimal “Sealable Training Run” Protocol

1. **Snapshot input**: T1 creates `dataset_snapshot_<date>.manifest.json` (hash list).  
2. **Refine** (T2): dedup + normalize + redact; output `refined_<id>/` + hashes.  
3. **Synthesize instructions** (T3): generate `bizra_instruct_v1.jsonl` with a **quality score** field per row.  
4. **Train** (T4): LoRA/QLoRA with **URP leases** (VRAM/RAM/time).  
5. **Evaluate** (T5): run *negative* tests + task suites; reject if regressions.  
6. **Register** (T6): sign model artifacts + write `MODEL_CARD.md`.  
7. **Deploy** (T7): swap PAT agent models via allowlisted mapping only.  
8. **Receipt** (T8): write a single receipt linking (snapshot → run config → metrics → artifacts).

---

## Training‑Specific Invariants (must hold)

- **I‑T1 (Provenance):** Every training sample references a source hash + origin category.
- **I‑T2 (Dedup):** No near‑duplicate rows across train/valid/test splits.
- **I‑T3 (Ethics):** Any sample that violates Ihsān is quarantined; counts are tracked.
- **I‑T4 (Repro):** Run config + seed + dataset snapshot uniquely reproduce metrics within tolerance.
- **I‑T5 (No silent regressions):** If eval fails → model is not registered.
- **I‑T6 (Sealing):** All scripts/configs used in training are tracked in git + referenced by commit SHA.

---

## Compartmentalization for `bizra-reasoner-7b` vs `bizra-planner-7b`

### `bizra-planner-7b`
- **Training diet:** JSON plans, tool schemas, constraints, failure handling, “refuse/ask clarification” patterns.
- **Primary eval:** plan validity, schema compliance, step minimality, tool‑call success rate.

### `bizra-reasoner-7b`
- **Training diet:** BIZRA architecture reasoning, safety proofs, ADR rationale, “why” chains, risk analysis.
- **Primary eval:** correctness under constraints, fewer hallucinated interfaces, better conflict resolution.

---

## What to implement first (fastest value)

1. **Receipts everywhere (T8):** one schema for data snapshot + training + eval.
2. **Eval Arena (T5):** regression gates + adversarial suite (negative tests).
3. **Registry discipline (T6):** signed artifacts + immutable mapping to PAT.

