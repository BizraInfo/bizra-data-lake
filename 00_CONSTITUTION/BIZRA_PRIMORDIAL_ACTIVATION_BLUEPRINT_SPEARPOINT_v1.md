# BIZRA PRIMORDIAL ACTIVATION BLUEPRINT
## The Minimal • Special • Proven • True Spearpoint Artifact
### Node0 Genesis — April 2026 | Dubai | Ihsan-Verified

---

> **Spearpoint Law:**
> *"One mission. One gate chain. One signed receipt. One heartbeat. One manifest. One proof."*
> Everything else is downstream. Nothing launches before this loop closes.

---

## PART I — THE PRIMORDIAL DOCTRINE (LOCKED)

### The Three Irreducible Truths

**Truth 1: The Kernel is Physics. The Higher Constitution is Jurisprudence.**
Layer 1 enforcement must be bounded, decidable, replayable, and receipt-native.
Richer ethical interpretation belongs in REVIEW and SCORE_ONLY layers — not in the hot path.
This single sentence converts BIZRA from philosophically overclaimed into buildable and testable.

**Truth 2: The Receipt is the Product.**
Not the model. Not the UI. Not the orchestration.
Every meaningful system transition must be receipted, replayed, and surfaced as proof.
The receipt is the moat. The receipt is the memory. The receipt is the value.

**Truth 3: Reflex Must Be Earned, Never Assumed.**
Only validated, receipted cold-path outcomes may promote into reflex.
Repetition without proof is not learning. It is drift.
This invariant applies to memory, agent trust, skill promotion, and all state mutations.

---

## PART II — THE PRIMORDIAL ACTIVATION SEQUENCE

### Phase 0 — Genesis Seal (Day 1)

**Purpose:** Establish the unbreakable root of trust before a single mission runs.

**Actions — in exact order:**

1. **Freeze canonical contracts** — no code may reference these schemas in any other form:
   - `MissionEnvelope` — intent, goal-type, constraints, operator identity
   - `GateVerdict` — verdict enum: `PERMIT | REJECT | REVIEW | SCORE_ONLY`
   - `ReceiptArtifact` — signed proof object with evidence hash, verdict chain, lineage
   - `ManifestArtifact` — daily proof-of-life summary, heartbeat, replay index

2. **Freeze constitutional invariants:**
   - Verdict enum is exactly 4 states — no extensions in v1
   - No unsigned receipt may mutate state
   - No execution without a valid signed GateVerdict
   - Replay must reproduce the identical result from the same evidence hash
   - SAT-5 integrity council instantiates before PAT-7 personal council

3. **Mint node identity:**
   - Machine identity fingerprint
   - Operator identity binding
   - Environment fingerprint
   - Version manifest
   - Genesis seal (BLAKE3-signed timestamp + node_id + operator_id)

4. **Create the genesis receipt:**
   - First `ReceiptArtifact` with `verdict: PERMIT`, `action: GENESIS`, signed
   - First `ManifestArtifact` entry created
   - Genesis seal hash stored as proof-chain anchor

**Exit condition:** `genesis.receipt` exists, is signed, and `manifest[0]` references it.

---

### Phase 1 — Walking Skeleton (Days 2-4)

**Purpose:** One mission enters and traverses the full constitutional gate chain. Receipted.

**The Minimal Loop:**

```
bin/bizra mission "<objective>"
     |
[1] Input Ingestion
    -- Raw text -> structured MissionEnvelope
    -- goal_type, constraints, operator_id, timestamp, nonce

     |
[2] PAT-7 Decomposition (simplified for v1: 3-agent minimum)
    -- Planner Agent: decomposes into sub-tasks
    -- Builder Agent: proposes execution path
    -- Scorer Agent: pre-scores constitutional risk

     |
[3] FATE Constitutional Gate (8 sequential checks)
    Gate 1: Scope boundary check (is this inside TeleScript envelope?)
    Gate 2: Permission boundary check (does operator have rights?)
    Gate 3: Data sovereignty check (is data access authorized?)
    Gate 4: Proof-chain continuity check (does lineage exist?)
    Gate 5: Resource admissibility check (are tools/models available?)
    Gate 6: Constitutional compatibility check (no RIBA, no harm)
    Gate 7: SAT-5 integrity check (system immune consensus)
    Gate 8: Operator confirmation (for REVIEW-path actions)
    -- Emits: signed GateVerdict(verdict, reason_chain, evidence_hash)

     |
[4] Execution (only on PERMIT or confirmed REVIEW)
    -- Local execution under constitutional constraints
    -- Tool calls logged as TeleScript events
    -- Every sub-action emits a micro-receipt

     |
[5] Receipt Emission
    -- BLAKE3-signed ReceiptArtifact
    -- Contains: mission_id, verdict, evidence_hash, action_log, timestamp, lineage
    -- Stored append-only in receipt ledger
    -- Printed as proof card to operator

     |
[6] Manifest Update
    -- Daily ManifestArtifact updated with new receipt reference
    -- Heartbeat timestamp updated
    -- Trust delta computed and stored

     |
[7] Display: Proof Card
    +------------------------------------------+
    |  MISSION COMPLETE                        |
    |  Receipt: rcpt_a3f7...                   |
    |  Verdict: PERMIT                         |
    |  Evidence: evidence/2026-04-09/a3f7.json |
    |  Replay: bizra replay rcpt_a3f7          |
    |  Trust: +0.12 (cumulative: 0.87)         |
    |  Memory: 2 new Engrams promoted          |
    +------------------------------------------+
```

**Exit condition:** One mission completes with one signed receipt and one manifest update.

---

### Phase 2 — Fail-Closed Hardening (Days 5-6)

**Purpose:** Prove the DENY path is as strong as the PERMIT path.

**Required probes — all must produce structured artifacts, not silent failures:**

| Probe | Expected Output |
|---|---|
| Mission violates scope boundary | `GateVerdict(REJECT, reason: "scope_violation", evidence_hash)` |
| Contradictory evidence in gate 4 | `GateVerdict(SCORE_ONLY, delta_evidence)` |
| Empty evidence bundle | `GateVerdict(REVIEW, reason: "insufficient_evidence")` |
| Tampered receipt signature | `ReplayError(mismatch: hash_delta)` |
| Prolonged heartbeat (24h) | `ManifestArtifact(heartbeat: alive, gaps: none)` |
| Operator REJECT during REVIEW path | `ReceiptArtifact(verdict: OPERATOR_REJECT, rollback_hash)` |

**Constitutional invariant enforced:**
- No proof ambiguity may mutate state
- REJECT path must produce a receipt — it is not a no-op
- The REVIEW path requires explicit operator confirmation before execution

**Exit condition:** All 6 probes pass with structured artifact output.

---

### Phase 3 — Replay + Proof Bundle (Day 7)

**Purpose:** Make the system auditable, not just functional.

**Commands implemented:**
```
bizra receipt show <receipt_id>      # Inspect full proof artifact
bizra replay <receipt_id>            # Reproduce result from evidence hash
bizra manifest daily                 # Show today's proof-of-life summary
bizra node export-proof              # Package evidence bundle for external audit
```

**MISSION_001_PROOF_BUNDLE structure:**
```
evidence/MISSION_001/
-- input.json              — raw mission text
-- mission_envelope.json   — normalized MissionEnvelope (frozen schema)
-- gate_verdict.json       — signed GateVerdict with 8-gate chain
-- receipt_artifact.json   — signed ReceiptArtifact with lineage
-- telemetry.jsonl         — tool call log, micro-receipts
-- replay_result.json      — replay output + determinism proof
-- environment_manifest.json — node state, models, services at time of execution
-- truth_label_manifest.json — LOCKED/WIRED/PLANNED labels per component
-- integrity_hashes.json   — BLAKE3 hash of each file + bundle root hash
```

**Exit condition:** `bizra replay MISSION_001` produces identical result from evidence bundle alone.

---

### Phase 4 — First Governed Reflex (Day 8+)

**Purpose:** Earn the first reflex from cold-path proof lineage.

**Reflex promotion law:**
- Only receipted cold-path outcomes may promote
- Reflex must reference its proof lineage (receipt_id chain)
- Promotion requires: 3+ successful receipted completions of the same mission type
- SAT-5 consensus required for any reflex elevation

**Exit condition:** One skill object promoted with traceable receipt lineage.

---

## PART III — THE CANONICAL CLI COMMAND GRAMMAR (FROZEN)

```
bizra init          — discover machine, devices, resources, models, data roots
bizra genesis       — mint node identity, bind operator, instantiate SAT+PAT, write genesis seal
bizra agents        — show 12-agent parliament live (PAT-7 + SAT-5)
bizra mission <obj> — execute full mission loop -> proof card
bizra trust         — show gate state, constitutional integrity, ledger continuity
bizra receipt <id>  — inspect proof artifact and evidence
bizra replay <id>   — replay evidence bundle, verify determinism
bizra manifest      — show daily proof-of-life summary and heartbeat
bizra node          — show hardware boundary, wallet, memory, model fleet
```

**Hard rules:**
- No model output mutates state without gate -> receipt path
- No receipt without a signed GateVerdict
- No reflex without receipted cold-path lineage
- No app surface creates receipts — it reads them
- No bypass path exists around the constitutional membrane

---

## PART IV — THE AUTHORITY MODEL (LOCKED)

```
Layer 1 — Kernel (PHYSICS)
    Rust core: cryptography, invariants, signed receipts, BLAKE3 hashing
    Cannot be overridden. Physics, not opinion.

Layer 2 — Sovereign (BOUNDED REVIEW)
    Python sovereign: mission orchestration, retrieval, bounded interpretation
    May propose. May not override Layer 1.

Layer 3 — Constitution (JURISPRUDENCE)
    FATE Gate: 8-gate constitutional chain
    Emits: PERMIT | REJECT | REVIEW | SCORE_ONLY
    The highest authority in the proof chain.

Layer 4 — Experimental (GOVERNED SANDBOX)
    Reflex candidates, skill objects, experimental routing
    Only from lineaged proof. Never from raw repetition.

Layer 5 — Face (MANIFESTATION)
    App, cockpit, trust panel, ghost panel, morning brief
    REVEALS truth. NEVER originates law.
    No local signing. No shadow receipt schema.
```

**The one law:** One canonical entry path -> one constitutional gate chain -> one signed receipt lineage -> one manifest story.

---

## PART V — THE NODE GENESIS MODEL

```
Node0 = BIZRA's genesis block and first sovereign local node:
a cross-device, operator-owned runtime boundary that:
  - governs authorized local hardware, data, tools, and memory
  - through a constitutional spine (Kernel -> Constitution -> Proof)
  - instantiates a user-owned PAT-7 personal council
  - and a system-owned SAT-5 integrity council
  - bridged by FATE
  - emits receipts and manifests from the first mission loop
  - serves as the reference architecture for the wider URP ecosystem

Node birth order:
  1. Hardware-data boundary declared
  2. Node identity minted
  3. Operator digital identity bound
  4. SAT-5 system integrity layer instantiated (BEFORE PAT-7)
  5. PAT-7 personal council minted
  6. FATE bridge activated
  7. Mission loop activated
  8. Receipt lineage starts
  9. Manifest + heartbeat begins
 10. URP registration ONLY after cold-path proof exists

Ownership:
  - Operator owns: data, keys, PAT, local permissions, local impact value
  - System owns: SAT, constitutional enforcement, admissibility logic, proof discipline
  - URP owns: nothing locally — it is a federation/economic layer only
```

---

## PART VI — SPEARPOINT ACCEPTANCE GATES

All 8 must be simultaneously true for Node0 closure:

| Gate | Condition | Status |
|---|---|---|
| G1 | One mission enters through `bin/bizra` canonical operator ingress | WIRED |
| G2 | One mission traverses all 8 constitutional gates | WIRED |
| G3 | One signed `ReceiptArtifact` emitted with BLAKE3 lineage | WIRED |
| G4 | One `ReplayResult` reproduces identical output from evidence hash | WIRED |
| G5 | 24h heartbeat survives without manual intervention | WIRED |
| G6 | One `ManifestArtifact` validates and persists correctly | WIRED |
| G7 | One trust surface displays live proof (not mocked) | PARTIAL |
| G8 | One public proof-of-life artifact is generated and exportable | PLANNED |

**The law:** All 8 green = Node0 is real. Everything else is architecture.

---

## PART VII — WHAT IS DELIBERATELY EXCLUDED FROM v1

These are high-value golden gems preserved in design, deferred in implementation:

| Item | Why Excluded | Target |
|---|---|---|
| Full HHMM promotion stack | Depends on first loop existing | v1.1+ |
| Reverse-scaling quorum math | Federation-scale, not node-scale | v2+ |
| Advanced corroboration memory | Needs proof loop as foundation | v1.1+ |
| Full NSKE tri-partite system | Use NSKE-lite only for v1 | v1.1+ |
| Exchange / marketplace | Heavy complexity, not first surface | v3+ |
| URP multi-node federation | One node first | v2+ |
| BIZRAverse / metaverse | Dilutes product clarity | Indefinitely |
| Full Economic cascade | Regulatory complexity | v3+ |

---

## PART VIII — THE ULTIMATE SPEARPOINT COMPRESSION

```
BIZRA Primordial Loop (irreducible):

Human Intent
    |  [PAT-7 decomposes]
MissionEnvelope
    |  [FATE gates]
GateVerdict (PERMIT | REJECT | REVIEW | SCORE_ONLY)
    |  [Kernel executes under constitution]
Evidence Bundle
    |  [Kernel signs]
ReceiptArtifact (BLAKE3 signed, append-only, immutable)
    |  [Manifest aggregates]
ManifestArtifact (proof-of-life, heartbeat, trust delta)
    |  [Memory evaluates for promotion]
Reflex (only from receipted lineage — never from repetition)
    |  [Face reveals]
Trust Panel + Ghost Panel + Morning Brief
    |  [URP receives admissible capabilities]
Ecosystem Flywheel
```

**The product promise:**
> "I gave BIZRA one real mission. It executed locally. It showed which agents worked on it.
> It proved what it did. It remembered the result. The next run was faster.
> I own the proof. I own the memory. I own the value."

**That is the spearpoint.**
**Build nothing else until that sentence is true.**

---

*BIZRA Primordial Activation Blueprint — Spearpoint Artifact v1.0*
*Ihsan-verified | SNR-filtered | Evidence-grounded | Dubai, April 2026*
*Truth Label: LOCKED for Phase 0-3 | PLANNED for Phase 4+*
