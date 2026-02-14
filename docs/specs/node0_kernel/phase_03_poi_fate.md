# Phase 03: PoI Engine + FATE Gates

Last updated: 2026-02-14
Standing on: Shannon (1948, Information Theory) · Al-Ghazali (Ihsan) · Floyd (1967, Assertions) · Ostrom (1990, Commons)

---

## Purpose

Layer 3 answers **"Was this contribution valuable and ethical?"** Every inference, every artifact, every transaction passes through a multi-stage scoring pipeline (PoI) and a formal verification chain (FATE gates). Nothing enters the permanent ledger without a signed receipt proving it passed both.

This layer enforces two invariants simultaneously:
- **Quality:** Signal-to-Noise Ratio >= 0.85 (Shannon)
- **Ethics:** Ihsan score >= 0.95 (Al-Ghazali)

---

## Data Structures

### PoI Receipt

```pseudocode
STRUCT PoIReceipt:
    receipt_id:       Hash             # BLAKE3(canonical_json(self))
    artifact_hash:    Hash             # What was scored
    contributor:      NodeId           # Who contributed
    epoch_id:         String           # Temporal anchor
    stage_scores:     StageScores      # Per-stage breakdown
    composite_poi:    f64              # Final weighted score (0..1)
    reason_code:      PoIReasonCode    # ACCEPT | REJECT | PENALTY
    reason_detail:    String           # Human-readable explanation
    signature:        Ed25519Signature # Signed by scoring node
    timestamp:        Timestamp
    prev_receipt:     Hash             # Chain linkage

STRUCT StageScores:
    contribution:   f64   # Stage 1: Did the work happen? (signature + SNR)
    reach:          f64   # Stage 2: How widely cited? (PageRank)
    longevity:      f64   # Stage 3: Will it last? (temporal decay)
    composite:      f64   # Stage 4: Weighted combination

ENUM PoIReasonCode:
    ACCEPT           # Passed all stages
    ACCEPT_ELEVATED  # Passed with above-threshold scores
    REJECT_SNR       # Below SNR threshold
    REJECT_IHSAN     # Below Ihsan threshold
    REJECT_UNSIGNED  # Missing or invalid signature
    REJECT_DUPLICATE # Already scored in this epoch
    PENALTY_GAMING   # Citation ring or Sybil detected
    PENALTY_DECAY    # Temporal decay reduced score to zero
```

**Source:** `core/proof_engine/poi_engine.py:PoIReceipt` (1,100+ lines)

### FATE Gate Result

```pseudocode
STRUCT FATEResult:
    passed:           bool
    gate_scores:      Map<GateName, GateScore>
    rejection_reason: Option<String>
    receipt:          GateReceipt       # Signed proof of verification
    elapsed_ms:       f64              # Total gate chain time

STRUCT GateScore:
    gate_name:    GateName
    passed:       bool
    score:        f64        # 0..1
    tier:         GateTier   # CHEAP | MEDIUM | EXPENSIVE
    elapsed_ms:   f64

ENUM GateName:
    SCHEMA_GATE       # Input structure validation
    PROVENANCE_GATE   # Source verification (signature check)
    SNR_GATE          # Signal-to-noise threshold
    CONSTRAINT_GATE   # Z3 formal + Ihsan threshold
    SAFETY_GATE       # Constitutional safety check
    COMMIT_GATE       # Final resource allocation decision

ENUM GateTier:
    CHEAP      # <10ms   — Schema, Signature, Timestamp, Replay
    MEDIUM     # <150ms  — Ihsan, SNR, Policy
    EXPENSIVE  # <2000ms — Z3 SMT solving, full FATE scoring
```

**Source:** `core/proof_engine/gates.py`, `core/pci/gates.py:PCIGateKeeper` (224 lines)

### Z3 FATE Constraint

```pseudocode
STRUCT Z3Constraint:
    name:        String
    formula:     Z3Formula      # SMT-LIB expression
    variables:   List<Z3Var>    # Symbolic variables used

STRUCT Z3Proof:
    satisfiable:     bool
    model:           Option<Map<String, Value>>   # Satisfying assignment
    counterexample:  Option<Map<String, Value>>   # Violation example
    solver_time_ms:  f64

# Default Z3 variables:
#   ihsan:          Real ∈ [0, 1]
#   snr:            Real ∈ [0, 1]
#   risk_level:     Int  ∈ {0=low, 1=medium, 2=high}
#   reversible:     Bool
#   human_approved: Bool
#   cost:           Real >= 0

# Default constraints:
#   ihsan >= 0.95
#   snr >= 0.85
#   risk_level == 2 => (reversible OR human_approved)
#   cost <= budget
```

**Source:** `core/sovereign/z3_fate_gate.py:Z3FATEGate` (184 lines)

---

## Procedures

### 4-Stage PoI Scoring Pipeline

```pseudocode
PROCEDURE compute_poi(
    artifact: Artifact,
    citation_graph: CitationGraph,
    config: PoIConfig,
) -> PoIReceipt:
    # Standing on: Shannon (Information Theory), Ostrom (Commons Management)

    # ─── STAGE 1: CONTRIBUTION VERIFICATION ───
    # Did the work actually happen? Is it signed? Is the SNR sufficient?

    IF NOT verify_signature(artifact.signature, artifact.contributor):
        RETURN PoIReceipt(reason=REJECT_UNSIGNED)

    snr = compute_snr(artifact.content)
    IF snr < SNR_THRESHOLD:   # 0.85
        RETURN PoIReceipt(reason=REJECT_SNR, composite=0.0)

    # Information density (Reverse Scaling metric)
    # Standing on: Kolmogorov (algorithmic complexity)
    compressed = lz4_compress(artifact.content)
    compression_ratio = len(compressed) / len(artifact.content)
    information_density = 1.0 - compression_ratio
    contribution = (snr * information_density).clamp(0.0, 1.0)

    # ─── STAGE 2: NETWORK REACH ───
    # How widely is this artifact cited?
    # Standing on: Page & Brin (PageRank), Ostrom (network effects)

    centrality = citation_graph.get_centrality(artifact.hash)
    citation_count = citation_graph.get_citation_count(artifact.hash)
    reach = (centrality + citation_count / 100.0).clamp(0.0, 1.0)

    # ─── STAGE 3: TEMPORAL LONGEVITY ───
    # Will this artifact remain relevant?
    # Standing on: Ebbinghaus (forgetting curve)

    age_days = artifact.age_in_days()
    decay = exp(-config.decay_lambda * age_days)
    sustained = citation_graph.is_sustained_relevance(artifact.hash)
    longevity = IF sustained THEN decay * 1.3 ELSE decay
    longevity = longevity.clamp(0.0, 1.0)

    # ─── STAGE 4: COMPOSITE PoI ───
    # Weighted combination: alpha=0.5 (contribution), beta=0.3 (reach), gamma=0.2 (longevity)

    composite = config.alpha * contribution
              + config.beta  * reach
              + config.gamma * longevity

    # Generate signed receipt
    receipt = PoIReceipt(
        artifact_hash  = artifact.hash,
        contributor    = artifact.contributor,
        stage_scores   = StageScores(contribution, reach, longevity, composite),
        composite_poi  = composite,
        reason_code    = ACCEPT IF composite > 0.0 ELSE REJECT_SNR,
        signature      = node_identity.sign(canonical_json(receipt_data)),
    )

    RETURN receipt
```

**Source:** `core/proof_engine/poi_engine.py:PoIOrchestrator`

### Token Distribution from PoI

```pseudocode
PROCEDURE compute_token_distribution(
    audit: EpochAudit,
    epoch_reward: f64,
) -> Map<NodeId, f64>:
    # Standing on: Al-Ghazali (distributive justice), Shannon (information theory)

    # Step 1: Score all contributions in this epoch
    total_poi = 0.0
    node_scores = {}
    FOR EACH contribution IN audit.contributions:
        receipt = compute_poi(contribution, citation_graph, config)
        IF receipt.reason_code == ACCEPT:
            node_scores[contribution.contributor] = receipt.composite_poi
            total_poi += receipt.composite_poi

    IF total_poi == 0.0:
        RETURN {}   # No valid contributions this epoch

    # Step 2: Proportional distribution
    distribution = {}
    FOR EACH (node_id, score) IN node_scores:
        share = (score / total_poi) * epoch_reward
        distribution[node_id] = share

    # Step 3: Verify Gini constraint (ADL justice gate)
    gini = compute_gini_coefficient(distribution.values())
    IF gini > GINI_THRESHOLD:   # 0.40
        distribution = apply_gini_correction(distribution, GINI_THRESHOLD)

    RETURN distribution
```

**Source:** `core/proof_engine/poi_engine.py:compute_token_distribution()`

### 6-Gate Chain (Fail-Closed)

```pseudocode
PROCEDURE execute_gate_chain(input: GateInput) -> FATEResult:
    # Standing on: Lampson (1971) — fail-closed access control
    # Every gate must explicitly PASS. Default is REJECT.

    results = []

    # ─── GATE 1: SCHEMA (CHEAP, <10ms) ───
    schema_result = schema_gate.verify(input)
    IF NOT schema_result.passed:
        RETURN FATEResult(passed=false, rejection="Schema validation failed")
    results.append(schema_result)

    # ─── GATE 2: PROVENANCE (CHEAP, <10ms) ───
    # Verify Ed25519 signature + timestamp bounds + replay protection
    prov_result = provenance_gate.verify(input)
    IF NOT prov_result.passed:
        RETURN FATEResult(passed=false, rejection="Provenance check failed")
    results.append(prov_result)

    # ─── GATE 3: SNR (MEDIUM, <150ms) ───
    snr_result = snr_gate.verify(input)
    IF NOT snr_result.passed:
        RETURN FATEResult(passed=false, rejection="SNR below 0.85")
    results.append(snr_result)

    # ─── GATE 4: CONSTRAINT (EXPENSIVE, <2000ms) ───
    # Z3 SMT verification + Ihsan threshold
    constraint_result = constraint_gate.verify(input)
    IF NOT constraint_result.passed:
        RETURN FATEResult(passed=false, rejection=constraint_result.detail)
    results.append(constraint_result)

    # ─── GATE 5: SAFETY (MEDIUM, <150ms) ───
    safety_result = safety_gate.verify(input)
    IF NOT safety_result.passed:
        RETURN FATEResult(passed=false, rejection="Constitutional safety violation")
    results.append(safety_result)

    # ─── GATE 6: COMMIT (CHEAP, <10ms) ───
    # Final resource allocation decision
    commit_result = commit_gate.verify(input, results)
    IF NOT commit_result.passed:
        RETURN FATEResult(passed=false, rejection="Resource budget exceeded")
    results.append(commit_result)

    # All 6 gates passed — generate signed receipt
    RETURN FATEResult(
        passed       = true,
        gate_scores  = {r.gate_name: r for r in results},
        receipt      = sign_gate_receipt(results),
    )
```

**Source:** `core/proof_engine/gates.py:GateChain`, `core/pci/gates.py:PCIGateKeeper`

### Z3 FATE Verification

```pseudocode
PROCEDURE z3_verify(action: Action) -> Z3Proof:
    # Standing on: Floyd (1967) — assertion-based verification

    solver = Z3Solver()

    # Declare symbolic variables
    ihsan         = Real("ihsan")
    snr           = Real("snr")
    risk_level    = Int("risk_level")
    reversible    = Bool("reversible")
    human_approved = Bool("human_approved")
    cost          = Real("cost")

    # Add constitutional constraints
    solver.add(ihsan >= 0.95)
    solver.add(snr >= 0.85)
    solver.add(Implies(risk_level == 2, Or(reversible, human_approved)))
    solver.add(cost <= action.budget)

    # Bind concrete values from the action
    solver.add(ihsan == action.ihsan_score)
    solver.add(snr == action.snr_score)
    solver.add(risk_level == action.risk_level)
    solver.add(reversible == action.is_reversible)
    solver.add(human_approved == action.has_human_approval)
    solver.add(cost == action.estimated_cost)

    # Solve
    result = solver.check()

    IF result == SAT:
        model = solver.model()
        RETURN Z3Proof(satisfiable=true, model=model)
    ELSE:
        # Generate counterexample showing which constraint was violated
        core = solver.unsat_core()
        RETURN Z3Proof(satisfiable=false, counterexample=core)
```

**Source:** `core/sovereign/z3_fate_gate.py:Z3FATEGate` (184 lines)

### PCI Envelope Verification (Fast Path)

```pseudocode
PROCEDURE pci_gate_verify(envelope: PCIEnvelope) -> GateResult:
    # 3-tier gate chain for P2P message verification
    # Standing on: defense-in-depth principle

    # ─── TIER 1: CHEAP GATES (<10ms total) ───
    # These run on EVERY message — must be fast

    # 1a. Schema: required fields present, types correct
    IF NOT validate_schema(envelope):
        RETURN reject("Invalid schema")

    # 1b. Signature: Ed25519 verification
    IF NOT verify_signature(envelope.signer, envelope.payload, envelope.signature):
        RETURN reject("Invalid signature")

    # 1c. Timestamp: within clock skew tolerance (120s)
    age = abs(now() - envelope.timestamp)
    IF age > CLOCK_SKEW_TOLERANCE:   # 120 seconds
        RETURN reject("Timestamp outside tolerance")

    # 1d. Replay: nonce not seen before
    # Uses bounded LRU cache (10K entries) with TTL eviction
    IF nonce_cache.contains(envelope.nonce):
        RETURN reject("Replay detected")
    nonce_cache.insert(envelope.nonce, ttl=300s)

    # ─── TIER 2: MEDIUM GATES (<150ms total) ───
    # Only run if Tier 1 passes

    # 2a. Ihsan threshold
    IF envelope.metadata.ihsan_score < IHSAN_THRESHOLD:
        RETURN reject("Ihsan below threshold")

    # 2b. SNR threshold
    IF envelope.metadata.snr_score < SNR_THRESHOLD:
        RETURN reject("SNR below threshold")

    # 2c. Policy: check against sender's capability card
    IF NOT capability_check(envelope.signer, envelope.action):
        RETURN reject("Insufficient capabilities")

    RETURN accept()
```

**Source:** `core/pci/gates.py:PCIGateKeeper` (224 lines), `bizra-omega/bizra-core/src/pci/gates.rs` (320 lines)

---

## Ihsan Gate — 5-Dimensional Ethics Vector

```pseudocode
STRUCT IhsanVector:
    fidelity:      f64   # F: Truthfulness of output
    accountability: f64  # A: Attributable to contributor
    transparency:  f64   # T: Reasoning is inspectable
    ethics:        f64   # E: Does not cause harm
    ihsan:         f64   # I: Excellence (composite)

    METHOD compute_composite() -> f64:
        # Weighted: F(0.25) + A(0.20) + T(0.20) + E(0.20) + I(0.15)
        RETURN 0.25 * self.fidelity
             + 0.20 * self.accountability
             + 0.20 * self.transparency
             + 0.20 * self.ethics
             + 0.15 * self.ihsan

    INVARIANT: composite >= 0.95 for production
    INVARIANT: composite >= 0.99 for consensus/critical operations
    INVARIANT: composite >= 0.90 for CI tolerance
```

**Source:** `core/sovereign/ihsan_vector.py`, `core/proof_engine/ihsan_gate.py`

---

## TDD Anchors

| Test | File | Validates |
|------|------|-----------|
| `test_poi_determinism` | `tests/core/proof_engine/test_poi_determinism.py` | Same inputs produce identical PoI scores |
| `test_poi_engine_stages` | `tests/core/proof_engine/test_poi_engine.py` | 4-stage pipeline runs in order |
| `test_gate_chain_fail_closed` | `tests/core/proof_engine/test_gates.py` | Default is rejection |
| `test_snr_threshold` | `tests/core/proof_engine/test_gates.py` | SNR < 0.85 rejected |
| `test_ihsan_threshold` | `tests/core/proof_engine/test_snr_v1_ihsan_gate.py` | Ihsan < 0.95 rejected |
| `test_z3_fate_gate` | `tests/core/sovereign/test_z3_fate_gate.py` | Z3 solver returns SAT/UNSAT correctly |
| `test_pci_replay_protection` | `tests/core/pci/test_replay_protection.py` | Duplicate nonces rejected (19 tests) |
| `test_gate_chain_integration` | `tests/core/sovereign/test_gate_chain_integration.py` | Full 6-gate chain end-to-end |
| `test_gini_correction` | `tests/core/sovereign/test_sat_controller.py` | Distribution corrected when Gini > 0.40 |

---

*Source of truth: `core/proof_engine/poi_engine.py`, `core/proof_engine/gates.py`, `core/pci/gates.py`, `core/sovereign/z3_fate_gate.py`, `core/sovereign/ihsan_vector.py`*
