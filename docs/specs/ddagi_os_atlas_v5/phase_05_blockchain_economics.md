# Phase 05 — Blockchain & Economics: BlockGraph, Dual Tokens, Resource Pool

> Source: Atlas v5.0 — Diagrams D7 (BlockGraph), D8 (SEED+BLOOM), D14 (URP)
> Status: SPECIFICATION SEALED | SNR: 0.95

---

## 1. Functional Requirements

### FR-050: BlockGraph DAG

Not a fork. Built from scratch as a directed acyclic graph for high-throughput,
content-addressed proof storage with constitutional governance in the consensus
layer.

**Genesis Block.** Contains the BLAKE3 hash of the Al-Bazrah Constitution,
anchoring the entire DAG to the covenant. All blocks carry Merkle proof chains
back to this root.

**DAG Structure.** Parallel branches enable concurrent block production. Merge
points are BFT consensus checkpoints validating branch consistency. Active tips
represent the latest valid state across non-finalized branches.

```
Genesis (Al-Bazrah hash)
    +-- Branch A --+
    +-- Branch B --+--> Merge Point (BFT checkpoint) --> Active Tips
    +-- Branch C --+
```

**Storage Model:**

| Tier    | Technology       | Purpose                                      |
|---------|------------------|----------------------------------------------|
| Hot     | SQLite / RocksDB | Local node: fast reads, recent blocks         |
| Warm    | IPFS (CID)       | Content-addressed: block data, proof payloads |
| Archive | Erasure Coding   | Reed-Solomon (k=4, n=7), survives 3 losses   |

Every block: `CID = BLAKE3(header || payload)`. IPFS provides deduplication.

**Cryptographic Audit.** Every block carries: Ed25519 signature, Merkle proof
to parents, ZK proofs where privacy is required, BLAKE3 hash chain.

### FR-051: Proof-of-Impact Consensus

PoI replaces PoW/PoS with quality-based consensus. Validators compete on
demonstrated impact, not computation or capital.

**Validation Pipeline:**
1. **Impact Assessment** -- Attestation with action receipt, before/after
   evidence, quality metrics (SNR, Ihsan, user feedback).
2. **Weighted Scoring** -- `validator_weight = impact_score * sqrt(stake)`.
   Square root dampens plutocratic influence.
3. **BFT Finality** -- `3f + 1` validators, `f` Byzantine faults tolerated.
   Finalized when weighted agreement > 2/3 supermajority.
4. **Cryptographic Commitment** -- Ed25519-signed assessments, BLAKE3 Merkle
   root of all validator signatures in the merge-point block.

**Impact Score Components:**

| Component           | Weight | Source                                       |
|---------------------|--------|----------------------------------------------|
| Task Quality        | 0.35   | SNR composite (atomic*0.4 + edge*0.3 + path*0.3) |
| User Satisfaction   | 0.25   | Explicit feedback + implicit engagement      |
| Network Benefit     | 0.20   | Pattern reusability, federation adoption     |
| Resource Efficiency | 0.10   | Compute/storage per unit of impact           |
| Novelty             | 0.10   | First-time pattern vs. cached reflex         |

### FR-052: SEED Token (Stable Utility)

Utility token for resource consumption. Designed for stability, not speculation.

**Earning:** Resource contribution (pledged + verified), task completion
(SNR >= 0.85), attestation validation (correct PoI votes earn validator fee).

**Spending:** Compute cycles (inference, batch), storage (IPFS pinning),
bandwidth (federation gossip, transfers).

**Peg:** `1 SEED = 1 minute of median inference`. Maintained via mint/burn:
above peg mints from reserve, below peg burns from transaction fees.

**Shariah Compliance:** No riba (Mudarabah profit-sharing only), usage-based
pricing, Zakat 2.5% at mint (`net + zakat = gross`), asset-backed (real
compute/storage capacity).

### FR-053: BLOOM Token (Impact Growth)

Governance and reputation token. Cannot be purchased -- only earned through
validated PoI attestations. No pre-mine, no ICO, no investor allocation.

**Governance:** Voting weight = `bloom_balance * reputation_multiplier`.
Quality amplifies voice.

**Reputation:** Higher BLOOM earns priority scheduling in the Resource Pool
and preference as PoI validator.

**Value Dynamics:** ADL Gini invariant (<= 0.35) enforced on BLOOM
distribution at every merge point. Quality-proportional governance resists
concentration.

**Eligibility Gate:** Ihsan >= 0.90 (`IHSAN_BLOOM_ELIGIBILITY`) required for
BLOOM minting.

### FR-054: PoI Reward Flow

```
ACTION -> ATTESTATION -> VALIDATE (BFT 3f+1) -> SCORE -> MINT
```

**Distribution Split:**

| Recipient     | Share | Purpose                                    |
|---------------|-------|--------------------------------------------|
| User (Actor)  | 70%   | Direct reward for productive work          |
| System (Pool) | 20%   | Infrastructure maintenance, validator fees |
| Reserve       | 10%   | Stability reserve, peg maintenance, Waqf   |

SEED minted for resource/task utility; BLOOM only for validated high-quality
impact. Both subject to Zakat: `net = gross * (1 - 0.025)`. The 2.5% flows
to the Universal Basic Compute (UBC) pool -- minimum compute access for all.

### FR-055: Reverse Scaling Economics

More participation leads to lower costs (opposite of extractive platforms):

```
More Users -> More Data -> Better AI -> More Value -> Lower Cost -> (loop)
```

**Theorem 2.5:** `GDP(N) = Theta(N / log N)`. Total GDP grows faster than
linearly; `log N` denominator is coordination overhead kept low by DAG
parallelism and reflex caching.

**Corollary 4.2:** With local models (`C_LLM ~ 0`), system is viable for
ALL cache hit rates `rho > 0`. Cloud dependency eliminated.

### FR-056: Universal Resource Pool

Aggregates node resources into a shared commons with dynamic pricing and
constitutional governance.

**Six Resource Categories (from FR-011):**

| Category   | Unit           | Measurement                         |
|------------|----------------|-------------------------------------|
| Compute    | CPU-seconds    | FLOPS benchmark at registration     |
| Storage    | GB-months      | Verified capacity, proof-of-space   |
| Bandwidth  | MB transferred | Measured at network layer           |
| Knowledge  | Attestations   | Verified expertise contributions    |
| Attention  | Hours          | Human feedback and verification     |
| Creativity | Artifacts      | Original content scored by PoI      |

**Dynamic Pricing:**
`price(r, t) = base_price * (demand(t) / supply(t)) ^ elasticity`
Elasticity per category: compute 0.5, storage 0.3, bandwidth 0.7. Harberger
tax (7%, `ADL_HARBERGER_TAX_RATE`) discourages hoarding via self-assessment.

**Leases.** Time-bounded, signed contracts on the BlockGraph. Violations
trigger automatic SEED penalty deduction.

**SLA Monitoring:** Compute (p99 latency, throughput), Storage (99.9%
availability), Bandwidth (throughput, jitter).

**Fraud Detection:** Proof-of-Resource challenges, anomaly detection
(reported vs. measured), cross-node validation.

**Auto-Rebalance:** On node departure, leased resources redistribute to
minimize Gini coefficient change. Justice-preserving by design.

---

## 2. Edge Cases

**EC-050: Genesis Block Corruption.**
Node presenting a genesis block with non-matching Al-Bazrah hash is
immediately expelled (`NodeStatus::Expelled`). Hash is hardcoded in the
binary. No recovery -- entire trust chain depends on this anchor.

**EC-051: PoI Attestation Spam.**
SNR floor (0.85) filters noise. Ihsan gate (0.90) blocks low-quality BLOOM
minting. BFT weighted scoring deprioritizes trivial contributions. Rate
limit: 100 attestations/hour/node (`ACTION_BUS_MAX_PER_HOUR`).

**EC-052: SEED Peg Depegging.**
Reserve fund (10% of minting) absorbs demand shocks. If exhausted: emergency
mode -- pause new minting, freeze at last stable price, auto-generate
governance recapitalization proposal.

**EC-053: Gini Threshold Breach During Merge.**
Post-merge BLOOM distribution exceeding ADL Gini threshold (0.35) rejects
the merge. Branch must redistribute via Zakat top-up or Harberger
acceleration. System never finalizes a state violating the justice invariant.

**EC-054: Resource Lease Orphan on Node Departure.**
Auto-rebalancer migrates lease to replacement node. If no replacement within
SLA grace (5 min), lessee receives SEED refund and penalty is deducted from
departed node's deposit.

---

## 3. Pseudocode

### 3.1 submit_poi(attestation)

```
FUNCTION submit_poi(attestation, node, blockgraph, committee):
    IF attestation IS None OR attestation.receipt IS None:
        RETURN PoIResult(REJECTED, "missing_attestation_or_receipt")
    IF NOT verify_ed25519(attestation.signature, node.verifying_key, attestation.payload_bytes()):
        RETURN PoIResult(REJECTED, "invalid_signature")

    IF attestation.snr_composite < SNR_THRESHOLD:  # 0.85
        RETURN PoIResult(REJECTED, "snr_below_floor")
    IF attestation.ihsan_score < IHSAN_GATE_MINIMUM:  # 0.85
        RETURN PoIResult(REJECTED, "ihsan_below_gate")

    IF blockgraph.count_attestations(node.id, window=3600) >= ACTION_BUS_MAX_PER_HOUR:
        RETURN PoIResult(REJECTED, "rate_limit_exceeded")
    IF NOT attestation.receipt.verify_merkle_chain(blockgraph.genesis_hash):
        RETURN PoIResult(REJECTED, "broken_merkle_chain")
    IF BLAKE3(attestation.evidence.before) == BLAKE3(attestation.evidence.after):
        RETURN PoIResult(REJECTED, "no_observable_impact")

    # BFT committee scoring
    votes = []
    FOR validator IN committee.select_validators(quorum_size=3*f+1):
        score = validator.assess_impact(attestation)
        votes.append(ValidatorVote(
            validator_id = validator.id, impact_score = score,
            weight = score * sqrt(validator.stake),
            signature = validator.sign_ed25519(attestation.id, score),
        ))

    total_weight = SUM(v.weight FOR v IN votes)
    IF total_weight == 0: RETURN PoIResult(REJECTED, "zero_validator_weight")
    final_score = SUM(v.impact_score * v.weight FOR v IN votes) / total_weight

    agreeing = SUM(v.weight FOR v IN votes IF v.impact_score > 0)
    IF agreeing / total_weight < 2/3:
        RETURN PoIResult(REJECTED, "no_supermajority")

    poi_block = PoIBlock(
        parent_tips=blockgraph.active_tips(), attestation=attestation,
        votes=votes, final_score=final_score,
        merkle_root=BLAKE3_merkle([v.signature FOR v IN votes]),
    )
    poi_block.signature = node.sign_ed25519(poi_block.canonical_bytes())
    blockgraph.append(poi_block)
    RETURN PoIResult(ACCEPTED, score=final_score, block_cid=poi_block.cid)
```

### 3.2 mint_tokens(impact_score)

```
FUNCTION mint_tokens(impact_score, node, token_ledger, poi_block_cid):
    IF impact_score <= 0: RETURN MintResult(REJECTED, "non_positive_impact")
    IF impact_score > 1.0: RETURN MintResult(REJECTED, "impact_exceeds_maximum")

    # SEED minting (utility reward)
    seed_gross   = impact_score * TOKENS_PER_COMPUTE_UNIT  # 100 base
    seed_zakat   = seed_gross * ZAKAT_RATE                 # 2.5%
    seed_net     = seed_gross - seed_zakat
    seed_user    = seed_net * 0.70
    seed_system  = seed_net * 0.20
    seed_reserve = seed_net * 0.10

    # BLOOM minting (governance reward, conditional)
    bloom_amount = 0
    IF node.ihsan_score >= IHSAN_BLOOM_ELIGIBILITY:  # 0.90
        bloom_gross = impact_score * BLOOM_MULTIPLIER
        bloom_amount = bloom_gross - (bloom_gross * ZAKAT_RATE)
        projected_gini = token_ledger.simulate_gini(node.id, bloom_amount, "BLOOM")
        IF projected_gini > ADL_GINI_THRESHOLD:  # 0.35
            bloom_amount = 0  # Justice invariant prevails

    token_ledger.begin_transaction()
    TRY:
        token_ledger.credit(node.id, "SEED", seed_user, proof=poi_block_cid)
        token_ledger.credit("SYSTEM_POOL", "SEED", seed_system, proof=poi_block_cid)
        token_ledger.credit("RESERVE", "SEED", seed_reserve, proof=poi_block_cid)
        token_ledger.credit("UBC_POOL", "SEED", seed_zakat, proof=poi_block_cid)
        IF bloom_amount > 0:
            token_ledger.credit(node.id, "BLOOM", bloom_amount, proof=poi_block_cid)
        token_ledger.commit()
    EXCEPT error:
        token_ledger.rollback()
        RETURN MintResult(ERROR, "ledger_write_failed", details=str(error))

    RETURN MintResult(SUCCESS, seed_minted=seed_net, seed_zakat=seed_zakat,
                      bloom_minted=bloom_amount,
                      distribution={user: seed_user, system: seed_system, reserve: seed_reserve})
```

### 3.3 allocate_resource(request)

```
FUNCTION allocate_resource(request, pool, node, pricing_engine):
    VALID_CATEGORIES = {"compute","storage","bandwidth","knowledge","attention","creativity"}
    IF request.category NOT IN VALID_CATEGORIES:
        RETURN AllocationResult(REJECTED, "invalid_resource_category")
    IF request.duration_seconds <= 0 OR request.duration_seconds > 86400:
        RETURN AllocationResult(REJECTED, "invalid_lease_duration")
    IF request.quantity <= 0:
        RETURN AllocationResult(REJECTED, "non_positive_quantity")

    price = dynamic_price(pool.supply(request.category), pool.demand(request.category),
                          pricing_engine.base_price(request.category),
                          pricing_engine.elasticity(request.category))
    total_cost = price.unit_price * request.quantity * (request.duration_seconds / 3600)

    IF node.seed_balance < total_cost:
        RETURN AllocationResult(REJECTED, "insufficient_seed",
                                required=total_cost, available=node.seed_balance)

    candidates = pool.find_providers(category=request.category,
                                      quantity=request.quantity,
                                      min_ihsan=IHSAN_GATE_MINIMUM)  # 0.85
    IF len(candidates) == 0:
        RETURN AllocationResult(REJECTED, "no_providers_available")

    provider = candidates[0]  # Highest Ihsan + lowest latency
    lease = ResourceLease(
        lease_id=uuid4(), lessee=node.id, lessor=provider.id,
        category=request.category, quantity=request.quantity,
        price_per_unit_hour=price.unit_price, total_cost=total_cost,
        start_time=now_utc(), end_time=now_utc() + request.duration_seconds,
        sla=request.sla OR default_sla(request.category),
    )
    lease.signature = node.sign_ed25519(lease.canonical_bytes())

    pool.begin_transaction()
    TRY:
        pool.escrow(node.id, total_cost, lease.lease_id)
        pool.activate_lease(lease)
        pool.decrement_supply(provider.id, request.category, request.quantity)
        pool.commit()
    EXCEPT error:
        pool.rollback()
        RETURN AllocationResult(ERROR, "allocation_failed", details=str(error))

    RETURN AllocationResult(SUCCESS, lease_id=lease.lease_id, provider=provider.id,
                            cost=total_cost, expires_at=lease.end_time)
```

### 3.4 dynamic_price(supply, demand)

```
FUNCTION dynamic_price(supply, demand, base_price, elasticity):
    MIN_MULT = 0.10; MAX_MULT = 10.0

    IF supply <= 0: RETURN PriceResult(unit_price=base_price * MAX_MULT, clamped=True)
    IF demand <= 0: RETURN PriceResult(unit_price=base_price * MIN_MULT, clamped=True)
    IF base_price <= 0: RAISE ValueError("base_price must be positive")
    IF elasticity < 0 OR elasticity > 1.0: RAISE ValueError("elasticity in [0,1]")

    ratio = demand / supply
    raw_price = base_price * (ratio ^ elasticity)
    clamped = CLAMP(raw_price, base_price * MIN_MULT, base_price * MAX_MULT)
    harberger_annual = clamped * ADL_HARBERGER_TAX_RATE  # 0.07

    RETURN PriceResult(unit_price=clamped, raw_price=raw_price, ratio=ratio,
                       elasticity=elasticity, harberger_annual=harberger_annual,
                       clamped=(clamped != raw_price))
```

---

## 4. TDD Anchors

```
TEST submit_poi_rejects_invalid_signature:
    attestation = make_attestation(snr=0.92, ihsan=0.96)
    attestation.signature = b"invalid_bytes"
    result = submit_poi(attestation, node, blockgraph, committee)
    ASSERT result.status == REJECTED AND "invalid_signature" IN result.reason

TEST submit_poi_rejects_below_snr_floor:
    attestation = make_attestation(snr=0.80, ihsan=0.96)
    result = submit_poi(attestation, node, blockgraph, committee)
    ASSERT result.status == REJECTED AND "snr_below_floor" IN result.reason

TEST submit_poi_accepts_valid_attestation:
    attestation = make_attestation(snr=0.92, ihsan=0.96, valid_sig=True)
    mock_committee(votes=[0.85, 0.90, 0.88, 0.91])
    result = submit_poi(attestation, node, blockgraph, committee)
    ASSERT result.status == ACCEPTED AND result.score > 0

TEST mint_tokens_applies_zakat_conservation:
    result = mint_tokens(impact_score=0.90, node, token_ledger, block_cid)
    ASSERT result.status == SUCCESS
    total = result.seed_minted + result.seed_zakat
    ASSERT abs(total - 0.90 * TOKENS_PER_COMPUTE_UNIT) < 1e-9  # net + zakat = gross

TEST mint_tokens_blocks_bloom_below_ihsan_eligibility:
    node.ihsan_score = 0.88  # Below IHSAN_BLOOM_ELIGIBILITY (0.90)
    result = mint_tokens(impact_score=0.95, node, token_ledger, block_cid)
    ASSERT result.bloom_minted == 0 AND result.seed_minted > 0

TEST mint_tokens_blocks_bloom_on_gini_breach:
    token_ledger.mock_simulate_gini(return_value=0.40)  # > ADL_GINI_THRESHOLD (0.35)
    node.ihsan_score = 0.96
    result = mint_tokens(impact_score=0.95, node, token_ledger, block_cid)
    ASSERT result.bloom_minted == 0  # Justice invariant enforced

TEST allocate_resource_rejects_insufficient_balance:
    node.seed_balance = 1.0
    request = make_request(category="compute", quantity=1000, duration=3600)
    result = allocate_resource(request, pool, node, pricing_engine)
    ASSERT result.status == REJECTED AND "insufficient_seed" IN result.reason

TEST dynamic_price_clamps_to_bounds:
    result = dynamic_price(supply=1, demand=10000, base_price=1.0, elasticity=0.5)
    ASSERT result.unit_price <= 10.0 AND result.clamped == True
    result = dynamic_price(supply=10000, demand=1, base_price=1.0, elasticity=0.5)
    ASSERT result.unit_price >= 0.10
```

---

## 5. Cross-References

### Python Modules
- `core/treasury/sat_economy.py` -- `SATRole`, `EconomicState`, `ZAKAT_RATE` (0.025), `SAT_PER_NODE` (5), `compute_gdp_scaling()` (Theorem 2.5), `sustainability_analysis()` (Corollary 4.2), `zakat_mint()`.
- `core/treasury/adl_invariant.py` -- `AdlGate`, `AdlInvariant`, `calculate_gini()`, `simulate_transaction_impact()`. Re-exports from `core.sovereign.adl_invariant`.
- `core/treasury/adl_kernel.py` -- `AdlEnforcer`, `IncrementalGini`, `NetworkGiniTracker`. Re-exports from `core.sovereign.adl_kernel`.
- `core/treasury/market_integration.py` -- Market dynamics. Re-exports from `core.sovereign.market_integration`.
- `core/treasury/treasury_controller.py` -- Treasury state machine. Re-exports from `core.sovereign.treasury_controller`.
- `core/integration/constants.py` -- `ADL_GINI_THRESHOLD` (0.35), `ADL_HARBERGER_TAX_RATE` (0.07), `ADL_MINIMUM_HOLDING` (1e-9), `ADL_GINI_MIN_ACCOUNTS` (5), `IHSAN_BLOOM_ELIGIBILITY` (0.90), `IHSAN_GATE_MINIMUM` (0.85), `UNIFIED_SNR_THRESHOLD` (0.85).
- `core/pci/gates.py` -- `PCIGateKeeper`, gate chain (SCHEMA > SIGNATURE > TIMESTAMP > REPLAY > IHSAN > SNR > POLICY).
- `core/pci/envelope.py` -- `PCIEnvelope` for signed inter-node PoI attestation messages.
- `core/proof_engine/evidence_ledger.py` -- Merkle-chained receipt log. PoI attestations append here before BlockGraph submission.

### Rust Crates
- `bizra-omega/bizra-core/src/islamic_finance.rs` -- `ZAKAT_RATE` (0.025), `NISAB_THRESHOLD` (1000.0), `MIN_MUDARIB_SHARE` (0.30), `MAX_RABBULMAL_SHARE` (0.70), `IslamicComplianceGate`, `ZakatCalculator`, `MudarabahContract`, `WaqfEndowment`.
- `bizra-omega/bizra-core/src/lib.rs` -- `IHSAN_THRESHOLD` (0.95), re-exports `islamic_finance`, `omega` (AdlInvariant), `pat` (minting), `pci`.
- `bizra-omega/bizra-core/src/omega.rs` -- `AdlInvariant`, `ADL_GINI_THRESHOLD`. Rust-side justice enforcement.
- `bizra-omega/bizra-core/src/pat/minting.rs` -- `AgentMintingEngine`, `AGENT_MINT_IHSAN_THRESHOLD`.
- `bizra-omega/bizra-resourcepool/src/lib.rs` -- `PoolNode`, `NodeClass`, `NodeStatus`, `IHSAN_THRESHOLD` (Decimal 0.95), `ZAKAT_RATE` (Decimal 0.025), `NISAB_THRESHOLD` (1M), `HARBERGER_TAX_RATE` (Decimal 0.07), `ADL_GINI_MAX` (Decimal 0.35), `TOKENS_PER_COMPUTE_UNIT` (100).
- `bizra-omega/bizra-proofspace/` -- `BizraBlock`, `ValidationResult`, `Verdict`. Block validation for the BlockGraph.

### Atlas v5 Phases
- Phase 00 -- System Overview (FR-002: L1 = BlockGraph + PoI + SEED/BLOOM; FR-003: steps 8-9 = on-chain proof + token minting)
- Phase 01 -- Sovereign Node (FR-011: 6 resource categories feed the Pool; FR-012: Ed25519 identity for block signing)
- Phase 06 -- Governance + Soul (FATE Gate validates financial transactions; Ihsan Wall gates BLOOM eligibility; ADL Gini enforced at merge points)

### Standing on Giants
- Nakamoto (2008): Proof-of-Work transformed into Proof-of-Impact
- Gini (1912): Gini coefficient -- ADL justice invariant (0.35)
- Harberger (1962): Self-assessed taxation -- prevents resource hoarding
- Rawls (1971): Veil of ignorance -- UBC pool ensures minimum compute for all
- Al-Ghazali (1095): Maqasid al-Shariah -- Zakat, no riba, asset-backed tokens
- Ibn Khaldun (1377): Asabiyyah -- network solidarity grows with shared impact
- Shannon (1948): SNR as quality floor for PoI acceptance
- Weyl & Posner (2018): Radical Markets -- Harberger tax mechanism design
- Yunus (2006): Social business -- profit-sharing over interest
