# Phase 67.02 — 15 Native Algorithms (Three Minds v2)
# ════════════════════════════════════════════════════

## Standing on Giants
- Al-Ghazali (1058-1111): Intent as ethical pre-gate (Niyyah)
- Ibn Khaldun (1332-1406): Asabiyyah + progressive inequality response
- Al-Khwarizmi (780-850): Algorithm as deterministic procedure
- Kahneman (2002): System 1/2 cognitive architecture
- Nakamoto (2008): Proof-of-Work → adapted to Proof-of-Impact

## Source

`last update/BIZRA_Native_Algorithms_v2_ThreeMinds.py` (990 lines, 37 self-tests)

## Target

```
core/constitutional/
├── __init__.py          # Re-exports
├── types.py             # ActionReceipt, WalletState, Proposal, etc.
├── fixed_point.py       # Spec 01
├── algorithms.py        # A1-A15 implementations
├── ticker.py            # process_tick() — 12-step heartbeat
└── declaration.py       # Spec 03
```

## Data Structures

```
MODULE types

IMPORT fixed_point: fp, fp_float, FP_PRECISION

@dataclass
CLASS ActionReceipt:
    """Immutable record of a verified human action.

    All scores are fixed-point integers (fp(0.95) = 950000).
    """
    receipt_id: bytes           # BLAKE3 hash of content
    actor_id: bytes             # Ed25519 public key
    action_type: str            # "contribution" | "attestation" | "governance" | ...
    timestamp: int              # Unix ms
    intent_score: int           # fp — Al-Ghazali gate (must be >= INTENT_FLOOR)
    efficiency_score: int       # fp
    impact_score: int           # fp
    reproducibility_score: int  # fp
    oracle_signature: bytes     # Ed25519 signature by validator
    metadata_hash: bytes        # BLAKE3 of metadata
    co_actors: List[bytes]      # Ed25519 keys of collaborators

@dataclass
CLASS WalletState:
    """Sovereign economic state of a node.

    Identity = reduction of all events (A14).
    """
    node_id: bytes                      # Ed25519 public key
    seed_balance: int = 0               # Fixed-point SEED token balance
    bloom_balance: int = 0              # Fixed-point BLOOM (soulbound)
    last_active: int = 0                # Last action timestamp
    total_actions: int = 0              # Lifetime action count
    ihsan_history: List[int] = []       # Last N ihsan scores (fixed-point)
    created_at: int = 0                 # Node creation timestamp
    attestations_given: Set[bytes] = {} # Nodes we attested
    attestations_received: Set[bytes] = {} # Nodes that attested us
    governance_votes: int = 0           # Lifetime governance votes
    cooperative_actions: int = 0        # Cooperative action count

@dataclass
CLASS Proposal:
    """Governance proposal for Shura (A8)."""
    proposal_id: bytes
    proposer: bytes                     # Ed25519 key
    description: str
    votes_for: int = 0                  # BLOOM-weighted
    votes_against: int = 0
    status: str = "active"              # active | passed | rejected | expired
    created_at: int = 0

@dataclass
CLASS Reflex:
    """Compiled intelligence pattern for O(1) lookup (A10)."""
    pattern_hash: bytes                 # BLAKE3 of input pattern
    action_chain: List[str]             # Pre-compiled action sequence
    confidence: int                     # Fixed-point confidence score
    last_used: int = 0                  # Timestamp
    use_count: int = 0

@dataclass
CLASS Attestation:
    """Mutual attestation record for Asabiyyah (A12/A15)."""
    attester: bytes
    attestee: bytes
    receipt_id: bytes                   # Receipt being attested
    timestamp: int
    signature: bytes                    # Ed25519 signature

@dataclass
CLASS Event:
    """Append-only event for the immutable log (A14)."""
    event_id: int                       # Sequential
    event_type: str                     # "mint" | "transfer" | "vote" | ...
    actor: bytes
    data: dict
    timestamp: int
    prev_hash: bytes                    # Chain link
    hash: bytes                         # BLAKE3(event_id + type + data + prev_hash)
```

## Algorithm Pseudocode

### A1: Ihsan Scorer (with Al-Ghazali Intent Gate)

```
CONSTANTS:
    IHSAN_FLOOR    = fp(0.95)   # from constants.py
    INTENT_FLOOR   = fp(0.90)   # Al-Ghazali pre-gate
    W_INTENT       = fp(0.25)
    W_EFFICIENCY   = fp(0.25)
    W_IMPACT       = fp(0.30)
    W_REPRODUCIBILITY = fp(0.20)

FUNCTION intent_gate(receipt: ActionReceipt) -> bool:
    """Al-Ghazali's correction: intent MUST pass before computation.

    Pre-gate — not a weight. If intent < 0.90, the receipt is rejected
    before any resource is spent on scoring.
    """
    RETURN receipt.intent_score >= INTENT_FLOOR

FUNCTION ihsan_score(receipt: ActionReceipt) -> int:
    """Compute Ihsan quality score. Returns fixed-point [0, 1.0].

    Precondition: intent_gate(receipt) == True
    """
    RETURN fp_clamp(
        fp_mul(W_INTENT, receipt.intent_score)
        + fp_mul(W_EFFICIENCY, receipt.efficiency_score)
        + fp_mul(W_IMPACT, receipt.impact_score)
        + fp_mul(W_REPRODUCIBILITY, receipt.reproducibility_score),
        FP_ZERO, FP_ONE
    )

FUNCTION full_ihsan_check(receipt: ActionReceipt) -> (bool, int):
    """Combined gate + score. Returns (passed, score).

    Three Minds correction: Al-Ghazali gate fires BEFORE scoring.
    """
    IF NOT intent_gate(receipt):
        RETURN (False, FP_ZERO)

    score = ihsan_score(receipt)
    RETURN (score >= IHSAN_FLOOR, score)
```

### A2: SEED Minter (Proof of Impact)

```
CONSTANTS:
    BASE_MINT = fp(1.0)

FUNCTION mint_seed(receipt: ActionReceipt, ihsan: int) -> int:
    """Mint SEED tokens from verified work.

    I-4: Only work of verified excellence produces value.
    """
    IF ihsan < IHSAN_FLOOR:
        RETURN FP_ZERO  # Below quality threshold → no mint

    # Efficiency bonus: faster work = more value
    efficiency_bonus = fp_div(receipt.efficiency_score, FP_ONE)

    RETURN fp_mul(BASE_MINT, fp_add(FP_ONE, fp_div(efficiency_bonus, fp(2))))
```

### A3: BLOOM Accumulator (Soulbound Governance)

```
CONSTANTS:
    BLOOM_ACCRUAL = fp(0.01)   # Per high-ihsan action
    BLOOM_DECAY   = fp(0.01)   # Per tick of inactivity

FUNCTION accrue_bloom(wallet: WalletState, ihsan: int) -> int:
    """Accrue BLOOM from sustained excellence. Soulbound — cannot transfer.

    I-5: Governance belongs to those who participate.
    """
    IF ihsan >= IHSAN_FLOOR:
        RETURN fp_add(wallet.bloom_balance, BLOOM_ACCRUAL)
    RETURN wallet.bloom_balance

FUNCTION decay_bloom(wallet: WalletState, current_time: int) -> int:
    """Decay BLOOM for inactive nodes. Use it or lose it.

    Al-Ghazali: governance without contribution is empty authority.
    """
    ticks_idle = (current_time - wallet.last_active) // TICK_INTERVAL
    IF ticks_idle <= 0:
        RETURN wallet.bloom_balance

    decay = fp_mul(BLOOM_DECAY, fp(ticks_idle))
    RETURN max(0, fp_sub(wallet.bloom_balance, decay))
```

### A4: Gini Enforcer (Khaldunian Curve + Ghazali Equity Factor)

```
CONSTANTS:
    GINI_HEALTHY = fp(0.35)   # I-3 threshold
    GINI_WARNING = fp(0.50)
    GINI_CRISIS  = fp(0.70)

FUNCTION compute_gini(balances: List[int]) -> int:
    """Compute Gini coefficient in fixed-point.

    Returns fp value in [0, 1.0]. 0 = perfect equality, 1 = total concentration.
    """
    n = len(balances)
    IF n <= 1: RETURN FP_ZERO

    sorted_b = sorted(balances)
    total = sum(sorted_b)
    IF total == 0: RETURN FP_ZERO

    cum_sum = 0
    weighted_sum = 0
    FOR i, b IN enumerate(sorted_b):
        cum_sum += b
        weighted_sum += (2 * (i + 1) - n - 1) * b

    RETURN fp_div(weighted_sum, fp_mul(fp(n), total))

FUNCTION khaldunian_throttle(gini: int) -> int:
    """Ibn Khaldun's progressive throttle (replaces binary gate).

    v1 BUG: Binary gate (gini > 0.35 → mint 0) caused economic death.
    v2 FIX: Progressive curve maintains activity while converging to equality.

    T8 proved: v2 earns 238 SEED vs v1's 0.00 SEED (23,844× improvement).
    """
    IF gini <= fp(0.30):
        RETURN FP_ONE  # Healthy: full minting

    IF gini <= fp(0.40):
        # Warning zone: quadratic dropoff
        excess = fp_sub(gini, fp(0.30))
        penalty = fp_mul(fp(4), fp_mul(excess, excess))
        RETURN fp_sub(FP_ONE, penalty)

    IF gini <= fp(0.50):
        RETURN fp(0.20)  # Stressed: reduced but not zero

    IF gini <= fp(0.70):
        RETURN fp(0.10)  # Crisis: minimal minting

    RETURN fp(0.01)  # Extreme: near-zero but never zero

FUNCTION ghazali_equity_factor(wallet: WalletState, mean_balance: int) -> int:
    """Newcomer advantage multiplier.

    Those below the mean earn MORE per unit of work.
    T9 proved: 3.27× for newcomers vs wealthy nodes.

    I-3: Wealth concentration shall not exceed Gini 0.35.
    """
    IF wallet.seed_balance >= mean_balance:
        RETURN FP_ONE  # At or above mean: standard rate

    IF wallet.seed_balance == 0:
        RETURN fp(EQUITY_FACTOR_MAX)  # Maximum newcomer boost

    ratio = fp_div(mean_balance, wallet.seed_balance)
    RETURN fp_clamp(ratio, fp(EQUITY_FACTOR_MIN), fp(EQUITY_FACTOR_MAX))

FUNCTION progressive_mint(receipt, ihsan, wallet, network_gini, mean_balance) -> int:
    """Full minting pipeline with all corrections applied.

    Three Minds integrated:
    1. Al-Ghazali: Intent gate (already passed if we're here)
    2. Ibn Khaldun: Khaldunian throttle on network Gini
    3. Al-Khwarizmi: All math in fixed-point
    """
    base = mint_seed(receipt, ihsan)
    IF base == 0: RETURN 0

    throttle = khaldunian_throttle(network_gini)
    equity = ghazali_equity_factor(wallet, mean_balance)

    RETURN fp_mul(fp_mul(base, throttle), equity)
```

### A5: Zakat Engine (Annual Purification)

```
CONSTANTS:
    ZAKAT_RATE = fp(0.025)    # 2.5% — I-7
    NISAB_THRESHOLD = fp(85)  # Minimum balance for Zakat (85g gold equivalent)

FUNCTION compute_zakat(wallet: WalletState) -> int:
    """I-7: Wealth above threshold shall be purified annually.

    Deterministic: exactly 2.5% of balance above nisab.
    Redistribution target: 50% to lowest-balance nodes.
    """
    IF wallet.seed_balance < NISAB_THRESHOLD:
        RETURN FP_ZERO  # Below nisab: exempt

    RETURN fp_mul(wallet.seed_balance, ZAKAT_RATE)
```

### A6: Backing Ratio (Reserve Health)

```
FUNCTION backing_ratio(total_seed: int, total_verified_work: int) -> int:
    """Every SEED must be backed by verified work.

    Ratio < 1.0 = inflation (currency unbacked).
    Ratio = 1.0 = perfect backing.
    Ratio > 1.0 = deflation (more work than tokens).
    """
    IF total_seed == 0: RETURN FP_ONE
    RETURN fp_div(total_verified_work, total_seed)
```

### A7: Demurrage (Idle Tax)

```
CONSTANTS:
    DEMURRAGE_RATE = fp(0.001)  # 0.1% per tick for idle balances

FUNCTION apply_demurrage(wallet: WalletState, current_time: int) -> int:
    """Tax idle wealth to incentivize circulation.

    Active nodes (recent action within TICK_INTERVAL): exempt.
    Idle nodes: lose 0.1% per tick.
    """
    ticks_idle = (current_time - wallet.last_active) // TICK_INTERVAL
    IF ticks_idle <= 0:
        RETURN wallet.seed_balance  # Active: no demurrage

    fee = fp_mul(wallet.seed_balance, fp_mul(DEMURRAGE_RATE, fp(ticks_idle)))
    RETURN max(0, fp_sub(wallet.seed_balance, fee))
```

### A8: Shura Governance (BLOOM-Weighted Voting)

```
FUNCTION shura_vote(proposal: Proposal, voter: WalletState, approve: bool) -> Proposal:
    """BLOOM-weighted governance. Soulbound = earned, not bought.

    I-5: Governance belongs to those who participate.
    """
    weight = voter.bloom_balance
    IF weight == 0:
        RETURN proposal  # No governance stake = no vote

    IF approve:
        proposal.votes_for = fp_add(proposal.votes_for, weight)
    ELSE:
        proposal.votes_against = fp_add(proposal.votes_against, weight)

    RETURN proposal

FUNCTION shura_resolve(proposal: Proposal) -> str:
    """Resolve proposal by BLOOM-weighted majority."""
    total = fp_add(proposal.votes_for, proposal.votes_against)
    IF total == 0:
        RETURN "expired"  # No participation

    approval_ratio = fp_div(proposal.votes_for, total)
    IF approval_ratio > fp(0.667):  # Supermajority
        RETURN "passed"
    RETURN "rejected"
```

### A9: Trust Monitor

```
FUNCTION trust_score(wallet: WalletState) -> int:
    """Composite trust score from historical ihsan performance."""
    IF len(wallet.ihsan_history) == 0:
        RETURN FP_ZERO

    total = sum(wallet.ihsan_history)
    avg = fp_div(total, fp(len(wallet.ihsan_history)))

    # Consistency bonus: low variance = higher trust
    variance = compute_variance(wallet.ihsan_history, avg)
    consistency = fp_sub(FP_ONE, fp_clamp(variance, FP_ZERO, FP_ONE))

    RETURN fp_mul(avg, fp_add(FP_ONE, fp_div(consistency, fp(2))))
```

### A10: Reflex Compiler (System-1 Cache)

```
FUNCTION compile_reflex(pattern: str, action_chain: List[str], confidence: int) -> Reflex:
    """Compile a verified pattern into O(1) cached reflex.

    Kahneman System-1: 90% of interactions hit cache.
    """
    RETURN Reflex(
        pattern_hash=blake3(pattern.encode()),
        action_chain=action_chain,
        confidence=confidence,
        last_used=now(),
        use_count=0
    )

FUNCTION reflex_lookup(cache: Dict[bytes, Reflex], pattern: str) -> Optional[Reflex]:
    """O(1) hash lookup. Returns None if no cached reflex."""
    key = blake3(pattern.encode())
    reflex = cache.get(key)
    IF reflex AND reflex.confidence >= IHSAN_FLOOR:
        reflex.last_used = now()
        reflex.use_count += 1
        RETURN reflex
    RETURN None
```

### A11: Identity Reducer

```
FUNCTION reduce_identity(events: List[Event]) -> WalletState:
    """Identity = reduction of all events. (Event Sourcing)

    Given the complete event log, recompute the current state.
    Deterministic: same events → same state.
    """
    wallet = WalletState(node_id=events[0].actor)
    FOR event IN events:
        APPLY event TO wallet  # Replay each event
    RETURN wallet
```

### A12: Consensus Oracle (Mutual Attestation)

```
FUNCTION attest(attester: WalletState, attestee: WalletState,
                receipt: ActionReceipt) -> Attestation:
    """Mutual attestation — the bond that builds Asabiyyah.

    Attester vouches for the quality of attestee's receipt.
    Both parties gain social credit (A15).
    """
    attestation = Attestation(
        attester=attester.node_id,
        attestee=attestee.node_id,
        receipt_id=receipt.receipt_id,
        timestamp=now(),
        signature=ed25519_sign(attester.private_key, receipt.receipt_id)
    )

    # Update social graph
    attester.attestations_given.add(attestee.node_id)
    attestee.attestations_received.add(attester.node_id)

    RETURN attestation
```

### A13: Chain Resolver (Topological Sort)

```
FUNCTION resolve_chain(events: List[Event]) -> List[Event]:
    """Topological sort of dependent events.

    Ensures causal ordering: no event processed before its dependencies.
    O(n) with topological sort.
    """
    graph = build_dependency_graph(events)
    RETURN topological_sort(graph)
```

### A14: Event Sourcer (Immutable History)

```
FUNCTION append_event(log: List[Event], event_type: str,
                      actor: bytes, data: dict) -> Event:
    """Append event to immutable log with hash chain.

    Integrity: each event includes hash of previous event.
    Merkle chain: any tampering breaks the chain.
    """
    prev_hash = log[-1].hash IF log ELSE b'\x00' * 32
    event_id = len(log)

    content = canonical_bytes(event_id, event_type, data, prev_hash)
    event_hash = blake3(content)

    event = Event(
        event_id=event_id,
        event_type=event_type,
        actor=actor,
        data=data,
        timestamp=now(),
        prev_hash=prev_hash,
        hash=event_hash
    )
    log.append(event)
    RETURN event
```

### A15: Asabiyyah Index (Social Cohesion)

```
CONSTANTS:
    ASABIYYAH_WEIGHTS = (fp(0.4), fp(0.3), fp(0.3))
    # (reciprocal_attestations, governance_votes, cooperative_actions)

FUNCTION asabiyyah_score(wallet: WalletState, network_size: int) -> int:
    """Ibn Khaldun's social cohesion metric for a single node.

    Measures: how connected, how participatory, how cooperative.
    T10 proved: monotonic growth with attestation activity.
    """
    IF network_size <= 1: RETURN FP_ZERO

    # Reciprocal attestations (both gave and received)
    reciprocal = len(wallet.attestations_given & wallet.attestations_received)
    max_reciprocal = network_size - 1
    a_reciprocal = fp_div(fp(reciprocal), fp(max_reciprocal))

    # Governance participation
    a_governance = fp_clamp(fp_div(fp(wallet.governance_votes), fp(10)), FP_ZERO, FP_ONE)

    # Cooperative actions
    a_cooperative = fp_clamp(fp_div(fp(wallet.cooperative_actions), fp(20)), FP_ZERO, FP_ONE)

    RETURN fp_add(
        fp_add(
            fp_mul(ASABIYYAH_WEIGHTS[0], a_reciprocal),
            fp_mul(ASABIYYAH_WEIGHTS[1], a_governance)
        ),
        fp_mul(ASABIYYAH_WEIGHTS[2], a_cooperative)
    )

FUNCTION network_asabiyyah(wallets: List[WalletState]) -> int:
    """Network-wide social cohesion score.

    Average of all individual Asabiyyah scores.
    Ibn Khaldun: "Asabiyyah is the pillar of civilization."
    """
    IF len(wallets) == 0: RETURN FP_ZERO

    total = sum(asabiyyah_score(w, len(wallets)) FOR w IN wallets)
    RETURN fp_div(total, fp(len(wallets)))
```

### process_tick() — 12-Step Heartbeat

```
FUNCTION process_tick(wallets: List[WalletState], receipts: List[ActionReceipt],
                      proposals: List[Proposal], event_log: List[Event],
                      reflex_cache: Dict) -> TickResult:
    """One heartbeat of the constitutional kernel.

    Runs all 15 algorithms in constitutional order.
    12 steps, deterministic, reproducible.
    """
    results = TickResult()

    # Step 1: Al-Ghazali Intent Gate — reject low-intent receipts
    valid_receipts = [r FOR r IN receipts IF intent_gate(r)]
    results.rejected = len(receipts) - len(valid_receipts)

    # Step 2: Ihsan Scoring — compute quality for valid receipts
    scored = [(r, ihsan_score(r)) FOR r IN valid_receipts]

    # Step 3: Compute network Gini
    balances = [w.seed_balance FOR w IN wallets]
    gini = compute_gini(balances)
    mean_balance = fp_div(sum(balances), fp(len(wallets))) IF wallets ELSE FP_ZERO

    # Step 4: Progressive Minting — SEED creation with all corrections
    FOR receipt, ihsan IN scored:
        wallet = find_wallet(wallets, receipt.actor_id)
        minted = progressive_mint(receipt, ihsan, wallet, gini, mean_balance)
        wallet.seed_balance = fp_add(wallet.seed_balance, minted)
        results.total_minted += minted

    # Step 5: BLOOM Accrual — governance token growth
    FOR receipt, ihsan IN scored:
        wallet = find_wallet(wallets, receipt.actor_id)
        wallet.bloom_balance = accrue_bloom(wallet, ihsan)

    # Step 6: BLOOM Decay — inactive governance weight reduction
    FOR wallet IN wallets:
        wallet.bloom_balance = decay_bloom(wallet, now())

    # Step 7: Demurrage — idle balance tax
    FOR wallet IN wallets:
        wallet.seed_balance = apply_demurrage(wallet, now())

    # Step 8: Zakat Collection — annual purification
    IF is_zakat_cycle():
        FOR wallet IN wallets:
            zakat_due = compute_zakat(wallet)
            wallet.seed_balance = fp_sub(wallet.seed_balance, zakat_due)
            results.zakat_pool += zakat_due

    # Step 9: Governance — resolve expired proposals
    FOR proposal IN proposals:
        IF proposal_expired(proposal):
            proposal.status = shura_resolve(proposal)

    # Step 10: Reflex Cache — compile new patterns
    FOR receipt, ihsan IN scored:
        IF ihsan >= fp(0.98):  # Only excellent work becomes reflex
            compile_reflex(receipt.action_type, [...], ihsan)

    # Step 11: Event Logging — immutable history
    FOR receipt, ihsan IN scored:
        append_event(event_log, "mint", receipt.actor_id, {
            "receipt_id": receipt.receipt_id.hex(),
            "ihsan": fp_float(ihsan),
            "minted": fp_float(minted)
        })

    # Step 12: Asabiyyah — network cohesion update
    results.network_asabiyyah = network_asabiyyah(wallets)
    results.network_gini = gini

    RETURN results
```

## Integration Points

| Algorithm | Existing Module | Action |
|-----------|----------------|--------|
| A1 Ihsan | `core/iaas/snr_v2_adapter.py` | Wire ihsan_score through SNR pipeline |
| A2 SEED | `core/treasury/token_minter.py` | Extend with progressive_mint |
| A3 BLOOM | (new) | Add to constitutional/ |
| A4 Gini | `core/integration/constants.py` → ADL_GINI_THRESHOLD | Wire khaldunian_throttle |
| A5 Zakat | `core/treasury/token_minter.py` → zakat deduction | Use compute_zakat |
| A10 Reflex | `core/living_memory/` | Wire cache through reflex compiler |
| A14 Events | `core/proof_engine/evidence_ledger.py` | Compatible — same chain model |
| A15 Asabiyyah | (new) | Add to constitutional/ |
