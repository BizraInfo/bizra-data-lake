# 07 — SEED Economics

> SEED = the proof-native token. Earned by impact, not purchased.
> BLOOM = soulbound reputation (non-transferable).
> Zakat = 2.5% redistribution at mint. Gini <= 0.35 hard gate.
> Harberger = 5% annual tax on declared marketplace value.

## Pseudocode: SEED Token

```
STRUCT SEEDToken:
    // SEED is the unit of economic exchange within the URP.
    // It is MINTED from proven impact, not bought or pre-allocated.
    balance:     f64
    minted:      f64      // lifetime total minted to this node
    burned:      f64      // lifetime total burned from this node
    zakat_paid:  f64      // lifetime zakat contributed

CONST SEED_PER_MISSION:    f64 = 1.0     // 1 SEED per successful mission
CONST BURN_PER_DEGRADED:   f64 = 0.1     // 0.1 SEED burned per degraded mission
CONST ZAKAT_RATE:          f64 = 0.025   // 2.5% at mint time
CONST ADL_GINI_THRESHOLD:  f64 = 0.35    // hard gate on wealth concentration
CONST HARBERGER_TAX_RATE:  f64 = 0.05    // 5% annual on declared marketplace value
```

## Pseudocode: Proof of Impact (PoI)

```
FUNCTION compute_seed_reward(request: ProofCarryingRequest) -> f64:
    // SEED is earned through Proof of Impact, not proof of work/stake.
    // Impact = quality × signal × volume, with diminishing returns.

    ihsan = request.ihsan_score          // quality score (>= 0.95 to cross FATE)
    snr = request.snr_score              // signal-to-noise (>= 0.85)
    actions = request.proof_trace.states_traversed.len()

    // PoI formula: ihsan × snr × log2(actions + 1)
    // log2 provides diminishing returns — prevents gaming by action count
    raw_impact = ihsan * snr * log2(actions as f64 + 1.0)

    // Normalize to SEED: 1.0 SEED per standard mission
    reward = raw_impact * SEED_PER_MISSION

    RETURN reward

FUNCTION compute_poi_score(mission_result: MissionResult) -> f64:
    // Proof of Impact score used for BLOOM reputation
    RETURN mission_result.ihsan
         * mission_result.snr
         * log2(mission_result.actions_count as f64 + 1.0)
```

## Pseudocode: SEED Minting

```
FUNCTION mint_seed(
    ledger: &mut SEEDLedger,
    node_id: NodeId,
    raw_reward: f64,
) -> MintResult:
    // Minting always deducts zakat first.
    // Minting is gated by Gini (Adl) threshold.

    // Step 1: Compute zakat deduction
    zakat = raw_reward * ZAKAT_RATE       // 2.5%
    net_reward = raw_reward - zakat

    // Step 2: Simulate post-mint Gini coefficient
    post_gini = simulate_gini_after(ledger, node_id, net_reward)

    // Step 3: Adl gate — reject if concentration too high
    // Exception: genesis mint is exempt (bootstrapping)
    IF post_gini > ADL_GINI_THRESHOLD AND NOT is_genesis_mint():
        current_share = ledger.balance(node_id) / ledger.total_supply()
        proposed_share = (ledger.balance(node_id) + net_reward) / (ledger.total_supply() + net_reward)

        IF proposed_share > current_share:
            RETURN MintResult::GiniRejected(post_gini)

    // Step 4: Credit the accounts
    ledger.credit(node_id, net_reward)
    ledger.credit(ZAKAT_POOL, zakat)

    // Step 5: Update lifetime counters
    ledger.add_minted(node_id, raw_reward)
    ledger.add_zakat(node_id, zakat)

    RETURN MintResult::Minted(net_reward, zakat)

ENUM MintResult:
    Minted(f64, f64)          // (net_reward, zakat_deducted)
    GiniRejected(f64)          // post-mint Gini would exceed threshold
```

## Pseudocode: SEED Burning

```
FUNCTION burn_seed(
    ledger: &mut SEEDLedger,
    node_id: NodeId,
    amount: f64,
    reason: BurnReason,
) -> BurnResult:
    // SEED is burned for degraded missions and constitutional penalties.
    // Burns reduce supply — they are deflationary.

    balance = ledger.balance(node_id)
    IF balance < amount:
        // Cannot burn more than balance — floor at zero
        actual_burn = balance
    ELSE:
        actual_burn = amount

    ledger.debit(node_id, actual_burn)
    ledger.add_burned(node_id, actual_burn)

    RETURN BurnResult::Burned(actual_burn, reason)

ENUM BurnReason:
    DegradedMission       // mission completed below threshold
    ConstitutionalPenalty  // violated a constitutional gate
    HarbergerUnderbid      // someone else claimed at declared price
```

## Pseudocode: Gini Coefficient

```
FUNCTION compute_gini(ledger: &SEEDLedger) -> f64:
    // Standard Gini coefficient over all node balances.
    // 0.0 = perfect equality, 1.0 = one node owns everything.

    balances = ledger.all_balances().sorted()
    n = balances.len()

    IF n == 0 OR n == 1:
        RETURN 0.0

    // Gini = (2 * sum(i * balance_i)) / (n * sum(balance_i)) - (n + 1) / n
    numerator = 0.0
    total = 0.0
    FOR (i, balance) IN balances.enumerate():
        numerator += (i + 1) as f64 * balance
        total += balance

    IF total == 0.0:
        RETURN 0.0

    gini = (2.0 * numerator) / (n as f64 * total) - (n as f64 + 1.0) / n as f64
    RETURN gini.clamp(0.0, 1.0)

FUNCTION simulate_gini_after(
    ledger: &SEEDLedger,
    node_id: NodeId,
    additional: f64,
) -> f64:
    // Simulate what Gini would be if we added `additional` to node_id
    simulated = ledger.clone()
    simulated.credit(node_id, additional)
    RETURN compute_gini(simulated)
```

## Pseudocode: Zakat Pool Distribution

```
FUNCTION distribute_zakat(ledger: &mut SEEDLedger):
    // Zakat pool is redistributed periodically to lowest-balance nodes.
    // This is the constitutional wealth redistribution mechanism.

    pool = ledger.balance(ZAKAT_POOL)
    IF pool == 0.0:
        RETURN

    // Find nodes below median balance
    median = ledger.median_balance()
    recipients = ledger.nodes_below(median)

    IF recipients.is_empty():
        RETURN

    // Equal distribution to all below-median nodes
    per_node = pool / recipients.len() as f64

    FOR node_id IN recipients:
        ledger.transfer(ZAKAT_POOL, node_id, per_node)
```

## Pseudocode: BLOOM Reputation

```
STRUCT BLOOMScore:
    // BLOOM = soulbound reputation. Cannot be transferred or traded.
    // Accumulates from proven impact over time.
    cumulative_poi:   f64       // lifetime Proof of Impact sum
    mission_count:    u64       // total missions completed
    streak:           u32       // consecutive successful missions
    tier:             BloomTier

ENUM BloomTier:
    Seed        // 0-10 PoI
    Sprout      // 10-50 PoI
    Sapling     // 50-200 PoI
    Tree        // 200-1000 PoI
    Forest      // 1000+ PoI

FUNCTION update_bloom(
    bloom: &mut BLOOMScore,
    poi: f64,
    mission_success: bool,
) -> BloomTier:
    bloom.cumulative_poi += poi
    bloom.mission_count += 1

    IF mission_success:
        bloom.streak += 1
    ELSE:
        bloom.streak = 0

    bloom.tier = MATCH bloom.cumulative_poi:
        0.0..10.0    => BloomTier::Seed,
        10.0..50.0   => BloomTier::Sprout,
        50.0..200.0  => BloomTier::Sapling,
        200.0..1000.0 => BloomTier::Tree,
        _            => BloomTier::Forest,

    RETURN bloom.tier

INVARIANT bloom_is_soulbound:
    // BLOOM scores CANNOT be transferred between nodes.
    // BLOOM CANNOT be traded on the marketplace.
    // BLOOM is earned only through proven impact.
    ASSERT bloom.is_transferable == false
    ASSERT bloom.is_tradeable == false
```

## Pseudocode: Harberger Tax

```
FUNCTION apply_harberger_tax(
    ledger: &mut SEEDLedger,
    marketplace: &Marketplace,
):
    // Harberger tax: nodes declaring marketplace prices pay 5% annual tax.
    // This prevents hoarding capabilities at artificially low prices.
    // If you underprice, someone can buy at your declared price.

    FOR listing IN marketplace.active_listings():
        annual_tax = listing.price_seed * HARBERGER_TAX_RATE
        daily_tax = annual_tax / 365.0
        days_since_last = days_between(listing.last_taxed, now())
        tax_due = daily_tax * days_since_last

        balance = ledger.balance(listing.provider)
        IF balance >= tax_due:
            ledger.debit(listing.provider, tax_due)
            ledger.credit(COMMONS_POOL, tax_due)
            listing.last_taxed = now()
        ELSE:
            // Cannot pay tax — listing is delisted
            marketplace.remove(listing.id)
```

## Sovereignty Tiers

```
FUNCTION compute_sovereignty_tier(
    seed_balance: f64,
    all_trust_pass: bool,
) -> SovereigntyTier:
    // Sovereignty tier is derived from SEED balance + constitutional compliance.
    // Determines what capabilities and privileges a node has.

    MATCH (seed_balance, all_trust_pass):
        (b, true) IF b >= 100.0 => SOVEREIGN
        (b, true) IF b >= 10.0  => CITIZEN
        (_, true)               => SEEDLING
        (_, false)              => DEGRADED

ENUM SovereigntyTier:
    DEGRADED   // constitutional violation — restricted capabilities
    SEEDLING   // new node, building up SEED
    CITIZEN    // established, participating in URP
    SOVEREIGN  // full sovereignty — can federate, marketplace, govern
```

## TDD Anchors

```
TEST seed_reward_includes_zakat:
    ledger = SEEDLedger::new()
    result = mint_seed(ledger, node_a, raw_reward=1.0)
    ASSERT result IS Minted(0.975, 0.025)   // net, zakat
    ASSERT ledger.balance(node_a) == 0.975
    ASSERT ledger.balance(ZAKAT_POOL) == 0.025

TEST seed_burn_reduces_balance:
    ledger = SEEDLedger::new()
    mint_seed(ledger, node_a, 10.0)
    burn_seed(ledger, node_a, 0.1, BurnReason::DegradedMission)
    ASSERT ledger.balance(node_a) == 10.0 * (1.0 - ZAKAT_RATE) - 0.1

TEST gini_rejects_concentration:
    ledger = SEEDLedger::new()
    mint_seed(ledger, node_a, 1000.0)  // genesis: exempt
    // Now node_a has most of the supply
    result = mint_seed(ledger, node_a, 1000.0)  // non-genesis
    ASSERT result IS GiniRejected

TEST gini_allows_fair_distribution:
    ledger = SEEDLedger::new()
    FOR node IN [node_a, node_b, node_c, node_d, node_e]:
        mint_seed(ledger, node, 10.0)
    gini = compute_gini(ledger)
    ASSERT gini < ADL_GINI_THRESHOLD

TEST poi_uses_log_diminishing_returns:
    // 10 actions should NOT yield 10x the reward of 1 action
    reward_1 = compute_seed_reward(make_request(actions=1))
    reward_10 = compute_seed_reward(make_request(actions=10))
    ASSERT reward_10 < reward_1 * 10   // diminishing
    ASSERT reward_10 > reward_1         // but still more

TEST bloom_is_non_transferable:
    bloom = BLOOMScore::new()
    update_bloom(bloom, poi=5.0, success=true)
    ASSERT bloom.is_transferable == false
    ASSERT bloom.tier == BloomTier::Seed

TEST bloom_tiers_advance:
    bloom = BLOOMScore::new()
    update_bloom(bloom, poi=15.0, success=true)
    ASSERT bloom.tier == BloomTier::Sprout
    update_bloom(bloom, poi=100.0, success=true)
    ASSERT bloom.tier == BloomTier::Sapling

TEST zakat_pool_redistributes:
    ledger = SEEDLedger::new()
    mint_seed(ledger, rich_node, 1000.0)
    mint_seed(ledger, poor_node, 1.0)
    pool_before = ledger.balance(ZAKAT_POOL)
    distribute_zakat(ledger)
    ASSERT ledger.balance(ZAKAT_POOL) == 0.0
    ASSERT ledger.balance(poor_node) > 1.0 * (1.0 - ZAKAT_RATE)

TEST harberger_delists_on_nonpayment:
    ledger = SEEDLedger::new()
    marketplace = Marketplace::new()
    marketplace.list(node_a, Capability::Compute, price=100.0)
    ledger.set_balance(node_a, 0.0)  // cannot pay tax
    apply_harberger_tax(ledger, marketplace)
    ASSERT marketplace.active_listings().len() == 0

TEST sovereignty_tiers:
    ASSERT compute_sovereignty_tier(0.0, true) == SEEDLING
    ASSERT compute_sovereignty_tier(50.0, true) == CITIZEN
    ASSERT compute_sovereignty_tier(100.0, true) == SOVEREIGN
    ASSERT compute_sovereignty_tier(1000.0, false) == DEGRADED
```
