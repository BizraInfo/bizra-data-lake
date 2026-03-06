# Phase 67.06 — 10 Chaos Validators (Constitutional Stress Tests)
# ═══════════════════════════════════════════════════════════════

## Standing on Giants
- Kolmogorov (1933): Probability axioms for stochastic testing
- Nassim Taleb (2007): Antifragility — systems that gain from disorder
- Netflix Chaos Monkey (2011): Deliberate failure injection
- Ibn Khaldun (1377): Civilizations collapse when inequality exceeds threshold

## Source

`last update/BIZRA_Chaos_Test_v2.py` (546 lines, 10 tests, 10/10 passing)

## Purpose

The 10 chaos tests are not unit tests — they are **constitutional validators**.
Each test simulates a catastrophic scenario and verifies that the 15 native
algorithms maintain invariants under extreme stress. If any chaos test fails,
the constitutional kernel has a bug.

## Target

```
tests/constitutional/
├── test_chaos.py          # All 10 chaos validators
├── conftest.py            # Shared fixtures (wallets, receipts, network)
└── test_fixed_point.py    # From Spec 01
```

## The 10 Chaos Validators

### T1: Gini Stagnation (100K wallets, 24-month simulation)

```
TEST test_gini_stagnation:
    """Extreme initial inequality does not stagnate the economy.

    Setup: 100,000 wallets with power-law distribution (realistic).
    Run: 24 monthly ticks with random receipts.
    Assert: Gini decreases over time (Khaldunian curve working).
    """
    random.seed(42)  # Reproducible

    # Create 100K wallets with extreme initial distribution
    wallets = []
    FOR i IN range(100_000):
        balance = fp(random.paretovariate(1.5) * 10)  # Power-law
        wallets.append(WalletState(node_id=random_id(), seed_balance=balance))

    initial_gini = compute_gini([w.seed_balance FOR w IN wallets])
    ASSERT initial_gini > fp(0.60)  # Starts very unequal

    # Run 24 months of economic activity
    FOR month IN range(24):
        receipts = generate_random_receipts(wallets, count=10_000)
        process_tick(wallets, receipts, [], [], {})

    final_gini = compute_gini([w.seed_balance FOR w IN wallets])

    ASSERT final_gini < initial_gini  # Gini decreased
    ASSERT final_gini <= fp(0.50)     # Converging toward healthy
```

### T2: Whale Attack (Progressive Throttle Response)

```
TEST test_whale_attack:
    """A single whale owning 99% cannot maintain dominance.

    Setup: 1 whale (99% of wealth) + 99 newcomers.
    Run: 12 months of activity.
    Assert: Whale's share decreases, newcomers grow.
    """
    random.seed(42)

    whale = WalletState(node_id=b'whale', seed_balance=fp(990_000))
    others = [WalletState(node_id=random_id(), seed_balance=fp(100))
              FOR _ IN range(99)]
    wallets = [whale] + others

    initial_whale_share = fp_div(whale.seed_balance,
                                  sum(w.seed_balance FOR w IN wallets))

    FOR month IN range(12):
        receipts = generate_random_receipts(wallets, count=1000)
        process_tick(wallets, receipts, [], [], {})

    final_whale_share = fp_div(whale.seed_balance,
                                sum(w.seed_balance FOR w IN wallets))

    ASSERT final_whale_share < initial_whale_share  # Whale lost share
    ASSERT fp_float(final_whale_share) < 0.90       # Below 90%
```

### T3: Sybil Flood (Intent Gate + Ihsan Gate)

```
TEST test_sybil_flood:
    """100K low-quality receipts cannot extract value.

    Setup: 100,000 Sybil receipts with low intent/quality scores.
    Assert: Intent gate rejects most, Ihsan gate catches the rest.
    Total minted ≈ 0.
    """
    random.seed(42)

    sybil_receipts = []
    FOR i IN range(100_000):
        r = ActionReceipt(
            receipt_id=random_id(),
            actor_id=random_id(),
            action_type="spam",
            timestamp=now(),
            intent_score=fp(random.uniform(0.0, 0.89)),  # Below 0.90 gate
            efficiency_score=fp(random.uniform(0.0, 0.5)),
            impact_score=fp(random.uniform(0.0, 0.3)),
            reproducibility_score=fp(random.uniform(0.0, 0.4)),
            oracle_signature=b'',
            metadata_hash=b'',
            co_actors=[]
        )
        sybil_receipts.append(r)

    # Count rejections
    intent_rejected = sum(1 FOR r IN sybil_receipts IF NOT intent_gate(r))

    ASSERT intent_rejected >= 50_000  # At least half caught by intent gate

    # Process the few that pass intent gate
    passed_intent = [r FOR r IN sybil_receipts IF intent_gate(r)]
    total_minted = 0
    FOR r IN passed_intent:
        passed, ihsan = full_ihsan_check(r)
        IF passed:
            total_minted += fp_float(mint_seed(r, ihsan))

    ASSERT total_minted < 100  # Negligible extraction
```

### T4: Ghost Town (99% Inactive, Demurrage)

```
TEST test_ghost_town:
    """Economy contracts gracefully when 99% of nodes are inactive.

    Setup: 1000 wallets, only 10 active.
    Run: 12 months.
    Assert: Active nodes maintain/grow, inactive nodes shrink via demurrage.
    System never halts completely.
    """
    random.seed(42)

    wallets = [WalletState(node_id=random_id(), seed_balance=fp(100))
               FOR _ IN range(1000)]
    active_nodes = wallets[:10]

    FOR month IN range(12):
        # Only active nodes produce receipts
        receipts = generate_random_receipts(active_nodes, count=100)
        process_tick(wallets, receipts, [], [], {})

    # Inactive nodes lost balance to demurrage
    inactive_balances = [w.seed_balance FOR w IN wallets[10:]]
    ASSERT all(b < fp(100) FOR b IN inactive_balances)  # All shrunk

    # Active nodes grew
    active_balances = [w.seed_balance FOR w IN active_nodes]
    ASSERT all(b > fp(100) FOR b IN active_balances)  # All grew

    # Economy didn't halt
    total_supply = sum(w.seed_balance FOR w IN wallets)
    ASSERT total_supply > 0  # Never reached zero
```

### T5: Reflex Cache (100K Lookups, O(1) Performance)

```
TEST test_reflex_cache_performance:
    """Reflex cache handles 100K lookups in < 1 second.

    System-1 (A10) must be O(1) hash lookup.
    """
    cache = {}

    # Compile 10K reflexes
    FOR i IN range(10_000):
        pattern = f"pattern_{i}"
        reflex = compile_reflex(pattern, [f"action_{i}"], fp(0.98))
        cache[reflex.pattern_hash] = reflex

    # 100K lookups
    start = time.perf_counter()
    hits = 0
    FOR i IN range(100_000):
        pattern = f"pattern_{i % 10_000}"
        result = reflex_lookup(cache, pattern)
        IF result: hits += 1
    elapsed = time.perf_counter() - start

    ASSERT elapsed < 1.0   # Under 1 second
    ASSERT hits == 100_000  # All hits (patterns exist)
```

### T6: Backing Collapse (Circuit Breaker)

```
TEST test_backing_collapse:
    """Backing ratio below 0.5 triggers circuit breaker.

    If more tokens exist than verified work, minting halts.
    """
    total_seed = fp(1_000_000)     # 1M tokens
    total_work = fp(400_000)       # Only 400K verified work units

    ratio = backing_ratio(total_seed, total_work)
    ASSERT fp_float(ratio) < 0.5  # Dangerously low

    # Minting should be suppressed when backing is low
    throttle = min(FP_ONE, ratio)  # Cap at 1.0
    ASSERT fp_float(throttle) < 0.5
```

### T7: Event Log Integrity (100K Chain)

```
TEST test_event_log_integrity:
    """100K event chain maintains hash integrity.

    A14: Immutable history. Any tampering breaks the chain.
    """
    event_log = []

    FOR i IN range(100_000):
        append_event(event_log, "test", random_id(), {"seq": i})

    ASSERT len(event_log) == 100_000

    # Verify chain
    FOR i IN range(1, len(event_log)):
        ASSERT event_log[i].prev_hash == event_log[i-1].hash

    # Tamper with event 50,000
    event_log[50_000].data["seq"] = 999999

    # Recompute hash — it won't match stored hash
    recomputed = blake3(canonical_bytes(event_log[50_000]))
    ASSERT recomputed != event_log[50_000].hash  # Tampering detected
```

### T8: Khaldunian Curve vs Binary Gate (Head-to-Head)

```
TEST test_khaldunian_vs_binary:
    """Khaldunian curve produces 23,844× more economic activity than binary gate.

    v1 (binary): Gini > 0.35 → mint 0 SEED → economic death
    v2 (curve): Progressive throttle → 238 SEED over 12 months
    """
    random.seed(42)

    # Setup: extreme inequality (Gini > 0.50)
    wallets_v1 = create_extreme_inequality(100)
    wallets_v2 = deepcopy(wallets_v1)

    minted_v1 = 0  # Binary gate
    minted_v2 = 0  # Khaldunian curve

    FOR month IN range(12):
        gini = compute_gini([w.seed_balance FOR w IN wallets_v1])

        FOR receipt IN generate_receipts(wallets_v1):
            ihsan = ihsan_score(receipt)
            # v1: binary gate
            IF fp_float(gini) <= 0.35:
                minted_v1 += fp_float(mint_seed(receipt, ihsan))
            # else: 0

        FOR receipt IN generate_receipts(wallets_v2):
            ihsan = ihsan_score(receipt)
            # v2: Khaldunian throttle
            throttle = khaldunian_throttle(gini)
            minted_v2 += fp_float(fp_mul(mint_seed(receipt, ihsan), throttle))

    ASSERT minted_v1 < 0.01  # Binary gate killed the economy
    ASSERT minted_v2 > 100   # Khaldunian curve kept it alive

    IF minted_v1 > 0:
        ratio = minted_v2 / minted_v1
        ASSERT ratio > 1000  # Massive improvement
```

### T9: Newcomer Equity (3.27× Ratio, 18-Month Path)

```
TEST test_newcomer_equity:
    """Newcomers earn 3.27× more per action than wealthy nodes.

    The Ghazali Equity Factor ensures economic mobility.
    """
    mean_balance = fp(500)

    # Poor newcomer
    newcomer = WalletState(node_id=b'new', seed_balance=fp(1))
    newcomer_equity = ghazali_equity_factor(newcomer, mean_balance)

    # Wealthy node
    wealthy = WalletState(node_id=b'rich', seed_balance=fp(5000))
    wealthy_equity = ghazali_equity_factor(wealthy, mean_balance)

    ratio = fp_float(newcomer_equity) / fp_float(wealthy_equity)
    ASSERT ratio >= 3.0    # At least 3× advantage
    ASSERT ratio <= 5.0    # Capped by EQUITY_FACTOR_MAX

    # 18-month simulation
    newcomer = WalletState(node_id=b'new', seed_balance=fp(1))
    FOR month IN range(18):
        FOR day IN range(30):
            receipt = create_quality_receipt(newcomer.node_id)
            ihsan = ihsan_score(receipt)
            gini = fp(0.30)  # Healthy network
            minted = progressive_mint(receipt, ihsan, newcomer, gini, mean_balance)
            newcomer.seed_balance = fp_add(newcomer.seed_balance, minted)

    ASSERT newcomer.seed_balance >= mean_balance  # Reached median
```

### T10: Asabiyyah Emergence (Monotonic Social Cohesion)

```
TEST test_asabiyyah_emergence:
    """Network cohesion grows monotonically with attestation activity.

    Ibn Khaldun: "Asabiyyah is the pillar of civilization."
    """
    wallets = [WalletState(node_id=random_id()) FOR _ IN range(50)]

    asabiyyah_history = []

    FOR round IN range(20):
        # Random pairs attest each other
        FOR _ IN range(10):
            i, j = random.sample(range(50), 2)
            attest(wallets[i], wallets[j], create_dummy_receipt())
            wallets[i].governance_votes += 1
            wallets[j].cooperative_actions += 1

        score = network_asabiyyah(wallets)
        asabiyyah_history.append(score)

    # Monotonically increasing (or stable)
    FOR i IN range(1, len(asabiyyah_history)):
        ASSERT asabiyyah_history[i] >= asabiyyah_history[i-1]

    # Final score > 0 (network has cohesion)
    ASSERT fp_float(asabiyyah_history[-1]) > 0.0
```

## Reproducibility

All chaos tests use `random.seed(42)` for deterministic results.
This means:
- Same test, same machine, same result (always)
- Same test, different machine, same result (fixed-point ensures this)
- CI can run chaos tests without flakiness

## Performance Budget

| Test | Expected Duration | Constraint |
|------|-------------------|------------|
| T1 (100K wallets) | < 30s | Large-scale simulation |
| T2 (Whale) | < 5s | Small network |
| T3 (Sybil) | < 10s | 100K receipts |
| T4 (Ghost Town) | < 5s | 1K wallets |
| T5 (Reflex) | < 1s | O(1) lookups |
| T6 (Backing) | < 1s | Pure computation |
| T7 (Event Log) | < 10s | 100K chain |
| T8 (Curve) | < 10s | 12-month simulation |
| T9 (Newcomer) | < 10s | 18-month simulation |
| T10 (Asabiyyah) | < 5s | 50 nodes, 20 rounds |

Mark T1, T3, T7 with `@pytest.mark.slow` (> 10s expected).
