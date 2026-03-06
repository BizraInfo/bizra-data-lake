# Phase 67.07 — TDD Anchors (Test-First Contracts)
# ═════════════════════════════════════════════════

## Standing on Giants
- Beck (2002): Test-Driven Development by Example
- Hoare (1969): Axiomatic basis for computer programming
- Lamport (1978): Correctness proofs for distributed systems

## Purpose

This file consolidates ALL test contracts for Phase 67. Every test listed here
MUST pass before the corresponding implementation is considered complete.
Tests are organized by module, with clear preconditions and assertions.

## Test Structure

```
tests/
├── constitutional/
│   ├── conftest.py                     # Shared fixtures
│   ├── test_fixed_point.py             # Spec 01: Fixed-point arithmetic
│   ├── test_algorithms.py              # Spec 02: 15 native algorithms
│   ├── test_declaration.py             # Spec 03: Declaration genesis
│   ├── test_sovereignty_cli.py         # Spec 04: CLI commands
│   ├── test_chaos.py                   # Spec 06: 10 chaos validators
│   └── test_ticker.py                  # process_tick() heartbeat
├── akis/
│   ├── test_extractor.py              # Spec 05: Source detection + extraction
│   ├── test_relevance.py              # Spec 05: BIZRA relevance scoring
│   └── test_formatter.py             # Spec 05: Output formatting
└── cross_language/
    └── test_fixed_point_parity.py     # Python ↔ Rust determinism
```

## Shared Fixtures (conftest.py)

```python
# tests/constitutional/conftest.py

import pytest
import random
from core.constitutional.fixed_point import fp, fp_float, FP_PRECISION
from core.constitutional.types import ActionReceipt, WalletState

@pytest.fixture
def deterministic_seed():
    """All chaos tests use seed 42 for reproducibility."""
    random.seed(42)
    yield
    # No teardown needed

@pytest.fixture
def quality_receipt():
    """A receipt that passes both intent gate and ihsan check."""
    return ActionReceipt(
        receipt_id=b'\x01' * 32,
        actor_id=b'\x02' * 32,
        action_type="contribution",
        timestamp=1741392000000,
        intent_score=fp(0.95),
        efficiency_score=fp(0.90),
        impact_score=fp(0.92),
        reproducibility_score=fp(0.88),
        oracle_signature=b'\x03' * 64,
        metadata_hash=b'\x04' * 32,
        co_actors=[]
    )

@pytest.fixture
def low_intent_receipt():
    """A receipt that fails the Al-Ghazali intent gate."""
    return ActionReceipt(
        receipt_id=b'\x05' * 32,
        actor_id=b'\x06' * 32,
        action_type="spam",
        timestamp=1741392000000,
        intent_score=fp(0.50),  # Below 0.90 floor
        efficiency_score=fp(0.90),
        impact_score=fp(0.90),
        reproducibility_score=fp(0.90),
        oracle_signature=b'\x07' * 64,
        metadata_hash=b'\x08' * 32,
        co_actors=[]
    )

@pytest.fixture
def newcomer_wallet():
    """A freshly initialized wallet with zero balance."""
    return WalletState(
        node_id=b'\x10' * 32,
        seed_balance=0,
        bloom_balance=0,
        created_at=1741392000000,
        last_active=1741392000000,
    )

@pytest.fixture
def wealthy_wallet():
    """A wallet with substantial balance."""
    return WalletState(
        node_id=b'\x20' * 32,
        seed_balance=fp(5000),
        bloom_balance=fp(10),
        created_at=1741000000000,
        last_active=1741392000000,
        total_actions=500,
        ihsan_history=[fp(0.96)] * 50,
    )

@pytest.fixture
def network_wallets(deterministic_seed):
    """50-node network for Asabiyyah and Gini tests."""
    wallets = []
    for i in range(50):
        balance = fp(random.uniform(10, 1000))
        wallets.append(WalletState(
            node_id=bytes([i]) * 32,
            seed_balance=balance,
            created_at=1741392000000,
            last_active=1741392000000,
        ))
    return wallets
```

## Test Contracts by Module

### Fixed-Point Arithmetic (13 tests)

```python
# tests/constitutional/test_fixed_point.py

class TestFpConversions:
    def test_fp_roundtrip_zero(self): ...
    def test_fp_roundtrip_one(self): ...
    def test_fp_roundtrip_fractional(self): ...
    def test_fp_negative_disallowed(self): ...

class TestFpArithmetic:
    def test_fp_mul_precision(self): ...
    def test_fp_mul_commutative(self): ...
    def test_fp_div_by_zero_returns_zero(self): ...
    def test_fp_div_precision(self): ...
    def test_fp_add_overflow_guard(self): ...

class TestFpDerived:
    def test_fp_clamp_below(self): ...
    def test_fp_clamp_above(self): ...
    def test_fp_clamp_within(self): ...
    def test_fp_weighted_avg(self): ...

class TestFpDeterminism:
    def test_1000_iterations_same_result(self): ...
```

### Native Algorithms (35 tests)

```python
# tests/constitutional/test_algorithms.py

class TestA1IntentGate:
    def test_passes_above_floor(self, quality_receipt): ...
    def test_rejects_below_floor(self, low_intent_receipt): ...
    def test_boundary_at_exactly_0_90(self): ...

class TestA1IhsanScore:
    def test_quality_receipt_above_threshold(self, quality_receipt): ...
    def test_score_in_range_zero_to_one(self): ...
    def test_weighted_sum_correct(self): ...

class TestA2SeedMinter:
    def test_mints_for_quality_work(self, quality_receipt): ...
    def test_zero_mint_below_threshold(self, low_intent_receipt): ...
    def test_efficiency_bonus(self): ...

class TestA3BloomAccumulator:
    def test_accrues_on_high_ihsan(self, newcomer_wallet): ...
    def test_no_accrual_below_threshold(self, newcomer_wallet): ...
    def test_decay_on_inactivity(self, wealthy_wallet): ...

class TestA4GiniEnforcer:
    def test_compute_gini_equal_distribution(self): ...
    def test_compute_gini_extreme_inequality(self): ...
    def test_khaldunian_throttle_healthy(self): ...
    def test_khaldunian_throttle_warning(self): ...
    def test_khaldunian_throttle_crisis(self): ...
    def test_khaldunian_throttle_never_zero(self): ...
    def test_ghazali_equity_newcomer_advantage(self, newcomer_wallet): ...
    def test_ghazali_equity_wealthy_standard(self, wealthy_wallet): ...
    def test_ghazali_equity_capped_at_max(self): ...

class TestA5ZakatEngine:
    def test_zakat_above_nisab(self, wealthy_wallet): ...
    def test_zakat_below_nisab_exempt(self, newcomer_wallet): ...
    def test_zakat_rate_exactly_2_5_percent(self): ...

class TestA7Demurrage:
    def test_active_node_exempt(self): ...
    def test_idle_node_taxed(self): ...
    def test_demurrage_never_negative(self): ...

class TestA8ShuraGovernance:
    def test_bloom_weighted_vote(self): ...
    def test_zero_bloom_no_vote(self): ...
    def test_supermajority_passes(self): ...
    def test_minority_rejected(self): ...

class TestA10ReflexCompiler:
    def test_compile_and_lookup(self): ...
    def test_cache_miss_returns_none(self): ...

class TestA14EventSourcer:
    def test_append_creates_chain(self): ...
    def test_chain_integrity(self): ...

class TestA15Asabiyyah:
    def test_score_zero_for_isolated_node(self): ...
    def test_score_increases_with_attestations(self): ...
    def test_network_asabiyyah_average(self, network_wallets): ...
```

### Declaration Genesis (7 tests)

```python
# tests/constitutional/test_declaration.py

class TestDeclaration:
    def test_hash_matches_canonical(self): ...
    def test_modified_text_fails(self): ...
    def test_genesis_event_structure(self): ...
    def test_covenant_chain_valid(self): ...
    def test_covenant_chain_broken(self): ...
    def test_all_seven_invariants(self): ...
    def test_constitutional_violation_exception(self): ...
```

### Sovereignty CLI (8 tests)

```python
# tests/constitutional/test_sovereignty_cli.py

class TestBizraInit:
    def test_creates_identity(self, tmp_path): ...
    def test_creates_wallet(self, tmp_path): ...
    def test_creates_event_log_with_genesis(self, tmp_path): ...
    def test_fails_on_tampered_declaration(self, tmp_path): ...

class TestBizraWork:
    def test_mints_seed(self, tmp_path): ...
    def test_rejects_low_intent(self, tmp_path): ...

class TestBizraAttest:
    def test_builds_asabiyyah(self, tmp_path): ...

class TestBizraStatus:
    def test_shows_trajectory(self, tmp_path): ...
```

### Chaos Validators (10 tests)

```python
# tests/constitutional/test_chaos.py (see Spec 06 for full pseudocode)

class TestChaos:
    @pytest.mark.slow
    def test_t1_gini_stagnation(self, deterministic_seed): ...
    def test_t2_whale_attack(self, deterministic_seed): ...
    @pytest.mark.slow
    def test_t3_sybil_flood(self, deterministic_seed): ...
    def test_t4_ghost_town(self, deterministic_seed): ...
    def test_t5_reflex_cache_performance(self): ...
    def test_t6_backing_collapse(self): ...
    @pytest.mark.slow
    def test_t7_event_log_integrity(self, deterministic_seed): ...
    def test_t8_khaldunian_vs_binary(self, deterministic_seed): ...
    def test_t9_newcomer_equity(self, deterministic_seed): ...
    def test_t10_asabiyyah_emergence(self, deterministic_seed): ...
```

### AKIS Pipeline (8 tests)

```python
# tests/akis/test_extractor.py

class TestSourceDetection:
    def test_youtube_video(self): ...
    def test_youtube_channel(self): ...
    def test_github(self): ...
    def test_generic_web(self): ...

class TestParsing:
    def test_vtt_deduplication(self): ...
    def test_vtt_strips_timestamps(self): ...

# tests/akis/test_relevance.py

class TestRelevanceScoring:
    def test_high_relevance(self): ...
    def test_low_relevance(self): ...
```

### Ticker (3 tests)

```python
# tests/constitutional/test_ticker.py

class TestProcessTick:
    def test_12_step_heartbeat(self, network_wallets): ...
    def test_intent_rejection_count(self): ...
    def test_gini_computed_correctly(self, network_wallets): ...
```

## Total Test Count

| Module | Tests |
|--------|------:|
| Fixed-point | 13 |
| Algorithms | 35 |
| Declaration | 7 |
| CLI | 8 |
| Chaos | 10 |
| AKIS | 8 |
| Ticker | 3 |
| Cross-language | 2 |
| **Total** | **86** |

## Implementation Order (TDD)

1. **fixed_point.py** — Write tests first, then implement. Foundation of everything.
2. **types.py** — Data structures. No logic, just structure.
3. **algorithms.py** — One algorithm at a time: test → implement → verify.
   Order: A1 → A4 → A2 → A5 → A3 → A7 → A8 → A10 → A14 → A15 → rest.
4. **declaration.py** — Genesis block handler.
5. **ticker.py** — Orchestrates all algorithms in process_tick().
6. **cli.py** — Sovereignty commands wired to implementations.
7. **chaos tests** — Final validation: all 10 must pass.
8. **akis/** — Knowledge extraction pipeline (parallel track).

## CI Integration

Add to `.github/workflows/ci.yml`:

```yaml
- name: Constitutional Tests
  run: |
    source .venv-linux/bin/activate
    pytest tests/constitutional/ -q --tb=short

- name: AKIS Tests
  run: |
    source .venv-linux/bin/activate
    pytest tests/akis/ -q --tb=short

- name: Chaos Validators (slow)
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  run: |
    source .venv-linux/bin/activate
    pytest tests/constitutional/test_chaos.py -q --tb=short -m slow
```
