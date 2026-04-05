# 01 — Engram Tiered Memory Hierarchy

**Status:** SPEC  
**Extends:** `bizra-ttrl/src/engram.rs` (v0.90.0 HashMap-based EngramCache)  
**Crate:** `bizra-ttrl`

---

## 1. Motivation

v0.90.0's EngramCache is a single-tier `HashMap<[u8;32], EngramEntry>` in process memory.
This works at small scale but creates a hard scaling ceiling: at 300B parameters with 25%
Engram allocation, 75B parameters (~150 GB at fp16) cannot fit in GPU HBM.

v0.91.0 introduces a three-tier memory hierarchy that mirrors classical computer architecture
(L1/L2/L3 cache) applied to knowledge rather than data:

```
L1: GPU HBM    — hot patterns, ~80% of lookups, <1ms latency
L2: Host DRAM  — primary Engram tables, ~15% of lookups, 1-5ms latency
L3: NVMe SSD   — cold patterns, ~3% of lookups, 10-50ms latency
Miss:          — prefetch prediction failed, ~2%, must recompute
```

## 2. Data Structures

```pseudocode
struct TieredEngramCache:
    l1_cache: LruHashMap<[u8;32], EngramEntry>    # capacity from config
    l2_store: DramBackedMap<[u8;32], EngramEntry>  # mmap or arena-allocated
    l3_store: SsdBackedMap<[u8;32], EngramEntry>   # file-backed
    
    # Telemetry
    l1_hits: u64
    l2_hits: u64
    l3_hits: u64
    misses: u64
    
    # Configuration
    l1_capacity: usize         # max entries in HBM (from config)
    l2_capacity: usize         # max entries in DRAM
    promotion_threshold: u64   # hit_count to promote L2→L1
    demotion_threshold_ms: u64 # time since last hit to demote L1→L2

struct EngramEntry:             # existing, unchanged
    value: String
    confidence: f64
    written_at_ms: u64
    hit_count: u64

enum TierHit:
    L1Hit { value: String, confidence: f64 }
    L2Hit { value: String, confidence: f64 }
    L3Hit { value: String, confidence: f64 }
    Miss
```

## 3. Lookup Flow

```pseudocode
fn lookup(cache, intent_bytes, min_confidence) -> TierHit:
    key = blake3("engram/v1:" + intent_bytes)
    
    # L1 check (hot path, GPU HBM)
    if entry = cache.l1_cache.get_mut(key):
        if entry.confidence >= min_confidence:
            entry.hit_count += 1
            cache.l1_hits += 1
            return L1Hit(entry.value, entry.confidence)
    
    # L2 check (DRAM)
    if entry = cache.l2_store.get_mut(key):
        if entry.confidence >= min_confidence:
            entry.hit_count += 1
            cache.l2_hits += 1
            # Promotion check: if hot enough, schedule L2→L1 promotion
            if entry.hit_count >= cache.promotion_threshold:
                schedule_promotion(cache, key, entry)
            return L2Hit(entry.value, entry.confidence)
    
    # L3 check (NVMe SSD)
    if entry = cache.l3_store.get(key):
        if entry.confidence >= min_confidence:
            cache.l3_hits += 1
            # L3 hits always promote to L2 (warm the working set)
            schedule_promotion_l3_to_l2(cache, key, entry)
            return L3Hit(entry.value, entry.confidence)
    
    cache.misses += 1
    return Miss
```

## 4. Tier Migration

```pseudocode
fn run_tier_migration(cache, now_ms):
    """Background task — runs periodically (every 1000 cycles or 60 seconds)."""
    
    # L1 → L2 demotion: entries not hit recently
    for (key, entry) in cache.l1_cache.entries():
        time_since_hit = now_ms - entry.last_hit_ms()
        if time_since_hit > cache.demotion_threshold_ms:
            cache.l2_store.insert(key, entry)
            cache.l1_cache.remove(key)
    
    # L2 → L3 demotion: entries with low hit rate
    for (key, entry) in cache.l2_store.entries():
        if entry.hit_count < MIN_L2_RETAIN_HITS:
            if cache.l2_store.len() > cache.l2_capacity * 0.9:  # only if near capacity
                cache.l3_store.insert(key, entry)
                cache.l2_store.remove(key)
    
    # L1 promotion queue: process scheduled promotions
    while promotion = cache.promotion_queue.pop():
        if cache.l1_cache.len() < cache.l1_capacity:
            cache.l1_cache.insert(promotion.key, promotion.entry)
            cache.l2_store.remove(promotion.key)
        else:
            # L1 full — evict LRU from L1 to L2, then promote
            evicted = cache.l1_cache.evict_lru()
            cache.l2_store.insert(evicted.key, evicted.entry)
            cache.l1_cache.insert(promotion.key, promotion.entry)
            cache.l2_store.remove(promotion.key)

fn schedule_promotion(cache, key, entry):
    cache.promotion_queue.push(PromotionRequest { key, entry: entry.clone() })

fn schedule_promotion_l3_to_l2(cache, key, entry):
    cache.l2_store.insert(key, entry.clone())
```

## 5. Confidence Decay

```pseudocode
fn decay_confidence(cache, decay_rate, now_ms):
    """Entries lose confidence over time unless refreshed by the Engram compiler."""
    
    for tier in [cache.l1_cache, cache.l2_store, cache.l3_store]:
        for (key, entry) in tier.entries_mut():
            age_hours = (now_ms - entry.written_at_ms) / 3_600_000
            entry.confidence *= (1.0 - decay_rate).pow(age_hours)
    
    # Evict entries below the constitutional floor
    cache.l1_cache.retain(|_, e| e.confidence >= ENGRAM_EVICTION_FLOOR)
    cache.l2_store.retain(|_, e| e.confidence >= ENGRAM_EVICTION_FLOOR)
    cache.l3_store.retain(|_, e| e.confidence >= ENGRAM_EVICTION_FLOOR)
```

## 6. Hit Rate Telemetry

```pseudocode
fn hit_rate(cache) -> TieredHitRate:
    total = cache.l1_hits + cache.l2_hits + cache.l3_hits + cache.misses
    if total == 0:
        return TieredHitRate::zero()
    return TieredHitRate {
        l1: cache.l1_hits as f64 / total as f64,     # target: 0.80
        l2: cache.l2_hits as f64 / total as f64,     # target: 0.15
        l3: cache.l3_hits as f64 / total as f64,     # target: 0.03
        miss: cache.misses as f64 / total as f64,    # target: 0.02
        total_queries: total,
    }
```

## 7. Backward Compatibility

The existing `EngramCache` API surface must be preserved:
- `insert()`, `lookup()`, `lookup_readonly()`, `record_hit()`, `evict_stale()`, `hit_rate()`
- `OmniKernel.engram_cache_mut()` continues to work
- The `TieredEngramCache` wraps the existing API; callers don't see tiers unless they ask

```pseudocode
impl TieredEngramCache:
    # Backward-compatible: delegates to L1 first, then cascades
    fn lookup_compat(self, intent_bytes, min_confidence) -> EngramResult:
        match self.lookup(intent_bytes, min_confidence):
            L1Hit(v, c) | L2Hit(v, c) | L3Hit(v, c) => EngramResult::Hit { v, c }
            Miss => EngramResult::Miss
```

## 8. TDD Anchors

```
TEST tiered_01: L1 hit returns without L2/L3 access
    INSERT entry into L1 with confidence 0.99
    LOOKUP same key with min_confidence 0.95
    ASSERT result is L1Hit
    ASSERT l1_hits == 1, l2_hits == 0

TEST tiered_02: L1 miss falls through to L2
    INSERT entry into L2 only (not L1)
    LOOKUP same key
    ASSERT result is L2Hit
    ASSERT l2_hits == 1

TEST tiered_03: L2 miss falls through to L3
    INSERT entry into L3 only
    LOOKUP same key
    ASSERT result is L3Hit

TEST tiered_04: complete miss returns Miss
    LOOKUP key that doesn't exist in any tier
    ASSERT result is Miss
    ASSERT misses == 1

TEST tiered_05: promotion from L2 to L1 after threshold hits
    INSERT entry into L2 with hit_count = promotion_threshold - 1
    LOOKUP twice (hits threshold)
    RUN tier_migration
    ASSERT entry now in L1, removed from L2

TEST tiered_06: demotion from L1 to L2 after inactivity
    INSERT entry into L1 with last_hit long ago
    RUN tier_migration with now_ms far in the future
    ASSERT entry now in L2, removed from L1

TEST tiered_07: confidence decay reduces entry confidence over time
    INSERT entry with confidence 0.99 at time T
    RUN decay_confidence at time T + 24h
    ASSERT entry.confidence < 0.99

TEST tiered_08: backward-compatible lookup_compat returns EngramResult
    INSERT entry into L2
    result = LOOKUP_COMPAT same key
    ASSERT result == EngramResult::Hit { value, confidence }

TEST tiered_09: hit_rate telemetry sums to 1.0
    PERFORM 100 mixed lookups across all tiers
    rate = HIT_RATE()
    ASSERT abs(rate.l1 + rate.l2 + rate.l3 + rate.miss - 1.0) < 1e-9

TEST tiered_10: L1 eviction under capacity pressure promotes LRU to L2
    FILL L1 to capacity
    INSERT new hot entry that triggers promotion
    RUN tier_migration
    ASSERT L1 count == capacity (not exceeded)
    ASSERT evicted entry exists in L2
```

## 9. Edge Cases

- **Cold start**: All tiers empty. First N lookups are all misses. Prefetch pipeline (spec 02)
  handles warm-up by pre-loading likely entries from L3 into L2.
- **L2 at capacity**: Demotion to L3 triggers. If L3 is also at capacity, lowest-hit entries
  are evicted entirely (data loss — the Engram compiler must re-populate).
- **Concurrent access**: `lookup_readonly` path must not block tier migration. Use RwLock
  with migration holding the write lock only during batch moves.
- **Memory pressure**: If host DRAM is constrained, L2 capacity should auto-shrink.
  Implement a watermark-based backpressure mechanism.
