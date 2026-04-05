# 02 — Deterministic Prefetch Pipeline

**Status:** SPEC  
**New module:** `bizra-ttrl/src/prefetch.rs`  
**Depends on:** 01_engram_tiered (L1/L2/L3 tiers)

---

## 1. Motivation

The tiered memory hierarchy introduces a latency gap: L2 DRAM access (1-5ms) and L3 NVMe
access (10-50ms) are orders of magnitude slower than L1 HBM (<1ms). Without mitigation,
every L2/L3 hit stalls the GPU pipeline waiting for data transfer over PCIe.

The prefetch pipeline eliminates this stall by **predicting** which Engram entries will be
needed in the next N tokens and transferring them from L2/L3 to L1 *before* the lookup
occurs. PCIe transfers overlap with GPU computation on the current token, masking the
latency entirely.

### Performance Targets

| Metric | Week 1 | Week 4 | Steady-state |
|--------|--------|--------|--------------|
| L1 Hit Rate | 45% | 70% | 81% |
| PCIe BW Utilization | 52% | 72% | 82% |
| Prefetch Miss Rate | 12% | 8% | 2% |
| Throughput Penalty | 27% | 8% | <3.1% |

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GPU Compute Pipeline                       │
│  Token N processing on GPU (attention, MoE routing, etc.)    │
└────────────────────────┬────────────────────────────────────┘
                         │ concurrent with
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Prefetch Pipeline (CPU)                     │
│                                                               │
│  1. Predict: which Engram keys will Token N+1..N+W need?     │
│  2. Check:   are those keys already in L1?                   │
│  3. Fetch:   issue async PCIe DMA for L2→L1 transfers        │
│  4. Track:   update prediction model with hit/miss feedback   │
└─────────────────────────────────────────────────────────────┘
```

## 3. Prediction Model

```pseudocode
struct PrefetchPredictor:
    # N-gram frequency table: maps (key_t-2, key_t-1) → predicted key_t
    bigram_table: HashMap<([u8;32], [u8;32]), PrefetchCandidate>
    
    # Fallback: unigram frequency (most commonly accessed keys)
    unigram_freq: BTreeMap<u64, [u8;32]>  # sorted by frequency descending
    
    # Sliding window of recent lookups
    recent_keys: CircularBuffer<[u8;32]>  # window size W (configurable)
    
    # Prediction accuracy tracking
    predictions_made: u64
    predictions_correct: u64

struct PrefetchCandidate:
    predicted_key: [u8;32]
    frequency: u64
    confidence: f64  # hit_rate of this prediction

fn predict_next_keys(predictor, current_key, prev_key, count) -> Vec<[u8;32]>:
    candidates = []
    
    # Strategy 1: Bigram lookup (highest accuracy)
    if bigram = predictor.bigram_table.get((prev_key, current_key)):
        if bigram.confidence >= PREFETCH_MIN_CONFIDENCE:
            candidates.push(bigram.predicted_key)
    
    # Strategy 2: Unigram fallback (fill remaining slots)
    for (freq, key) in predictor.unigram_freq.iter().take(count - candidates.len()):
        if key not in candidates:
            candidates.push(key)
    
    # Strategy 3: Locality heuristic (keys with similar BLAKE3 prefix)
    # Skip — this is a future optimization
    
    return candidates[..min(count, candidates.len())]
```

## 4. Prefetch Execution

```pseudocode
struct PrefetchPipeline:
    predictor: PrefetchPredictor
    in_flight: HashSet<[u8;32]>  # keys currently being transferred
    prefetch_window: usize       # how many tokens ahead to predict (default: 4)
    max_in_flight: usize         # PCIe concurrency limit (default: 8)
    
    # Telemetry
    transfers_initiated: u64
    transfers_completed: u64
    transfer_bytes: u64
    pcie_utilization: f64        # EMA of bandwidth fraction used

fn prefetch_step(pipeline, cache, current_key, prev_key):
    """Called once per token, concurrent with GPU computation."""
    
    # 1. Predict next keys
    predicted = predict_next_keys(
        pipeline.predictor,
        current_key,
        prev_key,
        pipeline.prefetch_window
    )
    
    # 2. Filter: skip keys already in L1 or already in-flight
    to_fetch = []
    for key in predicted:
        if not cache.l1_cache.contains(key) and key not in pipeline.in_flight:
            to_fetch.push(key)
    
    # 3. Respect PCIe concurrency limit
    available_slots = pipeline.max_in_flight - pipeline.in_flight.len()
    to_fetch = to_fetch[..min(available_slots, to_fetch.len())]
    
    # 4. Issue async transfers
    for key in to_fetch:
        if entry = cache.l2_store.get(key):
            issue_async_transfer(cache.l1_cache, key, entry)
            pipeline.in_flight.insert(key)
            pipeline.transfers_initiated += 1
        elif entry = cache.l3_store.get(key):
            # L3→L2 first, then L2→L1 in next cycle
            issue_async_transfer(cache.l2_store, key, entry)
            pipeline.in_flight.insert(key)
            pipeline.transfers_initiated += 1
    
    # 5. Collect completed transfers
    for key in poll_completed_transfers():
        pipeline.in_flight.remove(key)
        pipeline.transfers_completed += 1

fn update_predictor(pipeline, actual_key):
    """Called after each lookup to train the prediction model."""
    
    pipeline.predictor.predictions_made += 1
    
    # Was this key predicted?
    if actual_key in pipeline.last_predicted_set:
        pipeline.predictor.predictions_correct += 1
    
    # Update bigram table
    if len(pipeline.predictor.recent_keys) >= 2:
        prev2 = pipeline.predictor.recent_keys[-2]
        prev1 = pipeline.predictor.recent_keys[-1]
        bigram_key = (prev2, prev1)
        entry = pipeline.predictor.bigram_table.get_or_default(bigram_key)
        if entry.predicted_key == actual_key:
            entry.frequency += 1
            entry.confidence = entry.frequency as f64 / (entry.frequency + entry.miss_count) as f64
        else:
            entry.miss_count += 1
            # If new key is more frequent, replace prediction
            if should_replace(entry, actual_key):
                entry.predicted_key = actual_key
                entry.frequency = 1
                entry.miss_count = 0
    
    # Update unigram frequency
    pipeline.predictor.unigram_freq.increment(actual_key)
    
    # Append to sliding window
    pipeline.predictor.recent_keys.push(actual_key)
```

## 5. PCIe Bandwidth Management

```pseudocode
struct PcieBandwidthTracker:
    theoretical_bw_gbps: f64   # PCIe Gen4 x16 = ~25 GB/s, Gen5 = ~50 GB/s
    ema_alpha: f64             # smoothing factor (0.05)
    current_utilization: f64   # EMA of actual/theoretical

fn update_pcie_utilization(tracker, bytes_transferred, elapsed_ms):
    actual_gbps = (bytes_transferred as f64 / 1e9) / (elapsed_ms / 1000.0)
    instant_utilization = actual_gbps / tracker.theoretical_bw_gbps
    tracker.current_utilization = tracker.ema_alpha * instant_utilization
        + (1.0 - tracker.ema_alpha) * tracker.current_utilization

fn should_throttle(tracker) -> bool:
    # Don't consume more than 90% of PCIe — leave headroom for other DMA
    tracker.current_utilization > 0.90
```

## 6. Calibration Period

```pseudocode
fn is_calibrating(pipeline, start_ms, now_ms) -> bool:
    """First 2 weeks = calibration period. Throughput may be lower than v0.90.0."""
    elapsed_days = (now_ms - start_ms) / 86_400_000
    return elapsed_days < PREFETCH_CALIBRATION_DAYS  # 14

fn calibration_status(pipeline) -> CalibrationReport:
    accuracy = pipeline.predictor.predictions_correct as f64
        / max(1, pipeline.predictor.predictions_made) as f64
    return CalibrationReport {
        accuracy,
        l1_hit_rate: compute_l1_rate(pipeline),
        pcie_utilization: pipeline.pcie_tracker.current_utilization,
        is_mature: accuracy >= PREFETCH_MATURE_ACCURACY,  # 0.80
    }
```

## 7. TDD Anchors

```
TEST prefetch_01: bigram prediction returns correct key for known sequence
    TRAIN predictor with sequence [A, B, C] repeated 10 times
    PREDICT next after (B, C)
    ASSERT predicted == A

TEST prefetch_02: L1 miss triggers async L2→L1 transfer
    INSERT entry into L2 only
    PREDICT that entry's key
    RUN prefetch_step
    ASSERT key is in in_flight set
    COMPLETE transfer
    ASSERT entry now in L1

TEST prefetch_03: respects max_in_flight limit
    SET max_in_flight = 2
    PREDICT 5 keys not in L1
    RUN prefetch_step
    ASSERT in_flight.len() == 2, not 5

TEST prefetch_04: already-in-L1 keys are not re-fetched
    INSERT entry into L1
    PREDICT that key
    RUN prefetch_step
    ASSERT transfers_initiated unchanged

TEST prefetch_05: predictor accuracy improves with training
    RUN 100 cycles with repeated patterns
    ASSERT prediction accuracy > 0.70

TEST prefetch_06: PCIe throttle activates above 90%
    SET current_utilization = 0.92
    ASSERT should_throttle() == true
    SET current_utilization = 0.80
    ASSERT should_throttle() == false

TEST prefetch_07: calibration_status reports immature during first 14 days
    CREATE pipeline at time T
    CHECK calibration_status at T + 7 days
    ASSERT is_mature == false (accuracy too low)

TEST prefetch_08: L3 hit promotes to L2 first, then L1 in next cycle
    INSERT entry into L3 only
    PREDICT that key
    RUN prefetch_step
    ASSERT entry now in L2 (not yet L1)
    RUN prefetch_step again
    ASSERT entry promoted to L1 (if hit count met)

TEST prefetch_09: unigram fallback fills when bigram has no match
    TRAIN predictor with unigram frequencies only (no bigrams)
    PREDICT next keys
    ASSERT predicted keys are top-frequency unigrams

TEST prefetch_10: update_predictor tracks correct hit/miss
    PREDICT key X
    ACTUAL lookup is key X
    ASSERT predictions_correct incremented
    PREDICT key X
    ACTUAL lookup is key Y
    ASSERT predictions_correct NOT incremented
```

## 8. Edge Cases

- **Cold start (empty predictor)**: Unigram fallback is also empty. First N lookups are all
  misses. The pipeline gracefully does nothing until the predictor accumulates data.
- **Workload shift**: If the task category changes (entity-heavy → reasoning-heavy), the
  bigram table becomes stale. Implement exponential decay on bigram frequencies to adapt.
- **PCIe contention**: Other DMA operations (model weight loading, gradient sync) compete
  for PCIe bandwidth. The throttle mechanism prevents the prefetch from starving other
  transfers.
- **SSD failure**: If L3 reads fail, the prefetch skips L3 entries gracefully. The miss rate
  increases but the system continues to function with L1/L2 only.
