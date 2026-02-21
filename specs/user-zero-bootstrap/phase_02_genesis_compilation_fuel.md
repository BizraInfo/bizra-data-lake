# Phase 02: GENESIS Compilation Fuel

**Status:** SPEC | **Dependencies:** Phase 01 (`conversations_unified.parquet`)
**Goal:** Transform unified conversation history into `CompileSample` fuel for the
ReflexCompiler. Extract behavioral patterns, validate cross-platform, score by SNR,
and feed into the existing 4-gate compilation pipeline.

---

## 1. CompilationFuelExtractor

```pseudocode
struct CompilationFuelExtractor:
  conversations:  ParquetReader<ConversationTurn>
  extractors:     [IntentClassifier, TraitDetector, PreferenceMiner,
                   ExpertiseDetector, BehaviorAnalyzer, EmotionalToneDetector]
  snr_scorer:     SNRScorer
  guardian:       GuardianCouncil

  fn extract_all() -> Vec<CompileSample>:
    samples = []
    for conv in self.conversations.iter_conversations():
      for extractor in self.extractors:
        samples.extend(extractor.extract(conv))
    samples = self.cross_platform_validate(samples)
    samples = self.score_and_rank(samples)
    return samples
```

Each extractor produces records matching the `CompileSample` struct from
`bizra-omega/bizra-agent/src/reflex_compiler.rs` (lines 50-58):

```rust
// From reflex_compiler.rs -- the target struct
pub struct CompileSample {
    pub route_signature: String,     // e.g. "intent=Create,depth=0.8"
    pub path_signature: String,      // e.g. "classify>create"
    pub response_confidence: f32,
    pub context_richness: f32,
    pub guardian_approved: bool,
    pub ihsan_at_decision: f32,
    pub timestamp: u64,
}
```

Phase 02 extends with metadata for extraction provenance:

```pseudocode
struct CompileSampleExtended(CompileSample):
  extraction_type:      enum { Intent, Trait, Preference, Expertise, Behavior, Tone }
  source_platforms:     Vec<Platform>
  cross_validated:      bool
  pattern_hash:         [u8; 32]
  evidence_turn_ids:    Vec<[u8; 32]>
  extraction_confidence: f32
```

---

## 2. Pattern Extraction Categories

### 2.1 Intent Classification

```pseudocode
enum UserIntent: Create, Learn, Fix, Explore, Decide, Express

struct IntentClassifier:
  fn extract(conv: Conversation) -> Vec<CompileSample>:
    for turn in conv.user_turns():
      intent = classify_intent(turn.content)
      if intent.confidence >= 0.70:
        yield CompileSample {
          route_signature:     format!("intent={},depth={:.2}", intent.label, intent.depth),
          path_signature:      format!("classify>{}", intent.label.lowercase()),
          response_confidence: intent.confidence,
          context_richness:    turn.topics.len() / 5.0,
          guardian_approved:   true,
          ihsan_at_decision:   intent.confidence,
          timestamp:           turn.timestamp.to_epoch(),
        }
```

### 2.2 Trait Discovery

Personality patterns: communication style, formality, verbosity, humor,
technical depth, language mixing. Each measured as f32 in [0.0, 1.0].

```pseudocode
struct TraitDetector:
  fn extract(conv: Conversation) -> Vec<CompileSample>:
    for (trait_name, trait_value) in self.measure_traits(conv.user_turns()):
      yield CompileSample {
        route_signature:     format!("trait={},value={:.2}", trait_name, trait_value),
        response_confidence: trait_value,
        context_richness:    conv.user_turns().len() / 20.0,
        ...
      }
```

### 2.3 Preference Mining

```pseudocode
struct PreferenceMiner:
  fn extract(conv: Conversation) -> Vec<CompileSample>:
    // Explicit: "I prefer X", "I always use Y", "my go-to is Z"
    for match in regex_scan(PREFERENCE_PATTERNS, conv.user_turns()):
      yield CompileSample { route_signature: "pref=explicit,...", confidence: 0.90, ... }
    // Implicit: tech/tool mentioned 3+ times across conversations
    for (tech, count) in count_tech_mentions(conv.user_turns()).filter(|c| c >= 3):
      yield CompileSample { route_signature: "pref=implicit,...", confidence: count/10.0, ... }
```

### 2.4 Expertise Detection

Measures domain knowledge depth via: technical vocabulary, correction of
assistant errors, follow-up question depth, code snippet complexity.

```pseudocode
struct ExpertiseDetector:
  fn extract(conv: Conversation) -> Vec<CompileSample>:
    for (domain, depth, evidence) in detect_domains(conv):
      yield CompileSample {
        route_signature: format!("expertise={},depth={:.2}", domain, depth),
        response_confidence: depth,
        context_richness: evidence.len() / 20.0,
        ...
      }
```

### 2.5 Behavioral Patterns

Time-of-day active hours, session length distribution, topic transition graph.

```pseudocode
struct BehaviorAnalyzer:
  fn extract(conv: Conversation) -> Vec<CompileSample>:
    yield from compute_hour_distribution(conv).as_samples()
    yield from compute_session_length(conv).as_samples()
    yield from compute_topic_transitions(conv).as_samples()
```

### 2.6 Emotional Tone

Per-turn sentiment trajectory and trigger pattern detection.

```pseudocode
struct EmotionalToneDetector:
  fn extract(conv: Conversation) -> Vec<CompileSample>:
    sentiments = conv.user_turns().map(detect_sentiment)
    trajectory = compute_sentiment_trajectory(sentiments)
    yield CompileSample {
      route_signature: format!("tone=trajectory,slope={:.3}", trajectory.slope),
      response_confidence: trajectory.r_squared,
      ...
    }
```

---

## 3. Cross-Platform Validation

Same pattern seen on multiple platforms provides stronger evidence than
single-platform repetition.

```pseudocode
fn cross_platform_validate(samples: Vec<CompileSample>) -> Vec<CompileSample>:
  groups = group_by(samples, |s| s.pattern_hash)

  for (hash, group) in groups:
    platforms = group.flat_map(|s| s.source_platforms).unique()
    n = platforms.len()

    if n >= 3:
      for s in group:
        s.response_confidence *= 1.0 + 0.1 * (n - 1)   # up to 1.9x
        s.response_confidence = min(s.response_confidence, 1.0)
        s.cross_validated = true
    elif n == 2:
      for s in group:
        s.response_confidence *= 1.05
        s.cross_validated = true

  return samples
```

Example: "Direct communication style" detected on ChatGPT, Claude, and DeepSeek
(3 platforms) gets a 1.2x confidence boost, increasing the likelihood of passing
the ReflexCompiler's Ihsan gate.

---

## 4. SNR Scoring

Mirrors the Rust implementation in `reflex_compiler.rs` (lines 169-172):

```rust
// reflex_compiler.rs:169-172 -- existing formula
pub fn snr_score(response_confidence: f32, context_richness: f32, guardian_approved: bool) -> f32 {
    let guardian = if guardian_approved { 1.0 } else { 0.0 };
    (0.5 * response_confidence) + (0.3 * context_richness) + (0.2 * guardian)
}
```

Phase 02 adds cross-platform awareness:

```pseudocode
fn snr_score_phase02(sample: CompileSampleExtended) -> f32:
  guardian = 1.0 if sample.guardian_approved else 0.0
  base = 0.5 * sample.response_confidence
       + 0.3 * sample.context_richness
       + 0.2 * guardian

  if sample.cross_validated:
    platform_bonus = 0.15 * (sample.source_platforms.len() / 10.0)
    base *= (1.0 + platform_bonus)

  return clamp(base, 0.0, 1.0)
```

Reference: `core/iaas/snr_v2_adapter.py` (line 23) for the Python-side SNR
protocol bridge using `UNIFIED_IHSAN_THRESHOLD` from `core/integration/constants.py`.

---

## 5. ReflexCompiler Integration

CompileSamples feed into the existing 4-gate pipeline from `reflex_compiler.rs`
(lines 103-160, `evaluate()` method):

```pseudocode
Gate 1: Frequency     bucket.samples.len() >= 3           InsufficientSamples
Gate 2: Ihsan         avg_ihsan >= 0.95                   LowIhsan
Gate 3: SNR           avg_snr >= 0.90                     LowSnr
Gate 4: Consistency   path_variance < 0.10                PathVarianceHigh
```

Reference: `CompilerConfig` defaults at lines 12-27 of `reflex_compiler.rs`.

### Quarantine-Not-Evict

Samples failing a gate are quarantined per `reflex_cache.rs` (lines 36-65,
`QuarantineReason` enum: `GuardianVeto`, `RevalidationFailed`,
`PolicyHashMismatch`, `ManualInvalidation`, `MissingPolicyHash`).

```pseudocode
on gate failure:
  rule.quarantined = true
  rule.quarantine_reason = match failed_gate:
    Gate 1 -> None               # insufficient data; wait for more
    Gate 2 -> GuardianVeto
    Gate 3 -> RevalidationFailed
    Gate 4 -> RevalidationFailed

  reflex_cache.store_quarantined(rule)
  # Rule stays in cache for re-evaluation when new evidence arrives
```

The `ReflexCache` (line 101, `reflex_cache.rs`) uses LRU eviction for active
rules but retains quarantined rules separately.

---

## 6. Sigmoid Model Revision

### Current Model (Single-Platform)

```pseudocode
y(x) = L / (1 + e^(-k * (x - x0)))
L=1.0, k=0.003, x0=1500

y(500)=0.05  y(1500)=0.50  y(3000)=0.85  y(5000)=0.98
```

### Revised Model (Multi-Platform)

```pseudocode
y(x) = L / (1 + e^(-k * (x - x0)))
L=1.0, k=0.002, x0=3000

y(1000)=0.02  y(3000)=0.50  y(5000)=0.73  y(7000)=0.88  y(10000)=0.96

Cross-validated adjustment (1.5x effective count):
  User Zero: 7000 raw, ~30% cross-validated
  effective = 7000*0.70 + 7000*0.30*1.5 = 8050
  y(8050) = 0.92
```

**Why multi-platform does not plateau early:**
1. Orthogonal validation -- same preference, different phrasing, stronger evidence
2. Reduced style overfitting -- consistent behavior _despite_ different assistants
3. Broader intent coverage -- different platforms for different purposes
4. Temporal diversity -- platform switching provides natural train/test splits

---

## 7. Data Flow Diagram

```
  conversations_unified.parquet (Phase 01)
       |
       v
  +----------------------------------+
  |    CompilationFuelExtractor       |
  |    6 extractors:                  |
  |    Intent | Trait | Preference    |
  |    Expertise | Behavior | Tone    |
  +----------------------------------+
       |
       v  Vec<CompileSampleExtended>
  +----------------------------------+
  |    Cross-Platform Validation      |
  |    3+ platforms -> 1.2x boost    |
  |    2  platforms -> 1.05x boost   |
  +----------------------------------+
       |
       v
  +----------------------------------+
  |    SNR Scoring                    |
  |    0.5*conf + 0.3*rich + 0.2*grd |
  |    + cross-platform bonus (15%)  |
  +----------------------------------+
       |
       v
  +----------------------------------+
  |    ReflexCompiler (4-gate)        |
  |    Freq >= 3 | Ihsan >= 0.95     |
  |    SNR >= 0.90 | Var < 0.10      |
  +----------------------------------+
       |               |
       v               v
  [Compiled]      [Quarantined]
  ReflexRule      awaiting evidence
       |               |
       v               v
  +----------------------------------+
  |    ReflexCache                    |
  |    LRU active + quarantine store  |
  +----------------------------------+
       |
       v
  Compiled Agent Identity (System-1 reflexes for User Zero)
```

---

## 8. TDD Anchors

Tests marked [PROP] use property-based testing (Hypothesis/proptest).

```pseudocode
# --- Extraction Tests ---

test_intent_classification_accuracy:
  labeled = load("fixtures/labeled_intents.json")  # 200 turns
  predicted = IntentClassifier.classify_batch(labeled.turns)
  assert accuracy(predicted, labeled.intents) >= 0.90

test_trait_discovery_consistency:
  chatgpt = TraitDetector.extract(chatgpt_convs)
  claude  = TraitDetector.extract(claude_convs)
  deepseek = TraitDetector.extract(deepseek_convs)
  for trait in all_trait_names:
    assert variance([chatgpt[trait], claude[trait], deepseek[trait]]) < 0.15

test_preference_mining_explicit:
  conv = make_conversation([(User, "I prefer Rust over Go for systems programming")])
  samples = PreferenceMiner.extract(conv)
  assert any("pref=explicit" in s.route_signature for s in samples)

test_preference_mining_implicit:
  convs = [make_conversation_mentioning("pytest") for _ in range(5)]
  samples = PreferenceMiner.extract_across(convs)
  assert any("pref=implicit,tech=pytest" in s.route_signature for s in samples)

test_expertise_detection_depth:
  expert = ExpertiseDetector.extract(load("fixtures/rust_expert.json"))[0]
  beginner = ExpertiseDetector.extract(load("fixtures/rust_beginner.json"))[0]
  assert expert.response_confidence > beginner.response_confidence + 0.20

# --- Cross-Platform Tests ---

test_cross_platform_confidence_boost:
  sample = CompileSample { response_confidence: 0.80, source_platforms: [ChatGPT, Claude, DeepSeek] }
  result = cross_platform_validate([sample])
  assert result[0].response_confidence >= 0.80 * 1.15
  assert result[0].cross_validated == true

test_cross_platform_single_no_boost:
  sample = CompileSample { response_confidence: 0.80, source_platforms: [ChatGPT] }
  result = cross_platform_validate([sample])
  assert result[0].response_confidence == 0.80 and not result[0].cross_validated

# --- SNR Tests ---

[PROP] test_snr_scoring_range:
  forall conf in 0..1, rich in 0..1, approved in {true, false}:
    assert 0.0 <= snr_score_phase02(conf, rich, approved) <= 1.0

test_snr_scoring_distribution:
  scores = [snr_score_phase02(s) for s in generate_realistic_samples(10000)]
  assert 0.4 <= mean(scores) <= 0.8 and std(scores) >= 0.10

# --- Compiler Integration Tests ---

test_reflex_compiler_gate_pass:
  compiler = ReflexCompiler::new(32)
  trigger = compute_trigger_hash("intent=Create")
  for _ in range(5): compiler.record_success(trigger, high_quality_sample())
  result = compiler.evaluate(trigger, CompilerConfig::default(), policy_hash)
  assert result.is_ok()
  assert result.unwrap().compile_ihsan >= 0.95 and not result.unwrap().quarantined

test_reflex_compiler_gate_fail_quarantine:
  compiler = ReflexCompiler::new(32)
  trigger = compute_trigger_hash("low_quality")
  for _ in range(5): compiler.record_success(trigger, low_snr_sample())
  assert compiler.evaluate(trigger, CompilerConfig::default(), policy_hash).is_err()
  cache.store_quarantined(trigger, QuarantineReason::RevalidationFailed)
  assert cache.is_quarantined(trigger) and cache.contains(trigger)

test_sigmoid_multiplatform_above_single:
  for n in [5000, 7000]:
    single = sigmoid(n, k=0.003, x0=1500)
    multi  = sigmoid(n * 1.15, k=0.002, x0=3000)
    assert multi >= single * 0.95

# --- Idempotency Tests ---

test_compilation_fuel_idempotent:
  parquet = load("conversations_unified.parquet")
  assert CompilationFuelExtractor(parquet).extract_all()
      == CompilationFuelExtractor(parquet).extract_all()

test_incremental_extraction:
  batch_1 = extract(conversations[0:3000])
  batch_2 = extract(conversations[3000:5000])
  assert set(merge(batch_1, batch_2).pattern_hashes) == set(extract(conversations[0:5000]).pattern_hashes)
```

---

## 9. Open Questions

1. **Min conversations per platform for cross-validation.** Propose: 50 minimum
   before a platform contributes to cross-platform confidence boost.
2. **Incremental vs full rebuild.** Propose: incremental with generation tracking;
   cross-validation re-runs across all generations, extraction only for new data.
3. **GPU acceleration.** RTX 4090 available for embeddings + classification;
   CPU for pattern mining and behavioral analysis.
4. **Privacy boundaries.** User Zero = no gates. Multi-user = `PrivacyGate` hook.
   Design the hook point now, enforce later.
5. **Compilation target format.** Propose: produce `user_zero_profile.json`
   alongside compiled reflexes for debugging and marketing demonstration.
