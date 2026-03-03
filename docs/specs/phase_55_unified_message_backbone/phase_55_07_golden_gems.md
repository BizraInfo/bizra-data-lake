# Phase 55.7: Golden Gems — Deep Architectural Refinements

## 6 Architectural Insights That Modify the UMB Spec

Each gem below modifies a concrete implementation decision from phases 55.1–55.6.
These are corrections, not additions — they simplify the architecture.

---

## Gem 1: BlockGraph as Emergent Projection

**Modifies:** `phase_55_01_core_types.md` (Action enum), `phase_55_05_saga.md` (gate sequence)

**Before:** `Action::AttestToBlockGraph` is an explicit pipeline stage after gates pass.
The saga dispatches it, waits for completion, then transitions to Completed.

**After:** Remove `AttestToBlockGraph` from the Action enum. The BlockGraph is not a
step — it's a **read-side projection** over the signed Envelope chain that already exists.

Every Envelope has `id`, `causation_id`, and `signature`. This IS a hash chain:
```
Envelope_N.signature → causation_id → Envelope_N-1.signature → ...
```

### Implementation

```
STRUCT BlockGraphProjection
  message_store: Arc<MessageStore>
  latest_hash:   AtomicU64   // Pointer to latest block

  FN project_block(&self, saga_id: &SagaId) -> Block
    LET messages = self.message_store.by_correlation(saga_id)
    Block {
      hash:        merkle_root(messages.iter().map(|m| &m.signature)),
      parent:      self.latest_hash.load(Ordering::SeqCst),
      height:      self.latest_hash.load(Ordering::SeqCst) + 1,
      proof_chain: messages,  // THE MESSAGES ARE THE PROOF
    }

// Register as PostEmit hook — automatic, zero-cost attestation
STRUCT BlockGraphProjectionHook(Arc<BlockGraphProjection>)

IMPL Hook FOR BlockGraphProjectionHook
  FN hook_point(&self) -> HookPoint => PostEmit
  FN name(&self) -> &'static str => "blockgraph_projection"

  ASYNC FN intercept_event(&self, envelope: &Envelope<Event>) -> HookResult
    IF LET Event::ResponseSynthesized { request_id, .. } = &envelope.payload
      // Projection is O(1) amortized — messages already stored
      LET block = self.0.project_block(request_id)
      self.0.latest_hash.store(block.height, Ordering::SeqCst)
    HookResult::Continue
```

### Saga Modification

Remove the attestation step from the gate sequence. After all gates pass, the
saga transitions directly to Completed. The BlockGraph projection fires as a
PostEmit hook on the `ResponseSynthesized` event.

```
// BEFORE (phase_55_05):
Alpha4 → Alpha7 → Alpha8 → Alpha9 → Alpha10 → AttestToBlockGraph → Completed

// AFTER:
Alpha4 → Alpha7 → Alpha8 → Alpha9 → Alpha10 → Completed
// (BlockGraph projection fires automatically via PostEmit hook)
```

### TDD Anchor

```
TEST test_blockgraph_emerges_from_signed_messages
  // Process a request through the full pipeline
  // Verify that a Block was created without explicit attestation action
  LET backbone = build_backbone_with_projection()
  backbone.process_request("test query".into(), user).await.unwrap()
  LET block = backbone.blockgraph.latest_block()
  ASSERT block.proof_chain.len() > 0
  ASSERT block.proof_chain.iter().all(|m| m.signature.is_some())
  // Verify merkle root matches
  LET expected = merkle_root(block.proof_chain.iter().map(|m| &m.signature))
  ASSERT block.hash == expected
```

---

## Gem 2: Bus Topology as Organizational Authority

**Modifies:** Design heuristic for all future components.

**Rule:** When adding any new component to BIZRA, classify it:
- **Does it DECIDE?** → Connects to Action Bus (mpsc, one handler)
- **Does it OBSERVE?** → Subscribes to Event Bus (broadcast, many)
- **Does it do BOTH?** → Split into two: a decider + an observer

```
// Component classification table (extends phase_55_00)
//
// DECIDERS (Action Bus):        OBSERVERS (Event Bus):
//   PAT Intent Parser             SAT Monitor
//   PAT Task Decomposer           Telemetry/Tracing
//   Orchestrator                   Learning Layer
//   Agent Runtime                  BlockGraph Projection
//   Constitutional Gates           Budget Tracker
//   Output Synthesizer             Ihsan Scorer
//
// SOVEREIGNTY (your resource)   TRANSPARENCY (all informed)
//   SEED token                    BLOOM token
```

This is not new code — it's a **design constraint** documented for all future phases.

---

## Gem 3: Execution Cache (Reverse Scaling Foundation)

**Modifies:** `phase_55_06_backbone.md` (process_request flow)

**Before:** Every request goes through the full saga pipeline.
**After:** Before saga creation, check the Knowledge Graph for a cached execution.

### Implementation

```
STRUCT ExecutionCache
  store: DashMap<RequestSignature, CachedExecution>

STRUCT RequestSignature([u8; 32])  // blake3 hash of normalized intent

STRUCT CachedExecution
  trace_id:     TraceId
  response:     Response
  ihsan_score:  f64
  created_at:   Instant
  hit_count:    AtomicU64

IMPL ExecutionCache
  FN lookup(&self, signature: &RequestSignature) -> Option<&CachedExecution>
    self.store.get(signature)
      .filter(|c| c.ihsan_score >= IHSAN_THRESHOLD)
      .filter(|c| c.created_at.elapsed() < MAX_CACHE_AGE)

  FN store(&self, intent: &StructuredIntent, execution: CachedExecution)
    LET signature = intent.canonical_hash()
    self.store.insert(signature, execution)
```

### Backbone Modification

Insert cache check between step 1 (arrival event) and step 3 (action envelope):

```
ASYNC FN process_request(&self, raw_input, user_id) -> Result<Response>
  // ... steps 1-2 unchanged ...

  // NEW: Cache check before saga creation
  LET intent = self.quick_parse(&raw_input)  // Lightweight parse for signature
  IF LET Some(intent) = intent
    LET signature = intent.canonical_hash()
    IF LET Some(cached) = self.execution_cache.lookup(&signature)
      // Re-validate through constitutional gates (fast — no agent work)
      IF self.revalidate_cached(&cached).await?
        cached.hit_count.fetch_add(1, Ordering::Relaxed)
        self.event_bus.emit(Event::CacheHit { signature, original_trace: cached.trace_id }).await
        RETURN Ok(cached.response.clone())

  // Cache miss — full pipeline (steps 3-8 unchanged)
```

### New Event Variant

Add to `event.rs`:
```
Event::CacheHit {
  signature:      RequestSignature,
  original_trace: TraceId,
}
```

### Cache Saturation Economics

```
// Cache hit rate as function of historical request volume:
// hit_rate = 1 - e^(-lambda * N)
//
// At N = 10^2:   ~2%  → most computed fresh
// At N = 10^6:   ~40% → many partially cached
// At N = 10^9:   ~85% → most served from cache
// At N = 10^10:  ~99% → nearly all cache hits
//
// This is the mathematical foundation for reverse scaling.
```

### TDD Anchors

```
TEST test_cache_hit_bypasses_saga
  LET backbone = build_backbone_with_cache()
  // First request — full pipeline
  backbone.process_request("quantum computing breakthroughs".into(), user).await.unwrap()
  ASSERT backbone.saga_coordinator.total_sagas_created() == 1

  // Second identical request — cache hit, no saga
  backbone.process_request("quantum computing breakthroughs".into(), user).await.unwrap()
  ASSERT backbone.saga_coordinator.total_sagas_created() == 1  // Still 1

TEST test_cache_respects_ihsan_threshold
  // Cache an execution with low ihsan score
  cache.store(&intent, CachedExecution { ihsan_score: 0.80, .. })
  // Lookup should return None (below 0.95 threshold)
  ASSERT cache.lookup(&intent.canonical_hash()).is_none()

TEST test_cache_expires_after_max_age
  cache.store(&intent, CachedExecution { created_at: old_timestamp, .. })
  ASSERT cache.lookup(&intent.canonical_hash()).is_none()
```

---

## Gem 4: Daughter Test as System-Wide Interrupt

**Modifies:** `phase_55_03_hook_system.md` (HookPoint enum), saga gate sequence

**Before:** Daughter Test runs once as the final gate after α4-α10.
**After:** Daughter Test runs as a Hook at EVERY HookPoint with MAX priority.

### HookPoint Modification

Add `All` variant to HookPoint:
```
ENUM HookPoint
  PreDispatch
  PostDispatch
  PreEmit
  PostEmit
  OnError
  OnStateTransition
  PreGate
  PostGate
  All              // NEW: runs at every single hook point
```

### HookRegistry Modification

When running hooks at any point, also include hooks registered for `HookPoint::All`:

```
ASYNC FN run_hooks(&self, point: HookPoint, envelope) -> Result<..>
  // Get hooks for this specific point
  LET specific = self.hooks.get(&point)
  // Get hooks registered for All points
  LET universal = self.hooks.get(&HookPoint::All)
  // Merge and sort by priority (universal hooks run first if higher priority)
  LET merged = merge_by_priority(universal, specific)
  // ... run merged hooks in order ...
```

### DaughterTestHook

```
STRUCT DaughterTestHook
  evaluator: Box<dyn SafetyEvaluator>

IMPL Hook FOR DaughterTestHook
  FN hook_point(&self) -> HookPoint => HookPoint::All
  FN priority(&self) -> i32 => i32::MAX  // ALWAYS first
  FN name(&self) -> &'static str => "daughter_test"

  ASYNC FN intercept(&self, envelope: &Envelope<Action>) -> HookResult
    LET content = envelope.payload.extractable_content()
    IF LET Some(content) = content
      LET safety = self.evaluator.evaluate(&content).await
      IF safety < DAUGHTER_TEST_THRESHOLD
        RETURN HookResult::Abort {
          reason: format!(
            "DAUGHTER TEST VIOLATION at {} (safety: {:.3})",
            envelope.payload.message_type(), safety
          )
        }
    HookResult::Continue

  ASYNC FN intercept_event(&self, envelope: &Envelope<Event>) -> HookResult
    // Also check events — catch harmful content in notifications
    LET content = envelope.payload.extractable_content()
    IF LET Some(content) = content
      LET safety = self.evaluator.evaluate(&content).await
      IF safety < DAUGHTER_TEST_THRESHOLD
        RETURN HookResult::Abort {
          reason: format!("DAUGHTER TEST VIOLATION in event: {}", envelope.payload.message_type())
        }
    HookResult::Continue
```

### Remove from Saga Gate Sequence

The Daughter Test is no longer a saga step. Remove from `next_gate()`:
```
// BEFORE:
FN next_gate: Alpha4 → Alpha7 → Alpha8 → Alpha9 → Alpha10 → DaughterTest

// AFTER:
FN next_gate: Alpha4 → Alpha7 → Alpha8 → Alpha9 → Alpha10
// (Daughter Test runs at EVERY hook point automatically)
```

### TDD Anchors

```
TEST test_daughter_test_catches_harm_during_execution
  // Inject harmful content in an AgentCompleted event
  // Daughter Test hook should abort BEFORE the event reaches subscribers
  LET hook = DaughterTestHook::new(MockSafetyEvaluator::always_fail())
  LET result = hook.intercept_event(&harmful_event_envelope).await
  ASSERT matches!(result, HookResult::Abort { .. })

TEST test_daughter_test_runs_at_every_hook_point
  // Register DaughterTestHook, process a request
  // Verify hook was called at PreDispatch, PostDispatch, PreEmit, PostEmit
  LET call_log = Arc::new(Mutex::new(Vec::new()))
  LET hook = TrackingDaughterTestHook::new(call_log.clone())
  // ... register and process ...
  LET log = call_log.lock()
  ASSERT log.contains(&HookPoint::PreDispatch)
  ASSERT log.contains(&HookPoint::PostDispatch)
  ASSERT log.contains(&HookPoint::PreEmit)
  ASSERT log.contains(&HookPoint::PostEmit)
```

---

## Gem 5: AdaptiveGate<T> — Generic Threshold Component

**Modifies:** Constitutional gates, SNR filter, circuit breaker, safety monitor.

Four "different" components share identical structure:

```
Input → [threshold check] → Pass/Fail → [adjust threshold from outcome]
```

### Implementation

```
STRUCT AdaptiveGate<T: Evaluable>
  threshold:     f64
  learning_rate: f64
  history:       VecDeque<GateOutcome>
  max_history:   usize
  evaluator:     Box<dyn Fn(&T) -> f64 + Send + Sync>

TRAIT Evaluable: Send + Sync + 'static {}
IMPL Evaluable FOR AgentOutput {}
IMPL Evaluable FOR ComponentHealth {}
IMPL Evaluable FOR MessageContent {}

IMPL<T: Evaluable> AdaptiveGate<T>
  FN new(threshold: f64, evaluator: impl Fn(&T) -> f64 + Send + Sync + 'static) -> Self

  FN evaluate(&mut self, input: &T) -> GateDecision
    LET score = (self.evaluator)(input)
    LET decision = IF score >= self.threshold
      GateDecision::Pass { score }
    ELSE
      GateDecision::Fail { score, threshold: self.threshold }

    self.history.push_back(GateOutcome { score, decision: decision.clone() })
    IF self.history.len() > self.max_history
      self.history.pop_front()
      self.recalibrate()

    decision

  FN recalibrate(&mut self)
    LET false_positive_rate = self.history.iter()
      .filter(|o| o.was_false_positive()).count() as f64
      / self.history.len() as f64

    IF false_positive_rate > 0.01
      self.threshold *= 1.0 + self.learning_rate  // Tighten
    ELSE
      self.threshold *= 1.0 - (self.learning_rate * 0.1)  // Slightly loosen

// Instantiate all four components from one generic:
TYPE ConstitutionalGate = AdaptiveGate<AgentOutput>
TYPE SNRFilter          = AdaptiveGate<Signal>
TYPE CircuitBreaker     = AdaptiveGate<ComponentHealth>
TYPE SafetyMonitor      = AdaptiveGate<MessageContent>
```

### TDD Anchors

```
TEST test_adaptive_gate_passes_above_threshold
  LET mut gate = AdaptiveGate::new(0.95, |x: &f64| *x)
  ASSERT matches!(gate.evaluate(&0.97), GateDecision::Pass { .. })
  ASSERT matches!(gate.evaluate(&0.90), GateDecision::Fail { .. })

TEST test_adaptive_gate_tightens_on_false_positives
  LET mut gate = AdaptiveGate::new(0.50, evaluator)
  // Feed it inputs that pass but are marked as bad outcomes
  FOR _ IN 0..100 { gate.evaluate_with_outcome(&bad_input, false) }
  ASSERT gate.threshold > 0.50  // Threshold increased

TEST test_adaptive_gate_loosens_when_too_strict
  // All inputs are valid but get rejected → threshold loosens
```

---

## Gem 6: Emergence Over Addition (Meta-Principle)

**Not code — a permanent design constraint.**

> When tempted to add a new component, first ask:
> Can this behavior EMERGE from a new interaction between existing components?
> If yes, add the interaction, not the component.

This is how 26 components scale to 260 capabilities while staying at 26 components.

Applied to the 5 gems above:
- BlockGraph emerged from signed envelopes (no new component)
- Cache emerged from Knowledge Graph + request signatures (no new component)
- Daughter Test emerged from Hook system + priority (no new component)
- Four gate types emerged from one generic (reduced components)
- Bus topology classification emerged from existing mpsc/broadcast choice

**Document this in the project's CLAUDE.md as a standing architectural constraint.**

---

## Summary of Spec Modifications

| Phase File | Modification |
|-----------|-------------|
| `55_01_core_types.md` | Remove `Action::AttestToBlockGraph`, add `Event::CacheHit` |
| `55_03_hook_system.md` | Add `HookPoint::All`, add DaughterTestHook, modify registry |
| `55_05_saga.md` | Remove attestation step, remove Daughter Test from gate sequence |
| `55_06_backbone.md` | Add execution cache check before saga, add BlockGraphProjection |
