# Phase 55.6: Unified Backbone — Full Pipeline Wiring

## Week 5b Deliverable

Wire Action Bus + Event Bus + Hooks + State Machine + Saga Coordinator into the
`UnifiedMessageBackbone` — the single entry point for all request processing.
This completes Phase 7: first heartbeat through UMB, genesis block attested.

---

## Module: `backbone.rs`

### UnifiedMessageBackbone — The Entry Point

```
STRUCT UnifiedMessageBackbone
  FIELDS:
    action_bus:        Arc<ActionBus>
    event_bus:         Arc<EventBus>
    hook_registry:     Arc<HookRegistry>
    saga_coordinator:  Arc<SagaCoordinator>
    state_store:       Arc<StateStore>
    circuit_breakers:  Arc<CircuitBreakerHook>
    metrics:           BackboneMetrics

  FN builder() -> BackboneBuilder
    BackboneBuilder::new()

  ASYNC FN start(&self) -> JoinHandle<()>
    // Start the action bus dispatch loop
    LET action_bus = self.action_bus.clone()
    LET handle = tokio::spawn(async move { action_bus.run().await })

    // Wire saga coordinator as event subscriber
    self.event_bus.subscribe(Arc::new(SagaEventSubscriber(self.saga_coordinator.clone())))

    // Wire state store as event subscriber
    self.event_bus.subscribe(Arc::new(StateStoreSubscriber(self.state_store.clone())))

    handle

  ASYNC FN process_request(
    &self,
    raw_input: String,
    user_id: ActorId,
  ) -> Result<Response, SystemError>

    LET trace_id = TraceId::new()
    LET request_id = CorrelationId::new()

    // 1. Emit arrival event
    self.event_bus.emit(Event::RequestReceived {
      request_id: request_id.clone(),
      user_id: user_id.clone(),
    }).await?

    // 2. Create request state (typestate: Received)
    LET request = RequestState::<Received>::new(raw_input.clone(), trace_id.clone())
    self.state_store.store(request_id.clone(), DynRequestState::Received(request))

    // 3. Build action envelope
    LET envelope = Envelope::new(
      Action::ParseIntent { raw_input, user_id },
      trace_id,
      ActorId::system(),
      request_id.clone(),
    )

    // 4. Run pre-dispatch hooks (auth, budget, circuit breaker)
    LET envelope = self.hook_registry
      .run_hooks(HookPoint::PreDispatch, envelope)
      .await?

    // 5. Dispatch to action bus
    //    Handler emits events → saga coordinator picks up →
    //    drives: Parse → Decompose → Execute → Synthesize →
    //    Validate (α4→α10) → Attest → Complete
    self.action_bus.dispatch(envelope).await?

    // 6. Await saga completion (with timeout)
    LET outcome = self.saga_coordinator
      .await_completion(request_id.clone(), Duration::from_secs(30))
      .await?

    // 7. Post-completion hooks (learning, cleanup)
    self.event_bus.emit(Event::OutcomeLearned {
      lesson: outcome.extract_lesson(),
      confidence: outcome.quality_score(),
    }).await?

    // 8. Convert to response
    MATCH outcome
      SagaOutcome::Success(results) => Ok(Response::success(results))
      SagaOutcome::Compensated { .. } => Err(SystemError::request_compensated())
      SagaOutcome::Aborted(reason) => Err(SystemError::request_aborted(reason))

  FN status(&self) -> BackboneStatus
    BackboneStatus {
      active_sagas: self.saga_coordinator.active_count(),
      action_queue_depth: self.action_bus.queue_depth(),
      event_subscribers: self.event_bus.subscriber_count(),
      circuit_breakers: self.circuit_breakers.status(),
      hooks_registered: self.hook_registry.count(),
    }
```

### BackboneBuilder

```
STRUCT BackboneBuilder
  capacity:    usize
  hooks:       Vec<Arc<dyn Hook>>
  handlers:    Vec<Arc<dyn ActionHandler>>
  subscribers: Vec<Arc<dyn EventSubscriber>>

  FN new() -> Self
    Self { capacity: 1024, hooks: vec![], handlers: vec![], subscribers: vec![] }

  FN with_capacity(mut self, capacity: usize) -> Self
    self.capacity = capacity; self

  FN with_hook(mut self, hook: Arc<dyn Hook>) -> Self
    self.hooks.push(hook); self

  FN with_handler(mut self, handler: Arc<dyn ActionHandler>) -> Self
    self.handlers.push(handler); self

  FN with_subscriber(mut self, subscriber: Arc<dyn EventSubscriber>) -> Self
    self.subscribers.push(subscriber); self

  FN with_default_hooks(self) -> Self
    self
      .with_hook(Arc::new(AuthorizationHook::permissive()))
      .with_hook(Arc::new(BudgetEnforcementHook::new()))
      .with_hook(Arc::new(CircuitBreakerHook::new(5, Duration::from_secs(30))))
      .with_hook(Arc::new(TracingHook::new()))
      .with_hook(Arc::new(ConstitutionalPreCheckHook::new()))
      .with_hook(Arc::new(BlockGraphAttestationHook::new()))
      .with_hook(Arc::new(IhsanScoringHook::new(0.95)))

  FN build(self) -> Result<UnifiedMessageBackbone, BuildError>
    // Validation
    LET event_bus = Arc::new(EventBus::new(self.capacity * 4))
    LET action_bus = Arc::new(ActionBus::new(self.capacity, event_bus.clone()))

    // Register all handlers
    FOR handler IN &self.handlers
      action_bus.register_handler(handler.clone())?

    // Register all hooks
    LET hook_registry = HookRegistry::new()
    FOR hook IN &self.hooks
      hook_registry.register(hook.clone())

    // Register all subscribers
    FOR subscriber IN &self.subscribers
      event_bus.subscribe(subscriber.clone())

    LET circuit_breakers = self.hooks.iter()
      .find_map(|h| h.downcast_ref::<CircuitBreakerHook>().cloned())
      .unwrap_or_else(|| Arc::new(CircuitBreakerHook::default()))

    Ok(UnifiedMessageBackbone {
      action_bus,
      event_bus,
      hook_registry: Arc::new(hook_registry),
      saga_coordinator: Arc::new(SagaCoordinator::new(action_bus.clone())),
      state_store: Arc::new(StateStore::new()),
      circuit_breakers,
      metrics: BackboneMetrics::default(),
    })
```

### Metrics & Status

```
STRUCT BackboneMetrics
  requests_total:     Counter
  requests_succeeded: Counter
  requests_failed:    Counter
  request_duration:   Histogram
  active_sagas:       Gauge

STRUCT BackboneStatus
  active_sagas:       usize
  action_queue_depth: usize
  event_subscribers:  usize
  circuit_breakers:   HashMap<ActorId, CircuitState>
  hooks_registered:   usize
```

---

## Complete Request Flow (All 26 Components)

```
USER: "Research quantum computing breakthroughs in 2025"
  │
  ▼
┌─────────────────────────────────────────────────────────┐
│ UNIFIED MESSAGE BACKBONE                                │
│                                                         │
│  PRE-DISPATCH HOOKS:                                    │
│    [Authorization]    Ed25519 verified                   │
│    [Budget]           10,000 tokens allocated            │
│    [CircuitBreaker]   All components healthy             │
│    [Tracing]          TraceID assigned                   │
│    [ASPH Pre-Check]   No sycophancy risk                 │
│                                                         │
│  ACTION: ParseIntent → PAT.Planner                      │
│  EVENT:  IntentParsed { confidence: 0.94 }              │
│                                                         │
│  ACTION: DecomposeTask → PAT.Planner                    │
│  EVENT:  TaskDecomposed { 4 subtasks, DAG }             │
│                                                         │
│  SAGA CREATED → drives parallel agent execution         │
│    Step 1: Research      → PAT.Researcher               │
│    Step 2: Evaluate      → PAT.Evaluator    [after 1]   │
│    Step 3: Synthesize    → PAT.Integrator   [after 2]   │
│    Step 4: Quality check → PAT.Ethicist     [after 3]   │
│                                                         │
│  CONSTITUTIONAL GATES (sequential):                     │
│    α4 Fallback → α7 Verification → α8 Dark Matter →    │
│    α9 Attestation → α10 Binary                          │
│    Ihsan Score: computed at each gate                    │
│    Daughter Test: final check                            │
│                                                         │
│  ACTION: AttestToBlockGraph                              │
│  EVENT:  BlockGraphAttested { hash, block_height }      │
│                                                         │
│  POST-COMPLETION HOOKS:                                  │
│    [Learning]     Lesson extracted + stored               │
│    [Tracing]      Full trace committed                   │
│    [Budget]       Remaining tokens reported              │
│    [Attestation]  Proof-of-Impact recorded               │
│                                                         │
│  SAGA STATE: Completed → Response delivered              │
└─────────────────────────────────────────────────────────┘
```

---

## TDD Anchors

```
TEST test_backbone_end_to_end
  LET backbone = UnifiedMessageBackbone::builder()
    .with_default_hooks()
    .with_handler(Arc::new(MockIntentParser))
    .with_handler(Arc::new(MockTaskDecomposer))
    .with_handler(Arc::new(MockAgentRuntime))
    .with_handler(Arc::new(MockOutputSynthesizer))
    .with_handler(Arc::new(MockGateValidator))
    .with_handler(Arc::new(MockAttestor))
    .build().unwrap()
  backbone.start().await
  LET result = backbone.process_request(
    "Research quantum computing".into(),
    ActorId::agent("user-1"),
  ).await
  ASSERT result.is_ok()

TEST test_backbone_builder_validation
  // Builder with no handlers should fail validation
  LET result = UnifiedMessageBackbone::builder()
    .with_default_hooks()
    .build()
  ASSERT result.is_err()  // No action handlers registered

TEST test_backbone_rejects_expired_request
  // Envelope with zero TTL → rejected by pre-dispatch hooks
  LET result = backbone.process_request(/* expired */).await
  ASSERT result.is_err()

TEST test_backbone_status
  LET backbone = /* ... build ... */
  backbone.start().await
  LET status = backbone.status()
  ASSERT status.hooks_registered == 7  // Default hooks
  ASSERT status.active_sagas == 0

TEST test_backbone_concurrent_requests
  // Process 10 requests concurrently
  LET handles: Vec<_> = (0..10).map(|i|
    tokio::spawn(backbone.process_request(format!("Query {i}"), user.clone()))
  ).collect()
  LET results: Vec<_> = join_all(handles).await
  ASSERT results.iter().all(|r| r.unwrap().is_ok())

TEST test_backbone_graceful_shutdown
  LET backbone = /* ... build + start ... */
  // Process a request
  backbone.process_request("test".into(), user).await.unwrap()
  // Shutdown should drain in-flight work
  backbone.shutdown().await
  ASSERT backbone.status().active_sagas == 0
```

## Edge Cases

- Backbone shutdown during active saga → timeout triggers compensation
- Concurrent process_request calls → each gets its own CorrelationId
- Builder with duplicate handlers for same action type → BuildError
- process_request with empty string → still valid (parser decides)
- All hooks abort → request fails fast, no saga created
- Event bus full → broadcast drops oldest (slow subscribers lose events)
- Backbone with zero capacity → rejects all non-critical requests
