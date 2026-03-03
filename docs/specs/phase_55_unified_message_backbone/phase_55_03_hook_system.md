# Phase 55.3: Hook System — Lifecycle Interception

## Week 3 Deliverable

Implement the hook registry and 7 built-in hooks. Hooks intercept messages at
defined lifecycle points, enabling cross-cutting concerns (auth, budget, tracing,
constitutional checks) without modifying any component.

---

## Module: `hook.rs`

### HookPoint — Where Interception Occurs

```
ENUM HookPoint
  DERIVE: Clone, Debug, PartialEq, Eq, Hash

  PreDispatch         // Before action reaches handler
  PostDispatch        // After handler completes successfully
  PreEmit             // Before event is broadcast
  PostEmit            // After all subscribers notified
  OnError             // When any component errors
  OnStateTransition   // When any state machine transitions
  PreGate             // Before constitutional gate evaluation
  PostGate            // After gate decision (pass or fail)

// Lifecycle position:
//
//   User Request
//     │
//     ▼
//   [PreDispatch hooks]  ← auth, budget, circuit breaker
//     │
//     ▼
//   Action Handler executes
//     │
//     ▼
//   [PostDispatch hooks] ← tracing, metrics
//     │
//     ▼
//   [PreEmit hooks]      ← event validation
//     │
//     ▼
//   Event broadcast to subscribers
//     │
//     ▼
//   [PostEmit hooks]     ← learning, cleanup
```

### HookResult — What Interception Can Do

```
ENUM HookResult
  Continue
    // Pass through unchanged — most common result

  Transform(Envelope<Action>)
    // Modify the message before it continues
    // Use case: inject trace headers, adjust priority

  Abort { reason: String }
    // Kill the pipeline — message is NOT processed
    // Use case: auth failure, budget exceeded, circuit open

  Fork { additional: Vec<Action> }
    // Inject additional actions into the pipeline
    // Use case: trigger audit logging alongside primary action
```

### Hook Trait

```
TRAIT Hook: Send + Sync + 'static
  // Which lifecycle point this hook intercepts
  FN hook_point(&self) -> HookPoint

  // Higher priority runs first (default 0)
  FN priority(&self) -> i32
    0

  // Human-readable name for tracing
  FN name(&self) -> &'static str

  // The interception logic
  ASYNC FN intercept(&self, envelope: &Envelope<Action>) -> HookResult

  // Optional: intercept events (default: pass through)
  ASYNC FN intercept_event(&self, envelope: &Envelope<Event>) -> HookResult
    HookResult::Continue
```

### HookRegistry

```
STRUCT HookRegistry
  FIELDS:
    hooks: HashMap<HookPoint, BTreeMap<Reverse<i32>, Vec<Arc<dyn Hook>>>>
    // Sorted by priority (highest first), then insertion order
    metrics: HookMetrics

  FN register(&mut self, hook: Arc<dyn Hook>)
    LET point = hook.hook_point()
    LET priority = Reverse(hook.priority())
    self.hooks
      .entry(point)
      .or_default()
      .entry(priority)
      .or_default()
      .push(hook)

  ASYNC FN run_hooks(
    &self,
    point: HookPoint,
    envelope: Envelope<Action>,
  ) -> Result<Envelope<Action>, HookAbort>

    LET hooks = self.hooks.get(&point)
    IF hooks.is_none() THEN RETURN Ok(envelope)

    LET mut current = envelope
    LET mut forked_actions = Vec::new()

    FOR (_, hook_group) IN hooks.unwrap()
      FOR hook IN hook_group
        LET start = Instant::now()
        LET result = hook.intercept(&current).await
        self.metrics.record(hook.name(), point, start.elapsed())

        MATCH result
          HookResult::Continue =>
            // Pass through

          HookResult::Transform(new_envelope) =>
            current = new_envelope

          HookResult::Abort { reason } =>
            self.metrics.aborts.increment()
            RETURN Err(HookAbort {
              hook_name: hook.name(),
              hook_point: point,
              reason,
            })

          HookResult::Fork { additional } =>
            forked_actions.extend(additional)

    // Process forked actions (non-blocking)
    IF !forked_actions.is_empty()
      self.dispatch_forked(forked_actions).await

    Ok(current)

  ASYNC FN run_event_hooks(
    &self,
    point: HookPoint,
    envelope: Envelope<Event>,
  ) -> Result<Envelope<Event>, HookAbort>
    // Similar to run_hooks but for events
    // Events can be transformed but not forked (events are notifications, not commands)
```

---

## Built-in Hooks

### 1. AuthorizationHook (Component 24)

```
STRUCT AuthorizationHook
  verifying_keys: HashMap<ActorId, VerifyingKey>
  require_signature: bool  // Default: true in production, false in dev

IMPL Hook FOR AuthorizationHook
  FN hook_point(&self) -> HookPoint => PreDispatch
  FN priority(&self) -> i32 => 100  // Runs first — reject unauthorized early
  FN name(&self) -> &'static str => "authorization"

  ASYNC FN intercept(&self, envelope: &Envelope<Action>) -> HookResult
    IF !self.require_signature THEN RETURN HookResult::Continue

    MATCH &envelope.signature
      None =>
        HookResult::Abort { reason: "Unsigned message rejected".into() }
      Some(_) =>
        LET key = self.verifying_keys.get(&envelope.source)
        MATCH key
          None =>
            HookResult::Abort { reason: format!("Unknown actor: {}", envelope.source) }
          Some(key) =>
            IF envelope.verify(key)
              HookResult::Continue
            ELSE
              HookResult::Abort { reason: "Invalid signature".into() }
```

### 2. BudgetEnforcementHook (Component 20)

```
STRUCT BudgetEnforcementHook
  budgets: DashMap<CorrelationId, ResourceBudget>
  usage:   DashMap<CorrelationId, ResourceUsage>

IMPL Hook FOR BudgetEnforcementHook
  FN hook_point(&self) -> HookPoint => PreDispatch
  FN priority(&self) -> i32 => 90  // After auth, before processing
  FN name(&self) -> &'static str => "budget_enforcement"

  ASYNC FN intercept(&self, envelope: &Envelope<Action>) -> HookResult
    LET corr = &envelope.correlation_id

    LET budget = self.budgets.get(corr)
    IF budget.is_none() THEN RETURN HookResult::Continue  // No budget set

    LET usage = self.usage.entry(corr.clone()).or_default()

    IF usage.tokens_used >= budget.max_tokens
      RETURN HookResult::Abort {
        reason: format!("Budget exhausted: {}/{} tokens", usage.tokens_used, budget.max_tokens)
      }

    IF usage.api_calls >= budget.max_api_calls
      RETURN HookResult::Abort {
        reason: format!("API call limit: {}/{}", usage.api_calls, budget.max_api_calls)
      }

    HookResult::Continue
```

### 3. CircuitBreakerHook (Component 26)

```
STRUCT CircuitBreakerHook
  breakers: DashMap<ActorId, CircuitBreaker>

STRUCT CircuitBreaker
  state:         CircuitState
  failure_count: u32
  threshold:     u32          // Failures before opening
  reset_timeout: Duration     // Time before half-open
  last_failure:  Option<Instant>

ENUM CircuitState
  Closed      // Normal operation
  Open        // Rejecting all requests
  HalfOpen    // Allowing one probe request

IMPL Hook FOR CircuitBreakerHook
  FN hook_point(&self) -> HookPoint => PreDispatch
  FN priority(&self) -> i32 => 80
  FN name(&self) -> &'static str => "circuit_breaker"

  ASYNC FN intercept(&self, envelope: &Envelope<Action>) -> HookResult
    LET target = extract_target_actor(&envelope.payload)
    LET breaker = self.breakers.entry(target.clone()).or_insert_with(||
      CircuitBreaker::new(threshold: 5, reset_timeout: Duration::from_secs(30))
    )

    MATCH breaker.state
      CircuitState::Closed =>
        HookResult::Continue

      CircuitState::Open =>
        IF breaker.should_attempt_reset()
          breaker.state = CircuitState::HalfOpen
          HookResult::Continue  // Allow probe
        ELSE
          HookResult::Abort {
            reason: format!("Circuit open for {}: {} failures", target, breaker.failure_count)
          }

      CircuitState::HalfOpen =>
        HookResult::Continue  // Allow probe through

  // PostDispatch: record success/failure
  ASYNC FN on_result(&self, target: ActorId, success: bool)
    LET mut breaker = self.breakers.get_mut(&target)
    IF success
      breaker.failure_count = 0
      breaker.state = CircuitState::Closed
    ELSE
      breaker.failure_count += 1
      breaker.last_failure = Some(Instant::now())
      IF breaker.failure_count >= breaker.threshold
        breaker.state = CircuitState::Open
```

### 4. TracingHook (Component 25)

```
STRUCT TracingHook
  // Integrates with tracing crate spans

IMPL Hook FOR TracingHook
  FN hook_point(&self) -> HookPoint => PreDispatch  // Also PostDispatch, PreEmit, PostEmit
  FN priority(&self) -> i32 => 50  // Middle — after security, before processing
  FN name(&self) -> &'static str => "tracing"

  ASYNC FN intercept(&self, envelope: &Envelope<Action>) -> HookResult
    tracing::info_span!("umb.action",
      message_id = %envelope.id,
      trace_id = %envelope.trace_id,
      source = %envelope.source,
      action_type = envelope.payload.message_type(),
      priority = ?envelope.priority,
    ).in_scope(|| {
      tracing::info!("Action dispatched")
    })
    HookResult::Continue

  ASYNC FN intercept_event(&self, envelope: &Envelope<Event>) -> HookResult
    tracing::info_span!("umb.event",
      message_id = %envelope.id,
      trace_id = %envelope.trace_id,
      event_type = envelope.payload.message_type(),
    ).in_scope(|| {
      tracing::info!("Event emitted")
    })
    HookResult::Continue
```

### 5. ConstitutionalPreCheckHook (Component 23/ASPH)

```
STRUCT ConstitutionalPreCheckHook
  sycophancy_detector: SycophancyDetector
  bias_scanner:        BiasScanner

IMPL Hook FOR ConstitutionalPreCheckHook
  FN hook_point(&self) -> HookPoint => PreGate
  FN priority(&self) -> i32 => 100
  FN name(&self) -> &'static str => "constitutional_precheck"

  ASYNC FN intercept(&self, envelope: &Envelope<Action>) -> HookResult
    IF LET Action::ValidateOutput { output, .. } = &envelope.payload
      // Pre-screen for obvious failures before expensive gate evaluation
      IF self.sycophancy_detector.detect(&output.content) > 0.8
        RETURN HookResult::Abort {
          reason: "ASPH: Sycophancy detected (score > 0.8)".into()
        }
      IF self.bias_scanner.scan(&output.content).has_critical_bias()
        RETURN HookResult::Abort {
          reason: "Constitutional: Critical bias detected".into()
        }
    HookResult::Continue
```

### 6. BlockGraphAttestationHook (Component 22)

```
STRUCT BlockGraphAttestationHook
  ledger: Arc<Mutex<EvidenceLedger>>

IMPL Hook FOR BlockGraphAttestationHook
  FN hook_point(&self) -> HookPoint => PostGate
  FN priority(&self) -> i32 => 10  // Runs late — after all validation
  FN name(&self) -> &'static str => "blockgraph_attestation"

  ASYNC FN intercept_event(&self, envelope: &Envelope<Event>) -> HookResult
    IF LET Event::ResponseSynthesized { request_id, quality_score } = &envelope.payload
      LET receipt = ActionReceipt {
        request_id: request_id.clone(),
        quality_score: *quality_score,
        timestamp: envelope.timestamp,
        trace_id: envelope.trace_id.clone(),
        hash: blake3::hash(&envelope.signing_material()),
      }
      self.ledger.lock().await.append(receipt)
    HookResult::Continue
```

### 7. IhsanScoringHook

```
STRUCT IhsanScoringHook
  threshold: f64  // Default: 0.95 (from constants.py)

IMPL Hook FOR IhsanScoringHook
  FN hook_point(&self) -> HookPoint => PostDispatch
  FN priority(&self) -> i32 => 60
  FN name(&self) -> &'static str => "ihsan_scoring"

  ASYNC FN intercept_event(&self, envelope: &Envelope<Event>) -> HookResult
    IF LET Event::IhsanScoreComputed { score, target } = &envelope.payload
      IF *score < self.threshold
        // Emit breach event — doesn't abort, but alerts
        RETURN HookResult::Fork {
          additional: vec![Action::RecoverFromFailure {
            failed_action: Box::new(Action::ValidateOutput { /* ... */ }),
            error: SystemError::ihsan_breach(*score, self.threshold),
          }]
        }
    HookResult::Continue
```

---

## TDD Anchors

```
TEST test_hook_priority_ordering
  // Register hooks with priorities 10, 50, 100
  // Assert they run in order: 100, 50, 10 (highest first)
  LET execution_order = Arc::new(Mutex::new(Vec::new()))
  LET hooks = [
    OrderTrackingHook::new("low", 10, execution_order.clone()),
    OrderTrackingHook::new("mid", 50, execution_order.clone()),
    OrderTrackingHook::new("high", 100, execution_order.clone()),
  ]
  // Register in scrambled order
  registry.register(hooks[1])
  registry.register(hooks[0])
  registry.register(hooks[2])

  registry.run_hooks(HookPoint::PreDispatch, envelope).await.unwrap()
  ASSERT execution_order.lock() == ["high", "mid", "low"]

TEST test_hook_abort_stops_pipeline
  // Hook1 (priority 100): Continue
  // Hook2 (priority 50): Abort
  // Hook3 (priority 10): Continue (should NOT run)
  LET result = registry.run_hooks(HookPoint::PreDispatch, envelope).await
  ASSERT result.is_err()
  ASSERT hook3.was_called() == false

TEST test_hook_transform_modifies_envelope
  // Hook transforms priority from Normal to High
  LET env = Envelope::new(action, trace, source, corr)  // Normal priority
  LET result = registry.run_hooks(HookPoint::PreDispatch, env).await.unwrap()
  ASSERT result.priority == Priority::High

TEST test_hook_fork_injects_actions
  // Hook returns Fork with 2 additional actions
  // Assert both additional actions are dispatched
  LET result = registry.run_hooks(HookPoint::PreDispatch, env).await.unwrap()
  // forked_actions should contain 2 entries

TEST test_circuit_breaker_opens_after_threshold
  LET hook = CircuitBreakerHook::new(threshold: 3)
  // 3 failures for same actor
  hook.on_result(actor.clone(), false).await
  hook.on_result(actor.clone(), false).await
  hook.on_result(actor.clone(), false).await
  // 4th request should be rejected
  LET result = hook.intercept(&envelope).await
  ASSERT matches!(result, HookResult::Abort { .. })

TEST test_circuit_breaker_resets_after_timeout
  LET hook = CircuitBreakerHook::new(threshold: 1, reset_timeout: Duration::from_millis(50))
  hook.on_result(actor.clone(), false).await
  // Circuit is open
  ASSERT matches!(hook.intercept(&envelope).await, HookResult::Abort { .. })
  // Wait for reset timeout
  sleep(Duration::from_millis(100)).await
  // Should be half-open now — allow probe
  ASSERT matches!(hook.intercept(&envelope).await, HookResult::Continue)

TEST test_budget_enforcement_rejects_over_limit
  LET hook = BudgetEnforcementHook::new()
  hook.set_budget(corr.clone(), ResourceBudget { max_tokens: 100, .. })
  hook.record_usage(corr.clone(), ResourceUsage { tokens_used: 100, .. })
  LET result = hook.intercept(&envelope).await
  ASSERT matches!(result, HookResult::Abort { .. })

TEST test_ihsan_hook_forks_on_breach
  LET hook = IhsanScoringHook::new(threshold: 0.95)
  LET event = Event::IhsanScoreComputed { score: 0.80, target: 0.95 }
  LET result = hook.intercept_event(&event_envelope).await
  ASSERT matches!(result, HookResult::Fork { .. })

TEST test_tracing_hook_adds_spans
  // Verify tracing hook emits correct span fields
  // Use tracing-test subscriber to capture spans
```

## Edge Cases

- Hook that panics must not crash the pipeline (catch_unwind + log)
- Hook with same priority as another → insertion order determines execution order
- Empty HookRegistry → envelope passes through unchanged
- Hook at PreDispatch aborts → PostDispatch hooks do NOT run
- Multiple Transform results → they compose (each transform feeds the next)
- Fork during PostEmit → forked actions are queued for next dispatch cycle
