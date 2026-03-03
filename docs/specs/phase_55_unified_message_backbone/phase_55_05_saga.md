# Phase 55.5: Saga Coordinator — Multi-Agent Transactions

## Week 5a Deliverable

Implement the Saga Coordinator for multi-agent transactions with compensating
actions. Based on Garcia-Molina's Saga Pattern: distributed transactions use
sequences of local steps with compensating actions for rollback.

---

## Module: `saga.rs`

### Design Rationale (Garcia-Molina)

A multi-agent task is a distributed transaction. You can't atomically commit
across independent agents. Instead, define a sequence of local steps with
compensating actions for rollback:

```
Step 1: Research (compensate: discard results)
Step 2: Evaluate (compensate: clear evaluations)
Step 3: Synthesize (compensate: delete draft)
Step 4: Validate (compensate: mark invalid)
Step 5: Attest (compensate: revoke attestation)
```

If Step 3 fails, run compensations in reverse: clear evaluations, discard results.

### Saga Types

```
STRUCT SagaId(Uuid)
  DERIVE: Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize

STRUCT Saga
  FIELDS:
    id:              SagaId
    request_id:      CorrelationId
    trace_id:        TraceId
    steps:           Vec<SagaStep>
    current_step:    usize
    state:           SagaState
    compensations:   Vec<CompensatingAction>  // Rollback stack (LIFO)
    created_at:      Instant
    timeout:         Duration
    results:         HashMap<SubTaskId, AgentOutput>  // Collected per step

  FN new(request_id, trace_id, steps, timeout) -> Self
    Self {
      id: SagaId(Uuid::new_v4()),
      request_id, trace_id,
      steps,
      current_step: 0,
      state: SagaState::Running { active_steps: vec![] },
      compensations: Vec::new(),
      created_at: Instant::now(),
      timeout,
      results: HashMap::new(),
    }

STRUCT SagaStep
  subtask:        SubTask
  assigned_agent: ActorId
  budget:         ResourceBudget
  timeout:        Duration
  retry_policy:   RetryPolicy
  compensation:   CompensatingAction
  parallel_group: Option<u32>  // Steps in same group run concurrently

STRUCT CompensatingAction
  description: String
  action:      Action        // The undo action to dispatch
  timeout:     Duration

  ASYNC FN execute(&self, bus: &ActionBus) -> Result<(), SystemError>
    bus.dispatch(Envelope::new(
      self.action.clone(),
      TraceId::current(),
      ActorId::system(),
      CorrelationId::current(),
    ).with_priority(Priority::High)
     .with_ttl(self.timeout)
    ).await
    .map_err(|e| SystemError::compensation_failed(e))

STRUCT RetryPolicy
  max_retries:    u32       // Default: 3
  backoff:        Duration  // Default: 1s
  backoff_factor: f64       // Default: 2.0 (exponential)
  retry_count:    u32       // Current attempt

ENUM SagaState
  Running { active_steps: Vec<SubTaskId> }
  AwaitingGate { gate_id: GateId }
  Compensating { failed_at: usize, reason: String, current: usize }
  Completed { results: Vec<AgentOutput> }
  Aborted { reason: String, compensations_applied: usize, compensations_failed: usize }

ENUM SagaDecision
  Continue                           // No state change
  AdvanceTo(SagaState)              // Transition to new state
  DispatchAction(Action)            // Send an action
  DispatchActions(Vec<Action>)      // Send multiple actions (parallel)
```

### Saga Advancement (Event-Driven State Machine)

```
IMPL Saga
  ASYNC FN advance(&mut self, event: &Event, bus: &ActionBus) -> SagaDecision

    // Global timeout check
    IF self.created_at.elapsed() > self.timeout
      RETURN self.begin_compensation("Saga timeout exceeded", self.current_step)

    MATCH (&self.state, event)
      // Agent completed → check if group done → advance
      (SagaState::Running { active_steps }, Event::AgentCompleted { agent_id, output, .. }) =>
        LET step = self.find_step_for_agent(agent_id)
        self.results.insert(step.subtask.id.clone(), output.clone())
        self.compensations.push(step.compensation.clone())

        LET remaining = active_steps.iter()
          .filter(|s| !self.results.contains_key(s)).count()

        IF remaining == 0
          MATCH self.next_step_group()
            Some(steps) =>
              SagaDecision::DispatchActions(steps.iter().map(|s|
                Action::AssignSubtask {
                  task: s.subtask.clone(),
                  agent_id: s.assigned_agent.clone(),
                  budget: s.budget.clone(),
                }
              ).collect())
            None =>
              SagaDecision::DispatchAction(Action::ValidateOutput {
                output: self.collect_outputs(),
                gate_sequence: vec![
                  GateId::Alpha4Fallback, GateId::Alpha7Verification,
                  GateId::Alpha8DarkMatter, GateId::Alpha9Attestation,
                  GateId::Alpha10Binary,
                ],
              })
        ELSE
          SagaDecision::Continue

      // Agent failed → retry or compensate
      (SagaState::Running { .. }, Event::AgentFailed { agent_id, error, retryable }) =>
        LET step = self.find_step_for_agent(agent_id)
        IF *retryable && step.retry_policy.retry_count < step.retry_policy.max_retries
          step.retry_policy.retry_count += 1
          SagaDecision::DispatchAction(Action::ExecuteSkill {
            agent_id: step.assigned_agent.clone(),
            skill: step.subtask.skill.clone(),
            context: AgentContext::retry(step.retry_policy.retry_count),
          })
        ELSE
          self.begin_compensation(&format!("Agent {} failed: {}", agent_id, error), self.current_step)

      // Gate passed → next gate or attest
      (SagaState::AwaitingGate { gate_id }, Event::GatePassed { .. }) =>
        MATCH self.next_gate(gate_id)
          Some(next) => SagaDecision::AdvanceTo(SagaState::AwaitingGate { gate_id: next })
          None => SagaDecision::DispatchAction(Action::AttestToBlockGraph {
            trace: self.execution_trace(),
          })

      // Gate failed → compensate all
      (SagaState::AwaitingGate { gate_id }, Event::GateFailed { violation, .. }) =>
        self.begin_compensation(&format!("Gate {:?} failed: {:?}", gate_id, violation), self.steps.len())

      // BlockGraph attested → complete
      (_, Event::BlockGraphAttested { .. }) =>
        SagaDecision::AdvanceTo(SagaState::Completed {
          results: self.results.values().cloned().collect(),
        })

      // Daughter Test failed → IMMEDIATE abort (no negotiation)
      (_, Event::DaughterTestResult { passed: false, reason }) =>
        SagaDecision::AdvanceTo(SagaState::Aborted {
          reason: format!("DAUGHTER TEST VIOLATION: {}", reason),
          compensations_applied: 0, compensations_failed: 0,
        })

      // Budget exhausted → compensate
      (_, Event::ResourceBudgetExhausted { saga_id, .. }) IF *saga_id == self.id =>
        self.begin_compensation("Resource budget exhausted", self.current_step)

      _ => SagaDecision::Continue

  FN begin_compensation(&mut self, reason: &str, failed_at: usize) -> SagaDecision
    SagaDecision::AdvanceTo(SagaState::Compensating {
      failed_at, reason: reason.to_string(),
      current: self.compensations.len(),
    })

  ASYNC FN run_compensations(&mut self, bus: &ActionBus) -> (usize, usize)
    LET mut applied = 0; LET mut failed = 0
    FOR compensation IN self.compensations.iter().rev()  // LIFO
      MATCH compensation.execute(bus).await
        Ok(()) => applied += 1
        Err(e) => { tracing::error!("Compensation failed: {e}"); failed += 1 }
    (applied, failed)

  FN next_gate(&self, current: &GateId) -> Option<GateId>
    LET seq = [Alpha4Fallback, Alpha7Verification, Alpha8DarkMatter, Alpha9Attestation, Alpha10Binary]
    seq.iter().position(|g| g == current).and_then(|p| seq.get(p + 1).cloned())
```

### SagaCoordinator

```
STRUCT SagaCoordinator
  sagas:   DashMap<SagaId, Saga>
  waiters: DashMap<CorrelationId, Vec<oneshot::Sender<SagaOutcome>>>
  bus:     Arc<ActionBus>

  FN create_saga(&self, request_id, trace_id, steps, timeout) -> SagaId
    LET saga = Saga::new(request_id, trace_id, steps, timeout)
    LET id = saga.id.clone()
    self.sagas.insert(id.clone(), saga)
    id

  ASYNC FN on_event(&self, event: &Event)
    LET mut completed = Vec::new()
    FOR mut entry IN self.sagas.iter_mut()
      LET decision = entry.value_mut().advance(event, &self.bus).await
      MATCH decision
        SagaDecision::AdvanceTo(SagaState::Completed { results }) =>
          completed.push((entry.key().clone(), SagaOutcome::Success(results)))
        SagaDecision::AdvanceTo(SagaState::Aborted { reason, .. }) =>
          completed.push((entry.key().clone(), SagaOutcome::Aborted(reason)))
        SagaDecision::AdvanceTo(SagaState::Compensating { .. }) =>
          LET (a, f) = entry.value_mut().run_compensations(&self.bus).await
          completed.push((entry.key().clone(), SagaOutcome::Compensated { applied: a, failed: f }))
        SagaDecision::DispatchAction(action) =>
          self.bus.dispatch(Envelope::new(action, /* ... */)).await
        SagaDecision::DispatchActions(actions) =>
          FOR a IN actions { self.bus.dispatch(Envelope::new(a, /* ... */)).await }
        _ => {}
    FOR (id, outcome) IN completed
      self.sagas.remove(&id)
      IF LET Some((_, waiters)) = self.waiters.remove(&id)
        FOR w IN waiters { let _ = w.send(outcome.clone()); }

  ASYNC FN await_completion(&self, request_id, timeout) -> Result<SagaOutcome, SystemError>
    LET (tx, rx) = oneshot::channel()
    self.waiters.entry(request_id).or_default().push(tx)
    tokio::time::timeout(timeout, rx).await
      .map_err(|_| SystemError::saga_timeout())?
      .map_err(|_| SystemError::saga_cancelled())

ENUM SagaOutcome
  Success(Vec<AgentOutput>)
  Compensated { applied: usize, failed: usize }
  Aborted(String)
```

---

## TDD Anchors

```
TEST test_saga_happy_path
  LET saga = Saga::new(request_id, trace_id, steps, Duration::from_secs(10))
  // 3 agents complete → ValidateOutput dispatched → gates pass → attested
  LET d3 = saga.advance(&Event::AgentCompleted { .. }, &bus).await
  ASSERT matches!(d3, SagaDecision::DispatchAction(Action::ValidateOutput { .. }))
  FOR gate IN [Alpha4, Alpha7, Alpha8, Alpha9, Alpha10]
    saga.advance(&Event::GatePassed { gate_id: gate, .. }, &bus).await
  LET d = saga.advance(&Event::BlockGraphAttested { .. }, &bus).await
  ASSERT matches!(d, SagaDecision::AdvanceTo(SagaState::Completed { .. }))

TEST test_saga_compensation_on_failure
  // Step 1 succeeds, step 2 fails non-retryable → compensate step 1
  LET d = saga.advance(&Event::AgentFailed { retryable: false, .. }, &bus).await
  ASSERT matches!(d, SagaDecision::AdvanceTo(SagaState::Compensating { .. }))
  LET (applied, failed) = saga.run_compensations(&bus).await
  ASSERT applied == 1 && failed == 0

TEST test_saga_retry_with_backoff
  // 3 retryable failures → retry, 4th → compensate
  LET d = saga.advance(&Event::AgentFailed { retryable: true, .. }, &bus).await
  ASSERT matches!(d, SagaDecision::DispatchAction(Action::ExecuteSkill { .. }))

TEST test_saga_daughter_test_immediate_abort
  LET d = saga.advance(&Event::DaughterTestResult { passed: false, .. }, &bus).await
  ASSERT matches!(d, SagaDecision::AdvanceTo(SagaState::Aborted { .. }))

TEST test_saga_timeout
  LET saga = Saga::new(/* timeout: 10ms */); sleep(50ms)
  LET d = saga.advance(&any_event, &bus).await
  ASSERT matches!(d, SagaDecision::AdvanceTo(SagaState::Compensating { .. }))

TEST test_saga_gate_failure_triggers_compensation
  LET d = saga.advance(&Event::GateFailed { gate_id: Alpha7, .. }, &bus).await
  ASSERT matches!(d, SagaDecision::AdvanceTo(SagaState::Compensating { .. }))

TEST test_parallel_saga_steps
  // Steps in parallel_group=1 dispatch simultaneously
  // Group 2 waits until group 1 completes
```

## Edge Cases

- Saga with zero steps → immediately completes (no-op)
- All compensations fail → SagaState::Aborted with failed count
- Saga receives event for different saga → SagaDecision::Continue (ignore)
- Concurrent saga advancement → DashMap handles thread safety
- Double completion event → idempotent (second is ignored)
