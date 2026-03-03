# Phase 55.4: Typestate State Machine — Compile-Time Correctness

## Week 4 Deliverable

Implement a state machine where illegal transitions are impossible — they fail at
compile time, not runtime. Uses Rust's phantom type parameter pattern: each state
is a zero-sized type (ZST), and only valid transitions are implemented.

---

## Module: `state.rs`

### Design Rationale

Traditional state machines use a `state: State` enum field and match on it at
runtime. This has two problems:

1. Every transition must check the current state → runtime overhead
2. Invalid transitions compile fine — they just panic or return errors at runtime

The typestate pattern eliminates both. Each state is a distinct type. Methods for
transitioning are only implemented on the correct source state. Calling `.synthesize()`
on a `RequestState<Received>` literally does not compile.

### State Types (Zero-Sized)

```
// These types exist only for the type system. They occupy zero bytes at runtime.
// The compiler uses them to enforce valid transitions.

TRAIT State {}

STRUCT Received;     IMPL State FOR Received {}
STRUCT Parsed;       IMPL State FOR Parsed {}
STRUCT Decomposed;   IMPL State FOR Decomposed {}
STRUCT Executing;    IMPL State FOR Executing {}
STRUCT Synthesizing; IMPL State FOR Synthesizing {}
STRUCT Validating;   IMPL State FOR Validating {}
STRUCT Attesting;    IMPL State FOR Attesting {}
STRUCT Completed;    IMPL State FOR Completed {}
STRUCT Failed;       IMPL State FOR Failed {}

// State graph:
//
//   Received ──→ Parsed ──→ Decomposed ──→ Executing
//       │                                      │
//       ▼                                      ▼
//     Failed ◄── Validating ◄── Synthesizing ──┘
//       ▲            │
//       │            ▼
//       └─── Attesting ──→ Completed
//
// Note: ANY state can transition to Failed (error at any point)
// Only Attesting → Completed (all gates passed)
```

### RequestState

```
STRUCT RequestState<S: State>
  FIELDS:
    id:           CorrelationId
    trace_id:     TraceId
    data:         RequestData
    entered_at:   Instant          // When this state was entered
    history:      Vec<StateEntry>  // Audit trail of all transitions
    _state:       PhantomData<S>   // Zero-sized state marker

  // INVARIANT: _state is never read at runtime. It exists only for the compiler.

STRUCT RequestData
  raw_input:    Option<String>
  intent:       Option<StructuredIntent>
  dag:          Option<TaskDAG>
  results:      Vec<AgentOutput>
  error:        Option<SystemError>
  violation:    Option<ViolationType>
  ihsan_score:  Option<f64>
  attestation:  Option<[u8; 32]>
  metadata:     HashMap<String, Value>

STRUCT StateEntry
  from_state:  &'static str
  to_state:    &'static str
  timestamp:   Instant
  duration_ms: f64      // Time spent in previous state
  trigger:     String   // What caused the transition
```

### Valid Transitions

```
// Each IMPL block defines which transitions are valid FROM that state.
// Transitions not implemented here cannot be called — compile error.

IMPL RequestState<Received>
  FN new(raw_input: String, trace_id: TraceId) -> Self
    Self {
      id: CorrelationId::new(),
      trace_id,
      data: RequestData { raw_input: Some(raw_input), ..Default::default() },
      entered_at: Instant::now(),
      history: vec![],
      _state: PhantomData,
    }

  FN parse(self, intent: StructuredIntent) -> RequestState<Parsed>
    // Emits Event::IntentParsed
    LET mut history = self.history
    history.push(StateEntry {
      from_state: "Received",
      to_state: "Parsed",
      timestamp: Instant::now(),
      duration_ms: self.entered_at.elapsed().as_secs_f64() * 1000.0,
      trigger: format!("Intent parsed: {}", intent.task_type),
    })

    RequestState {
      id: self.id,
      trace_id: self.trace_id,
      data: self.data.with_intent(intent),
      entered_at: Instant::now(),
      history,
      _state: PhantomData,
    }

  FN fail(self, error: SystemError) -> RequestState<Failed>
    self.transition_to_failed(error, "Received")

IMPL RequestState<Parsed>
  FN decompose(self, dag: TaskDAG) -> RequestState<Decomposed>
    // Emits Event::TaskDecomposed
    LET mut history = self.history
    history.push(StateEntry {
      from_state: "Parsed",
      to_state: "Decomposed",
      timestamp: Instant::now(),
      duration_ms: self.entered_at.elapsed().as_secs_f64() * 1000.0,
      trigger: format!("DAG created: {} subtasks", dag.subtask_count()),
    })

    RequestState {
      id: self.id,
      trace_id: self.trace_id,
      data: self.data.with_dag(dag),
      entered_at: Instant::now(),
      history,
      _state: PhantomData,
    }

  FN fail(self, error: SystemError) -> RequestState<Failed>
    self.transition_to_failed(error, "Parsed")

IMPL RequestState<Decomposed>
  FN begin_execution(self) -> RequestState<Executing>
    // Starts the Saga — dispatches actions to agents
    self.transition("Decomposed", "Executing", "Saga started")

  FN fail(self, error: SystemError) -> RequestState<Failed>
    self.transition_to_failed(error, "Decomposed")

IMPL RequestState<Executing>
  FN synthesize(self, results: Vec<AgentOutput>) -> RequestState<Synthesizing>
    LET data = self.data.with_results(results)
    self.transition_with_data("Executing", "Synthesizing", "All agents completed", data)

  FN fail(self, error: SystemError) -> RequestState<Failed>
    // Triggers compensating actions (Saga rollback)
    self.transition_to_failed(error, "Executing")

IMPL RequestState<Synthesizing>
  FN validate(self) -> RequestState<Validating>
    self.transition("Synthesizing", "Validating", "Response synthesized")

  FN fail(self, error: SystemError) -> RequestState<Failed>
    self.transition_to_failed(error, "Synthesizing")

IMPL RequestState<Validating>
  FN attest(self) -> RequestState<Attesting>
    // ONLY reachable if ALL constitutional gates passed
    self.transition("Validating", "Attesting", "All gates passed")

  FN reject(self, violation: ViolationType) -> RequestState<Failed>
    LET data = self.data.with_violation(violation.clone())
    self.transition_to_failed(
      SystemError::gate_violation(violation),
      "Validating",
    )

IMPL RequestState<Attesting>
  FN complete(self, attestation: [u8; 32]) -> RequestState<Completed>
    LET data = self.data.with_attestation(attestation)
    self.transition_with_data("Attesting", "Completed", "BlockGraph attested", data)

IMPL RequestState<Completed>
  // Terminal state — no further transitions
  FN result(&self) -> &RequestData
    &self.data

  FN total_duration(&self) -> Duration
    self.history.first()
      .map(|first| first.timestamp.elapsed())
      .unwrap_or_default()

IMPL RequestState<Failed>
  // Terminal state — only introspection
  FN error(&self) -> &SystemError
    self.data.error.as_ref().unwrap()

  FN failed_at_state(&self) -> &str
    self.history.last()
      .map(|e| e.from_state)
      .unwrap_or("Unknown")
```

### Shared Transition Helper

```
// Private helper — avoids boilerplate in each IMPL
IMPL<S: State> RequestState<S>
  FN transition<T: State>(
    self,
    from: &'static str,
    to: &'static str,
    trigger: impl Into<String>,
  ) -> RequestState<T>
    LET mut history = self.history
    history.push(StateEntry {
      from_state: from,
      to_state: to,
      timestamp: Instant::now(),
      duration_ms: self.entered_at.elapsed().as_secs_f64() * 1000.0,
      trigger: trigger.into(),
    })
    RequestState {
      id: self.id,
      trace_id: self.trace_id,
      data: self.data,
      entered_at: Instant::now(),
      history,
      _state: PhantomData,
    }

  FN transition_to_failed(self, error: SystemError, from: &'static str) -> RequestState<Failed>
    LET data = self.data.with_error(error)
    LET mut history = self.history
    history.push(StateEntry {
      from_state: from,
      to_state: "Failed",
      timestamp: Instant::now(),
      duration_ms: self.entered_at.elapsed().as_secs_f64() * 1000.0,
      trigger: format!("Error: {}", data.error.as_ref().unwrap()),
    })
    RequestState {
      id: self.id,
      trace_id: self.trace_id,
      data,
      entered_at: Instant::now(),
      history,
      _state: PhantomData,
    }
```

### StateStore — Event-Sourced Persistence

```
STRUCT StateStore
  active:    DashMap<CorrelationId, DynRequestState>
  completed: DashMap<CorrelationId, RequestData>
  event_log: Mutex<Vec<StateTransitionEvent>>

// DynRequestState wraps the typestate in a runtime-compatible enum
// for storage. The typestate is used during processing; the store
// captures snapshots.
ENUM DynRequestState
  Received(RequestState<Received>)
  Parsed(RequestState<Parsed>)
  Decomposed(RequestState<Decomposed>)
  Executing(RequestState<Executing>)
  Synthesizing(RequestState<Synthesizing>)
  Validating(RequestState<Validating>)
  Attesting(RequestState<Attesting>)
  Completed(RequestState<Completed>)
  Failed(RequestState<Failed>)

  FN state_name(&self) -> &'static str
    MATCH self
      Received(_)     => "Received"
      Parsed(_)       => "Parsed"
      Decomposed(_)   => "Decomposed"
      Executing(_)    => "Executing"
      Synthesizing(_) => "Synthesizing"
      Validating(_)   => "Validating"
      Attesting(_)    => "Attesting"
      Completed(_)    => "Completed"
      Failed(_)       => "Failed"
```

---

## TDD Anchors

```
TEST test_valid_lifecycle_compiles
  // The happy path must compile and execute correctly
  LET req = RequestState::<Received>::new("test query".into(), TraceId::new())
  LET req = req.parse(intent)
  LET req = req.decompose(dag)
  LET req = req.begin_execution()
  LET req = req.synthesize(results)
  LET req = req.validate()
  LET req = req.attest()
  LET req = req.complete(attestation_hash)
  ASSERT req.total_duration() > Duration::ZERO

TEST test_invalid_transition_does_not_compile
  // This test is a COMPILE-TIME test.
  // It exists as a comment — if uncommented, it must fail to compile.
  //
  // let req = RequestState::<Received>::new("test".into(), TraceId::new());
  // req.synthesize(vec![]);    // ERROR: no method `synthesize` for RequestState<Received>
  // req.complete(hash);        // ERROR: no method `complete` for RequestState<Received>
  // req.attest();              // ERROR: no method `attest` for RequestState<Received>

TEST test_any_state_can_fail
  // Received → Failed
  LET req = RequestState::<Received>::new("test".into(), TraceId::new())
  LET failed = req.fail(SystemError::new("parse error"))
  ASSERT failed.failed_at_state() == "Received"

  // Parsed → Failed
  LET req = RequestState::<Received>::new("test".into(), TraceId::new())
    .parse(intent)
  LET failed = req.fail(SystemError::new("decompose error"))
  ASSERT failed.failed_at_state() == "Parsed"

  // Executing → Failed
  LET req = RequestState::<Received>::new("test".into(), TraceId::new())
    .parse(intent).decompose(dag).begin_execution()
  LET failed = req.fail(SystemError::new("agent timeout"))
  ASSERT failed.failed_at_state() == "Executing"

TEST test_history_records_all_transitions
  LET req = RequestState::<Received>::new("test".into(), TraceId::new())
    .parse(intent)
    .decompose(dag)
    .begin_execution()
    .synthesize(results)
    .validate()
    .attest()
    .complete(hash)

  ASSERT req.data.history.len() == 7  // 7 transitions
  ASSERT req.data.history[0].from_state == "Received"
  ASSERT req.data.history[0].to_state == "Parsed"
  ASSERT req.data.history[6].from_state == "Attesting"
  ASSERT req.data.history[6].to_state == "Completed"

TEST test_state_durations_are_positive
  LET req = RequestState::<Received>::new("test".into(), TraceId::new())
  sleep(Duration::from_millis(10))
  LET req = req.parse(intent)
  ASSERT req.data.history[0].duration_ms >= 10.0

TEST test_completed_is_terminal
  // RequestState<Completed> has no transition methods except introspection
  LET req = /* ... full lifecycle ... */ .complete(hash)
  // req.fail(error)  // Would not compile — Completed has no fail()
  ASSERT req.result().attestation.is_some()

TEST test_failed_is_terminal
  LET req = RequestState::<Received>::new("test".into(), TraceId::new())
    .fail(SystemError::new("test error"))
  // req.parse(intent)  // Would not compile — Failed has no parse()
  ASSERT req.error().message == "test error"

TEST test_request_state_consumes_self
  // Transitions consume self — you cannot use a state after transitioning
  LET req = RequestState::<Received>::new("test".into(), TraceId::new())
  LET parsed = req.parse(intent)
  // req.parse(intent2)  // COMPILE ERROR: use of moved value `req`
  // This prevents "time travel" — once you've moved forward, you can't go back

TEST test_dyn_request_state_wrapping
  LET req = RequestState::<Executing>::new(/* ... */)
  LET dyn_state = DynRequestState::Executing(req)
  ASSERT dyn_state.state_name() == "Executing"
```

## Edge Cases

- `PhantomData<S>` is zero-sized — `size_of::<RequestState<Received>>() == size_of::<RequestState<Parsed>>()`
- History grows linearly — for long-lived requests, consider capping at N entries
- StateStore's DynRequestState is the runtime escape hatch — minimize its use
- `transition_to_failed` must work from ANY state — verify with one test per state
- Clock monotonicity: `entered_at.elapsed()` can theoretically be negative on clock skew
- Thread safety: RequestState is `Send` but not `Sync` (moved between threads, not shared)
