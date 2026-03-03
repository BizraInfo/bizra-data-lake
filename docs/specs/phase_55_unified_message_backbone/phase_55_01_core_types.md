# Phase 55.1: Core Types — Envelope, Identity, Action, Event

## Week 1 Deliverable

Define the foundational type system that every UMB component compiles against.
Rust's type system encodes correctness at compile time — invalid message flows,
illegal state transitions, and unhandled events become compilation errors.

---

## Module: `envelope.rs`

### Message Trait

```
TRAIT Message
  REQUIRES: Clone + Send + Sync + 'static

  FN message_type() -> &'static str
    // Returns a dot-separated topic string
    // e.g., "action.parse_intent", "event.gate_failed"
```

### Identity Types

```
STRUCT MessageId
  inner: Uuid (v4)
  DERIVE: Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize

  FN new() -> Self
    Self(Uuid::new_v4())

STRUCT TraceId
  inner: Uuid (v4)
  DERIVE: Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize
  // Distributed tracing — spans entire request lifecycle
  // Maps to Component 25 (Observability)

STRUCT ActorId
  inner: String
  DERIVE: Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize

  FN system() -> Self
    Self("system".into())

  FN agent(name: &str) -> Self
    Self(format!("agent:{}", name))

STRUCT CorrelationId
  inner: Uuid (v4)
  DERIVE: Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize
  // Groups all messages related to a single user request
```

### Priority Levels

```
ENUM Priority
  DERIVE: Clone, Debug, PartialOrd, Ord, PartialEq, Eq, Serialize, Deserialize

  Background  = 0   // Deferred work, learning updates
  Normal      = 1   // Standard request processing
  High        = 2   // Elevated priority tasks
  Critical    = 3   // Constitutional gate violations
  SystemPanic = 4   // Daughter Test failures, irrecoverable

  // INVARIANT: SystemPanic messages bypass all non-essential hooks
  // INVARIANT: Critical+ messages get dedicated channel capacity
```

### Envelope

```
STRUCT Envelope<T: Message>
  DERIVE: Clone, Debug

  FIELDS:
    id:             MessageId        // Unique per message
    trace_id:       TraceId          // Shared across request lifecycle
    source:         ActorId          // Who sent this
    timestamp:      SystemTime       // When created
    signature:      Option<Signature>  // Ed25519 (Component 24)
    payload:        T                // The actual message content
    causation_id:   Option<MessageId>  // What caused this message
    correlation_id: CorrelationId    // Groups related messages
    ttl:            Duration         // Prevents zombie messages
    priority:       Priority         // Dispatch ordering

  FN new(payload, trace_id, source, correlation_id) -> Self
    Self {
      id: MessageId::new(),
      trace_id,
      source,
      timestamp: SystemTime::now(),
      signature: None,
      payload,
      causation_id: None,
      correlation_id,
      ttl: Duration::from_secs(30),  // Default 30s TTL
      priority: Priority::Normal,
    }

  FN with_causation(mut self, cause: MessageId) -> Self
    self.causation_id = Some(cause)
    self

  FN with_priority(mut self, priority: Priority) -> Self
    self.priority = priority
    self

  FN with_ttl(mut self, ttl: Duration) -> Self
    self.ttl = ttl
    self

  FN sign(mut self, signing_key: &SigningKey) -> Self
    // Sign blake3(id + trace_id + timestamp + payload_type)
    let digest = blake3::hash(self.signing_material())
    self.signature = Some(signing_key.sign(digest.as_bytes()))
    self

  FN verify(&self, verifying_key: &VerifyingKey) -> bool
    MATCH self.signature
      Some(sig) => {
        let digest = blake3::hash(self.signing_material())
        verifying_key.verify(digest.as_bytes(), &sig).is_ok()
      }
      None => false

  FN is_expired(&self) -> bool
    SystemTime::now()
      .duration_since(self.timestamp)
      .map(|elapsed| elapsed > self.ttl)
      .unwrap_or(true)
```

---

## Module: `action.rs`

### Action Enum — Imperative Commands ("Do This")

```
ENUM Action
  DERIVE: Clone, Debug, Serialize, Deserialize

  // Component 1: Intent Parser
  ParseIntent {
    raw_input: String,
    user_id:   ActorId,
  }

  // Component 2: Task Decomposer
  DecomposeTask {
    intent:      StructuredIntent,
    constraints: Vec<Constraint>,
  }

  // Component 4: Orchestrator
  AssignSubtask {
    task:     SubTask,
    agent_id: ActorId,
    budget:   ResourceBudget,
  }

  ReplanExecution {
    saga_id: SagaId,
    reason:  ReplanReason,
  }

  // Component 5: Agent Runtime
  ExecuteSkill {
    agent_id: ActorId,
    skill:    SkillInvocation,
    context:  AgentContext,
  }

  // Component 9: Constitutional Gates
  ValidateOutput {
    output:        AgentOutput,
    gate_sequence: Vec<GateId>,
  }

  // Component 10: Error Recovery
  RecoverFromFailure {
    failed_action: Box<Action>,
    error:         SystemError,
  }

  // Component 11: Output Synthesis
  SynthesizeResponse {
    partial_results: Vec<AgentOutput>,
    request_id:      CorrelationId,
  }

  // Component 22: Audit
  AttestToBlockGraph {
    trace: ExecutionTrace,
  }

IMPL Message FOR Action
  FN message_type(&self) -> &'static str
    MATCH self
      ParseIntent { .. }       => "action.parse_intent"
      DecomposeTask { .. }     => "action.decompose_task"
      AssignSubtask { .. }     => "action.assign_subtask"
      ReplanExecution { .. }   => "action.replan_execution"
      ExecuteSkill { .. }      => "action.execute_skill"
      ValidateOutput { .. }    => "action.validate_output"
      RecoverFromFailure { .. } => "action.recover_failure"
      SynthesizeResponse { .. } => "action.synthesize_response"
      AttestToBlockGraph { .. } => "action.attest_blockgraph"
    // Exhaustive — Rust compiler enforces all variants handled
```

---

## Module: `event.rs`

### Event Enum — Declarative Notifications ("This Happened")

```
ENUM Event
  DERIVE: Clone, Debug, Serialize, Deserialize

  // === Lifecycle Events ===
  RequestReceived {
    request_id: CorrelationId,
    user_id:    ActorId,
  }

  IntentParsed {
    intent:     StructuredIntent,
    confidence: f64,
  }

  TaskDecomposed {
    dag:           TaskDAG,
    critical_path: Vec<SubTaskId>,
  }

  // === Agent Events ===
  AgentStarted {
    agent_id: ActorId,
    task:     SubTaskId,
  }

  AgentCompleted {
    agent_id: ActorId,
    output:   AgentOutput,
    duration: Duration,
  }

  AgentFailed {
    agent_id:  ActorId,
    error:     SystemError,
    retryable: bool,
  }

  // === Gate Events (Components 9/23) ===
  GatePassed {
    gate_id:   GateId,
    score:     f64,
    threshold: f64,
  }

  GateFailed {
    gate_id:   GateId,
    score:     f64,
    violation: ViolationType,
  }

  IhsanScoreComputed {
    score:  f64,
    target: f64,
  }

  DaughterTestResult {
    passed: bool,
    reason: String,
  }

  // === System Events ===
  ResourceBudgetExhausted {
    saga_id: SagaId,
    spent:   ResourceUsage,
  }

  CircuitBreakerTripped {
    component:    ActorId,
    failure_rate: f64,
  }

  // === Completion Events ===
  ResponseSynthesized {
    request_id:    CorrelationId,
    quality_score: f64,
  }

  BlockGraphAttested {
    attestation_hash: [u8; 32],
    block_height:     u64,
  }

  // === Meta-Learning (Component 12) ===
  OutcomeLearned {
    lesson:     Lesson,
    confidence: f64,
  }

IMPL Message FOR Event
  FN message_type(&self) -> &'static str
    MATCH self
      RequestReceived { .. }       => "event.request_received"
      IntentParsed { .. }          => "event.intent_parsed"
      TaskDecomposed { .. }        => "event.task_decomposed"
      AgentStarted { .. }          => "event.agent_started"
      AgentCompleted { .. }        => "event.agent_completed"
      AgentFailed { .. }           => "event.agent_failed"
      GatePassed { .. }            => "event.gate_passed"
      GateFailed { .. }            => "event.gate_failed"
      IhsanScoreComputed { .. }    => "event.ihsan_score"
      DaughterTestResult { .. }    => "event.daughter_test"
      ResourceBudgetExhausted { .. } => "event.budget_exhausted"
      CircuitBreakerTripped { .. }   => "event.circuit_breaker"
      ResponseSynthesized { .. }     => "event.response_synthesized"
      BlockGraphAttested { .. }      => "event.blockgraph_attested"
      OutcomeLearned { .. }          => "event.outcome_learned"
```

---

## Supporting Types

```
STRUCT StructuredIntent
  task_type:  String           // "research", "code", "analyze", etc.
  entities:   Vec<String>      // Extracted entities
  confidence: f64              // Parser confidence [0.0, 1.0]

STRUCT SubTask
  id:          SubTaskId
  description: String
  skill:       SkillInvocation
  dependencies: Vec<SubTaskId>  // DAG edges

STRUCT ResourceBudget
  max_tokens:    u64
  max_duration:  Duration
  max_api_calls: u32

STRUCT AgentOutput
  content:       String
  quality_score: f64
  sources:       Vec<SourceRef>
  metadata:      HashMap<String, Value>

STRUCT ExecutionTrace
  request_id: CorrelationId
  trace_id:   TraceId
  steps:      Vec<TraceStep>
  total_time: Duration

STRUCT SystemError
  code:      ErrorCode
  message:   String
  retryable: bool
  source:    Option<Box<dyn std::error::Error + Send + Sync>>

ENUM GateId
  Alpha4Fallback
  Alpha7Verification
  Alpha8DarkMatter
  Alpha9Attestation
  Alpha10Binary
  IhsanGate
  DaughterTest

ENUM ViolationType
  ThresholdBreach { expected: f64, actual: f64 }
  ConstitutionalViolation { article: String }
  DaughterTestFailure { reason: String }
  SecurityViolation { detail: String }
```

---

## TDD Anchors

```
TEST test_message_id_uniqueness
  // Two MessageIds must never collide
  LET ids: HashSet = (0..10_000).map(|_| MessageId::new()).collect()
  ASSERT ids.len() == 10_000

TEST test_envelope_creation
  LET action = Action::ParseIntent { raw_input: "hello".into(), user_id: ActorId::system() }
  LET env = Envelope::new(action, TraceId::new(), ActorId::system(), CorrelationId::new())
  ASSERT env.priority == Priority::Normal
  ASSERT env.causation_id.is_none()
  ASSERT !env.is_expired()

TEST test_envelope_signing_verification
  LET (signing_key, verifying_key) = generate_ed25519_keypair()
  LET env = Envelope::new(action, trace, source, corr).sign(&signing_key)
  ASSERT env.verify(&verifying_key)
  // Tamper with envelope — verification must fail
  LET tampered = env.with_priority(Priority::SystemPanic)
  ASSERT !tampered.verify(&verifying_key)

TEST test_envelope_expiry
  LET env = Envelope::new(action, trace, source, corr)
    .with_ttl(Duration::from_millis(1))
  sleep(Duration::from_millis(10))
  ASSERT env.is_expired()

TEST test_priority_ordering
  ASSERT Priority::Background < Priority::Normal
  ASSERT Priority::Normal < Priority::High
  ASSERT Priority::High < Priority::Critical
  ASSERT Priority::Critical < Priority::SystemPanic

TEST test_action_message_type_exhaustive
  // Every Action variant must return a unique message_type
  LET types = all_action_variants().map(|a| a.message_type()).collect::<HashSet>()
  ASSERT types.len() == ACTION_VARIANT_COUNT

TEST test_event_message_type_exhaustive
  // Every Event variant must return a unique message_type
  LET types = all_event_variants().map(|e| e.message_type()).collect::<HashSet>()
  ASSERT types.len() == EVENT_VARIANT_COUNT

TEST test_actor_id_constructors
  ASSERT ActorId::system().inner == "system"
  ASSERT ActorId::agent("researcher").inner == "agent:researcher"

TEST test_envelope_causation_chain
  LET env1 = Envelope::new(action1, trace, source, corr)
  LET env2 = Envelope::new(action2, trace, source, corr)
    .with_causation(env1.id.clone())
  ASSERT env2.causation_id == Some(env1.id)
```

## Edge Cases

- Envelope with zero TTL must be immediately expired
- Signature verification on unsigned envelope returns false (not panic)
- SystemPanic priority must be highest — no priority can exceed it
- Action/Event `message_type()` must be deterministic (same variant = same string)
- Clone of Envelope must deep-clone the payload
- MessageId across threads must be collision-free (UUID v4 guarantees)
