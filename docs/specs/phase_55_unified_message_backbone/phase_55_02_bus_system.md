# Phase 55.2: Bus System — Action Bus (mpsc) + Event Bus (broadcast)

## Week 2 Deliverable

Implement the two message-passing primitives. Actions use mpsc (one handler per
action type — no ambiguity). Events use broadcast (many subscribers react to one
event — fan-out).

---

## Module: `action_bus.rs`

### Design Rationale

Actions are imperative commands: "Parse this intent", "Execute this skill",
"Validate this output". Each action type has exactly ONE handler. If two
components could handle the same action, that's a design error — resolve it
at compile time, not runtime.

Uses `tokio::sync::mpsc` — bounded channel with backpressure.

### ActionHandler Trait

```
TRAIT ActionHandler: Send + Sync + 'static
  // Which action types this handler processes
  FN handles(&self) -> Vec<&'static str>

  // Process an action, returning events that resulted
  ASYNC FN handle(&self, envelope: Envelope<Action>) -> Result<Vec<Event>, SystemError>
```

### ActionBus

```
STRUCT ActionBus
  FIELDS:
    handlers:     HashMap<&'static str, Arc<dyn ActionHandler>>
    tx:           mpsc::Sender<Envelope<Action>>
    rx:           Mutex<mpsc::Receiver<Envelope<Action>>>  // Single consumer
    capacity:     usize                                     // Backpressure threshold
    metrics:      ActionBusMetrics
    event_bus:    Arc<EventBus>  // Forward resulting events

  CONST DEFAULT_CAPACITY: usize = 1024
  CONST HIGH_PRIORITY_CAPACITY: usize = 256  // Separate channel for Critical+

  FN new(capacity: usize, event_bus: Arc<EventBus>) -> Self
    LET (tx, rx) = mpsc::channel(capacity)
    Self {
      handlers: HashMap::new(),
      tx, rx: Mutex::new(rx),
      capacity,
      metrics: ActionBusMetrics::default(),
      event_bus,
    }

  FN register_handler(&mut self, handler: Arc<dyn ActionHandler>) -> Result<(), BusError>
    FOR action_type IN handler.handles()
      IF self.handlers.contains_key(action_type)
        RETURN Err(BusError::DuplicateHandler {
          action_type,
          existing: self.handlers[action_type].type_name(),
        })
      self.handlers.insert(action_type, handler.clone())
    Ok(())

  ASYNC FN dispatch(&self, envelope: Envelope<Action>) -> Result<(), BusError>
    // Validate envelope
    IF envelope.is_expired()
      self.metrics.expired.increment()
      RETURN Err(BusError::MessageExpired(envelope.id))

    // Route to handler
    LET action_type = envelope.payload.message_type()
    IF !self.handlers.contains_key(action_type)
      self.metrics.unhandled.increment()
      RETURN Err(BusError::NoHandler(action_type))

    // Backpressure: try_send for non-critical, send for critical+
    MATCH envelope.priority
      Priority::Critical | Priority::SystemPanic =>
        self.tx.send(envelope).await
          .map_err(|_| BusError::ChannelClosed)?
      _ =>
        self.tx.try_send(envelope)
          .map_err(|e| MATCH e {
            TrySendError::Full(_) => BusError::Backpressure,
            TrySendError::Closed(_) => BusError::ChannelClosed,
          })?

    self.metrics.dispatched.increment()
    Ok(())

  ASYNC FN run(&self)
    // Main dispatch loop — runs as a tokio task
    LOOP
      LET envelope = self.rx.lock().await.recv().await
      MATCH envelope
        Some(env) =>
          LET action_type = env.payload.message_type()
          LET handler = self.handlers[action_type].clone()

          // Spawn handler execution (don't block the dispatch loop)
          LET event_bus = self.event_bus.clone()
          LET metrics = self.metrics.clone()
          tokio::spawn(async move {
            LET start = Instant::now()
            MATCH handler.handle(env).await
              Ok(events) =>
                metrics.succeeded.increment()
                metrics.handle_time.record(start.elapsed())
                // Forward resulting events to event bus
                FOR event IN events
                  event_bus.emit(event).await
              Err(error) =>
                metrics.failed.increment()
                // Emit error event
                event_bus.emit(Event::AgentFailed {
                  agent_id: ActorId::system(),
                  error,
                  retryable: false,
                }).await
          })

        None => BREAK  // Channel closed, shutdown
```

### Backpressure Strategy

```
ENUM BackpressureStrategy
  DropOldest       // Discard oldest queued message (lossy)
  RejectNew        // Reject new message with error (default)
  BlockSender      // Block sender until space available
  OverflowToDisk   // Spill to WAL for later processing

// Default: RejectNew for Normal/Background, BlockSender for Critical+
// SystemPanic messages ALWAYS get through — they use a separate channel
```

---

## Module: `event_bus.rs`

### Design Rationale

Events are declarative notifications: "Intent was parsed", "Agent completed",
"Gate failed". Many subscribers can react to one event. This is where the
system becomes self-aware — components learn about each other without coupling.

Uses `tokio::sync::broadcast` — multi-producer, multi-consumer.

### EventSubscriber Trait

```
TRAIT EventSubscriber: Send + Sync + 'static
  // Name for tracing/debugging
  FN name(&self) -> &'static str

  // Which event types this subscriber cares about
  FN subscribes_to(&self) -> Vec<&'static str>

  // Minimum priority to receive (filter low-priority noise)
  FN min_priority(&self) -> Priority
    Priority::Background  // Default: receive everything

  // Handle an event
  ASYNC FN on_event(&self, envelope: Envelope<Event>) -> Result<(), SystemError>
```

### EventBus

```
STRUCT EventBus
  FIELDS:
    tx:          broadcast::Sender<Envelope<Event>>
    subscribers: RwLock<Vec<Arc<dyn EventSubscriber>>>
    capacity:    usize
    metrics:     EventBusMetrics

  CONST DEFAULT_CAPACITY: usize = 4096  // Larger than action bus — many subscribers

  FN new(capacity: usize) -> Self
    LET (tx, _) = broadcast::channel(capacity)
    Self {
      tx,
      subscribers: RwLock::new(Vec::new()),
      capacity,
      metrics: EventBusMetrics::default(),
    }

  FN subscribe(&self, subscriber: Arc<dyn EventSubscriber>)
    self.subscribers.write().push(subscriber)

  ASYNC FN emit(&self, event: Event) -> Result<(), BusError>
    LET envelope = Envelope::new(
      event,
      TraceId::current(),   // From tracing context
      ActorId::system(),
      CorrelationId::current(),
    )
    self.emit_envelope(envelope).await

  ASYNC FN emit_envelope(&self, envelope: Envelope<Event>) -> Result<(), BusError>
    LET event_type = envelope.payload.message_type()
    LET priority = envelope.priority.clone()

    // Broadcast to channel
    LET receiver_count = self.tx.send(envelope.clone())
      .map_err(|_| BusError::NoSubscribers)?

    self.metrics.emitted.increment()
    self.metrics.fanout.record(receiver_count)

    // Also dispatch to registered subscribers (filtered)
    LET subscribers = self.subscribers.read()
    FOR sub IN subscribers.iter()
      IF sub.min_priority() > priority
        CONTINUE  // Skip — below subscriber's priority threshold

      IF !sub.subscribes_to().contains(&event_type)
        && !sub.subscribes_to().contains(&"*")  // Wildcard subscriber
        CONTINUE  // Skip — not interested in this event type

      LET sub = sub.clone()
      LET env = envelope.clone()
      tokio::spawn(async move {
        IF LET Err(e) = sub.on_event(env).await
          tracing::warn!(
            subscriber = sub.name(),
            error = %e,
            "Subscriber failed to handle event"
          )
      })

    Ok(())

  FN receiver(&self) -> broadcast::Receiver<Envelope<Event>>
    self.tx.subscribe()
```

### Topic Filtering

```
// Subscribers declare interest via topic patterns:
//
// Exact:    "event.gate_passed"           — only gate passed events
// Prefix:   "event.agent_*"              — all agent events
// Wildcard: "*"                           — all events (e.g., tracing)
//
// The event bus matches topics using a simple prefix tree for O(1) lookup
// at scale. For small subscriber counts (<100), linear scan is fine.

STRUCT TopicMatcher
  exact:    HashSet<&'static str>
  prefixes: Vec<&'static str>
  wildcard: bool

  FN matches(&self, topic: &str) -> bool
    IF self.wildcard THEN RETURN true
    IF self.exact.contains(topic) THEN RETURN true
    self.prefixes.iter().any(|p| topic.starts_with(p))
```

---

## Module: `bus_error.rs`

```
ENUM BusError
  DuplicateHandler { action_type: &'static str, existing: String }
  NoHandler(&'static str)
  NoSubscribers
  MessageExpired(MessageId)
  Backpressure
  ChannelClosed
  SerializationError(String)

  IMPL Display, Error FOR BusError
```

---

## Metrics

```
STRUCT ActionBusMetrics
  dispatched: Counter
  succeeded:  Counter
  failed:     Counter
  expired:    Counter
  unhandled:  Counter
  handle_time: Histogram    // Duration distribution
  queue_depth: Gauge        // Current messages waiting

STRUCT EventBusMetrics
  emitted:     Counter
  fanout:      Histogram    // Subscribers reached per event
  dropped:     Counter      // Slow subscribers that lagged behind
  subscriber_errors: Counter
```

---

## TDD Anchors

```
TEST test_action_bus_dispatch_to_handler
  // Register a handler that handles "action.parse_intent"
  // Dispatch a ParseIntent action
  // Assert handler received the action
  LET (tx, rx) = oneshot::channel()
  LET handler = MockHandler::new(vec!["action.parse_intent"], tx)
  LET mut bus = ActionBus::new(16, event_bus)
  bus.register_handler(Arc::new(handler))

  LET action = Action::ParseIntent { raw_input: "test".into(), user_id: ActorId::system() }
  bus.dispatch(Envelope::new(action, trace, source, corr)).await.unwrap()

  LET received = rx.await.unwrap()
  ASSERT received.payload.message_type() == "action.parse_intent"

TEST test_action_bus_rejects_duplicate_handler
  LET handler1 = MockHandler::new(vec!["action.parse_intent"])
  LET handler2 = MockHandler::new(vec!["action.parse_intent"])
  LET mut bus = ActionBus::new(16, event_bus)
  bus.register_handler(Arc::new(handler1)).unwrap()
  ASSERT bus.register_handler(Arc::new(handler2)).is_err()

TEST test_action_bus_rejects_unhandled_action
  LET bus = ActionBus::new(16, event_bus)  // No handlers registered
  LET result = bus.dispatch(envelope).await
  ASSERT matches!(result, Err(BusError::NoHandler(_)))

TEST test_action_bus_backpressure
  LET bus = ActionBus::new(2, event_bus)  // Tiny capacity
  // Fill the channel
  bus.dispatch(envelope1).await.unwrap()
  bus.dispatch(envelope2).await.unwrap()
  // Third should get backpressure
  LET result = bus.dispatch(envelope3).await
  ASSERT matches!(result, Err(BusError::Backpressure))

TEST test_action_bus_critical_bypasses_backpressure
  LET bus = ActionBus::new(2, event_bus)
  bus.dispatch(envelope1).await.unwrap()
  bus.dispatch(envelope2).await.unwrap()
  // Critical messages block until space available (don't reject)
  LET critical = envelope3.with_priority(Priority::Critical)
  // This should eventually succeed (not return Backpressure)
  // Use timeout to prevent hanging
  LET result = timeout(Duration::from_secs(1), bus.dispatch(critical)).await
  // Result depends on whether consumer is running

TEST test_action_bus_rejects_expired_message
  LET env = Envelope::new(action, trace, source, corr)
    .with_ttl(Duration::from_millis(1))
  sleep(Duration::from_millis(10)).await
  LET result = bus.dispatch(env).await
  ASSERT matches!(result, Err(BusError::MessageExpired(_)))

TEST test_event_bus_fanout
  // Subscribe 3 subscribers to the same event type
  // Emit one event
  // Assert all 3 received it
  LET counters = [AtomicU32::new(0); 3]
  FOR i IN 0..3
    bus.subscribe(Arc::new(CountingSubscriber::new(
      "event.gate_passed",
      counters[i].clone(),
    )))

  bus.emit(Event::GatePassed { gate_id: GateId::Alpha4Fallback, score: 0.97, threshold: 0.95 }).await.unwrap()
  sleep(Duration::from_millis(50)).await  // Let spawned tasks complete

  FOR counter IN counters
    ASSERT counter.load(Ordering::SeqCst) == 1

TEST test_event_bus_topic_filtering
  // subscriber1 subscribes to "event.gate_*"
  // subscriber2 subscribes to "event.agent_*"
  // Emit GatePassed — only subscriber1 receives it
  // Emit AgentCompleted — only subscriber2 receives it

TEST test_event_bus_priority_filtering
  // subscriber with min_priority = Critical
  // Emit Normal priority event — subscriber does NOT receive it
  // Emit Critical priority event — subscriber DOES receive it

TEST test_event_bus_wildcard_subscriber
  // subscriber with subscribes_to = ["*"]
  // Emit any event — subscriber receives it
  // This is how the tracing hook works

TEST test_event_bus_subscriber_failure_isolation
  // One subscriber throws an error
  // Other subscribers still receive the event
  // Error is logged but does not propagate
```

## Edge Cases

- Action Bus with zero capacity must reject immediately
- Event Bus broadcast lag (slow subscriber) drops old events, not new ones
- Handler that panics must not crash the dispatch loop (catch_unwind)
- Empty handler list for an action type = BusError::NoHandler
- Event with no subscribers = BusError::NoSubscribers (log warning, don't fail)
- Concurrent register_handler during dispatch must not deadlock
- Shutdown: drain remaining messages before closing channels
