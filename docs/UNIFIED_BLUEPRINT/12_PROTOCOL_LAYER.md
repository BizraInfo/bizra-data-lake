# Module 12 — Protocol Layer

> **Domain:** UMB, wire formats, transport, IPC, A2A protocol
> **Source Specs:** SAP v0 (wire mapping), Phase 48 (Rust workspace), UCF
> **Key Paths:** `core/a2a/`, `core/protocols/`, `bizra-omega/iceoryx-bridge/`

## 12.1 Agent-to-Agent Protocol (A2A)

**Status:** [x] BUILT
**Path:** `core/a2a/`

Direct agent communication. Discovery, negotiation, and structured
message exchange between agents within a node.

---

## 12.2 Protocol Interface Contracts

**Status:** [x] BUILT
**Path:** `core/protocols/`

Structural typing via Python Protocol classes. Defines interface contracts
that modules must satisfy without inheritance coupling.

---

## 12.3 UCF EventBus (Unified Concurrency Fabric)

**Status:** [x] BUILT
**Paths:**
- Python: `core/sovereign/event_bus.py`
- Rust: `bizra-omega/bizra-hooks/src/event_bus.rs` (8 shards, FNV-1a)
- Bridge: `PyEventBridge` (Rust) -> `RustEventBridge` (Python wrapper)

8 namespace shards for O(N/8) dispatch. 12 canonical subscriber topics.
Graceful fallback: returns None when PyO3 not built.

---

## 12.4 Zero-Copy IPC (iceoryx2)

**Status:** [~] PARTIAL
**Path:** `bizra-omega/iceoryx-bridge/`
**Built:** Rust crate with iceoryx2 bindings
**Gap:** No Python consumer, no integration with sovereign runtime

### TDD Anchor
```
#[test]
fn test_iceoryx_publish_subscribe() {
    let publisher = IceoryxPublisher::new("bizra/events");
    let subscriber = IceoryxSubscriber::new("bizra/events");
    publisher.send(b"test_event");
    let msg = subscriber.recv_timeout(Duration::from_secs(1));
    assert_eq!(msg.unwrap(), b"test_event");
}
```

---

## 12.5 Action Bus (Event->Action->Receipt)

**Status:** [~] PARTIAL
**Path:** `bizra-omega/bizra-action/`
**Built:** Rust crate with Event->Action->Receipt pipeline
**Gap:** Limited Python integration, no persistent action log

---

## 12.6 Unified Message Bus (UMB)

**Status:** [ ] NOT BUILT
**Spec:** Single message bus unifying all transport (EventBus, Redis, IPC, network)
**Gap:** Currently 3+ separate transports (EventBus, Redis pub/sub, HTTP). No unified bus.

### Pseudocode
```
class UnifiedMessageBus:
    """Single bus abstracting all transport layers"""

    def __init__(self):
        self.transports = {
            "local": EventBusTransport(),        # In-process
            "ipc": IceoryxTransport(),            # Same machine, zero-copy
            "cluster": RedisTransport(),          # Same cluster
            "federation": GossipTransport(),      # Cross-network
        }

    def publish(self, topic: str, message: Message, scope: str = "auto"):
        transport = self._select_transport(scope, topic)
        transport.publish(topic, message.serialize())

    def subscribe(self, topic: str, handler: Callable, scope: str = "auto"):
        transport = self._select_transport(scope, topic)
        transport.subscribe(topic, lambda data: handler(Message.deserialize(data)))

    def _select_transport(self, scope, topic):
        if scope == "auto":
            return self._infer_optimal_transport(topic)
        return self.transports[scope]
```

---

## 12.7 Wire Format (SAP v0)

**Status:** [ ] NOT BUILT
**Spec:** `specs/sap-v0/README.md` — wire mapping for 9 canonical types
**Gap:** No serialization format defined. No protobuf/msgpack/CBOR schemas.

### Pseudocode
```
# Wire format using CBOR for compact binary + JSON for debug
class SAPWireFormat:
    VERSION = 1

    @staticmethod
    def encode(message: SAPMessage) -> bytes:
        envelope = {
            "version": SAPWireFormat.VERSION,
            "type": message.type_name,
            "payload": message.to_dict(),
            "signature": message.sign(),
            "timestamp": now_utc_iso(),
        }
        return cbor2.dumps(envelope)

    @staticmethod
    def decode(data: bytes) -> SAPMessage:
        envelope = cbor2.loads(data)
        assert envelope["version"] == SAPWireFormat.VERSION
        msg = SAP_TYPES[envelope["type"]].from_dict(envelope["payload"])
        assert msg.verify(envelope["signature"])
        return msg
```

---

## 12.8 WebSocket Real-Time Channel

**Status:** [ ] NOT BUILT
**Spec:** Required for live frontend updates and agent-to-UI streaming
**Gap:** No WebSocket server, no real-time push to frontend

### Pseudocode
```
# FastAPI WebSocket endpoint
@app.websocket("/ws/sovereign")
async def sovereign_websocket(websocket: WebSocket):
    await websocket.accept()
    # Subscribe to EventBus topics relevant to this client
    topics = ["mission.status", "agent.update", "sovereignty.score"]
    async for event in event_bus.subscribe_many(topics):
        await websocket.send_json({
            "topic": event.topic,
            "data": event.payload,
            "timestamp": event.timestamp,
        })
```

---

## Completion

| Feature | Status | Coverage |
|---------|--------|----------|
| 12.1 A2A Protocol | BUILT | Full |
| 12.2 Protocol Contracts | BUILT | Typing |
| 12.3 UCF EventBus | BUILT | 8 shards |
| 12.4 iceoryx IPC | PARTIAL | Rust only |
| 12.5 Action Bus | PARTIAL | No Python |
| 12.6 UMB | NOT BUILT | Zero |
| 12.7 Wire Format | NOT BUILT | Zero |
| 12.8 WebSocket | NOT BUILT | Zero |
| **TOTAL** | **3/8 + 2P + 3N** | **50%** |
