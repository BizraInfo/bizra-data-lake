# Phase 68.01 — Python Action Bus (CQRS Command Side)

## Context

The Rust ActionBus (`bizra-agent/src/action_bus.rs`) is complete with permit-
guarded dispatch and receipt chains. The Python side has MissionOrchestrator
which calls ChannelDispatcher directly — no CQRS separation. This spec adds
a Python ActionBus that mirrors the Rust architecture.

---

## 1. Requirements

### FR-1: Two-Phase Execution
Every action goes through: `Propose -> Gate -> Execute -> Verify -> Receipt`

### FR-2: Idempotency
Same action_id retried must not double-mint or double-execute.
Implementation: `_executed: set[str]` tracks completed action IDs.

### FR-3: Capability Masks
TeleScript restrictions travel with the action. No action executes without
a capability check.

### FR-4: Cancel + Rollback
First-class messages, not afterthoughts.

### FR-5: Event Emission
Every state transition emits an event via EventBus.

---

## 2. Data Types

```python
# core/bus/types.py

@dataclass(frozen=True)
class ActionEnvelope:
    """Immutable action proposal."""
    action_id: str              # blake3(content) hex
    kind: str                   # "hda.file.organize.v1"
    channel: str                # "desktop" | "browser" | "file" | "llm"
    payload: dict               # channel-specific data
    capabilities: tuple[str, ...]  # ("file_read", "file_write")
    telescript: dict             # {"allow_paths": [...], "deny_paths": [...]}
    budget: ActionBudget        # time + token limits
    correlation_id: str         # mission ID linkage
    actor_id: bytes             # ed25519 public key
    timestamp: int              # unix ms

@dataclass(frozen=True)
class ActionBudget:
    """Resource limits for a single action."""
    time_ms: int = 5000         # max wall-clock time
    s2_tokens_max: int = 1200   # max LLM tokens (System-2)
    retry_max: int = 2          # max retry attempts

class ActionStatus(Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    DENIED = "denied"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ROLLED_BACK = "rolled_back"

@dataclass(frozen=True)
class ActionReceipt:
    """Immutable proof that an action executed."""
    action_id: str
    status: ActionStatus
    outcome_hash: bytes         # blake3 of post-state
    ihsan_score: int            # fixed-point
    guardian_approved: bool
    timestamp: int
    prev_receipt_hash: bytes    # merkle chain link
    receipt_hash: bytes         # blake3(canonical form)
```

---

## 3. ActionBus Class — Pseudocode

```
CLASS ActionBus:
    INIT(event_bus, telescript, fate_gate, channels):
        self.event_bus = event_bus
        self.telescript = telescript     # TeleScriptEngine
        self.fate_gate = fate_gate       # FATE evaluator
        self.channels = channels         # dict[str, ChannelExecutor]
        self._executed = set()           # idempotency guard
        self._receipt_chain = []         # merkle chain
        self._pending = asyncio.Queue(maxsize=256)  # backpressure

    ASYNC propose(action: ActionEnvelope) -> ActionReceipt:
        """Full lifecycle: propose -> gate -> execute -> verify -> receipt."""

        # Step 0: Idempotency check
        IF action.action_id IN self._executed:
            RETURN self._find_receipt(action.action_id)

        # Step 1: TeleScript capability check
        allowed = self.telescript.check(action.capabilities, action.telescript)
        IF NOT allowed:
            receipt = self._make_receipt(action, DENIED, reason="telescript_denied")
            AWAIT self.event_bus.publish("policy.telescript.denied", {
                "action_id": action.action_id,
                "kind": action.kind,
                "capabilities": action.capabilities,
            })
            RETURN receipt

        # Step 2: FATE gate evaluation
        fate_result = self.fate_gate.evaluate(action)
        IF fate_result.denied:
            receipt = self._make_receipt(action, DENIED, reason=fate_result.reason)
            AWAIT self.event_bus.publish("policy.fate.vetoed", {
                "action_id": action.action_id,
                "reason_codes": fate_result.reason_codes,
            })
            RETURN receipt

        # Step 3: Emit intent event
        AWAIT self.event_bus.publish("action.intent", {
            "action_id": action.action_id,
            "kind": action.kind,
            "channel": action.channel,
        })

        # Step 4: Execute via channel
        channel = self.channels.get(action.channel)
        IF channel IS None:
            RETURN self._make_receipt(action, FAILED, reason="no_channel")

        TRY:
            result = AWAIT asyncio.wait_for(
                channel.execute(action),
                timeout=action.budget.time_ms / 1000.0
            )
        EXCEPT asyncio.TimeoutError:
            result = ChannelResult(success=False, reason="timeout")
        EXCEPT Exception as e:
            result = ChannelResult(success=False, reason=str(e))

        # Step 5: Build receipt
        status = COMPLETED IF result.success ELSE FAILED
        receipt = self._make_receipt(
            action, status,
            outcome_hash=result.outcome_hash,
            ihsan_score=result.ihsan_score,
        )

        # Step 6: Mark executed (idempotency)
        self._executed.add(action.action_id)

        # Step 7: Emit receipt event
        topic = "action.receipt" IF result.success ELSE "action.receipt.failed"
        AWAIT self.event_bus.publish(topic, receipt.to_dict())

        RETURN receipt

    ASYNC cancel(action_id: str) -> ActionReceipt:
        """Cancel a pending action."""
        IF action_id IN self._executed:
            RAISE AlreadyExecuted(action_id)
        receipt = self._make_receipt_by_id(action_id, CANCELLED)
        AWAIT self.event_bus.publish("action.cancelled", {"action_id": action_id})
        RETURN receipt

    DEF _make_receipt(action, status, **kwargs) -> ActionReceipt:
        """Build receipt with merkle chain link."""
        prev_hash = self._receipt_chain[-1].receipt_hash IF self._receipt_chain ELSE b"\x00" * 32
        canonical = json.dumps({
            "action_id": action.action_id,
            "status": status.value,
            "outcome_hash": kwargs.get("outcome_hash", b"").hex(),
            "prev": prev_hash.hex(),
        }, sort_keys=True, separators=(",", ":")).encode()
        receipt_hash = hashlib.blake2b(canonical, digest_size=32).digest()

        receipt = ActionReceipt(
            action_id=action.action_id,
            status=status,
            outcome_hash=kwargs.get("outcome_hash", b""),
            ihsan_score=kwargs.get("ihsan_score", FP_ZERO),
            guardian_approved=(status != DENIED),
            timestamp=int(time.time() * 1000),
            prev_receipt_hash=prev_hash,
            receipt_hash=receipt_hash,
        )
        self._receipt_chain.append(receipt)
        RETURN receipt
```

---

## 4. Channel Executor Protocol

```python
# core/bus/channels.py

class ChannelExecutor(Protocol):
    """Any channel that can execute actions."""

    async def execute(self, action: ActionEnvelope) -> ChannelResult: ...
    def supports(self, kind: str) -> bool: ...

@dataclass
class ChannelResult:
    success: bool
    outcome_hash: bytes = b""
    ihsan_score: int = 0        # fixed-point
    artifacts: list[str] = field(default_factory=list)
    reason: str = ""
```

### Built-in Channels

| Channel | Kind Prefix | Implementation |
|---------|-------------|----------------|
| `desktop` | `hda.*` | HDAClient (AHK JSON-RPC, port 9743) |
| `file` | `file.*` | Sandboxed file ops (pathlib + allow_paths) |
| `browser` | `web.*` | BrowserMCPClient (existing) |
| `llm` | `llm.*` | InferenceGateway (existing tiered fallback) |
| `proof` | `proof.*` | Screenshot/diff verifier |

---

## 5. Integration with MissionOrchestrator

```
CURRENT (Phase 57):
  MissionOrchestrator -> ChannelDispatcher -> [channels directly]

PHASE 68:
  MissionOrchestrator -> ActionBus.propose() -> TeleScript -> FATE -> Channel
                                              -> EventBus.emit()
                                              -> Receipt chain
```

The MissionOrchestrator's `_execute_channel()` method wraps each subtask
in an ActionEnvelope and routes through ActionBus instead of calling
channels directly. This adds:
- Capability gating (TeleScript)
- Constitutional veto (FATE)
- Receipt chain (merkle-linked proof)
- Event emission (for subscribers)
- Idempotency (replay-safe)

---

## 6. TDD Anchors (16 tests)

```python
class TestActionBusPropose:
    def test_propose_emits_intent_event()
    def test_propose_returns_receipt_on_success()
    def test_propose_denied_by_telescript()
    def test_propose_denied_by_fate_gate()
    def test_propose_timeout_returns_failed()
    def test_propose_channel_not_found()

class TestIdempotency:
    def test_duplicate_action_returns_same_receipt()
    def test_duplicate_action_does_not_re_execute()

class TestReceiptChain:
    def test_receipt_chain_links_prev_hash()
    def test_receipt_chain_genesis_has_zero_prev()
    def test_receipt_hash_deterministic()

class TestCancel:
    def test_cancel_pending_action()
    def test_cancel_already_executed_raises()

class TestBackpressure:
    def test_queue_maxsize_blocks_when_full()

class TestEventEmission:
    def test_success_emits_action_receipt()
    def test_failure_emits_action_receipt_failed()
    def test_deny_emits_policy_event()
```

---

## 7. Non-Goals

- **No persistence in ActionBus itself.** The EventLog (A14) is the durable
  store. ActionBus is in-memory with receipts emitted to EventBus.
- **No distributed ActionBus.** Node0 is single-process. Federation uses
  gossip to share receipts, not actions.
- **No replacing Rust ActionBus.** Python ActionBus mirrors Rust for the
  Python runtime. When PyO3 bridge is active, can delegate to Rust.
