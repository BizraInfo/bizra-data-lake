# Phase 69.02 — Wiring Sprint: Sprints 3-6 (ActionBus, Security, Omega, Capsules)

Continues from `phase_69_01_wiring_sprint.md` (Sprints 1-2).

---

## Sprint 3: ActionBus (~300 LOC, 16 tests)

### 3.1 `core/bus/action_bus.py`

```python
"""ActionBus — CQRS Command Pipeline with Constitutional Gates."""

from __future__ import annotations
import hashlib
import json
import time
from typing import Protocol

from core.bus.telescript import TeleScriptEngine
from core.bus.topics import TopicRegistry
from core.bus.types import ActionEnvelope, ActionStatus, BusActionReceipt
from core.sovereign.event_bus import EventBus

class ChannelExecutor(Protocol):
    """Contract for channel implementations."""
    async def execute(self, action: ActionEnvelope) -> dict: ...

class ActionBus:
    INIT(event_bus, telescript_engine, fate_gate, channels, topic_registry):
        self._event_bus = event_bus
        self._telescript = telescript_engine
        self._fate = fate_gate
        self._channels: dict[str, ChannelExecutor] = channels
        self._topics = topic_registry
        self._seen_ids: set[str] = set()
        self._receipt_chain: str = "genesis"

    ASYNC propose(action: ActionEnvelope) -> BusActionReceipt:
        """7-step CQRS lifecycle."""

        # Step 1: Idempotency check
        IF action.action_id IN self._seen_ids:
            RETURN self._get_cached_receipt(action.action_id)
        self._seen_ids.add(action.action_id)

        # Step 2: TeleScript capability check
        verdict = self._telescript.check(
            requested=action.capabilities,
            action_telescript=action.telescript,
            file_path=action.payload.get("path"),
        )
        IF NOT verdict.allowed:
            self._event_bus.emit("policy.telescript.denied", {
                "action_id": action.action_id,
                "reason": verdict.reason,
            })
            RETURN self._deny_receipt(action, verdict.reason)

        # Step 3: FATE gate evaluation
        fate_result = self._fate.evaluate(action)
        IF NOT fate_result.allowed:
            self._event_bus.emit("policy.fate.vetoed", {
                "action_id": action.action_id,
            })
            RETURN self._deny_receipt(action, "FATE gate denied")

        # Step 4: Emit intent event
        self._event_bus.emit("action.intent", {
            "action_id": action.action_id,
            "kind": action.kind,
            "channel": action.channel,
        })

        # Step 5: Channel execution
        channel = self._channels.get(action.channel)
        IF channel IS None:
            RETURN self._fail_receipt(action, f"Unknown channel: {action.channel}")

        TRY:
            start = time.monotonic_ns()
            outcome = AWAIT channel.execute(action)
            duration_ms = (time.monotonic_ns() - start) / 1_000_000
        EXCEPT Exception:
            RETURN self._fail_receipt(action, "Channel execution failed")

        # Step 6: Build receipt (merkle-chained)
        receipt = self._build_receipt(action, outcome, duration_ms)

        # Step 7: Emit receipt event
        self._event_bus.emit("action.receipt", {
            "receipt_id": receipt.receipt_id,
            "action_id": action.action_id,
            "status": receipt.status.value,
        })

        RETURN receipt

    DEF _build_receipt(action, outcome, duration_ms) -> BusActionReceipt:
        outcome_hash = hashlib.blake2b(
            json.dumps(outcome, sort_keys=True).encode()
        ).hexdigest()

        receipt_content = f"{action.action_id}:{outcome_hash}:{self._receipt_chain}"
        receipt_id = hashlib.blake2b(receipt_content.encode()).hexdigest()

        receipt = BusActionReceipt(
            receipt_id=receipt_id,
            action_id=action.action_id,
            status=ActionStatus.COMPLETED,
            outcome_hash=outcome_hash,
            ihsan_score=outcome.get("ihsan", 0.0),
            prev_receipt_hash=self._receipt_chain,
            guardian_verdict="allowed",
            duration_ms=duration_ms,
        )

        self._receipt_chain = receipt_id
        RETURN receipt
```

### 3.2 Integration: Wire MissionOrchestrator

```python
# core/sovereign/mission.py — PATCH
# Replace direct channel calls with ActionBus.propose()

# OLD:
result = await self.channel_dispatcher.dispatch(channel, task)

# NEW:
action = ActionEnvelope(
    action_id=blake3(f"{mission.mission_id}:{channel}:{task}"),
    kind=f"mission.{channel}",
    channel=channel,
    payload=task,
    capabilities=self._channel_capabilities(channel),
    telescript={},
    budget=ActionBudget(),
    correlation_id=mission.mission_id,
    actor_id=self._signer.public_key,
    timestamp=int(time.time() * 1000),
)
receipt = await self.action_bus.propose(action)
```

### 3.3 TDD Anchors (16 tests)

See spec 68.01 for the full 16-test contract.

---

## Sprint 4: Security Patches (~180 LOC, parallel with Sprint 2)

**Agent audit corrections:** WebSocket auth and 127.0.0.1 binding already
exist in both bridges. The real security gaps are: hardcoded credentials,
unauthenticated episodic memory, exception leakage, and open telemetry routes.

### 4.1 Remove Hardcoded Credentials [HIGH]

```python
# golden_gems/algebraic_effects.py — PATCH

# OLD (line 104-107):
self.valid_tokens = valid_tokens or {
    "bizra_secret_123": ["admin", "user"],
    "user_token_456": ["user"],
}

# NEW:
if not valid_tokens:
    raise ValueError("valid_tokens must be provided — no default credentials")
self.valid_tokens = valid_tokens
```

### 4.2 Auth-Gate SEL Episodes + Telemetry [HIGH]

```python
# core/sovereign/api.py — PATCH

# Add auth to these unauthenticated routes:
# /v1/sel/episodes       — exposes user query history (PII)
# /v1/sel/episodes/{hash} — exposes specific episodes
# /v1/spearpoint/stats   — leaks mission history
# /v1/judgment/stats     — leaks verdict telemetry
# /v1/judgment/stability — leaks entropy + dominant verdict
# /v1/suggestions        — leaks living memory proactive data
# /v1/token/balance      — exposes account balances

# Pattern: Add request param + auth check
async def sel_episodes(request: Request):
    _authenticate_http_request(request)
    # ... existing handler ...
```

### 4.3 API Exception Sanitization (39 locations) [HIGH]

```python
# core/sovereign/api.py — PATCH PATTERN (apply to all 39 locations)

# OLD:
except Exception as e:
    return JSONResponse({"error": str(e)}, status_code=500)

# NEW:
except Exception:
    logger.exception("Internal error in <endpoint_name>")
    return JSONResponse({"error": "Internal server error"}, status_code=500)
```

### 4.4 Bridge Token Startup Validation [MEDIUM]

```javascript
// filedfs/bizra-bridge.mjs — PATCH (add at startup)
if (!process.env.BIZRA_BRIDGE_TOKEN) {
    console.error('FATAL: BIZRA_BRIDGE_TOKEN not set — refusing to start');
    process.exit(1);
}
```

### 4.5 TDD Anchors (8 tests)

```python
class TestSecurityPatches:
    def test_hardcoded_credentials_removed()
    def test_auth_handler_requires_tokens()
    def test_sel_episodes_requires_auth()
    def test_sel_episodes_hash_requires_auth()
    def test_spearpoint_stats_requires_auth()
    def test_judgment_stats_requires_auth()
    def test_api_error_no_internal_details()
    def test_api_error_logs_full_traceback()
```

---

## Sprint 5: OmegaLoop + Config (~750 LOC, 26 tests)

### 5.1 `core/bus/omega_loop.py` (~400 LOC)

Full implementation from spec 68.02:
- OmegaLoopState, OmegaStatus, LoopBudget, ProofCondition
- `OmegaLoop.run()` — proof-based iteration
- Budget enforcement (300s time, 50K tokens, 100 actions)
- EventLog persistence for resumability
- 14 TDD anchors

### 5.2 `core/config/loader.py` (~350 LOC)

Full implementation from spec 68.03:
- ConfigLoader with 3-scope YAML merge
- SSoT validation (config >= constants.py thresholds)
- Ed25519 signature verification for federation configs
- 12 TDD anchors

---

## Sprint 6: Capsule Runtime (~300 LOC, 10 tests)

### 6.1 `core/bus/capsule_runtime.py`

Full implementation from spec 68.04:
- CapsuleRegistry auto-discovery
- CapsuleRuntime workflow execution via ActionBus
- Variable resolution ($step.result)
- Proof condition checking
- 10 TDD anchors

---

## Total Effort Summary

| Sprint | LOC | Tests | Dependencies |
|--------|-----|-------|-------------|
| 1: Asabiyyah-Gini | ~100 | 12 | None (can start immediately) |
| 2: Bus Foundation | ~450 | 34 | Sprint 1 (for economy.asabiyyah topic) |
| 3: ActionBus | ~300 | 16 | Sprint 2 (needs types + telescript + topics) |
| 4: Security | ~180 | 8 | None (parallel with any sprint) |
| 5: OmegaLoop + Config | ~750 | 26 | Sprint 3 (needs ActionBus) |
| 6: Capsule Runtime | ~300 | 10 | Sprint 3 (needs ActionBus) |
| **Total** | **~2,080** | **106** | |

Combined with existing Phase 67-68 TDD anchors: **162 + 103 = 265 total tests**
for the sovereign synthesis.

---

## Implementation Order (Critical Path)

```
Week 1:  Sprint 1 (Asabiyyah) ─────────────────── Sprint 4 (Security)
             │                                        │
             v                                        v
Week 2:  Sprint 2 (Bus Foundation) ──────────────── [patches merged]
             │
             v
Week 3:  Sprint 3 (ActionBus)
             │
             ├──────────────┐
             v              v
Week 4:  Sprint 5 (Omega)  Sprint 6 (Capsules)
```

Sprint 1 and Sprint 4 can run in parallel.
Sprint 5 and Sprint 6 can run in parallel after Sprint 3.

---

## Non-Goals for Phase 69

- **No AKIS pipeline** — standalone, lower SNR, defer to Phase 70
- **No federation wiring** — requires bus architecture first
- **No Rust cross-runtime sync** — `topics.json` export is prepared but
  Rust validation is Phase 70 work
- **No v3-memory HNSW** — separate experimental track
- **No Reverse Scale Hypothesis testing** — requires multi-node federation
