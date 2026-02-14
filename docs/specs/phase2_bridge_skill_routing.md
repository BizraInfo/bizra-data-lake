# Phase 2: Desktop Bridge Skill Routing + PCI Receipts

> **Status:** Specification
> **Depends on:** Phase 1 (Desktop Bridge — 3-command surface, COMPLETE)
> **Files affected:** `core/bridges/desktop_bridge.py`, new: `core/bridges/bridge_receipt.py`

---

## 1. Problem Statement

Phase 1 proved the bridge works (102 tests, Rust integration, AHK client). But it only
exposes 3 hardcoded methods: `ping`, `status`, `sovereign_query`. The Sovereign Runtime has
39 skills and 35+ slash commands — none reachable from the desktop.

**Goal:** Route any skill invocation through the bridge, gated by FATE + Rust GateChain,
producing a signed PCI receipt for every execution.

---

## 2. Functional Requirements

### FR-1: `invoke_skill` JSON-RPC method

```
method: "invoke_skill"
params: {
  "skill": "deep-research",
  "inputs": {"topic": "quantum error correction"},
  "ihsan_score": 0.97
}
```

Returns `SkillInvocationResult.to_dict()` from `core/skills/router.py`.

### FR-2: `list_skills` JSON-RPC method

```
method: "list_skills"
params: {"filter": "available"}  // optional
```

Returns list of `{name, description, status, agent, tags}` from `SkillRegistry`.

### FR-3: PCI receipt for every bridge command

Every command (including existing `ping`, `status`, `sovereign_query`) produces a signed
`Receipt` from `core/proof_engine/receipt.py`. The receipt is:
- Included in the JSON-RPC response under `"receipt"` key
- Written to `sovereign_state/bridge_receipts/` as `.json` files
- Signed with the node's `SimpleSigner` (dev) or Ed25519 (production)

### FR-4: `get_receipt` JSON-RPC method

```
method: "get_receipt"
params: {"receipt_id": "r-1234567890"}
```

Returns the full receipt for a previous command.

### FR-5: AHK chord expansion

Add to `bin/bizra_bridge.ahk`:
- `Ctrl+B, I` → `invoke_skill` (input dialog: skill name + optional args)
- `Ctrl+B, L` → `list_skills` (tooltip: top 10 available skills)

---

## 3. Non-Functional Requirements

| Requirement | Target | Rationale |
|-------------|--------|-----------|
| Latency overhead | < 5ms per receipt | Receipt generation must not slow bridge response |
| Receipt storage | < 1 KB per receipt | BLAKE3 digests + compact JSON |
| Skill list response | < 50ms | Registry lookup is in-memory |
| Backward compatibility | 100% | Existing `ping`/`status`/`sovereign_query` unchanged |
| Test count | +20 new tests | Cover new methods + receipt generation |

---

## 4. Constraints

- **No new Python dependencies.** Receipt signing uses existing `core/proof_engine/receipt.py`.
- **SkillRouter is lazy-loaded.** Do not import at module level (same pattern as InferenceGateway).
- **Receipts are append-only.** Never delete or modify a receipt after creation.
- **Receipt signing key from env.** `BIZRA_RECEIPT_PRIVATE_KEY_HEX` or fallback `SimpleSigner`.
- **Max 500 lines per file.** `bridge_receipt.py` handles receipt logic, `desktop_bridge.py` stays as router.

---

## 5. Edge Cases

| Case | Behavior |
|------|----------|
| Unknown skill name | Return error with code `-32602`, receipt status `REJECTED` |
| Skill suspended | Return error with code `-32603`, receipt status `AMBER_RESTRICTED` |
| SkillRegistry import fails | Degrade gracefully, `invoke_skill`/`list_skills` return import error |
| Receipt storage dir missing | Auto-create `sovereign_state/bridge_receipts/` |
| Receipt signing fails | Log warning, return response without `receipt` field (non-fatal) |
| Ihsan score below threshold | SkillRouter rejects, receipt status `REJECTED`, reason in receipt |
| Concurrent receipt writes | Use atomic write (write to `.tmp`, rename) |

---

## 6. Pseudocode

### 6.1 Bridge Receipt Module (`core/bridges/bridge_receipt.py`)

```
MODULE bridge_receipt

IMPORT Receipt, ReceiptStatus, SimpleSigner, Metrics FROM core.proof_engine.receipt
IMPORT blake3_digest, canonical_bytes FROM core.proof_engine.canonical
IMPORT UNIFIED_IHSAN_THRESHOLD FROM core.integration.constants

CONSTANT RECEIPT_DIR = Path("sovereign_state/bridge_receipts")

FUNCTION load_signer() -> SimpleSigner:
    # TDD Anchor: test_load_signer_from_env
    # TDD Anchor: test_load_signer_fallback
    key_hex = env("BIZRA_RECEIPT_PRIVATE_KEY_HEX", default="")
    IF key_hex:
        RETURN SimpleSigner(bytes.fromhex(key_hex))
    RETURN SimpleSigner(b"bizra-bridge-node0-dev-key")

CLASS BridgeReceiptEngine:
    INIT(signer=None, receipt_dir=RECEIPT_DIR):
        self.signer = signer OR load_signer()
        self.receipt_dir = receipt_dir
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        self._cache = {}  # receipt_id -> Receipt (LRU, max 100)

    FUNCTION create_receipt(
        method: str,
        query_data: dict,
        result_data: dict,
        fate_score: float,
        snr_score: float,
        gate_passed: str,
        status: ReceiptStatus,
        duration_ms: float,
        reason: str | None = None,
    ) -> Receipt:
        # TDD Anchor: test_create_accepted_receipt
        # TDD Anchor: test_create_rejected_receipt
        # TDD Anchor: test_receipt_has_valid_signature

        receipt_id = f"br-{int(time.time() * 1000)}-{method}"

        query_digest = blake3_digest(canonical_bytes(query_data))
        policy_digest = blake3_digest(b"bizra-bridge-v1")
        payload_digest = blake3_digest(canonical_bytes(result_data))

        receipt = Receipt(
            receipt_id=receipt_id,
            status=status,
            query_digest=query_digest,
            policy_digest=policy_digest,
            payload_digest=payload_digest,
            snr=snr_score,
            ihsan_score=fate_score,
            gate_passed=gate_passed,
            reason=reason,
            metrics=Metrics(duration_ms=duration_ms),
        )
        receipt.sign_with(self.signer)

        # Persist (non-blocking)
        self._persist(receipt)
        self._cache[receipt_id] = receipt

        RETURN receipt

    FUNCTION _persist(receipt: Receipt):
        # TDD Anchor: test_receipt_persisted_to_disk
        # TDD Anchor: test_atomic_write
        path = self.receipt_dir / f"{receipt.receipt_id}.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(receipt.to_dict(), indent=2))
        tmp.rename(path)

    FUNCTION get_receipt(receipt_id: str) -> Receipt | None:
        # TDD Anchor: test_get_receipt_from_cache
        # TDD Anchor: test_get_receipt_from_disk
        # TDD Anchor: test_get_receipt_not_found
        IF receipt_id IN self._cache:
            RETURN self._cache[receipt_id]
        path = self.receipt_dir / f"{receipt_id}.json"
        IF path.exists():
            RETURN deserialize(path.read_text())
        RETURN None
```

### 6.2 Skill Routing in Desktop Bridge

```
IN DesktopBridge._dispatch():

    # Extend handler map
    handlers = {
        "ping":             self._handle_ping,
        "status":           self._handle_status,
        "sovereign_query":  self._handle_sovereign_query,
        "invoke_skill":     self._handle_invoke_skill,     # NEW
        "list_skills":      self._handle_list_skills,       # NEW
        "get_receipt":      self._handle_get_receipt,        # NEW
    }

IN DesktopBridge.__init__():
    self._receipt_engine = None   # BridgeReceiptEngine (lazy)
    self._skill_router = None     # SkillRouter (lazy)

FUNCTION _get_receipt_engine() -> BridgeReceiptEngine | None:
    # TDD Anchor: test_receipt_engine_lazy_load
    IF self._receipt_engine IS None:
        TRY:
            FROM core.bridges.bridge_receipt IMPORT BridgeReceiptEngine
            self._receipt_engine = BridgeReceiptEngine()
        EXCEPT ImportError:
            PASS
    RETURN self._receipt_engine

FUNCTION _get_skill_router() -> SkillRouter | None:
    # TDD Anchor: test_skill_router_lazy_load
    IF self._skill_router IS None:
        TRY:
            FROM core.skills.router IMPORT SkillRouter
            self._skill_router = SkillRouter()
        EXCEPT ImportError:
            PASS
    RETURN self._skill_router
```

### 6.3 `invoke_skill` Handler

```
ASYNC FUNCTION _handle_invoke_skill(params: dict) -> dict:
    # TDD Anchor: test_invoke_skill_success
    # TDD Anchor: test_invoke_skill_unknown
    # TDD Anchor: test_invoke_skill_ihsan_blocked
    # TDD Anchor: test_invoke_skill_produces_receipt

    VALIDATE params has "skill" (string, non-empty)
    skill_name = params["skill"]
    inputs = params.get("inputs", {})
    ihsan = params.get("ihsan_score", 1.0)

    # FATE gate
    fate_result = self._validate_fate(f"invoke_skill:{skill_name}")
    IF NOT fate_result.get("passed"):
        receipt = self._emit_receipt("invoke_skill", params, fate_result,
                                     ReceiptStatus.REJECTED, "FATE gate")
        RETURN {"error": "FATE blocked", "fate": fate_result, "receipt": receipt}

    # Rust gate
    rust_gates = self._validate_rust_gates(skill_name)
    IF rust_gates AND NOT rust_gates.get("passed"):
        receipt = self._emit_receipt("invoke_skill", params, rust_gates,
                                     ReceiptStatus.REJECTED, "Rust GateChain")
        RETURN {"error": "Rust gate blocked", "rust_gates": rust_gates, "receipt": receipt}

    # Route to SkillRouter
    router = self._get_skill_router()
    IF router IS None:
        RETURN {"error": "SkillRouter not available"}

    start = time.monotonic()
    result = AWAIT router.invoke(skill_name, inputs, ihsan_score=ihsan)
    duration = (time.monotonic() - start) * 1000

    # Receipt
    status = ReceiptStatus.ACCEPTED IF result.success ELSE ReceiptStatus.REJECTED
    receipt = self._emit_receipt("invoke_skill", params, result.to_dict(),
                                 status, result.skill_name, duration)

    response = result.to_dict()
    response["receipt"] = receipt
    RETURN response
```

### 6.4 `list_skills` Handler

```
ASYNC FUNCTION _handle_list_skills(params: dict) -> dict:
    # TDD Anchor: test_list_skills_returns_skills
    # TDD Anchor: test_list_skills_filter
    # TDD Anchor: test_list_skills_no_registry

    router = self._get_skill_router()
    IF router IS None:
        RETURN {"error": "SkillRegistry not available", "skills": []}

    filter_status = params.get("filter", None)
    skills = router.registry.list_skills(filter_status)

    RETURN {
        "skills": [
            {
                "name": s.manifest.name,
                "description": s.manifest.description,
                "status": s.status.value,
                "agent": s.manifest.agent,
                "tags": s.manifest.tags,
            }
            FOR s IN skills
        ],
        "count": len(skills),
    }
```

### 6.5 `get_receipt` Handler

```
ASYNC FUNCTION _handle_get_receipt(params: dict) -> dict:
    # TDD Anchor: test_get_receipt_found
    # TDD Anchor: test_get_receipt_not_found

    VALIDATE params has "receipt_id" (string)

    engine = self._get_receipt_engine()
    IF engine IS None:
        RETURN {"error": "Receipt engine not available"}

    receipt = engine.get_receipt(params["receipt_id"])
    IF receipt IS None:
        RAISE ValueError(f"Receipt not found: {params['receipt_id']}")

    RETURN receipt.to_dict()
```

### 6.6 Receipt Emission Helper

```
FUNCTION _emit_receipt(method, query_data, result_data, status, gate, duration=0) -> dict | None:
    # TDD Anchor: test_emit_receipt_returns_dict
    # TDD Anchor: test_emit_receipt_none_when_engine_missing

    engine = self._get_receipt_engine()
    IF engine IS None:
        RETURN None

    TRY:
        receipt = engine.create_receipt(
            method=method,
            query_data=query_data IF isinstance(query_data, dict) ELSE {"raw": str(query_data)},
            result_data=result_data IF isinstance(result_data, dict) ELSE {"raw": str(result_data)},
            fate_score=result_data.get("fate_score", 0.0) IF isinstance(result_data, dict) ELSE 0.0,
            snr_score=0.95,  # Default from constitution
            gate_passed=gate,
            status=status,
            duration_ms=duration,
        )
        RETURN {"receipt_id": receipt.receipt_id, "status": receipt.status.value}
    EXCEPT Exception AS exc:
        logger.warning(f"Receipt emission failed: {exc}")
        RETURN None
```

### 6.7 AHK Client Extensions

```
; Ctrl+B, I -> invoke_skill
key = ih.Input
IF key = "i" OR key = "I":
    DoInvokeSkill()

FUNCTION DoInvokeSkill():
    ; Step 1: Ask for skill name
    skill := InputBox("Skill name:", "BIZRA Invoke Skill", "W400 H120")
    IF skill.Result != "OK" OR skill.Value = "":
        RETURN

    ; Step 2: Ask for inputs (optional JSON)
    inputs := InputBox("Inputs (JSON, optional):", "BIZRA Skill Inputs", "W400 H120")
    params_str := '{"skill":"' escaped_skill '"'
    IF inputs.Result = "OK" AND inputs.Value != "":
        params_str .= ',"inputs":' inputs.Value
    params_str .= '}'

    ShowTooltip("BIZRA: Invoking " skill.Value "...")
    response := SendCommand("invoke_skill", params_str)
    ; Parse and show result (same pattern as DoSovereignQuery)

; Ctrl+B, L -> list_skills
IF key = "l" OR key = "L":
    DoListSkills()

FUNCTION DoListSkills():
    response := SendCommand("list_skills")
    ; Parse skills array, show top 10 as tooltip
```

---

## 7. Test Plan

### 7.1 `tests/core/bridges/test_bridge_receipt.py` (NEW)

```
TDD Anchors:
├── TestBridgeReceiptEngine
│   ├── test_create_accepted_receipt          # Receipt fields correct
│   ├── test_create_rejected_receipt          # Rejected status + reason
│   ├── test_receipt_has_valid_signature      # SimpleSigner verifies
│   ├── test_receipt_persisted_to_disk        # JSON file exists
│   ├── test_atomic_write                     # No partial writes
│   ├── test_get_receipt_from_cache           # In-memory hit
│   ├── test_get_receipt_from_disk            # Disk fallback
│   ├── test_get_receipt_not_found            # Returns None
│   └── test_receipt_dir_auto_created         # mkdir on init
├── TestLoadSigner
│   ├── test_load_signer_from_env             # Reads env var
│   └── test_load_signer_fallback             # Default key
```

### 7.2 `tests/core/bridges/test_bridge_skills.py` (NEW)

```
TDD Anchors:
├── TestInvokeSkill
│   ├── test_invoke_skill_success             # Mock router returns OK
│   ├── test_invoke_skill_unknown             # Skill not found
│   ├── test_invoke_skill_ihsan_blocked       # Below threshold
│   ├── test_invoke_skill_produces_receipt    # receipt field in response
│   └── test_invoke_missing_param             # No "skill" in params
├── TestListSkills
│   ├── test_list_skills_returns_skills       # Skills listed
│   ├── test_list_skills_filter               # Filter by status
│   └── test_list_skills_no_registry          # Graceful degradation
├── TestGetReceipt
│   ├── test_get_receipt_found                # Returns receipt dict
│   └── test_get_receipt_not_found            # Error response
├── TestReceiptEmission
│   ├── test_emit_receipt_returns_dict        # receipt_id + status
│   ├── test_emit_receipt_none_when_missing   # Engine not loaded
│   └── test_existing_commands_get_receipts   # ping/status/query
```

---

## 8. Implementation Order

```
Step 1: core/bridges/bridge_receipt.py
        ├── BridgeReceiptEngine
        ├── load_signer()
        └── Receipt creation + persistence + retrieval

Step 2: tests/core/bridges/test_bridge_receipt.py
        └── All receipt TDD anchors (11 tests)

Step 3: core/bridges/desktop_bridge.py
        ├── Add invoke_skill handler
        ├── Add list_skills handler
        ├── Add get_receipt handler
        ├── Add _emit_receipt helper
        ├── Wire receipt generation into ping/status/sovereign_query
        └── Lazy-load SkillRouter + BridgeReceiptEngine

Step 4: tests/core/bridges/test_bridge_skills.py
        └── All skill routing TDD anchors (13 tests)

Step 5: bin/bizra_bridge.ahk
        ├── Add Ctrl+B,I (invoke_skill)
        └── Add Ctrl+B,L (list_skills)

Step 6: Verify
        ├── pytest tests/core/bridges/ -v  (target: 126+ tests all green)
        └── Manual AHK test: Ctrl+B,L shows skills
```

---

## 9. Verification

```bash
PYTHONPATH=/mnt/c/BIZRA-DATA-LAKE pytest tests/core/bridges/test_bridge_receipt.py -v
PYTHONPATH=/mnt/c/BIZRA-DATA-LAKE pytest tests/core/bridges/test_bridge_skills.py -v
PYTHONPATH=/mnt/c/BIZRA-DATA-LAKE pytest tests/core/bridges/ -v  # Full regression (126+ tests)
```

---

## 10. Deferred (Phase 3+)

Command palette GUI, chord sequence engine, context-sensitive hotkeys, Hunter scan,
WebSocket upgrade, streaming responses — all deferred until skill routing proves impact.

---

## 11. Giants

Fowler (EIP routing) | Cockburn (Hexagonal adapters) | Lamport (receipt ordering) |
Boyd (OODA: observe before scaling) | Al-Ghazali (Ihsan: no unaudited execution)
