# Phase 20: RDVE Actuator Layer — Reasoning-to-Desktop Bridge

## Formal Foundation

The Actuator Layer bridges abstract reasoning (RDVE) to concrete desktop events,
treating the operating system as a verifiable state machine where every interaction
is audited against Ihsan (excellence) thresholds.

### Actuator Mapping

The transition from strategic plan to desktop event:

```
pi: S x G -> A

Where:
  S = observed state (UI fingerprint from UIA)
  G = goal (from RDVE mission)
  A = optimal action (AHK instruction)
```

### Bicameral Memory Stack

1. **Volatile Context**: Real-time structural fingerprints via UI Automation (UIA)
2. **Permanent Procedures**: Audited AHK skills with high-density instruction sets

### Shannon Entropy Gate

Generated instructions must meet information density threshold before execution:

```
H(X) = -SUM(P(x_i) * log2(P(x_i)))  for i in 1..n

Threshold: H > 3.5 bits/char (prevents repetitive/low-signal code)
```

---

## Existing Infrastructure (What We Have)

### Python Side (COMPLETE)

| Component | File | Status | Lines |
|-----------|------|--------|-------|
| Desktop Bridge TCP server | `core/bridges/desktop_bridge.py` | BUILT | 1,019 |
| Bridge receipt engine | `core/bridges/bridge_receipt.py` | BUILT | ~200 |
| FATE gate integration | `core/elite/hooks.py` | BUILT | N/A |
| Rust GateChain (PyO3) | `bizra-omega/bizra-core` | BUILT | N/A |
| Shannon Entropy calculator | `core/uers/entropy.py` | BUILT | 688 |
| RDVE skill registration | `core/spearpoint/rdve_skill.py` | BUILT | N/A |
| Auto-researcher (RDVE) | `core/spearpoint/auto_researcher.py` | BUILT | ~500 |
| Skill Router | `core/skills/router.py` | BUILT | N/A |
| Test suite | `tests/core/bridges/test_desktop_bridge.py` | BUILT | 537 |

### AHK Side (V1 COMPLETE)

| Component | File | Status |
|-----------|------|--------|
| Bridge client | `bin/bizra_bridge.ahk` | BUILT (491 lines) |
| TCP Winsock2 connection | `bin/bizra_bridge.ahk` | BUILT |
| Auth envelope (token+ts+nonce) | `bin/bizra_bridge.ahk` | BUILT |
| Chord hotkeys (Ctrl+B,Q/S/I/L) | `bin/bizra_bridge.ahk` | BUILT |
| Reconnect with exponential backoff | `bin/bizra_bridge.ahk` | BUILT |

### What's Missing (Phase 20 Scope)

| Component | Description | Priority |
|-----------|-------------|----------|
| Shannon entropy gate on dispatch | Pre-flight entropy check before sending instructions to AHK | P0 |
| UIA fingerprinting method | `get_context` JSON-RPC method returning structural fingerprint | P0 |
| Actuator `execute` method | New bridge method for executing verified AHK instructions | P0 |
| Skill registration protocol | JSON schema for registering AHK skills in RDVE skills ledger | P1 |
| Fingerprint baseline store | BLAKE3-hashed baselines for VS Code, Chrome, Terminal | P1 |
| Command Palette overlay | Fuzzy-search GUI in AHK for system-wide skill invocation | P2 |

---

## Module 1: Shannon Entropy Gate on Bridge Dispatch (P0)

### File: `core/bridges/desktop_bridge.py`

**Insert after** `_validate_fate` method, **before** method handlers.

### Existing Infrastructure

```
core/uers/entropy.py:
  EntropyCalculator.text_entropy(text) -> EntropyMeasurement
  EntropyMeasurement.value: float       # Shannon entropy in bits/char
  EntropyMeasurement.normalized: float  # 0-1 normalized
```

### Pseudocode

```
CONSTANT ACTUATOR_ENTROPY_THRESHOLD = 3.5  # bits/char minimum

METHOD _validate_entropy(self, instruction: str) -> dict[str, Any]:
    """Shannon entropy gate — blocks low-signal instructions."""
    IF NOT instruction.strip():
        RETURN {"passed": False, "entropy": 0.0, "error": "Empty instruction"}

    TRY:
        FROM core.uers.entropy IMPORT EntropyCalculator
        calc = EntropyCalculator()
        measurement = calc.text_entropy(instruction)

        passed = measurement.value >= ACTUATOR_ENTROPY_THRESHOLD
        RETURN {
            "passed": passed,
            "entropy": round(measurement.value, 4),
            "normalized": round(measurement.normalized, 4),
            "threshold": ACTUATOR_ENTROPY_THRESHOLD,
            "unique_chars": measurement.metadata.get("unique_chars", 0),
        }
    EXCEPT ImportError:
        # Fail-open if entropy module unavailable (non-critical)
        RETURN {"passed": True, "entropy": -1.0, "error": "entropy_module_unavailable"}
```

### TDD Anchors

```
test_entropy_gate_blocks_low_signal
    bridge._validate_entropy("aaaaaaaaaa")
    ASSERT result["passed"] is False
    ASSERT result["entropy"] < 3.5

test_entropy_gate_passes_high_signal
    bridge._validate_entropy("import json; data = fetch('api/v1'); process(data)")
    ASSERT result["passed"] is True
    ASSERT result["entropy"] >= 3.5

test_entropy_gate_empty_instruction
    bridge._validate_entropy("")
    ASSERT result["passed"] is False

test_entropy_gate_threshold_boundary
    # Exactly at threshold
    ASSERT result["threshold"] == 3.5
```

---

## Module 2: Actuator Execute Method (P0)

### File: `core/bridges/desktop_bridge.py`

**Add** to handlers dict in `_dispatch()`.

### Pseudocode

```
METHOD _handle_actuator_execute(self, params: dict) -> dict[str, Any]:
    """
    Execute a verified instruction via the AHK actuator.

    Pipeline: FATE gate -> Entropy gate -> Rust gate -> Sign -> Dispatch receipt
    """
    IF NOT isinstance(params, dict) OR "code" NOT IN params:
        RAISE ValueError("Missing 'code' in params")

    code = str(params["code"]).strip()
    intent = str(params.get("intent", "execute"))
    target_app = params.get("target_app")  # Optional: expected window context

    IF NOT code:
        RAISE ValueError("Empty instruction code")

    # Gate 1: FATE (Ihsan threshold)
    fate_result = self._validate_fate(f"actuator_execute:{intent}")
    IF NOT fate_result.get("passed", False):
        RETURN {"error": "FATE gate blocked", "fate": fate_result}

    # Gate 2: Shannon Entropy (information density)
    entropy_result = self._validate_entropy(code)
    IF NOT entropy_result.get("passed", False):
        receipt = self._emit_receipt(
            "actuator_execute", params, entropy_result,
            "rejected", "SHANNON_ENTROPY",
            reason=f"Low entropy: {entropy_result.get('entropy', 0):.2f} < {ACTUATOR_ENTROPY_THRESHOLD}"
        )
        RETURN {
            "error": "Shannon entropy gate blocked",
            "entropy": entropy_result,
            "receipt": receipt,
        }

    # Gate 3: Rust GateChain (Schema + Ihsan + SNR)
    rust_gates = self._validate_rust_gates(code)
    IF rust_gates AND NOT rust_gates.get("passed", True):
        RETURN {"error": "Rust gate chain blocked", "rust_gates": rust_gates}

    # Sign instruction with BLAKE3
    content_hash = self._blake3_digest(code)

    # Build response (instruction is validated but NOT executed server-side)
    # The AHK client receives the signed, validated instruction and executes it
    RETURN {
        "status": "SEALED",
        "intent": intent,
        "target_app": target_app,
        "content_hash": content_hash,
        "fate": fate_result,
        "entropy": entropy_result,
        "instruction_length": len(code),
    }
```

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Bridge validates but does NOT execute code | Separation of concerns: Python validates, AHK executes |
| Three-gate pipeline (FATE + Shannon + Rust) | Defense-in-depth: no single gate bypass compromises system |
| BLAKE3 content hash returned | AHK client can verify instruction integrity |
| `target_app` is optional | Enables contextual validation in future (UIA fingerprint match) |

### TDD Anchors

```
test_actuator_execute_missing_code
    resp = dispatch("actuator_execute", {})
    ASSERT "error" in resp  # ValueError

test_actuator_execute_low_entropy_blocked
    resp = dispatch("actuator_execute", {"code": "aaaa"})
    ASSERT result["error"] == "Shannon entropy gate blocked"
    ASSERT "receipt" in result

test_actuator_execute_valid_instruction
    resp = dispatch("actuator_execute", {
        "code": "Send('^s'); WinWait('Save'); Click(200, 150)",
        "intent": "save_file",
        "target_app": "Code.exe",
    })
    ASSERT result["status"] == "SEALED"
    ASSERT result["content_hash"] is not None
    ASSERT result["entropy"]["passed"] is True

test_actuator_execute_fate_blocked
    # Mock FATE gate to fail
    resp = dispatch("actuator_execute", {"code": "valid instruction here"})
    ASSERT result["error"] == "FATE gate blocked"
```

---

## Module 3: UIA Context Fingerprinting Method (P0)

### File: `core/bridges/desktop_bridge.py`

**Add** `get_context` to handlers dict.

### Pseudocode

```
METHOD _handle_get_context(self, params: dict) -> dict[str, Any]:
    """
    Request UI context fingerprint from the AHK actuator client.

    This is a PROXY method: the bridge relays the request to AHK,
    which uses UIA-v2 to extract the structural fingerprint.

    The bridge does NOT connect to UIA directly (runs in WSL/Linux).
    """
    # This method returns the schema that the AHK client should fill
    # The actual UIA data comes FROM the AHK client via get_context calls
    RETURN {
        "schema_version": "1.0",
        "expected_fields": [
            "title",       # Active window title
            "class",       # Window class name
            "process",     # Process name
            "structure",   # UIA element tree (JSON)
            "hash",        # BLAKE3 hash of structure
        ],
        "note": "Context data populated by AHK actuator client via UIA-v2",
    }
```

### AHK Side: Context Extraction (Enhancement to `bin/bizra_bridge.ahk`)

```
; New function: extract UI context fingerprint
GetUIContext() {
    activeWin := WinGetID("A")
    IF (!activeWin) {
        RETURN '{"error":"no_active_window"}'
    }

    title := WinGetTitle(activeWin)
    class := WinGetClass(activeWin)
    process := WinGetProcessName(activeWin)

    ; Build context JSON
    RETURN '{' .
        '"title":"' . JsonEscape(title) . '",' .
        '"class":"' . JsonEscape(class) . '",' .
        '"process":"' . JsonEscape(process) . '"' .
    '}'
}
```

### TDD Anchors

```
test_get_context_returns_schema
    resp = dispatch("get_context", {})
    ASSERT result["schema_version"] == "1.0"
    ASSERT "expected_fields" in result
    ASSERT "title" in result["expected_fields"]

test_get_context_schema_fields_complete
    fields = result["expected_fields"]
    ASSERT "title" in fields
    ASSERT "class" in fields
    ASSERT "process" in fields
    ASSERT "structure" in fields
    ASSERT "hash" in fields
```

---

## Module 4: AHK Skill Registration Protocol (P1)

### File: `core/spearpoint/actuator_skills.py` (NEW)

### Pseudocode

```
@dataclass
CLASS ActuatorSkillManifest:
    """Schema for registering an AHK skill in the RDVE skills ledger."""
    name: str                              # e.g., "browser_navigate"
    description: str                       # Human-readable purpose
    target_app: str                        # Window class or process name
    ahk_code: str                          # AHK v2 instruction template
    entropy_score: float                   # Pre-computed Shannon entropy
    parameters: dict[str, str]             # Parameter name -> type
    requires_context: bool = False         # Whether UIA fingerprint is needed
    min_ihsan: float = 0.95                # Minimum Ihsan score to invoke

    METHOD validate(self) -> bool:
        """Validate manifest meets constitutional constraints."""
        IF self.entropy_score < 3.5:
            RETURN False
        IF self.min_ihsan < 0.90:
            RETURN False
        IF NOT self.name OR NOT self.ahk_code:
            RETURN False
        RETURN True


CLASS ActuatorSkillLedger:
    """Registry of verified AHK skills available to RDVE."""

    CONSTRUCTOR():
        self._skills: dict[str, ActuatorSkillManifest] = {}

    METHOD register(self, manifest: ActuatorSkillManifest) -> bool:
        IF NOT manifest.validate():
            RETURN False
        self._skills[manifest.name] = manifest
        RETURN True

    METHOD get(self, name: str) -> Optional[ActuatorSkillManifest]:
        RETURN self._skills.get(name)

    METHOD list_all(self) -> list[ActuatorSkillManifest]:
        RETURN list(self._skills.values())

    METHOD resolve_for_app(self, app: str) -> list[ActuatorSkillManifest]:
        """Find all skills targeting a specific application."""
        RETURN [s for s in self._skills.values() if s.target_app == app]
```

### Pre-Registered Skills (Phase 20 Baseline)

```
SKILLS = [
    ActuatorSkillManifest(
        name="vscode_save",
        description="Save current file in VS Code",
        target_app="Code.exe",
        ahk_code="Send('^s')",
        entropy_score=4.2,
        parameters={},
    ),
    ActuatorSkillManifest(
        name="browser_navigate",
        description="Navigate to URL in active browser",
        target_app="chrome.exe",
        ahk_code="Send('^l'); Sleep(100); Send('{url}'); Send('{Enter}')",
        entropy_score=5.1,
        parameters={"url": "str"},
    ),
    ActuatorSkillManifest(
        name="terminal_command",
        description="Execute command in active terminal",
        target_app="WindowsTerminal.exe",
        ahk_code="Send('{command}'); Send('{Enter}')",
        entropy_score=4.8,
        parameters={"command": "str"},
        requires_context=True,
    ),
]
```

### TDD Anchors

```
test_skill_manifest_validation_passes
    m = ActuatorSkillManifest(name="test", ..., entropy_score=4.0)
    ASSERT m.validate() is True

test_skill_manifest_low_entropy_fails
    m = ActuatorSkillManifest(name="test", ..., entropy_score=2.0)
    ASSERT m.validate() is False

test_skill_ledger_register_and_get
    ledger = ActuatorSkillLedger()
    ledger.register(valid_manifest)
    ASSERT ledger.get("test") is not None

test_skill_ledger_resolve_for_app
    ledger = ActuatorSkillLedger()
    ledger.register(vscode_skill)
    ledger.register(chrome_skill)
    ASSERT len(ledger.resolve_for_app("Code.exe")) == 1
```

---

## Implementation Order

| Step | Module | File | Est. Lines | Dependencies |
|------|--------|------|-----------|--------------|
| 1 | Shannon entropy gate | `desktop_bridge.py` | ~25 | `core/uers/entropy.py` (exists) |
| 2 | `actuator_execute` method | `desktop_bridge.py` | ~55 | Step 1 |
| 3 | `get_context` method | `desktop_bridge.py` | ~20 | None |
| 4 | AHK context extraction | `bin/bizra_bridge.ahk` | ~25 | None |
| 5 | Actuator skill ledger | `core/spearpoint/actuator_skills.py` | ~80 | None |
| 6 | Tests for steps 1-5 | `tests/core/bridges/test_actuator_layer.py` | ~120 | Steps 1-5 |

**Total: ~325 lines across 4 files.**

---

## Verification Plan

```bash
# Step 1-3: Python bridge extensions
python3 -m pytest tests/core/bridges/test_desktop_bridge.py -v --timeout=60

# Step 5: Skill ledger
python3 -m pytest tests/core/bridges/test_actuator_layer.py -v --timeout=60

# Step 6: Full regression
python3 -m pytest tests/ -q --timeout=60 \
  -m "not requires_ollama and not requires_gpu and not slow and not requires_network" \
  --ignore=tests/root_legacy --ignore=tests/e2e_http
```

---

## Sovereign Constraint Verification

| Constraint | Mechanism | Module |
|------------|-----------|--------|
| Ihsan >= 0.95 | FATE gate weighted evaluation | `_validate_fate()` (exists) |
| Shannon H > 3.5 | Entropy calculator pre-flight | `_validate_entropy()` (new) |
| Loopback 127.0.0.1 | Invariant in constructor | `DesktopBridge.__init__` (exists) |
| Structural context | UIA-v2 fingerprint via AHK | `get_context` (new) |
| Proof-carrying | BLAKE3 content hash + receipt | `_emit_receipt()` (exists) |
| Replay protection | Nonce + timestamp + HMAC | `_validate_auth()` (exists) |
| Rate limiting | Token bucket (20 req/s) | `TokenBucket` (exists) |

---

## Giants Protocol

| Giant | Contribution |
|-------|-------------|
| Shannon | Entropy audit ensures instruction quality before execution |
| Bernstein | Ed25519 signing of receipts (via `core.pci.crypto`) |
| Boyd | OODA loop: Observe (UIA) -> Orient (RDVE) -> Decide (gates) -> Act (AHK) |
| Al-Ghazali | Ihsan constraint: excellence is not optional |
| Lamport | Nonce-based replay protection follows Byzantine principles |
