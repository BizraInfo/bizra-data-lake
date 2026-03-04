# Phase 65.1: Genesis Boot — Cold Initialization

> Standing on Giants: Lamport (distributed identity, 1978) · Nakamoto (genesis block, 2008) · Al-Ghazali (constitutional floor, 1095)

## 1. Purpose

Cold boot a BIZRA node from `[UNINITIALIZED]` to `[ROOTED]` in < 5 seconds.
Three sequential stages: sovereignty establishment, constitution loading, HDA layer boot.

**Entry State**: `[UNINITIALIZED]` — no keypair, no ledger, no reflexes.
**Exit State**: `[ROOTED]` — identity sealed, constitution loaded, kinetic layer armed.
**Target Latency**: < 5 seconds total genesis time.

---

## 2. Pseudocode

### 2.1 Sovereignty Establishment

```
FUNCTION establish_sovereignty(user_name: str, config: Config) -> Identity:
    """Generate Ed25519 keypair and create genesis block."""

    # Step 1: Generate Ed25519 keypair
    # Source: core/pci/crypto.py:generate_keypair()
    private_key_hex, public_key_hex = generate_keypair()

    # Step 2: Persist to sovereign_state/identity/credentials.json
    # Source: core/sovereign/mission.py:_load_or_create_node_signer()
    state_dir = config.sovereign_state_dir / "identity"
    state_dir.mkdir(parents=True, exist_ok=True)

    credentials = {
        "node_id": generate_node_id(public_key_hex),  # "BIZRA-{hash[:8]}"
        "private_key": private_key_hex,
        "public_key": public_key_hex,
        "created_at": utc_now_iso(),
        "source": "genesis"
    }
    persist_json(state_dir / "credentials.json", credentials)

    # Step 3: Initialize BlockGraph with genesis block
    # Source: core/proof_engine/evidence_ledger.py
    ledger = EvidenceLedger(
        path=config.sovereign_state_dir / "evidence_chain.jsonl",
        validate_on_append=True
    )

    genesis_receipt = {
        "type": "GENESIS",
        "height": 0,
        "timestamp": utc_now_iso(),
        "constitution_hash": blake3_hash(CONSTITUTION_TEXT),
        "node_id": credentials["node_id"],
        "public_key": public_key_hex,
        "reason_codes": ["GENESIS_BLOCK"]
    }
    ledger.append(receipt=genesis_receipt)

    # Step 4: Mint initial IMPT balance (genesis grant)
    # Source: core/treasury/token_minter.py
    minter = TokenMinter.create(
        ledger=config.sovereign_state_dir / "token_ledger.jsonl"
    )
    minter.mint(
        recipient=credentials["node_id"],
        amount=GENESIS_IMPT_GRANT,  # 100 IMPT
        reason="genesis_grant"
    )

    RETURN Identity(
        node_id=credentials["node_id"],
        public_key=public_key_hex,
        state="SOVEREIGNTY_SEALED"
    )
```

### 2.2 Constitution Loading

```
FUNCTION load_constitution(config: Config) -> ConstitutionalState:
    """Load FATE gate constraints and Ihsan 8D tensor thresholds."""

    # Import constitutional thresholds (single source of truth)
    # Source: core/integration/constants.py
    FROM core.integration.constants IMPORT (
        UNIFIED_IHSAN_THRESHOLD,        # 0.95
        STRICT_IHSAN_THRESHOLD,         # 0.99
        UNIFIED_SNR_THRESHOLD,          # 0.85
        ADL_GINI_THRESHOLD              # 0.35
    )

    # Step 1: Load Lyapunov constraint
    lyapunov = LyapunovConstraint(
        condition="nabla_V_dot_f_x <= 0",
        description="System entropy must not increase"
    )

    # Step 2: Load TeleScript permissions
    # Source: core/governance/constitutional_gate.py
    telescript = TeleScriptPermissions(
        allow=[config.user_documents_path],
        deny=["/system32", "/etc/shadow", "/Windows/System32"]
    )

    # Step 3: Initialize Ihsan 8D tensor thresholds
    ihsan_dimensions = {
        "moral_clarity":       0.90,
        "epistemic_humility":  0.70,
        "temporal_awareness":  0.85,
        "resource_efficiency": 0.75,
        "user_alignment":      0.95,
        "transparency":        0.80,
        "reversibility":       0.90,
        "harm_minimization":   0.99
    }

    RETURN ConstitutionalState(
        lyapunov=lyapunov,
        telescript=telescript,
        ihsan_dimensions=ihsan_dimensions,
        state="CONSTITUTIONAL"
    )
```

### 2.3 HDA Kinetic Layer Boot

```
FUNCTION boot_hda(config: Config) -> HDAState:
    """Initialize Universal Action Bus with AHK + UIA backends."""

    # Source: core/sovereign/mission.py (HDAClient)
    # Source: core/bridges/desktop_bridge.py

    # Step 1: Initialize action bus routes
    action_bus = ActionBus()
    action_bus.register_route("keyboard.*", AHKKeyboardController())
    action_bus.register_route("mouse.*", AHKMouseController())
    action_bus.register_route("file.*", PowerShellFileManager())
    action_bus.register_route("app.*", UIAApplicationController())

    # Step 2: Verify UIA connectivity (Windows Accessibility Tree)
    IF platform.is_windows():
        uia_status = UIA.verify_connectivity()
    ELSE:
        uia_status = "MOCK"  # Non-Windows: HDA degrades to file-only

    # Step 3: Verify AHK engine readiness
    # Source: BIZRA_HDA_PORT (config, default 9743)
    ahk_status = HDAClient(
        host="127.0.0.1",
        port=config.hda_port
    ).health_check()

    RETURN HDAState(
        action_bus=action_bus,
        uia_status=uia_status,
        ahk_status=ahk_status,
        state="KINETIC_ARMED"
    )
```

### 2.4 Genesis Orchestrator

```
FUNCTION genesis_boot(user_name: str, config: Config) -> SystemState:
    """Complete genesis: sovereignty + constitution + HDA = ROOTED."""

    t0 = monotonic_clock()

    # Sequential — each depends on the previous
    identity = establish_sovereignty(user_name, config)
    constitution = load_constitution(config)
    hda = boot_hda(config)

    elapsed = monotonic_clock() - t0

    RETURN SystemState(
        identity=identity,
        constitution=constitution,
        hda=hda,
        state="ROOTED",
        genesis_time_ms=elapsed,
        impt_balance=GENESIS_IMPT_GRANT,
        temperature=INITIAL_TEMPERATURE,      # T = 2.0
        epistemic_entropy=MAX_ENTROPY,         # H_max
        reflexes_compiled=0
    )
```

---

## 3. Data Structures

```
@dataclass
class Identity:
    node_id: str               # "BIZRA-{hex[:8]}"
    public_key: str            # Ed25519 public key hex
    state: str                 # "SOVEREIGNTY_SEALED"

@dataclass
class ConstitutionalState:
    lyapunov: LyapunovConstraint
    telescript: TeleScriptPermissions
    ihsan_dimensions: dict[str, float]   # 8 dimensions
    state: str                            # "CONSTITUTIONAL"

@dataclass
class HDAState:
    action_bus: ActionBus
    uia_status: str            # "CONNECTED" | "MOCK"
    ahk_status: str            # "READY" | "UNAVAILABLE"
    state: str                 # "KINETIC_ARMED"

@dataclass
class SystemState:
    identity: Identity
    constitution: ConstitutionalState
    hda: HDAState
    state: str                 # "ROOTED"
    genesis_time_ms: float
    impt_balance: float        # Starting at 100
    temperature: float         # T = 2.0 (hot)
    epistemic_entropy: float   # H_max (no knowledge)
    reflexes_compiled: int     # 0 at genesis
```

---

## 4. Constants (No Hardcoding)

```
# All from core/integration/constants.py or config
GENESIS_IMPT_GRANT     = config.get("genesis_impt_grant", 100)
INITIAL_TEMPERATURE    = config.get("initial_temperature", 2.0)
MAX_ENTROPY            = log2(config.get("action_space_size", 128))
```

---

## 5. TDD Anchors

### Existing Tests
- `tests/core/sovereign/test_hardening_track1.py::TestPersistentNodeSigner` — signer persistence
- `tests/core/proof_engine/test_evidence_ledger.py` — genesis block, chain integrity
- `tests/core/treasury/test_token_minter.py` — genesis mint, zakat deduction

### New Tests Required

```python
# tests/core/sovereign/test_lifecycle_genesis.py

class TestGenesisBoot:
    """Phase 65.1: Genesis boot produces ROOTED state."""

    def test_genesis_creates_identity(self, tmp_path):
        """Sovereignty establishment produces valid Ed25519 identity."""
        state = genesis_boot("Dr. Sarah Chen", make_config(tmp_path))
        assert state.identity.state == "SOVEREIGNTY_SEALED"
        assert len(state.identity.public_key) == 64  # 32 bytes hex

    def test_genesis_creates_ledger_with_block_zero(self, tmp_path):
        """BlockGraph has exactly one genesis block after boot."""
        state = genesis_boot("test", make_config(tmp_path))
        ledger_path = tmp_path / "state" / "evidence_chain.jsonl"
        assert ledger_path.exists()
        lines = ledger_path.read_text().strip().split("\n")
        assert len(lines) == 1
        genesis = json.loads(lines[0])
        assert genesis["type"] == "GENESIS"

    def test_genesis_mints_impt(self, tmp_path):
        """Genesis grant mints exactly GENESIS_IMPT_GRANT IMPT."""
        state = genesis_boot("test", make_config(tmp_path))
        assert state.impt_balance == 100  # from config

    def test_genesis_under_5_seconds(self, tmp_path):
        """Genesis completes in < 5 seconds."""
        state = genesis_boot("test", make_config(tmp_path))
        assert state.genesis_time_ms < 5000

    def test_genesis_temperature_is_hot(self, tmp_path):
        """Initial temperature is 2.0 (full exploration)."""
        state = genesis_boot("test", make_config(tmp_path))
        assert state.temperature == 2.0

    def test_genesis_entropy_is_maximum(self, tmp_path):
        """Initial entropy is maximum (no user knowledge)."""
        state = genesis_boot("test", make_config(tmp_path))
        assert state.epistemic_entropy > 0
        assert state.reflexes_compiled == 0
```

---

## 6. Error Handling

```
ON keypair generation failure:
    LOG error, RETRY once with different entropy source
    IF retry fails: ABORT genesis, state remains [UNINITIALIZED]

ON ledger creation failure:
    LOG error, state_dir may be read-only
    ABORT genesis with specific error message

ON HDA boot failure:
    WARN "HDA unavailable — degrading to file-only mode"
    Continue with HDAState(uia_status="MOCK", ahk_status="UNAVAILABLE")
    System still reaches [ROOTED] with degraded kinetic capability
```

---

## 7. Sequence Diagram

```
User                  Genesis                 Crypto        Ledger        HDA
  │                      │                      │             │            │
  ├─ bizra init ────────>│                      │             │            │
  │                      ├─ generate_keypair() ─>│             │            │
  │                      │<── (priv, pub) ──────│             │            │
  │                      │                      │             │            │
  │                      ├─ persist credentials ─────────────>│            │
  │                      ├─ create genesis block ────────────>│            │
  │                      │<── block_0 hash ──────────────────│            │
  │                      │                      │             │            │
  │                      ├─ load constitution ──>│             │            │
  │                      │                      │             │            │
  │                      ├─ boot_hda() ──────────────────────────────────>│
  │                      │<── KINETIC_ARMED ────────────────────────────-│
  │                      │                      │             │            │
  │<── [ROOTED] ────────│                      │             │            │
```
