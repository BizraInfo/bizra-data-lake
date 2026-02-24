# Cold-Start Bootstrap API Reference

> **Version:** 1.0.0 | **Date:** 2026-02-22
> **BIZRA** (بذرة): Proactive Dynamic DDAGI OS — every human is a node, every node is a seed, every seed has infinite potential.

## Python APIs

### Genesis Gate — `bizra-normalizers/genesis_gate.py`

#### `NodeMaturityStage` (Enum)

Progressive maturity stages for node onboarding.

```python
from genesis_gate import NodeMaturityStage

NodeMaturityStage.SEED      # Zero data, immediate pass
NodeMaturityStage.SPROUT    # 10+ TEACH atoms
NodeMaturityStage.GROWING   # 100+ messages OR 1+ import
NodeMaturityStage.ROOTED    # 3+ provider imports
```

#### `GenesisGateConfig.for_cold_start()`

Factory method returning a gate config that allows zero-data users to pass.

```python
from genesis_gate import GenesisGateConfig

config = GenesisGateConfig.for_cold_start()
# GenesisGateConfig(min_cv=0.0, min_nodes=0, min_elite_nodes=0, fail_closed=True)
```

#### `GenesisGateConfig.for_stage(stage)`

Factory method returning gate config calibrated for a specific maturity stage.

```python
config = GenesisGateConfig.for_stage(NodeMaturityStage.GROWING)
# GenesisGateConfig(min_cv=0.5, min_nodes=3, min_elite_nodes=1, ...)
```

#### `determine_maturity_stage(atom_count, message_count, provider_count)`

Determine current node maturity based on interaction metrics.

```python
from genesis_gate import determine_maturity_stage

stage = determine_maturity_stage(atom_count=15, message_count=5, provider_count=0)
# NodeMaturityStage.SPROUT
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `atom_count` | `int` | `0` | Number of TEACH atoms stored |
| `message_count` | `int` | `0` | Total messages processed |
| `provider_count` | `int` | `0` | Distinct provider imports |

---

### Node Template — `core/pat/onboarding.py`

#### `NodeTemplate` (Dataclass)

MMORPG-inspired node template. Every new node inherits this configuration.

```python
from core.pat.onboarding import NodeTemplate

# Default template (standard user)
template = NodeTemplate.default()
template.pat_count           # 7 (Personal Agentic Team)
template.sat_count           # 5 (System Agentic Team)
template.initial_seed_grant  # 100.0
template.ihsan_floor         # 0.95
template.snr_minimum         # 0.85

# Alpha-100 template (early adopter)
alpha = NodeTemplate.alpha_100()
alpha.initial_seed_grant     # 500.0
alpha.max_inference_tokens_per_day  # 100_000

# Fork from existing template
forked = NodeTemplate.fork_from(alpha, initial_seed_grant=250.0)
```

#### Key Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `pat_count` | `int` | `7` | Personal Agentic Team size |
| `sat_count` | `int` | `5` | System Agentic Team size |
| `initial_seed_grant` | `float` | `100.0` | SEED tokens on creation |
| `ihsan_floor` | `float` | `0.95` | Ihsan quality threshold |
| `adl_gini_threshold` | `float` | `0.35` | Anti-concentration gate |
| `snr_minimum` | `float` | `0.85` | Signal quality floor |
| `desktop_rpc_enabled` | `bool` | `True` | AHK bridge access |
| `tool_call_enabled` | `bool` | `True` | Tool call channel |
| `llm_call_locked` | `bool` | `True` | Locked until GROWING |
| `file_op_locked` | `bool` | `True` | Locked until ROOTED |
| `browser_nav_locked` | `bool` | `True` | Locked until ROOTED |
| `bootstrap_reflex_count` | `int` | `4` | Pre-seeded reflexes |
| `max_cpu_percent` | `float` | `25.0` | Resource limit |
| `max_memory_mb` | `int` | `512` | Memory cap |
| `max_inference_tokens_per_day` | `int` | `50_000` | Inference budget |
| `initial_stage` | `str` | `"seed"` | Starting maturity stage |

---

### Token Ledger — `core/token/ledger.py`

#### `TokenLedger.genesis_grant(node_id, amount, epoch_id, memo)`

Issue initial SEED tokens to a new node via `GENESIS_MINT`.

```python
from core.token.ledger import TokenLedger

ledger = TokenLedger(db_path="tokens.db")
receipt = ledger.genesis_grant(
    node_id="node-abc123",
    amount=100.0,
    memo="Welcome to BIZRA"
)
receipt.tx_hash      # SHA-256 hash-chained receipt
receipt.new_balance  # 100.0
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `node_id` | `str` | required | Target node identifier |
| `amount` | `float` | `100.0` | SEED tokens to grant |
| `epoch_id` | `str` | `GENESIS_EPOCH_ID` | Epoch identifier |
| `memo` | `str` | auto-generated | Human-readable memo |

**Raises:** `ValueError` if `node_id` is empty or `amount <= 0`.

---

### Stereoscopic Engine — `bizra-normalizers/engine.py`

#### `AutonomousSNRGoTEngine.compile_from_atoms(atoms, session_id)`

Compile identity signals from pre-extracted memory atoms. This is the self-compilation feedback loop entry point.

```python
from engine import AutonomousSNRGoTEngine

engine = AutonomousSNRGoTEngine()
report = engine.compile_from_atoms(
    atoms=[
        {"kind": "fact", "content": "I am the founder of BIZRA", "confidence": 0.95},
        {"kind": "expertise", "content": "Distributed systems", "confidence": 0.90},
        {"kind": "preference", "content": "Concise communication", "confidence": 0.88},
    ],
    session_id="node-abc123"
)
report.snr_score         # Float: signal quality
report.identity_signals  # List of compiled signals
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `atoms` | `list[dict]` | required | List of `{kind, content, confidence, ?timestamp}` |
| `session_id` | `str` | `"local"` | Session/node identifier |

**Returns:** `StereoscopicReport` with compiled identity signals.

**Atom Kind Mapping** (Rust `AtomKind` -> Python `FragmentKind`):

| Rust AtomKind | Python FragmentKind |
|---------------|-------------------|
| `fact` | `FACT` |
| `preference` | `PREFERENCE` |
| `pattern` | `PATTERN` |
| `relationship` | `RELATIONSHIP` |
| `goal` | `GOAL` |
| `expertise` | `EXPERTISE` |
| `context` | `EMOTION` |
| `principle` | `STYLE` |
| `temporal` | `TEMPORAL` |
| `negation` | `FACT` |

---

### Custom Provider Registration — `bizra-normalizers/normalizers/__init__.py`

#### `register_provider(name, parser, is_conversation_platform)`

Register a custom conversation export parser.

```python
from normalizers import register_provider
from normalizers.base import GenericJsonlParser

# Register a custom parser for local model exports
register_provider(
    name="my_local_model",
    parser=GenericJsonlParser(
        role_field="role",
        content_field="content",
        model_field="model"
    ),
    is_conversation_platform=True,
)
```

#### `unregister_provider(name)`

Remove a custom provider registration. Cannot remove built-in providers.

#### `registered_providers() -> list[str]`

List all registered provider names (built-in + custom).

#### `custom_providers() -> dict[str, PlatformParser]`

Return only custom-registered providers.

---

### Atlas Tier Report — `scripts/atlas/atlas_gap_report.py`

#### `user_tier_report(tier_name)`

Query which capabilities are unlocked for a given user tier.

```python
from scripts.atlas.atlas_gap_report import user_tier_report

report = user_tier_report("growing")
report["capabilities_unlocked"]  # ['chat', 'teach', 'memory_recall', ...]
report["capabilities_locked"]    # ['desktop_actions', 'full_action_bus', ...]
report["next_tier"]              # 'rooted'
report["unlock_criteria"]        # '100+ atoms or synthesis complete'
report["available_priorities"]   # ['P0', 'P1']
```

#### CLI Usage

```bash
python scripts/atlas/atlas_gap_report.py --user-tier growing
```

---

## Rust APIs

### Bootstrap Reflexes — `bizra-omega/bizra-agent/src/reflex_cache.rs`

#### `ReflexCache::load_bootstrap_rules() -> usize`

Seed universal bootstrap rules when the cache is empty. Returns count of rules loaded (4 on cold start, 0 if cache already populated).

```rust
let mut cache = ReflexCache::new();
let loaded = cache.load_bootstrap_rules();
assert_eq!(loaded, 4); // greeting, help, remember, profile_recall
```

#### Bootstrap Rule Definitions

| Trigger Pattern | Agent | Action Chain | Confidence |
|----------------|-------|-------------|------------|
| `bootstrap:greeting` | Diplomat | `GreetUser>GenerateResponse` | 0.95 |
| `bootstrap:help` | Scholar | `RetrieveContext>GenerateResponse` | 0.90 |
| `bootstrap:remember` | Oracle | `RecallMemory>GenerateResponse` | 0.92 |
| `bootstrap:profile_recall` | Oracle | `ProfileRecall>GenerateResponse` | 0.95 |

#### `is_bootstrap_rule(rule) -> bool`

Check if a rule was seeded by bootstrap (policy_hash == all zeros).

---

### Self-Compilation Bridge — `bizra-omega/bizra-memory/src/bridge.rs`

#### `ConversationTurnWire`

Wire format mirroring the Python `ConversationTurn` schema.

```rust
pub struct ConversationTurnWire {
    pub provider: String,          // "bizra_self"
    pub conversation_id: String,   // "session-{node_id}"
    pub turn_id: String,           // FNV-1a deterministic hash
    pub role: String,              // "user"
    pub content: ExtractionContent,
    pub timestamp: u64,
    pub model: String,             // "sovereign-node"
    pub kind: AtomKind,
    pub confidence: f32,
}
```

#### `export_atoms_as_turns(atoms, session_id) -> Vec<ConversationTurnWire>`

Convert memory atoms into conversation turns for stereoscopic compilation.

```rust
use bizra_memory::{export_atoms_as_turns, ConversationTurnWire};

let atoms = vec![
    (AtomKind::Fact, "I am a software engineer", 0.95f32, 1000u64),
];
let turns = export_atoms_as_turns(&atoms, "node-abc");
assert_eq!(turns[0].provider, "bizra_self");
```

---

### Action Receipt — `bizra-omega/bizra-agent/src/action_types.rs`

#### `ActionReceipt.outcome_hash`

Optional SHA-256 hash proving action outcome (not just attempt).

```rust
pub struct ActionReceipt {
    // ... existing fields ...
    pub outcome_hash: Option<[u8; 32]>, // NEW: proves action effect
}
```

**Serialization:** Format v=2 with backward compatibility. The `from_jsonl()` parser treats `outcome` as optional, allowing v1 receipts to load without it.

---

### Self-Compilation Trigger — `bizra-omega/bizra-node/src/node.rs`

#### `SELF_COMPILE_INTERVAL`

Constant set to `50`. Every 50 commands processed, the node triggers self-compilation.

#### `BizraNode::trigger_self_compilation()`

Queries the memory pipeline for all stored atoms, converts them to `ConversationTurnWire` via `export_atoms_as_turns()`, and logs the export count. Production path will write JSONL or call the Python engine via FFI.

---

## Desktop Bridge APIs

### AHK Bridge — `filedfs/ahk_bridge.ahk`

#### Screenshot Infrastructure

| Function | Description |
|----------|-------------|
| `CaptureScreenshotHash(label)` | Full-screen capture via GDI+, returns SHA-256 hex |
| `HashFile(path)` | SHA-256 of file bytes via Windows CNG |

#### Enhanced `HandleActuatorExecute`

Returns enriched response with verification fields:

```json
{
  "status": "ok",
  "pre_hash": "a1b2c3...",
  "post_hash": "d4e5f6...",
  "state_changed": true,
  "outcome_confirmed": true
}
```

### Desktop Bridge — `core/bridges/desktop_bridge.py`

#### `_verify_action_outcome(action_id, pre_hash, post_hash, intent, target)`

Intent-aware action verification with confidence scoring.

| Confidence | Condition |
|-----------|-----------|
| 0.95 | Full SHA-256 hashes with confirmed outcome |
| 0.60 | Partial hash data (one hash missing) |
| 0.50 | Timestamp-only fallback (no screenshots) |
