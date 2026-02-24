# Cold-Start Bootstrap Integration Guide

> **Version:** 1.0.0 | **Date:** 2026-02-22
> **BIZRA** (بذرة): Proactive Dynamic DDAGI OS — every human is a node, every node is a seed, every seed has infinite potential.
> **Audience:** Engineers integrating with or extending the cold-start bootstrap system

## Quick Start

### For a New User (Onboarding Flow)

```
1. User opens BIZRA -> OnboardingFlow.jsx renders
2. Step 1 (Verify): Identity keypair generated
3. Step 2 (Provider): Optional conversation import
4. Step 3 (Teach): 7-question interview -> 25-35 TEACH atoms
5. Step 4 (Chat): First sovereign message
6. Step 5 (Dashboard): KnowsMeGauge + QualityTierBadge shown
```

### For Developers (Running Tests)

```bash
# Python (core + scripts)
pytest tests/core/ tests/scripts/ -x -q --timeout=60

# Rust (full workspace)
cd bizra-omega && cargo test --workspace --release

# Atlas tier report
python scripts/atlas/atlas_gap_report.py --user-tier seed
```

## Integration Points

### 1. Onboarding -> Memory Pipeline

The `TeachStep.jsx` interview generates TEACH atoms that flow into the Rust memory pipeline.

```
TeachStep.jsx                    Rust Runtime
-----------                      ------------
User answers 7 questions    ->   handle_teach() [handler.rs:100]
Each answer parsed into     ->   Atoms stored in memory pipeline
  1-5 atoms per question    ->   ProfileSnapshot updated
Total: 25-35 atoms          ->   knows_me_score = section_count / 8.0
```

**Protocol format:** `TEACH\t{kind}\t{content}\t{ihsan}\t{timestamp}`

**10 atom kinds:** `fact`, `preference`, `pattern`, `relationship`, `goal`, `expertise`, `context`, `principle`, `temporal`, `negation`

### 2. Memory Pipeline -> Self-Compilation

Every 50 commands, the Rust node triggers self-compilation.

```
bizra-node/src/node.rs           bizra-memory/src/bridge.rs
--------------------             -------------------------
execute() increments counter ->  (every 50th command)
trigger_self_compilation()   ->  export_atoms_as_turns()
  reads all stored atoms     ->  Vec<ConversationTurnWire>
  groups by AtomKind         ->  provider="bizra_self"
                             ->  model="sovereign-node"

                                 bizra-normalizers/engine.py
                                 ---------------------------
                             ->  compile_from_atoms()
                             ->  ConversationTurn objects
                             ->  self.compile() produces
                             ->  StereoscopicReport
```

### 3. Genesis Gate -> Tier Progression

The gate uses `NodeMaturityStage` to determine thresholds.

```python
# In compile_stereoscopic_graph.py
# CLI: python compile_stereoscopic_graph.py --gate-profile seed

from genesis_gate import GenesisGateConfig, NodeMaturityStage

# Determine stage from node metrics
stage = determine_maturity_stage(
    atom_count=15,     # from TEACH + extraction
    message_count=5,   # from command history
    provider_count=0,  # from import count
)
# -> NodeMaturityStage.SPROUT

# Get gate config for this stage
config = GenesisGateConfig.for_stage(stage)
# -> min_cv=0.0, min_nodes=1, min_elite_nodes=0
```

### 4. Token Economy -> Node Provisioning

The `OnboardingWizard` now issues a genesis SEED grant during onboarding.

```python
# In core/pat/onboarding.py

class OnboardingWizard:
    def __init__(self, template: Optional[NodeTemplate] = None):
        self._template = template or NodeTemplate.default()

    def onboard(self, display_name, ...):
        # Steps 1-6: identity, agents, credentials, first query...
        # Step 7 (NEW): Issue genesis grant
        self._issue_genesis_grant(node_id)
        # -> TokenLedger.genesis_grant(node_id, template.initial_seed_grant)
        # -> GENESIS_MINT from __UBC_POOL__
```

### 5. Bootstrap Reflexes -> Runtime

Bootstrap reflexes are loaded during `ReflexCache` initialization when the cache is empty.

```rust
// In bizra-agent/src/runtime.rs (init path)
let mut reflex_cache = ReflexCache::new();
let loaded = reflex_cache.load_bootstrap_rules();
// loaded == 4 on first run, 0 on subsequent runs

// During message routing:
// "Hello!" -> trigger hash matches bootstrap:greeting
//          -> routed to Diplomat agent directly
//          -> skips full 7-agent orchestration
//          -> ~10x faster first response
```

### 6. AHK Perception-Action Loop

```
Agent Decision         Action Bus            AHK Bridge
--------------         ----------            ----------
Decide action     ->   Validate Permit  ->   Pre-screenshot hash
                                         ->   Execute intent
                                         ->   100ms settle delay
                                         ->   Post-screenshot hash
                                         ->   state_changed?
                                         ->   outcome_confirmed?

Desktop Bridge         Event Bus             Memory
--------------         ---------             ------
verify_action_outcome  PostDeliver hook  ->   Learn from receipt
confidence scoring     action.receipt    ->   Memory extraction
outcome_hash           fires callback   ->   Feed back to pipeline
```

### 7. Custom Provider Integration

```python
from normalizers import register_provider
from normalizers.base import GenericJsonlParser, GenericOpenAIParser

# For JSONL exports (one JSON object per line)
register_provider("lm_studio", GenericJsonlParser(
    role_field="role",
    content_field="content",
    model_field="model",
    timestamp_field="created_at",
))

# For OpenAI-compatible API format
register_provider("ollama", GenericOpenAIParser())

# Check what's registered
from normalizers import registered_providers, custom_providers
print(registered_providers())   # [..., "lm_studio", "ollama"]
print(custom_providers())       # {"lm_studio": ..., "ollama": ...}
```

### 8. Frontend Components

#### KnowsMeGauge (App.jsx)

8-segment radial gauge in the right sidebar. Segments light up as profile sections are populated.

| Segment | Color | Profile Section |
|---------|-------|----------------|
| Facts | `#60A5FA` (blue) | Factual assertions |
| Preferences | `#A78BFA` (purple) | User preferences |
| Goals | `#FB923C` (orange) | Stated goals |
| Expertise | `#22D3EE` (cyan) | Domain expertise |
| Patterns | `#FBBF24` (gold) | Behavioral patterns |
| Relationships | `#34D399` (green) | Social graph |
| Principles | `#F0D68A` (gold) | Core principles |
| Context | `#F472B6` (pink) | Situational context |

#### QualityTierBadge (App.jsx)

Color-coded tier badge with capability chips and growth roadmap.

| Tier | Color | Hex |
|------|-------|-----|
| Seed | Brown | `#8B7355` |
| Sprout | Green | `#4CAF50` |
| Growing | Blue | `#2196F3` |
| Rooted | Purple | `#9C27B0` |
| Flourishing | Gold | `#FFD700` |

#### GrowthRoadmap (App.jsx)

Expandable vertical timeline showing all 5 tiers with current position highlighted.

## File Reference

### Modified Files (Complete List)

```
Python:
  bizra-normalizers/genesis_gate.py          # Tiered gate + NodeMaturityStage
  bizra-normalizers/compile_stereoscopic_graph.py  # --gate-profile CLI
  bizra-normalizers/normalizers/__init__.py   # register_provider()
  bizra-normalizers/normalizers/base.py       # GenericJsonlParser, GenericOpenAIParser
  bizra-normalizers/engine.py                 # compile_from_atoms()
  core/pat/onboarding.py                      # NodeTemplate + genesis grant
  core/token/ledger.py                        # genesis_grant()
  core/bridges/desktop_bridge.py              # verify_action_outcome()
  scripts/atlas/atlas_gap_report.py           # user_tier_report()

Rust:
  bizra-omega/bizra-agent/src/reflex_cache.rs    # Bootstrap reflexes
  bizra-omega/bizra-memory/src/bridge.rs         # ConversationTurnWire
  bizra-omega/bizra-memory/src/lib.rs            # Re-exports
  bizra-omega/bizra-node/src/node.rs             # Self-compilation trigger
  bizra-omega/bizra-agent/src/action_types.rs    # outcome_hash field
  bizra-omega/bizra-node/src/action_executor.rs  # Audit log update

Frontend:
  filedfs/onboarding/steps/TeachStep.jsx     # 7-question interview
  filedfs/App.jsx                            # KnowsMeGauge + QualityTierBadge
  filedfs/onboarding/steps/DashboardStep.jsx # Tier badge on completion
  filedfs/ahk_bridge.ahk                     # Screenshot capture

Tests:
  tests/scripts/test_atlas_gap_report.py     # 14 new tests
  scripts/atlas/__init__.py                  # Package init (new)
```

### Not Modified (Preserved)

- SAP v0 protocol, receipt chain format
- Event Bus architecture, TeleScript 9 primitives
- Token constants (ZAKAT_RATE=0.025, HARBERGER_TAX_RATE, supply caps)
- Existing 10 provider parsers
- Genesis seed format
- HHMM temporal layers

## Error Handling

| Scenario | Behavior |
|----------|----------|
| `genesis_grant()` with empty `node_id` | Raises `ValueError` |
| `genesis_grant()` with negative `amount` | Raises `ValueError` |
| `user_tier_report()` with invalid tier | Raises `ValueError` |
| Bootstrap rules on populated cache | Returns 0 (no-op) |
| Self-compilation on empty pipeline | Logs "no atoms", returns |
| AHK screenshot failure | Falls back to timestamp-based hash |
| Custom provider name collision | Raises `ValueError` |
| v1 receipt without outcome_hash | Loads with `outcome_hash: None` |
