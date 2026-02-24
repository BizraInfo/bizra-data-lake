# Phase 25: Genesis Bootstrap Pipeline

> One-command BIZRA node genesis — from identity to sovereignty in 12 steps.

## Context

Every BIZRA node begins with a Genesis ceremony. The `GenesisOrchestrator` wires together 12 isolated steps into a single sequential pipeline: identity minting, hardware scan, PAT/SAT activation, token allocation, URP pledge, HDA bridge, mobile pairing, guild join, quest acceptance, Ihsan targeting, and state persistence.

Each step is isolated — failure in one does not block subsequent steps. Every step records timing and details, producing an auditable `GenesisResult` receipt with deterministic SHA-256 hash.

Standing on Giants: Nakamoto (2008) — genesis block, Lamport (1978) — ordered execution, Shannon (1948) — SNR quality, Al-Ghazali — Ihsan floor.

## Package Structure

```
core/genesis/
  __init__.py            # 85 lines — package exports
  types.py               # 147 lines — GenesisConfig, GenesisResult, GenesisStep
  orchestrator.py        # 462 lines — 12-step pipeline
  hardware.py            # 170 lines — HardwareScanner (RAM, VRAM, GPU)
  state_persistence.py   # 473 lines — SovereignState JSON persistence
  urp.py                 # 94 lines — Universal Resource Pledge
  mobile_pairing.py      # 76 lines — Mobile device pairing stub
  cli.py                 # 180 lines — CLI interface
```

Total: ~1,687 lines.

## Data Types

### GenesisConfig

Maps 1:1 to CLI flags from the dream command.

```
DATACLASS GenesisConfig:
  identity_genesis: bool = False       # Mint Ed25519 keypair + identity card
  hardware_scan: bool = False          # Detect RAM, VRAM, GPU
  pat_count: int = 7                   # Personal Agentic Taskforce count
  sat_count: int = 5                   # Sovereign Agent Taskforce count
  hda_bridge: bool = False             # AutoHotkey-Rust IPC bridge
  mobile_pair: Optional[str] = None    # "Z Fold 6:SM-F956B"
  guild_join: Optional[str] = None     # "agriculture"
  quest_accept: Optional[str] = None   # "001-sustainable-water"
  ihsan_target: float = 0.999          # Constitutional trajectory target
  architect_name: str = "MoMo"         # Node architect identity
```

### GenesisStep

```
DATACLASS GenesisStep:
  name: str                           # Step identifier
  status: PENDING | RUNNING | SUCCESS | FAILED | SKIPPED
  duration_ms: float                  # Execution timing
  details: Dict[str, Any]            # Step-specific outputs
  error: Optional[str]               # Error message if failed
```

### GenesisResult

```
DATACLASS GenesisResult:
  steps: List[GenesisStep]           # All 12 steps
  node_id: str                       # Minted node ID
  genesis_hash: str                  # SHA-256 receipt hash (first 16 hex chars)
  total_duration_ms: float
  success: bool                      # True if failed_steps == 0
  created_at: str                    # ISO 8601 UTC

  PROPERTY successful_steps -> int
  PROPERTY failed_steps -> int
  PROPERTY skipped_steps -> int

  METHOD compute_hash():
    payload = JSON.dumps({node_id, [successful step names], created_at})
    genesis_hash = SHA256(payload)[:16]
```

## 12-Step Pipeline

```
FUNCTION orchestrator.run() -> GenesisResult:
  result = GenesisResult()
  start = monotonic()

  # Each step guarded by config flags and wrapped in error isolation
  FOR step_name, step_fn, condition IN [
    ("identity_genesis",  _step_identity_genesis,  config.identity_genesis),
    ("hardware_scan",     _step_hardware_scan,      config.hardware_scan),
    ("pat_activation",    _step_pat_activation,     ALWAYS),
    ("sat_activation",    _step_sat_activation,     ALWAYS),
    ("token_allocation",  _step_token_allocation,   ALWAYS),
    ("urp_pledge",        _step_urp_pledge,         hardware_info IS NOT None),
    ("hda_bridge",        _step_hda_bridge,         config.hda_bridge),
    ("mobile_pair",       _step_mobile_pair,        config.mobile_pair),
    ("guild_join",        _step_guild_join,          config.guild_join),
    ("quest_accept",      _step_quest_accept,        config.quest_accept),
    ("ihsan_target",      _step_ihsan_target,        ALWAYS),
    ("state_persist",     _step_state_persist,       ALWAYS),
  ]:
    IF condition:
      step = _run_step(step_name, step_fn)  # Isolated try/except
      result.steps.append(step)

  result.total_duration_ms = (monotonic() - start) * 1000
  result.success = result.failed_steps == 0
  result.compute_hash()
  RETURN result
```

### Step Details

| Step | Module | Imports | Key Output |
|------|--------|---------|------------|
| 1. Identity Genesis | `core.pat.minting` | `generate_identity_keypair`, `mint_genesis_node` | node_id, genesis_hash, sovereignty_score |
| 2. Hardware Scan | `core.genesis.hardware` | `HardwareScanner.scan()` | ram_gb, vram_gb, gpu name |
| 3. PAT Activation | (confirmation only) | — | pat_count, ihsan |
| 4. SAT Activation | (confirmation only) | — | sat_count, urp_pledged |
| 5. Token Allocation | `core.token.mint` | `TokenMinter.genesis_mint()` | receipt count (deferred if unavailable) |
| 6. URP Pledge | `core.genesis.urp` | `pledge_resources()` | ram + vram pledged |
| 7. HDA Bridge | `core.bridges` | `SovereignBridge` | bridge status |
| 8. Mobile Pairing | `core.genesis.mobile_pairing` | `pair_mobile()` | device_name, routing |
| 9. Guild Join | `core.guild.registry` | `GuildRegistry.join_guild()` | guild name, online count |
| 10. Quest Accept | `core.quest.engine` | `QuestEngine.accept_quest()` | quest_id, reward |
| 11. Ihsan Target | (computation only) | — | target, current, trajectory, estimated_cycles |
| 12. State Persist | `core.genesis.state_persistence` | `save_sovereign_state()` | state_dir, files_written |

### Error Isolation Pattern

```
FUNCTION _run_step(name, step_fn) -> GenesisStep:
  step = GenesisStep(name=name, status=RUNNING)
  start = monotonic()
  TRY:
    details = step_fn()
    step.status = SUCCESS
    step.details = details
  EXCEPT Exception as e:
    step.status = FAILED
    step.error = str(e)
    log.warning("Step '%s' failed: %s", name, e)
  step.duration_ms = elapsed
  RETURN step
```

## State Persistence

Written to `sovereign_state/genesis/`:

```
sovereign_state/genesis/
  identity.json          # Signed identity card
  pat_manifest.json      # PAT-7 agent roster
  sat_manifest.json      # SAT-5 agent roster
  urp_pledge.json        # Universal Resource Pledge
  hardware.json          # Hardware fingerprint
  genesis_receipt.json   # Full ceremony record
  recovery_phrase.txt    # BIP39-style 24-word phrase (DELETE AFTER COPYING)
  .keystore/
    sovereign.enc        # Encrypted private key (PBKDF2 + XOR)
```

All files: 0o600 permissions. `.keystore/sovereign.enc`: owner read/write only.

## Terminal Output Formatting

The `format_output()` method produces dream CLI output:

```
  Genesis block minted: 0xabcdef1234... (BIZRA-NODE0-GENESIS)
  Hardware scanned: 128GB RAM, 16GB VRAM (RTX 4090)
  PAT-7 instantiated: 42ms latency, Ihsan 0.98
  SAT-5 active: URP 128GB + 16GB VRAM pledged
  Token genesis allocation complete
  URP pledge: 128GB RAM + 16GB VRAM
  HDA bridge: AutoHotkey<->Rust IPC ready
  Z Fold 6 paired: proximity routing enabled
  Guild joined: #agriculture (3 nodes online)
  Quest accepted: "001-sustainable-water" (reward 50 $IMP)
  Ihsan target: 0.999 (current 0.98, trajectory +0.003/cycle)
  State persisted: sovereign_state/genesis

BIZRA Omega-v7.0 LIVE. You are Node0. The forest grows when you do.
```

## Cross-Module Dependencies

```
core.genesis
  imports -> core.pat.identity_card     (keypair generation)
  imports -> core.pat.minting           (node minting)
  imports -> core.token.mint            (token allocation)
  imports -> core.bridges               (HDA bridge check)
  imports -> core.guild.registry        (guild join)
  imports -> core.quest.engine          (quest accept)
  imports -> core.integration.constants (Ihsan thresholds)
```

## TDD Anchors

### test_genesis_orchestrator.py

```
TEST full_pipeline_all_steps_succeed:
  config = GenesisConfig(identity_genesis=True, hardware_scan=True,
                         guild_join="agriculture", quest_accept="001-sustainable-water")
  result = GenesisOrchestrator(config).run()
  ASSERT result.success == True
  ASSERT result.failed_steps == 0
  ASSERT result.genesis_hash != ""
  ASSERT len(result.genesis_hash) == 16

TEST step_isolation_failure_does_not_block:
  # With identity_genesis=True but no core.pat available
  config = GenesisConfig(identity_genesis=True)
  result = GenesisOrchestrator(config).run()
  # Identity step fails, but remaining steps still run
  ASSERT len(result.steps) >= 4  # pat, sat, token, ihsan, persist still run

TEST config_flags_control_steps:
  config = GenesisConfig()  # All defaults (minimal)
  result = GenesisOrchestrator(config).run()
  step_names = [s.name for s in result.steps]
  ASSERT "identity_genesis" NOT IN step_names
  ASSERT "hardware_scan" NOT IN step_names
  ASSERT "pat_activation" IN step_names  # Always runs

TEST genesis_hash_is_deterministic:
  config = GenesisConfig()
  result1 = GenesisOrchestrator(config).run()
  result2 = GenesisOrchestrator(config).run()
  # Hashes differ because created_at differs
  # But same result re-hashed is stable
  hash1 = result1.compute_hash()
  hash2 = result1.compute_hash()
  ASSERT hash1 == hash2

TEST format_output_contains_all_steps:
  config = GenesisConfig()
  orch = GenesisOrchestrator(config)
  result = orch.run()
  output = orch.format_output(result)
  ASSERT "BIZRA" IN output
  ASSERT "Ihsan" IN output or "target" IN output

TEST ihsan_trajectory_computation:
  config = GenesisConfig(ihsan_target=0.999)
  orch = GenesisOrchestrator(config)
  orch._current_ihsan = 0.98
  details = orch._step_ihsan_target()
  ASSERT details["target"] == 0.999
  ASSERT details["estimated_cycles"] > 0
```

### test_state_persistence.py (existing — 27 tests)

Already covers save/load/encryption/phrase roundtrips.

## Edge Cases

- Token allocation failure is non-critical — returns `{"receipts": 0, "success": True, "note": "deferred"}`
- URP pledge skipped if hardware scan didn't run (no hardware info)
- Mobile pairing is optional (`config.mobile_pair = None` skips step)
- State persist creates directory if it doesn't exist
- Duplicate guild membership returns success with "Already a member" message
