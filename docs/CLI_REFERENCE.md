# BIZRA CLI Reference

Last updated: 2026-03-14

## Entry Point

```bash
python -m core.sovereign [command] [options]
```

All commands run through the Sovereign Engine unified CLI.

## Commands

### Node Lifecycle

| Command | Description |
|---------|-------------|
| `activate` | Full genesis activation: ceremony + orchestrator + heartbeat |
| `genesis` | Bootstrap a new BIZRA node (one-command genesis) |
| `onboard` | Create sovereign identity (interactive wizard) |
| `status` | Show system status and health |
| `doctor` | Run system health check |
| `version` | Show version |

### Queries and Missions

| Command | Description |
|---------|-------------|
| `query "..."` | Run a single query through the reasoning engine |
| `mission "..."` | Execute a mission with full pipeline |
| *(no command)* | Interactive REPL mode |

### Server and Gateway

| Command | Description |
|---------|-------------|
| `serve` | Run API server |
| `gateway` | Run messaging gateway (Telegram) |
| `bridge start` | Start desktop bridge (AHK IPC) |
| `bridge ping` | Ping the running bridge |
| `bridge status` | Get bridge status |

### Token Economy

| Command | Description |
|---------|-------------|
| `wallet` | View token wallet balances |
| `tokens` | View token supply and stats |

### Sovereignty

| Command | Description |
|---------|-------------|
| `sovereignty init` | Generate keypair, sign the Covenant |
| `sovereignty work` | Do verified work, earn SEED |
| `sovereignty attest` | Vouch for another node's work |
| `sovereignty status` | See your sovereign state |
| `sovereignty ledger` | Show event ledger |
| `sovereignty reset` | Delete node (irreversible) |

### Tools

| Command | Description |
|---------|-------------|
| `dashboard` | View node identity |
| `impact` | Sovereignty progression tracker |
| `hunter scan` | Scan for bounty vulnerabilities |
| `hunter report` | Generate vulnerability report |
| `hunter list` | List previous scan results |
| `test` | Run integration tests |

---

## Command Details

### activate

Full genesis activation pipeline: ceremony (BLAKE3 cryptographic identity) -> orchestrator (12-step bootstrap) -> heartbeat (boot + first breath) -> activation receipt (evidence artifact).

```bash
# With seed phrase (deterministic, reproducible)
python -m core.sovereign activate --seed-phrase "my-seed"

# With seed file
python -m core.sovereign activate --seed-file ~/.bizra/seed.bin

# Skip first breath (boot only)
python -m core.sovereign activate --seed-phrase "my-seed" --skip-breath

# Skip orchestrator (ceremony + heartbeat only)
python -m core.sovereign activate --skip-orchestrator

# Verify existing activation artifacts
python -m core.sovereign activate --verify --data-dir sovereign_state/genesis

# Output as JSON
python -m core.sovereign activate --seed-phrase "my-seed" --json
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--seed-file PATH` | — | Path to 32+ byte seed file |
| `--seed-phrase PHRASE` | — | Passphrase to derive seed via BLAKE3 |
| `--data-dir DIR` | `sovereign_state/genesis` | Output directory for genesis artifacts |
| `--skip-orchestrator` | false | Skip 12-step orchestrator |
| `--skip-breath` | false | Skip first breath (boot only) |
| `--verify` | false | Verify existing artifacts instead of creating |
| `--json` | false | Output result as JSON |

**Output artifacts** (written to `--data-dir`):

- `node0_genesis.json` — Full ceremony output
- `genesis_hash.txt` — Root Merkle hash
- `pat_roster.txt` — 7 PAT agents
- `sat_roster.txt` — 5 SAT agents
- `activation_receipt.json` — Composite receipt with all sub-hashes
- `boot_receipt.json` — Node0 boot receipt

### genesis

Bootstrap a new BIZRA node with configurable agent counts, hardware scanning, and guild membership.

```bash
# Minimal genesis
python -m core.sovereign genesis --identity-genesis --pat-7 --sat-5

# Full genesis with all options
python -m core.sovereign genesis --identity-genesis --hardware-scan \
    --pat-7 --sat-5 --hda-bridge \
    --mobile-pair "Z Fold 6:SM-F956B" \
    --guild-join agriculture \
    --quest-accept 001-sustainable-water \
    --ihsan-target 0.999

# Full SAT-49 operating profile
python -m core.sovereign genesis --identity-genesis --sat-49

# Strict bootstrap (fail-closed, default)
python -m core.sovereign genesis --identity-genesis --strict-bootstrap

# Degraded mode (diagnostic only)
python -m core.sovereign genesis --identity-genesis --allow-degraded
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--identity-genesis` | false | Mint genesis identity (Node0 = Block0) |
| `--hardware-scan` | false | Scan and fingerprint local hardware |
| `--pat-7` | false | Instantiate 7 PAT agents |
| `--sat-5` | false | Instantiate 5 SAT agents |
| `--sat-49` | false | Full SAT-49 operating profile |
| `--pat-count N` | 7 | Custom PAT agent count |
| `--sat-count N` | 5 | Custom SAT agent count |
| `--sat-mode` | auto | SAT profile: `mini5` or `full49` |
| `--hda-bridge` | false | Initialize AutoHotkey-Rust IPC bridge |
| `--mobile-pair DEVICE` | — | Mobile device to pair |
| `--guild-join GUILD` | — | Guild to join |
| `--quest-accept QUEST` | — | Quest to accept |
| `--ihsan-target FLOAT` | 0.999 | Ihsan excellence target |
| `--strict-bootstrap` | true | Fail-closed: reject deferred/stub steps |
| `--allow-degraded` | false | Allow degraded steps (diagnostic only) |
| `--architect NAME` | MoMo | Genesis architect name |
| `--json` | false | Output as JSON |

### serve

```bash
python -m core.sovereign serve --port 8080
python -m core.sovereign serve --host 0.0.0.0 --port 8080
```

### query

```bash
python -m core.sovereign query "What is sovereignty?"
python -m core.sovereign query "Explain the FATE gate" --json
```

### mission

```bash
python -m core.sovereign mission "Analyze security posture"
python -m core.sovereign mission "Build weekly report" --json
```

---

## Node0 Activation Script

For operational use, the standalone activation script provides additional capabilities:

```bash
python scripts/node0_activate.py              # Start full node
python scripts/node0_activate.py status       # Check status + auto-load LLM fleet
python scripts/node0_activate.py mission "task"  # Assign mission
```

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `BIZRA_DATA_LAKE_ROOT` | Root path (auto-detected) |
| `BIZRA_BRIDGE_PORT` | Desktop bridge port (default: 9742) |
| `BIZRA_HDA_PORT` | AHK HDA port (default: 9743) |
| `BIZRA_BRIDGE_TOKEN` | Bridge authentication token |
| `BIZRA_AUTH_ALLOW_ANONYMOUS` | Allow anonymous API access (testing) |
| `LMSTUDIO_HOST` | LM Studio host (auto-detected from WSL gateway) |

---

## Constitutional Thresholds

All thresholds enforced at runtime (source: `core/integration/constants.py`):

| Threshold | Value | Context |
|-----------|-------|---------|
| Ihsan (production) | 0.95 | Minimum excellence score |
| Ihsan (strict) | 0.99 | Consensus/critical path |
| SNR (minimum) | 0.85 | Signal-to-noise floor |
| SNR (elite) | 0.98 | T0 operations |
| ADL Gini | <= 0.35 | Justice distribution gate |
| Zakat | 2.5% | Annual token deduction |
| Harberger | 5% | Annual self-assessed tax |
