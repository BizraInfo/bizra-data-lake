# Phase 6D — Build, Test, Verify

## Purpose

Compile, lint, format, build release, and live test the sovereign cockpit.

---

## Build Sequence

```bash
cd /mnt/c/BIZRA-DATA-LAKE/bizra-omega

# Step 1: Format
cargo fmt -p bizra-cli

# Step 2: Clippy (zero warnings)
cargo clippy -p bizra-cli -- -D warnings

# Step 3: Workspace test (ensure no regressions)
cargo test -p bizra-cli

# Step 4: Release build
cargo build -p bizra-cli --release

# Step 5: Binary size check (should be < 3.5 MB)
ls -lh target/release/bizra

# Step 6: Live test
target/release/bizra tui
```

---

## Live Test Checklist

### Visual Verification

| Zone | Check | Expected |
|------|-------|----------|
| Header | Trust indicator visible | "● SOVEREIGN" in gold |
| Header | Model count visible | "N models" text |
| Parliament | PAT-7 agents listed | P0-P6 with icons and roles |
| Parliament | SAT-5 agents listed | S0-S4 with icons and roles |
| Ghost Feed | Greeting shown | Time-appropriate greeting |
| Ghost Feed | Runtime state | "Ready" in green |
| Ghost Feed | Recommendations | At least 1 recommendation |
| Trust Rail | Verdict banner | "SOVEREIGN" or "DEGRADED" |
| Trust Rail | 13 checks | All with ✓ or ✗ marks |
| Trust Rail | Categories | [Constitutional Law], [Topology], [Genesis], [Ledger], [Substrate] |
| Substrate | CPU info | Real CPU name + cores |
| Substrate | RAM | Real amount + usage % |
| Substrate | GPU | Real GPU if present |
| Substrate | Models | Count + text/vision split |
| Receipt Rail | Chain status | Valid/broken indicator |
| Receipt Rail | Today summary | Count + complete |
| Receipt Rail | Recent list | Last N receipts with IDs |
| Receipt Rail | Manifest seal | BLAKE3 hex if receipts exist |
| Status Bar | Manifest summary | "N/N✓ today" |

### Interaction Verification

| Key | Expected |
|-----|----------|
| `q` | Quit TUI cleanly |
| `Tab` | Cycle through views |
| `1` | Return to Dashboard |
| `r` | Refresh data (status message appears) |
| `2` | Show Agents view (existing) |
| `3` | Show Chat view (existing) |

### Edge Cases

| Scenario | Expected |
|----------|----------|
| No receipts in ledger | Receipt rail shows "No receipts yet" |
| No models installed | Ghost recommends installing models |
| No GPU detected | Substrate omits GPU line |
| Terminal < 80 cols | Graceful degradation (no panic) |
| Terminal < 24 rows | Panels truncate (no panic) |

---

## Regression Checks

```bash
# Existing CLI commands still work:
target/release/bizra init
target/release/bizra genesis
target/release/bizra agents
target/release/bizra node
target/release/bizra trust
target/release/bizra manifest
target/release/bizra receipt --verify
target/release/bizra brief
target/release/bizra help
```

All 11 CLI commands must still function identically.
The TUI changes should not affect any non-TUI code paths.

---

## Quality Gates (must-pass before commit)

1. `cargo fmt -p bizra-cli -- --check` → no diffs
2. `cargo clippy -p bizra-cli -- -D warnings` → 0 warnings
3. `cargo test -p bizra-cli` → all pass
4. `cargo build -p bizra-cli --release` → clean build
5. `ls -lh target/release/bizra` → < 3.5 MB
6. `target/release/bizra tui` → 7-zone cockpit renders with real data
7. `target/release/bizra brief` → unchanged output (no regression)

---

## Commit Message Template

```
feat(cli): Phase 6 — Living Terminal Cockpit

7-zone sovereign mission control wired to proven CLI backends:
  [1] Header — trust verdict + model count
  [2] Parliament — PAT-7 + SAT-5 live roster
  [3] Ghost Feed — greeting, runtime, recommendations
  [4] Trust Rail — 13-check constitutional surface
  [5] Substrate — CPU/RAM/GPU/models
  [6] Receipt Rail — chain integrity + manifest seal
  [7] Status Bar — manifest summary

Data layer: gather_dashboard_data() collects all truth in one pass.
5-second periodic refresh + manual refresh (r key).
5 new widgets: ParliamentPanel, GhostFeed, TrustRail, SubstratePanel, ReceiptRail.
No new crates. No new authority. Same backends as CLI commands.
```
