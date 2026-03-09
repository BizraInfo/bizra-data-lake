# BIZRA Universal Installer v2.0 — Integration Guide

## Files Created

The following new Rust modules implement the v2.0 spec:

### Source Modules (`bizra-omega/bizra-installer/src/`)

| File | Spec Section | Description |
|------|-------------|-------------|
| `device_profile.rs` | §5.1-5.3 | Full `DeviceProfile` (OS, arch, CPU, RAM, GPU API, disk, locale, screen, touch), 7-tier `ModelTier`, `InstallFootprint`, detection functions |
| `i18n.rs` | §4 | Language tiers (1/2/3), 17 locales (10 Tier-1, 7 Tier-2), RTL support, `I18nManager` with fallback chain, `StringBundle` JSON loader |
| `health_check.rs` | §16 | 7-point health check: runtime, model, identity, ledger, agents, language packs, disk space. `HealthCheckReport` with pass/fail/warn |
| `install_receipt.rs` | §17 | SHA-256 hash-chained `InstallReceipt`, `DeviceSummary`, `ModelSelection`, `InstalledComponent`. JSONL append storage |
| `install_flow.rs` | §3.2, §6-12 | 6-step state machine: DETECT→GREET→ADAPT→INSTALL→IDENTITY→ALIVE. `InstallState`, `InstallOptions`, step executors |
| `urp.rs` | §20 | `ResourcePledge`, `URPState`, SEED minting with 2.5% Zakat floor, resource scoring, `ADL_GINI_THRESHOLD=0.35` |
| `profiles.rs` | §15 | `UserProfile`, `ProfileRegistry` (max 8), PIN with salted SHA-256, constant-time comparison, create/switch/remove |
| `self_update.rs` | §14 | `UpdateManifest`, delta patch vs full replace strategy, semver comparison, SHA-256 verification |

### Locale Bundles (`bizra-omega/bizra-installer/locales/`)

| File | Content |
|------|---------|
| `en/installer.json` | 48 English installer strings |
| `ar/installer.json` | 48 Arabic installer strings (native, RTL) |
| `en/errors.json` | 14 English error messages (E001-E014) |
| `ar/errors.json` | 14 Arabic error messages (E001-E014) |

### Integration Tests (`bizra-omega/bizra-installer/tests/`)

| File | Tests |
|------|-------|
| `installer_v2_tests.rs` | 10 integration tests: full flow, Arabic RTL, micro-node, model fallback, URP seed/zakat, multi-user profiles, receipt chains, self-update strategy, i18n coverage |

## Required Manual Edits

### 1. `src/lib.rs` — Add new module declarations

```rust
// Existing modules (keep as-is)
pub mod alpha100;
pub mod binary_fetch;
pub mod config;
pub mod policy;
pub mod provider;

// NEW: Universal Installer v2.0 modules
pub mod device_profile;
pub mod health_check;
pub mod i18n;
pub mod install_flow;
pub mod install_receipt;
pub mod profiles;
pub mod self_update;
pub mod urp;
```

### 2. `Cargo.toml` — No changes needed

All new modules use dependencies already in `Cargo.toml`:
- `serde`, `serde_json` — serialization
- `sha2` — SHA-256 hashing
- `chrono` — timestamps
- `dirs` — home directory detection

### 3. `src/main.rs` — Add new CLI subcommands (optional, Phase II)

The install flow can be integrated as a new `Install` subcommand group:

```rust
// In the Cli enum, add:
#[command(subcommand)]
Install(InstallCommands),

// Where:
#[derive(Subcommand)]
enum InstallCommands {
    /// Run the full 6-step install flow
    Start {
        #[arg(long)]
        locale: Option<String>,
        #[arg(long)]
        offline: bool,
        #[arg(long)]
        non_interactive: bool,
    },
    /// Run post-install health check
    HealthCheck,
    /// Show/manage profiles
    Profile {
        #[command(subcommand)]
        action: ProfileAction,
    },
    /// Check for updates
    Update {
        #[arg(long)]
        check_only: bool,
    },
}
```

## Test Verification

After applying `lib.rs` edits:

```bash
cd bizra-omega
cargo test -p bizra-installer --lib        # Unit tests (in each module)
cargo test -p bizra-installer --test installer_v2_tests  # Integration tests
cargo clippy -p bizra-installer -- -D warnings
```

## Module Dependency Graph

```
install_flow
├── device_profile (detect_device, DeviceProfile, ModelTier)
├── i18n (resolve_locale, I18nManager)
├── health_check (run_health_check, HealthCheckReport)
└── install_receipt (InstallReceipt, DeviceSummary, ModelSelection)

urp   (standalone — no internal deps)
profiles (standalone — no internal deps)
self_update (standalone — no internal deps)
```

## Constitutional Compliance

- **Ihsān gate**: Health check enforces ≥0.95 quality before ALIVE
- **Adl Gini**: URP has `ADL_GINI_THRESHOLD = 0.35` + `ZAKAT_RATE = 0.025`
- **Fail-closed**: Missing health checks = install blocked
- **Evidence chain**: Every install produces hash-chained `InstallReceipt`
- **Language sovereignty**: Arabic is Tier-1, RTL-native from first interaction
- **Security**: PIN verification uses constant-time comparison, SHA-256 salted
