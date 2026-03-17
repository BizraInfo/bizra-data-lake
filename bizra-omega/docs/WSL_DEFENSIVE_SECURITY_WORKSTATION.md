# WSL Defensive Security Workstation

This repository is running in **Ubuntu 24.04 on WSL2**. Use this setup to build a professional, authorized **defensive AppSec and audit workstation** for BIZRA. It is intentionally scoped to hardening, testing, analysis, supply-chain verification, and observability, not offensive exploitation.

## Current Baseline
Detected locally:

- Present: `rustc`, `cargo`, `python3`, `pip3`, `docker`, `gh`, `node`, `npm`, `nmap`, `jq`, `cargo-audit`
- Missing from the baseline: `shellcheck`, `cargo-llvm-cov`, `cargo-nextest`, `cargo-deny`, `cargo-fuzz`, `cargo-machete`, `cargo-outdated`, `semgrep`

## Bootstrap
Run the repo bootstrap script:

```bash
scripts/ops/bootstrap_defensive_security_wsl.sh
```

Useful flags:

```bash
scripts/ops/bootstrap_defensive_security_wsl.sh --base-only
scripts/ops/bootstrap_defensive_security_wsl.sh --skip-semgrep
```

The script installs:

- System packages: `clang`, `llvm`, `libz3-dev`, `shellcheck`, `ripgrep`, `fd-find`, `hyperfine`, `strace`, `lsof`, `pipx`
- Rust tooling: `cargo-audit`, `cargo-deny`, `cargo-nextest`, `cargo-llvm-cov`, `cargo-fuzz`, `cargo-outdated`, `cargo-machete`
- Python tooling: `semgrep` via `pipx`

## Recommended Security Workflow
Use these commands as the standard BIZRA security-quality loop:

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo audit
cargo llvm-cov --workspace --lcov --output-path lcov.info
semgrep --config auto .
```

For Hunter-specific work:

```bash
cargo test -p bizra-hunter
cargo bench -p bizra-hunter
```

## Safe Scope
Allowed focus:

- Secure coding
- Dependency and supply-chain review
- Coverage, fuzzing, and regression testing
- Observability and audit trails
- Authorized vulnerability analysis and non-weaponized proof generation

Out of scope for this workstation:

- Exploit kits
- Credential attacks
- Payload generation for intrusion
- Stealth, persistence, or unauthorized access tooling

## Next Repo Upgrades
After the bootstrap, the highest-value next steps are:

1. Wire `cargo-nextest`, coverage, Semgrep, and `cargo-deny` into CI.
2. Add SBOM/provenance generation to the release pipeline.
3. Replace deploy placeholders with real staging smoke tests and rollback.
4. Close the P0 coverage gaps documented in `docs/TEST_EXPANSION_PLAN.md`.
