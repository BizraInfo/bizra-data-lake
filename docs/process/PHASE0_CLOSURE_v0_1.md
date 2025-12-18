# Phase-0 Integrity Closure v0.1

Truth: MEASURED (this document references reproducible commands + evidence artifacts + hashes)
Truth Labels: VERIFIED | MEASURED | TARGET | DERIVED

## Scope (SoT Boundary)

- Core repo: `C:\BIZRA-Dual-Agentic-system--main`
- Node0 repo: `C:\BIZRA-Dual-Agentic-system--main\bizra-genesis-node`

## Ihsān Policy (Canonical)

- Constitution: `constitution/ihsan_v1.yaml`
- Tiered thresholds: `threshold_policy` in `constitution/ihsan_v1.yaml`
  - `thresholds_by_env`: `development=0.80`, `ci=0.90`, `production=0.95`
  - `thresholds_by_artifact_class`: `code=0.95`, `docs=0.80`, `config=0.90`, `data=0.90`, `evidence=0.95`
  - Combine rule: `max(env_threshold, artifact_class_threshold)`
- Enforced fail-closed:
  - Rust: `src/ihsan.rs` (`should_enforce()`), applied in `src/bridge.rs` and `src/pat_enhanced.rs`
  - Python: `bizra-genesis-node/bizra_kernel/ihsan_vector.py` (`threshold_for(...)`)

## Phase-0 Gates (Deterministic)

- CI workflow (fail-closed): `.github/workflows/phase0_integrity.yml`
- Rust core: `cargo fmt --check`, `cargo clippy -- -D warnings`, `cargo test`
- Python gates: portability, truth labels, lexicon append-only + tamper test, Node0 secure-defaults lint, Ihsān parity
- Node0 backend: reproducible compile (SQLx offline) + warning budget gate
- Formal invariance proof (conditional): `docs/blueprints/PHASE0_ALIGNMENT_INVARIANCE_PROOF.md`

## Evidence Artifacts (Receipts)

### Phase-0 Gate Run (Local Receipt)

- Consolidated local gate output: `evidence/phase0/phase0_gates_run.log`
- Integrity: `evidence/phase0/phase0_gates_hashes.txt`

### Secrets Scan (gitleaks)

- Tool: gitleaks `8.30.0` (`evidence/phase0/gitleaks_version.txt`)
- Output: `evidence/phase0/gitleaks_report.json` (redacted report)
- Integrity: `evidence/phase0/gitleaks_hashes.txt`
- False-positive allowlist (minimally scoped): `.gitleaksignore`
  - Justification: public token contract addresses in seed memory graphs
  - Scope: 4 fingerprints only; any new findings fail closed

### Genesis Seal Boundary (GenesisManifest)

- Profile (allowlist): `constitution/genesis_manifest_profile_v1.yaml`
- Generator/verifier: `tools/genesis_manifest.py`
- Manifest: `evidence/genesis/GENESIS_MANIFEST.json`
- Integrity: `evidence/genesis/GENESIS_MANIFEST.sha256.txt`
- Receipt: `evidence/genesis/GENESIS_MANIFEST_receipt.txt`
  - Note: current receipt uses `--source worktree` to bind to filesystem bytes; use `--source git` when sealing an immutable commit.

### Node0 Warning Ratchet

- Gate: `tools/node0_warning_budget.py`
- Policy: cap is set in CI via `NODE0_MAX_WARNINGS` in `.github/workflows/phase0_integrity.yml`
- Current measured status: `warnings=203` under `max_warnings=205` (run locally)

## Closure Verdict (Ihsān / Amānah / ʿAdl)

- Phase-0 gates are now enforceable and evidence-backed (fail-closed, no silent “simulation theater”).
- Remaining work is explicitly bounded to P1+: warning burn-down, “Real adapter” receipt enforcement, and Genesis immutability/tagging.
