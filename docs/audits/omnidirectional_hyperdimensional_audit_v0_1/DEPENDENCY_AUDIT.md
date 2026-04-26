# Dependency Audit — BIZRA v0.1

**Scope:** Rust, Python, Node dependencies; lockfiles; SBOM; license / audit-tooling gaps; supply-chain risk.

---

## 1. Inventory

| Ecosystem | Manifests found | Lockfiles found |
|---|---:|---:|
| Rust (`Cargo.toml`) | 41 | 5 |
| Python (`requirements.txt` family) | 9 files | — (pip does not mandate lockfiles) |
| Python (`pyproject.toml`) | 3 | — |
| Node (`package.json`) | 8 | 2 (`package-lock.json`) |

Full inventory in `artifacts/dependencies.json`.

## 2. Gap analysis (machine-detected)

| # | Gap | Impact |
|---|---|---|
| G1 | **Rust workspace without Cargo.lock — `filedfs/Cargo.toml`** | Non-reproducible builds in that tree. |
| G2 | **Rust workspace without Cargo.lock — `desktop/rust/Cargo.toml`** | Non-reproducible builds in that tree. |
| G3 | **SBOM artifact not located in repo** (no `*.spdx.json` / `*.cdx.json` found) | No machine-verifiable supply-chain attestation. |

## 3. Rust-specific

- **Primary workspace:** `bizra-omega/` — has `Cargo.lock` ✅. 25 crates.
- **Secondary workspaces:** `filedfs/`, `desktop/rust/`, + a few more without top-level locks. These are lower-priority but should still pin.
- **Clippy gate:** `cargo clippy --workspace --all-targets -- -D warnings` per CLAUDE.md — zero warnings enforced in CI. ✅
- **`uninlined_format_args` lint:** flagged as pre-existing Rust 1.91 lint across workspace. Not a security issue; tech debt.

## 4. Python-specific

- **Primary config:** `pyproject.toml` — ruff, black, isort, mypy centralized. ✅
- **`requirements.txt` + `requirements.flywheel.txt`** exist. Pinning disciplines vary.
- **Pinned deps detected:** none strictly pinned by `==` across surveyed files (many use `>=`). This is common but non-reproducible.
- **Optional installs:** `[dev]`, `[full]` extras defined in pyproject.toml.

**Action:** adopt `pip-tools` or `uv pip compile` to generate pinned `requirements.lock` for the core runtime surface.

## 5. Node-specific

- **Frontend:** `frontend/package.json` + `frontend/package-lock.json` ✅ (lockfile present).
- Other `package.json` files in the repo are smaller / localized — verify each has a sibling lock or is trivial.

## 6. License / audit tooling

- **pip-audit:** listed in CI stages per CLAUDE.md.
- **cargo-audit:** listed in CI stages per CLAUDE.md.
- **Trivy:** scans containers.
- **License enforcement:** not visible in this audit (no `cargo-deny` config, no `licensee` run). Recommend adding `cargo-deny` with an explicit license allow-list.

## 7. Supply-chain risk surface

| Layer | Risk posture |
|---|---|
| Rust | MEDIUM — primary workspace pinned; secondary workspaces not. |
| Python | MEDIUM — no strict pins in surveyed `requirements.txt`; no runtime lockfile. |
| Node | LOW — lockfile present for the frontend surface. |
| Container | Declared pinned via SHA in GitHub Actions (per CLAUDE.md) ✅. |
| SBOM | NOT PRODUCED — gap G3. |

## 8. Recommendations (ranked)

| # | Action | Effort |
|---|---|---|
| DR1 | Generate Cargo.lock for `filedfs/` and `desktop/rust/` | S |
| DR2 | Add `cargo-deny` with license + advisory checks to CI | M |
| DR3 | Adopt `uv pip compile` / `pip-tools` for Python lockfile | M |
| DR4 | Emit SBOM (SPDX or CycloneDX) on every release | M |
| DR5 | Verify Node secondary `package.json` files have locks | S |
| DR6 | Publish `SUPPLY_CHAIN.md` — declared approach + artifacts | S |

---

**Supply-chain verdict:** tooling exists (`pip-audit`, `cargo-audit`, Trivy) but attestation (SBOM) and some reproducibility guarantees (Python lockfile, secondary Rust lockfiles) are missing. Closing these gaps is a prerequisite for any "production-grade" public claim.
