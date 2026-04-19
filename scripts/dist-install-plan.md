# BIZRA cargo-dist Install Plan (Cycle-8 Day 3 Draft)

**Scope:** Day 3 of the First Fire 12-day plan — packaging preflight.
**Status:** DRAFT. No installer is generated until `cargo dist init` runs
against the config in `bizra-omega/Cargo.toml` `[workspace.metadata.dist]`.

**Non-conflict note:** this file is the Day 3 packaging plan, NOT the
already-existing `scripts/install.sh` (the node1-reproducibility bundle
installer, shipped in commit `4267296d`). Those two installers serve
different audiences and will eventually be either merged or kept
distinct per operator review.

---

## Shape of the future `bizra.ai/install.sh`

When `cargo dist init` generates the installer, it will be served at the
public URL `https://bizra.ai/install.sh`. Operators run:

```sh
curl -fsSL https://bizra.ai/install.sh | sh
```

The cargo-dist-generated installer performs this flow:

1. **Detect platform** — uname + arch → one of
   `{x86_64-unknown-linux-gnu, aarch64-apple-darwin, x86_64-apple-darwin, x86_64-pc-windows-msvc}`
2. **Fetch `dist-manifest.json`** from the GitHub release page — signed
   manifest produced by cargo-dist's release CI.
3. **Fetch the tarball** for the detected platform.
4. **Verify SHA-256** — the installer computes SHA-256 of the tarball
   and compares against the declared digest in `dist-manifest.json`.
   Abort on mismatch (cryptographic modality).
5. **Extract** to a staging directory.
6. **Install** both `bizra-cognition-gateway` and `dema` binaries to
   `$HOME/.cargo/bin` (or `$HOME/.bizra/bin` if cargo is absent).
7. **Print** the installed version + the SHA-256 of each binary for
   the operator's own post-install audit.

## Four-Modality verification checklist (post-install)

| Modality | Verification step | Who can verify |
|---|---|---|
| **Cryptographic** | `sha256sum $(which dema)` == declared SHA in manifest | anyone with sha256sum |
| **Empirical** | `dema --version && dema chain` runs; same output on any identical install | anyone with dema installed |
| **Formal (TESTED)** | `cargo test -p bizra-cognition` (after git clone) → 309/309 green. Full Isabelle/HOL-grade formal proof is Horizon, not T=0. | anyone with Rust toolchain |
| **Economic / Witness** | `dema chain` head matches witness peer's observed head (Day 4 work). Witness-grade detectability only; bonded stake / slashing / DAO / challenge-period economics are Horizon / Layer B. | anyone with a witness URL |

## Config location

- **Workspace config:** `bizra-omega/Cargo.toml` →
  `[workspace.metadata.dist]` block (see Day 3 commit for full content).
- **Binary targets:** `bizra-omega/bizra-cognition-gateway/Cargo.toml` →
  `[[bin]]` entries for `bizra-cognition-gateway` and `dema` already
  declared pre-Cycle-8.
- **CI workflow:** will be generated at `.github/workflows/release.yml`
  on first `cargo dist init` run. Not in Day 3 scope.

## Day 3 deliverables (this commit)

- `[workspace.metadata.dist]` block added to `bizra-omega/Cargo.toml`.
- `scripts/dist-install-plan.md` (this file).
- No CI workflow generated yet (requires cargo-dist locally).
- No binaries built yet (no new CI run triggered).
- `scripts/install.sh` UNCHANGED (node1 installer preserved).

## Day 4+ open items (not in scope today)

- `cargo install cargo-dist` (requires operator approval — `ask` gate).
- `cargo dist init --yes` — generates `.github/workflows/release.yml`
  and refines `[workspace.metadata.dist]` against current cargo-dist
  schema.
- Run `cargo dist plan` locally to emit the first `dist-manifest.json`
  preview, audit its shape.
- Resolve the `install.sh` vs `dist-install.sh` naming collision with
  the existing node1 installer — either merge into one universal
  installer or keep distinct with clear purpose labels.

## Rollback

```sh
git revert HEAD   # on cycle-8/seal-primitive-days-1-2
```

The cargo-dist config is inert until `cargo dist` commands are run
against it. Revert leaves no trace in CI, no published release, no
binary drift.

Close it. Prove it. Reveal it.
