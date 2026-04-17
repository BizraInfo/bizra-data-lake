# Cycle-6 Landing Guide for Claude Code

بسم الله الرحمن الرحيم

## Package contents (5 files)

1. `trust_compiler.rs` — kernel (755 lines, 8 unit tests)
2. `gateway_v03_compile.rs` — gateway routes (248 lines)
3. `dema_cli_v02_organize.rs` — CLI additions (289 lines)
4. `e2e-trust-compiler-test.sh` — integration test (220 lines)
5. `trust-compiler-e2e.yml` — CI workflow (67 lines)

Total: 1,579 lines across the full vertical.

## Landing order (execute sequentially)

### Step 1: Land trust_compiler.rs

```bash
cp trust_compiler.rs bizra-omega/bizra-cognition/src/trust_compiler.rs
```

Add to `bizra-omega/bizra-cognition/src/lib.rs`:
```rust
pub mod trust_compiler;
```

Add `tempfile` to dev-dependencies in `bizra-omega/bizra-cognition/Cargo.toml`:
```toml
[dev-dependencies]
tempfile = "3"
```

Verify:
```bash
cargo test -p bizra-cognition --lib trust_compiler
# Expected: 8 tests, all green
```

### Step 2: Land gateway compile routes

Merge `gateway_v03_compile.rs` into `bizra-omega/bizra-cognition-gateway/src/main.rs`:

1. Add the new DTOs (CompileRequest, CompileResponse, etc.)
2. Add the handler functions (handle_compile, handle_organize)
3. Add routes to the router:
   ```rust
   .route("/compile", post(handle_compile))
   .route("/organize", post(handle_organize))
   ```
4. Change the shared state from `Arc<RwLock<CognitionRuntime>>` to
   `Arc<RwLock<TrustCompiler>>` — or keep both and give /compile its
   own TrustCompiler instance. The minimal-risk path: keep existing
   /mission route unchanged, add /compile as a parallel path with
   its own TrustCompiler.

Verify:
```bash
cargo test -p bizra-cognition-gateway
cargo build -p bizra-cognition-gateway --release
```

### Step 3: Land dema CLI v0.2

Merge `dema_cli_v02_organize.rs` additions into
`bizra-omega/bizra-cognition-gateway/src/bin/dema.rs`:

1. Add `Organize` and `Compile` variants to the Commands enum
2. Add the handler functions
3. Add the `categorize_ext` function
4. Wire into the main match block

Verify:
```bash
cargo build -p bizra-cognition-gateway --bin dema --release
./target/release/dema organize --dry-run /tmp/test-dir
```

### Step 4: Land E2E test

```bash
mkdir -p scripts
cp e2e-trust-compiler-test.sh scripts/
chmod +x scripts/e2e-trust-compiler-test.sh
```

Run locally:
```bash
scripts/e2e-trust-compiler-test.sh
# Expected: ALL tests passed
```

### Step 5: Land CI workflow

```bash
cp trust-compiler-e2e.yml .github/workflows/
```

### Step 6: Commit and push

Three focused commits:
```bash
# Commit 1: kernel
git add bizra-omega/bizra-cognition/src/trust_compiler.rs \
        bizra-omega/bizra-cognition/src/lib.rs \
        bizra-omega/bizra-cognition/Cargo.toml
git commit -m "feat(cognition): trust compiler — admissibility-first compilation (Cycle-6 G1)"

# Commit 2: gateway + CLI
git add bizra-omega/bizra-cognition-gateway/
git commit -m "feat(gateway): /compile + /organize + dema organize (Cycle-6 G2)"

# Commit 3: E2E + CI
git add scripts/e2e-trust-compiler-test.sh .github/workflows/trust-compiler-e2e.yml
git commit -m "test(e2e): trust compiler full-vertical test + CI workflow (Cycle-6 G3)"
```

Push:
```bash
git push origin main
```

## Post-landing verification

```bash
# Full test suite
cargo test --workspace                    # all Rust tests
scripts/e2e-trust-compiler-test.sh        # E2E vertical

# The Mumo test
./target/release/bizra-cognition-gateway &
./target/release/dema organize ~/Downloads
./target/release/dema chain
```

## Success condition

When `dema organize ~/Downloads` runs and `dema chain` shows
N sub-receipts (one per file) + 5 gate verdicts + 1 final receipt,
Cycle-6 is CLOSED. The trust compiler compiled trust about the
real world for the first time.
