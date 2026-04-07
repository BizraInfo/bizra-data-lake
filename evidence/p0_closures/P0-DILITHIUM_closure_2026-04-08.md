# P0-DILITHIUM Closure Receipt — 2026-04-08

## Status: CLOSED

## Four-Condition Acceptance Gate

### 1. Gate exists in CI
- **Workflow:** `.github/workflows/ci.yml`
- **Z3 install step:** line 430 (`Install Z3 (required by fate-binding)`)
- **Test step:** line 451 (`cargo test --workspace --release`)
- **Stage:** Rust test matrix (runs all workspace crates including fate-binding)

### 2. Gate checks the right thing
- **Crate:** `fate-binding` (bizra-omega/fate-binding/)
- **Implementation:** `src/dilithium.rs` — ML-DSA-87 (NIST post-quantum, successor to Dilithium-5)
- **Library:** `pqcrypto-mldsa` crate (mldsa87 keypair, sign, verify)
- **20 tests across 4 modules:**
  - `dilithium::tests` (6): keypair generation, sign/verify, invalid signature rejection, serialization, public JSON excludes secret key, signed data
  - `capability_card::tests` (3): card creation, threshold rejection, card signing + verification
  - `gate_chain::tests` (3): all-pass chain, Ihsan gate fail, SNR gate pass
  - `z3_ihsan::tests` (3): formal Z3 Ihsan verification pass/fail, proof certificate generation
- **Also includes:** Z3 formal verification of Ihsan constraints (not just Dilithium signatures)

### 3. Failure is observable and blocks correctly
- **On test failure:** `cargo test --workspace` returns non-zero, CI job `test-rust` fails
- **Wired into final gate:** `test_rust` result checked in CI summary (ci.yml line 2003)
- **Z3 dependency:** If Z3 is missing, fate-binding compilation fails → entire Rust test stage fails

### 4. Proof it currently passes
- **Local run (2026-04-08 09:02 GST):**
  ```
  test result: ok. 20 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.10s
  ```

## Spearpoint Reference
- Spearpoint: b08f2208 (BIZRA-STS-001)
- Day: 2
- Date: 2026-04-08
- P0 registry: D5 deliverable
