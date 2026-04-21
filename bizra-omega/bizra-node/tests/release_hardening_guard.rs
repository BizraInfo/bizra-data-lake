//! # Release Hardening Regression Guard — Sprint A.1 (2026-04-21)
//!
//! **[ENFORCEMENT: PROVEN]** — This runtime test asserts the BLAKE3
//! production feature is active by computing `content_hash` on a known
//! input and verifying the output matches the expected BLAKE3 digest.
//! If the `production` feature regresses in `bizra-node`'s
//! `bizra-action` dependency, this test fails and the build breaks in
//! CI. Verification artifact: `cargo test -p bizra-node --test
//! release_hardening_guard`.
//!
//! Shipped `bizra-node` binaries MUST emit BLAKE3-hashed action receipts.
//! The `production` feature on `bizra-action` is CRITICAL for canonical
//! receipt integrity and ZANN_ZERO / CLAIM_MUST_BIND compliance.
//!
//! Without `production` enabled:
//! - `bizra_action::receipt::content_hash` falls back to FNV-1a (NOT cryptographic)
//! - Action receipts produce collision-trivial content hashes
//!
//! History: before 2026-04-21, `bizra-action.default = []` and release
//! workflows built without `--features`. Shipped binaries produced FNV-1a
//! action-receipt hashes silently. Sprint A.1 closes that hole by making
//! the `production` feature mandatory at the `bizra-node` dependency level.
//!
//! Scope note on the `signing` feature —
//! **[OPTIMIZATION: PLANNED]** for Ed25519 signing coverage:
//! the `signing` feature is also enabled on the dep, but its runtime code
//! path lives in `bizra-action::saga::ReceiptChain` (in `saga.rs`), NOT
//! in `bizra-action::receipt::ReceiptChain` (which always stores
//! `signature: [0u8; 64]`). The saga-side signing is additionally gated
//! by the `saga` feature, which is not currently enabled in
//! `bizra-node` / `bizra-agent`. Keeping `signing` enabled brings
//! ed25519-dalek in as a compiled dep so it is ready when `saga` is
//! eventually turned on; this guard does not yet exercise the Ed25519
//! signing path and that is intentional (see A.4 δ-2/δ-3 queue).

use bizra_action::receipt::content_hash;

/// [ENFORCEMENT: PROVEN] — runtime regression guard for PR #44 Sprint A.1.
/// Fails the build if the `production` feature on `bizra-action` is lost
/// from the `bizra-node` Cargo.toml dep spec.
#[test]
fn content_hash_uses_blake3_not_fnv_fallback() {
    // If this test fails, `bizra-node/Cargo.toml` has lost the
    // `features = ["production", ...]` spec on the `bizra-action` dep.
    // Action-receipt content hashes would degrade to non-cryptographic FNV-1a.
    let input: &[u8] = b"BIZRA-A1-HARDENING-GUARD-2026-04-21";
    let expected = *blake3::hash(input).as_bytes();
    let actual = content_hash(input);

    assert_eq!(
        actual, expected,
        "content_hash is not BLAKE3 — `production` feature not active on bizra-action. \
         Check bizra-omega/bizra-node/Cargo.toml: the `bizra-action` dep MUST have \
         features = [\"production\", \"signing\"]."
    );
}
