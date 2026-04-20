//! # Release Hardening Regression Guard — Sprint A.1 (2026-04-21)
//!
//! Shipped `bizra-node` binaries MUST emit BLAKE3-hashed action receipts.
//! The `production` feature on `bizra-action` is CRITICAL for canonical
//! receipt integrity and ZANN_ZERO / CLAIM_MUST_BIND compliance.
//!
//! Without `production` enabled:
//! - `bizra_action::receipt::content_hash` falls back to FNV-1a (NOT cryptographic)
//! - Action receipts produce collision-trivial content hashes
//!
//! This guard is a RUNTIME check: it computes `content_hash` on a known input
//! and asserts the output matches the expected BLAKE3 digest. If `production`
//! is disabled, the FNV-1a placeholder produces a different digest and this
//! test fails.
//!
//! History: before 2026-04-21, `bizra-action.default = []` and release
//! workflows built without `--features`. Shipped binaries produced FNV-1a
//! action-receipt hashes silently. Sprint A.1 closes that hole by making the
//! `production` feature mandatory at the `bizra-node` dependency level.
//!
//! Note: the `signing` feature is also enabled on the dep, but its code path
//! lives behind the `saga` feature which is not currently in scope for the
//! default action-bus path. Keeping it enabled prevents future drift when
//! `saga` is turned on.

use bizra_action::receipt::content_hash;

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
