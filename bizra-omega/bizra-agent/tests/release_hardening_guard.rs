//! # Release Hardening Regression Guard (bizra-agent mirror) — Sprint A.1 (2026-04-21)
//!
//! **[ENFORCEMENT: PROVEN]** — This runtime test asserts the BLAKE3
//! production feature is active for `bizra-agent`'s `bizra-action`
//! dependency by computing `content_hash` on a known input and verifying
//! the output matches the expected BLAKE3 digest. If the `production`
//! feature regresses in `bizra-agent`'s `bizra-action` dep spec, this
//! test fails and the build breaks in CI.
//!
//! Mirror rationale: `bizra-agent` (OmniKernel, PAT runtime) independently
//! depends on `bizra-action` and would independently lose the hardened
//! path if its Cargo.toml drifts. The parallel guard in
//! `bizra-node/tests/release_hardening_guard.rs` catches `bizra-node`'s
//! dep spec; this one catches `bizra-agent`'s. Both are needed because
//! the two crates are independently releasable.
//!
//! See `bizra-node/tests/release_hardening_guard.rs` for the full
//! scope note on the `signing` feature (Ed25519 signing lives in
//! `bizra-action::saga::ReceiptChain`, not in `receipt::ReceiptChain`,
//! and is additionally gated by the `saga` feature — not exercised here
//! by design; see A.4 δ-2/δ-3 queue).

use bizra_action::receipt::content_hash;

/// [ENFORCEMENT: PROVEN] — runtime regression guard for PR #44 Sprint A.1.
/// Fails the build if the `production` feature on `bizra-action` is lost
/// from the `bizra-agent` Cargo.toml dep spec.
#[test]
fn content_hash_uses_blake3_not_fnv_fallback() {
    // If this test fails, `bizra-agent/Cargo.toml` has lost the
    // `features = ["production", ...]` spec on the `bizra-action` dep.
    // Action-receipt content hashes would degrade to non-cryptographic FNV-1a.
    let input: &[u8] = b"BIZRA-A1-AGENT-HARDENING-GUARD-2026-04-21";
    let expected = *blake3::hash(input).as_bytes();
    let actual = content_hash(input);

    assert_eq!(
        actual, expected,
        "content_hash is not BLAKE3 — `production` feature not active on bizra-action. \
         Check bizra-omega/bizra-agent/Cargo.toml: the `bizra-action` dep MUST have \
         features = [\"production\", \"signing\"]."
    );
}
