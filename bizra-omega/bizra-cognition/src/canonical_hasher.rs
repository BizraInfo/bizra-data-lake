// bizra-cognition/src/canonical_hasher.rs
// Domain-separated BLAKE3 hashing for the cognition substrate.
// Self-contained — uses blake3 crate directly (same as bizra-core's
// orphan canonical_hasher.rs, which will be merged later).

pub type Blake3Hash = [u8; 32];

pub fn blake3_domain(domain: &str, data: &[u8]) -> Blake3Hash {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain.as_bytes());
    hasher.update(data);
    *hasher.finalize().as_bytes()
}

/// Simple BLAKE3 hash of a byte slice (no domain separation).
/// Used by thought_graph for content-addressing observations and predictions.
pub fn blake3_chain(data: &[u8]) -> Blake3Hash {
    *blake3::hash(data).as_bytes()
}
