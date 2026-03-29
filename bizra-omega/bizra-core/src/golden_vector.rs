//! # BIZRA Golden Vector — Cross-Language Sealing Test (Rust side)
//!
//! This module defines the canonical test vector and verifies that Rust
//! produces the identical hash as Python for the same input.
//!
//! The golden vector is FROZEN. If this hash changes, cross-language sealing
//! is broken and ALL receipts become untrustworthy.
//!
//! Standing on: Merkle (1979), Aumasson (2015), Shannon (1948)

use blake3::Hasher;

/// Domain prefix for golden vector hashing.
pub const DOMAIN_GOLDEN_VECTOR: &str = "bizra-golden-vector-v1";

/// Fixed-point precision multiplier (P = 1,000,000).
pub const FIXED_POINT_P: f64 = 1_000_000.0;

// ── The Golden Vector (FROZEN — never modify) ──────────────
/// Frozen mission ID for cross-language sealing test.
pub const GOLDEN_MISSION_ID: &str = "golden-vector-v1";
/// Frozen initiator ID (node0 genesis identity).
pub const GOLDEN_INITIATOR_ID: &str = "node0-genesis";
/// Frozen payload (Basmala — first words of every Surah).
pub const GOLDEN_PAYLOAD: &[u8] = b"In the Name of Allah, Most Gracious, Most Merciful";
/// Frozen Ihsan score for deterministic hashing.
pub const GOLDEN_IHSAN_SCORE: f64 = 0.984700;
/// Frozen timestamp (2024-03-28T00:00:00Z) for reproducibility.
pub const GOLDEN_TIMESTAMP: u64 = 1711584000000;

/// The FROZEN digest. Python and Rust must produce this exact hash.
pub const GOLDEN_DIGEST_HEX: &str =
    "966725c27200cdd28632e1f10a09ca7f982491be5842c8eb1264650a32a51205";

/// Serialize the golden vector into canonical bytes.
///
/// Field order: mission_id, initiator_id, payload, ihsan_fixed, timestamp
/// Encoding:
///   - Strings: UTF-8 bytes prefixed with u32le length
///   - Bytes: raw bytes prefixed with u32le length
///   - Float: round(value * FIXED_POINT_P) as u64le
///   - Int: u64le
pub fn serialize_golden_vector() -> Vec<u8> {
    let mut buf = Vec::with_capacity(128);

    // mission_id: length-prefixed UTF-8
    let mid = GOLDEN_MISSION_ID.as_bytes();
    buf.extend_from_slice(&(mid.len() as u32).to_le_bytes());
    buf.extend_from_slice(mid);

    // initiator_id: length-prefixed UTF-8
    let iid = GOLDEN_INITIATOR_ID.as_bytes();
    buf.extend_from_slice(&(iid.len() as u32).to_le_bytes());
    buf.extend_from_slice(iid);

    // payload: length-prefixed bytes
    buf.extend_from_slice(&(GOLDEN_PAYLOAD.len() as u32).to_le_bytes());
    buf.extend_from_slice(GOLDEN_PAYLOAD);

    // ihsan_score: fixed-point u64le (round, not truncate)
    let ihsan_fixed = (GOLDEN_IHSAN_SCORE * FIXED_POINT_P).round() as u64;
    buf.extend_from_slice(&ihsan_fixed.to_le_bytes());

    // timestamp: u64le
    buf.extend_from_slice(&GOLDEN_TIMESTAMP.to_le_bytes());

    buf
}

/// Compute the golden vector hash using domain-separated BLAKE3.
///
/// `hash = BLAKE3(domain + ":" + serialized_data)`
pub fn compute_golden_digest() -> [u8; 32] {
    let serialized = serialize_golden_vector();
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_GOLDEN_VECTOR.as_bytes());
    hasher.update(b":");
    hasher.update(&serialized);
    hasher.finalize().into()
}

/// Compute and return the golden vector hex digest.
pub fn golden_digest_hex() -> String {
    compute_golden_digest()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_golden_vector_produces_frozen_digest() {
        let digest = golden_digest_hex();
        assert_eq!(
            digest, GOLDEN_DIGEST_HEX,
            "FATAL: Golden vector digest mismatch. Cross-language sealing broken.\n\
             Expected: {}\n\
             Got:      {}",
            GOLDEN_DIGEST_HEX, digest
        );
    }

    #[test]
    fn test_golden_vector_serialization_deterministic() {
        let a = serialize_golden_vector();
        let b = serialize_golden_vector();
        assert_eq!(a, b, "Serialization is not deterministic");
    }

    #[test]
    fn test_golden_vector_serialization_length() {
        let serialized = serialize_golden_vector();
        // 4 + 16 + 4 + 13 + 4 + 50 + 8 + 8 = 107 bytes
        assert_eq!(serialized.len(), 107, "Serialized length mismatch");
    }

    #[test]
    fn test_golden_vector_fixed_point_rounding() {
        let ihsan_fixed = (GOLDEN_IHSAN_SCORE * FIXED_POINT_P).round() as u64;
        assert_eq!(ihsan_fixed, 984700, "Fixed-point rounding incorrect");
    }

    #[test]
    fn test_golden_vector_idempotency() {
        let d1 = golden_digest_hex();
        let d2 = golden_digest_hex();
        assert_eq!(d1, d2, "Hash is not idempotent");
    }

    #[test]
    fn test_golden_vector_domain_separation() {
        let serialized = serialize_golden_vector();

        // Standard domain
        let mut h1 = Hasher::new();
        h1.update(DOMAIN_GOLDEN_VECTOR.as_bytes());
        h1.update(b":");
        h1.update(&serialized);
        let d1: [u8; 32] = h1.finalize().into();

        // Different domain
        let mut h2 = Hasher::new();
        h2.update(b"different-domain");
        h2.update(b":");
        h2.update(&serialized);
        let d2: [u8; 32] = h2.finalize().into();

        assert_ne!(d1, d2, "Domain separation failed");
    }
}
