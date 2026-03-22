// bizra-core/src/canonical_hasher.rs
// ============================================================
// Universal Canonical Hasher — Idempotency + Security + Precision
// ============================================================
//
// THE CRITICAL LAYER: ensures that the same input ALWAYS produces
// the same hash, the same receipt, the same proof — regardless of
// platform, serialization order, or floating-point representation.
//
// Problems this solves:
//   1. Domain collision: different types with same bytes → same hash
//   2. Float drift: 0.95 on x86 ≠ 0.95 on ARM without fixed-point
//   3. Truncation: `(0.9499999 * 1_000_000) as u64` = 949999, not 950000
//   4. Serialization order: HashMap iteration order varies across runs
//   5. Missing domain prefix: bare BLAKE3 hashes can collide cross-type
//
// Standing on Giants:
//   Merkle (1979):  domain-separated hash trees
//   Bernstein (2006): domain separation in crypto (HKDF, Ed25519)
//   Aumasson (2015): BLAKE3 with keyed/derive modes
//   IEEE 754:       fixed-point for deterministic float hashing
// ============================================================

use blake3::Hasher;

/// Fixed-point precision multiplier (P = 1,000,000).
/// 6 decimal places — sufficient for Ihsān (0.950000), SNR, Gini.
pub const FIXED_POINT_P: f64 = 1_000_000.0;

// ── Domain Constants ──────────────────────────────────────
// Every subsystem gets a unique domain prefix.
// Domain separation prevents cross-type hash collisions:
//   BLAKE3("mission:" + data) ≠ BLAKE3("action:" + data)
//   even when `data` is identical.

/// Autopoietic state canonicalization.
pub const DOMAIN_CANONICAL: &[u8] = b"bizra-canonical-v1:";
/// Mission receipt hashing.
pub const DOMAIN_MISSION: &[u8] = b"bizra-mission-v1:";
/// Action receipt hashing.
pub const DOMAIN_ACTION: &[u8] = b"bizra-action-v1:";
/// Skill tree state hashing.
pub const DOMAIN_SKILL: &[u8] = b"bizra-skill-v1:";
/// Experience ledger episodes.
pub const DOMAIN_SEL: &[u8] = b"bizra-sel-v1:";
/// Experience ledger chain links.
pub const DOMAIN_SEL_CHAIN: &[u8] = b"bizra-sel-chain-v1:";
/// Saga receipt chain.
pub const DOMAIN_SAGA: &[u8] = b"bizra-saga-v1:";
/// Reflex trigger hashing.
pub const DOMAIN_REFLEX: &[u8] = b"bizra-reflex-v1:";
/// Genesis block hashing.
pub const DOMAIN_GENESIS: &[u8] = b"bizra-genesis-v1:";
/// ProofSpace block hashing.
pub const DOMAIN_PROOF: &[u8] = b"bizra-proof-v1:";
/// Node identity hashing.
pub const DOMAIN_IDENTITY: &[u8] = b"bizra-identity-v1:";

// ── Safe Fixed-Point Conversion ───────────────────────────
// ROUNDS instead of truncating. This fixes the audit finding:
//   truncating: (0.9499999 * 1_000_000) as u64 = 949999  ← WRONG
//   rounding:   (0.9499999 * 1_000_000).round() = 950000  ← CORRECT

/// Convert f64 to fixed-point u64 with ROUNDING (not truncation).
/// Precision: 6 decimal places (P = 1,000,000).
///
/// # Determinism guarantee
/// IEEE 754 mandates that `round()` is deterministic for the same
/// bit-pattern input. Since we control the domain (0.0..=1.0 for
/// Ihsān, 0.0..=1.0 for SNR, 0.0..=0.35 for Gini), overflow is
/// impossible.
#[inline]
pub fn to_fixed(value: f64) -> u64 {
    (value * FIXED_POINT_P).round() as u64
}

/// Convert fixed-point u64 back to f64.
#[inline]
pub fn from_fixed(fixed: u64) -> f64 {
    fixed as f64 / FIXED_POINT_P
}

// ── CanonicalHasher ───────────────────────────────────────
// Builder-pattern wrapper around BLAKE3 that enforces:
//   1. Domain separation (mandatory first update)
//   2. Fixed-point for all floats
//   3. Little-endian for all integers
//   4. Deterministic field ordering (caller responsibility)

/// A domain-separated BLAKE3 hasher for canonical state reduction.
///
/// # Usage
/// ```ignore
/// let hash = CanonicalHasher::new(DOMAIN_MISSION)
///     .update_bytes(&mission_id)
///     .update_u8(state as u8)
///     .update_u64(timestamp)
///     .update_f64(ihsan_score)
///     .update_optional_bytes(previous_hash.as_ref())
///     .finalize();
/// ```
///
/// # Guarantees
/// - Domain prefix is always the first data fed to BLAKE3
/// - Floats are converted to fixed-point via `to_fixed()` (rounds, not truncates)
/// - Integers are little-endian
/// - Optional fields use a presence byte (0x00 absent, 0x01 present)
pub struct CanonicalHasher {
    inner: Hasher,
}

impl CanonicalHasher {
    /// Create a new hasher with mandatory domain separation.
    pub fn new(domain: &[u8]) -> Self {
        let mut inner = Hasher::new();
        inner.update(domain);
        Self { inner }
    }

    /// Feed raw bytes.
    pub fn update_bytes(mut self, data: &[u8]) -> Self {
        self.inner.update(data);
        self
    }

    /// Feed a u8 value.
    pub fn update_u8(mut self, value: u8) -> Self {
        self.inner.update(&[value]);
        self
    }

    /// Feed a u32 (little-endian).
    pub fn update_u32(mut self, value: u32) -> Self {
        self.inner.update(&value.to_le_bytes());
        self
    }

    /// Feed a u64 (little-endian).
    pub fn update_u64(mut self, value: u64) -> Self {
        self.inner.update(&value.to_le_bytes());
        self
    }

    /// Feed an f64 as fixed-point u64 (ROUNDS, not truncates).
    /// This is the ONLY correct way to hash floats in BIZRA.
    pub fn update_f64(mut self, value: f64) -> Self {
        self.inner.update(&to_fixed(value).to_le_bytes());
        self
    }

    /// Feed an f32 as fixed-point u64 (ROUNDS, not truncates).
    pub fn update_f32(mut self, value: f32) -> Self {
        self.inner.update(&to_fixed(value as f64).to_le_bytes());
        self
    }

    /// Feed a string (UTF-8 bytes prefixed with length).
    /// Length prefix prevents "ab" + "cd" == "abc" + "d" collision.
    pub fn update_str(mut self, value: &str) -> Self {
        let len = value.len() as u64;
        self.inner.update(&len.to_le_bytes());
        self.inner.update(value.as_bytes());
        self
    }

    /// Feed an optional 32-byte hash (presence byte + data).
    /// Absent: 0x00. Present: 0x01 + 32 bytes.
    pub fn update_optional_hash(mut self, value: Option<&[u8; 32]>) -> Self {
        match value {
            Some(hash) => {
                self.inner.update(&[0x01]);
                self.inner.update(hash);
            }
            None => {
                self.inner.update(&[0x00]);
            }
        }
        self
    }

    /// Feed an optional f64 (presence byte + fixed-point).
    pub fn update_optional_f64(mut self, value: Option<f64>) -> Self {
        match value {
            Some(v) => {
                self.inner.update(&[0x01]);
                self.inner.update(&to_fixed(v).to_le_bytes());
            }
            None => {
                self.inner.update(&[0x00]);
            }
        }
        self
    }

    /// Feed an optional string (presence byte + length-prefixed UTF-8).
    pub fn update_optional_str(mut self, value: Option<&str>) -> Self {
        match value {
            Some(s) => {
                self.inner.update(&[0x01]);
                let len = s.len() as u64;
                self.inner.update(&len.to_le_bytes());
                self.inner.update(s.as_bytes());
            }
            None => {
                self.inner.update(&[0x00]);
            }
        }
        self
    }

    /// Feed a boolean (0x00 or 0x01).
    pub fn update_bool(mut self, value: bool) -> Self {
        self.inner.update(&[value as u8]);
        self
    }

    /// Finalize and produce the 32-byte canonical hash.
    pub fn finalize(self) -> [u8; 32] {
        *self.inner.finalize().as_bytes()
    }

    /// Finalize and produce a hex-encoded hash string.
    pub fn finalize_hex(self) -> String {
        self.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }
}

// ── Tests ─────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_same_input_same_hash() {
        let h1 = CanonicalHasher::new(DOMAIN_MISSION)
            .update_bytes(&[1, 2, 3])
            .update_u64(1000)
            .update_f64(0.95)
            .finalize();
        let h2 = CanonicalHasher::new(DOMAIN_MISSION)
            .update_bytes(&[1, 2, 3])
            .update_u64(1000)
            .update_f64(0.95)
            .finalize();
        assert_eq!(h1, h2, "Same input must produce same hash");
    }

    #[test]
    fn domain_separation_different_domains_different_hashes() {
        let data = &[1, 2, 3];
        let h_mission = CanonicalHasher::new(DOMAIN_MISSION).update_bytes(data).finalize();
        let h_action = CanonicalHasher::new(DOMAIN_ACTION).update_bytes(data).finalize();
        let h_skill = CanonicalHasher::new(DOMAIN_SKILL).update_bytes(data).finalize();
        assert_ne!(h_mission, h_action);
        assert_ne!(h_mission, h_skill);
        assert_ne!(h_action, h_skill);
    }

    #[test]
    fn fixed_point_rounds_not_truncates() {
        // The critical audit fix: 0.9499999 must round to 950000, not truncate to 949999
        assert_eq!(to_fixed(0.9499999), 950000);
        assert_eq!(to_fixed(0.95), 950000);
        assert_eq!(to_fixed(0.950001), 950001);
        assert_eq!(to_fixed(1.0), 1_000_000);
        assert_eq!(to_fixed(0.0), 0);
        // Edge case: 0.3500005 → 350001 (Gini threshold boundary)
        assert_eq!(to_fixed(0.3500005), 350001);
    }

    #[test]
    fn fixed_point_roundtrip() {
        let original = 0.95;
        let fixed = to_fixed(original);
        let restored = from_fixed(fixed);
        assert!((original - restored).abs() < 1e-6);
    }

    #[test]
    fn optional_none_vs_some_differ() {
        let h_none = CanonicalHasher::new(DOMAIN_MISSION)
            .update_optional_hash(None)
            .finalize();
        let h_some = CanonicalHasher::new(DOMAIN_MISSION)
            .update_optional_hash(Some(&[0u8; 32]))
            .finalize();
        assert_ne!(h_none, h_some, "None and Some([0;32]) must differ");
    }

    #[test]
    fn string_length_prefix_prevents_collision() {
        // "ab" + "cd" ≠ "abc" + "d" because length prefix differs
        let h1 = CanonicalHasher::new(DOMAIN_MISSION)
            .update_str("ab").update_str("cd").finalize();
        let h2 = CanonicalHasher::new(DOMAIN_MISSION)
            .update_str("abc").update_str("d").finalize();
        assert_ne!(h1, h2, "Length-prefixed strings must not collide");
    }

    #[test]
    fn bool_encoding() {
        let h_true = CanonicalHasher::new(DOMAIN_MISSION).update_bool(true).finalize();
        let h_false = CanonicalHasher::new(DOMAIN_MISSION).update_bool(false).finalize();
        assert_ne!(h_true, h_false);
    }

    #[test]
    fn finalize_hex_is_64_chars() {
        let hex = CanonicalHasher::new(DOMAIN_MISSION)
            .update_u64(42)
            .finalize_hex();
        assert_eq!(hex.len(), 64);
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn all_domains_are_unique() {
        let domains: Vec<&[u8]> = vec![
            DOMAIN_CANONICAL, DOMAIN_MISSION, DOMAIN_ACTION,
            DOMAIN_SKILL, DOMAIN_SEL, DOMAIN_SEL_CHAIN,
            DOMAIN_SAGA, DOMAIN_REFLEX, DOMAIN_GENESIS,
            DOMAIN_PROOF, DOMAIN_IDENTITY,
        ];
        for (i, a) in domains.iter().enumerate() {
            for (j, b) in domains.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "Domain {i} and {j} must be unique");
                }
            }
        }
    }

    #[test]
    fn f32_and_f64_same_value_same_hash() {
        let h32 = CanonicalHasher::new(DOMAIN_MISSION).update_f32(0.95f32).finalize();
        let h64 = CanonicalHasher::new(DOMAIN_MISSION).update_f64(0.95f64).finalize();
        // f32 0.95 → 0.949999988... → to_fixed rounds to 950000
        // f64 0.95 → 0.95 exactly   → to_fixed rounds to 950000
        assert_eq!(h32, h64, "f32 and f64 of 0.95 must produce same fixed-point hash");
    }

    #[test]
    fn optional_f64_none_vs_zero() {
        let h_none = CanonicalHasher::new(DOMAIN_MISSION).update_optional_f64(None).finalize();
        let h_zero = CanonicalHasher::new(DOMAIN_MISSION).update_optional_f64(Some(0.0)).finalize();
        assert_ne!(h_none, h_zero, "None and Some(0.0) must differ");
    }
}
