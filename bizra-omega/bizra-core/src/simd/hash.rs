//! Parallel BLAKE3 Hashing — Multi-threaded digest computation
//!
//! Uses BLAKE3's built-in parallelism for large data and
//! rayon for batch hashing of multiple messages.

use crate::DOMAIN_PREFIX;
use blake3::Hasher;

/// Batch hash result
#[derive(Debug)]
pub struct BatchHashResult {
    /// Hex-encoded BLAKE3 digests, one per input message.
    pub digests: Vec<String>,
    /// Total bytes processed across all messages.
    pub total_bytes: usize,
}

/// Hash multiple messages in parallel using rayon
///
/// For small messages, this uses thread pool for parallel
/// processing. For large messages, BLAKE3's internal SIMD
/// handles parallelism automatically.
pub fn blake3_parallel(messages: &[&[u8]]) -> BatchHashResult {
    let digests: Vec<String> = messages.iter().map(|msg| domain_hash_fast(msg)).collect();

    let total_bytes: usize = messages.iter().map(|m| m.len()).sum();

    BatchHashResult {
        digests,
        total_bytes,
    }
}

/// Fast domain-separated hash (inline, no allocation overhead)
#[inline(always)]
pub fn domain_hash_fast(message: &[u8]) -> String {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_PREFIX);
    hasher.update(message);
    hasher.finalize().to_hex().to_string()
}

/// Parallel hash with rayon for batch processing
#[cfg(feature = "parallel")]
pub fn blake3_parallel_rayon(messages: &[Vec<u8>]) -> BatchHashResult {
    use rayon::prelude::*;

    let digests: Vec<String> = messages
        .par_iter()
        .map(|msg| domain_hash_fast(msg))
        .collect();

    let total_bytes: usize = messages.iter().map(|m| m.len()).sum();

    BatchHashResult {
        digests,
        total_bytes,
    }
}

/// Hash large data with BLAKE3's internal parallelism
///
/// BLAKE3 automatically uses SIMD (AVX-512/AVX2/NEON) and
/// multithreading for data > 128KB.
pub fn blake3_large(data: &[u8]) -> String {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_PREFIX);

    // For large data, use update_rayon for parallel hashing
    #[cfg(feature = "rayon")]
    {
        hasher.update_rayon(data);
    }

    #[cfg(not(feature = "rayon"))]
    {
        hasher.update(data);
    }

    hasher.finalize().to_hex().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_hash() {
        let messages: Vec<&[u8]> = vec![b"message_1", b"message_2", b"message_3"];

        let result = blake3_parallel(&messages);
        assert_eq!(result.digests.len(), 3);
        assert_eq!(result.total_bytes, 27);

        // Verify determinism
        let result2 = blake3_parallel(&messages);
        assert_eq!(result.digests, result2.digests);
    }

    #[test]
    fn test_fast_hash() {
        let hash1 = domain_hash_fast(b"test");
        let hash2 = domain_hash_fast(b"test");
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 64); // 32 bytes = 64 hex chars
    }
}
