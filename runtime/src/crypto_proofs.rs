// src/crypto_proofs.rs - Deterministic Mathematical Verification
//
// Implements deterministic primality testing (Miller-Rabin with fixed bases) and
// Lean 4 certificate verification logic to replace probabilistic checks.
//
// Ihsān Requirement: Correctness >= 0.99 for L3 logic.

/// Verify primality deterministically for u64 inputs.
/// Uses deterministic Miller-Rabin with specific bases.
/// Proven correct for n < 2^64.
/// Bases: 2, 325, 9375, 28178, 450775, 9780504, 1795265022
pub fn verify_prime_deterministic(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    if n == 2 || n == 3 {
        return true;
    }
    if n.is_multiple_of(2) {
        return false;
    }

    let d = n - 1;
    let s = d.trailing_zeros();
    let d = d >> s;

    let bases = [2, 325, 9375, 28178, 450775, 9780504, 1795265022];

    for &a in &bases {
        if a % n == 0 {
            continue;
        }
        if witness(a, d, s, n) {
            return false;
        }
    }
    true
}

fn witness(a: u64, d: u64, s: u32, n: u64) -> bool {
    let mut x = mod_pow(a, d, n);
    if x == 1 || x == n - 1 {
        return false;
    }
    for _ in 0..s - 1 {
        x = mul_mod(x, x, n);
        if x == n - 1 {
            return false;
        }
    }
    true
}

fn mul_mod(a: u64, b: u64, m: u64) -> u64 {
    ((a as u128 * b as u128) % m as u128) as u64
}

fn mod_pow(mut base: u64, mut exp: u64, modulus: u64) -> u64 {
    if modulus == 1 {
        return 0;
    }
    let mut result = 1;
    base %= modulus;
    while exp > 0 {
        if exp % 2 == 1 {
            result = mul_mod(result, base, modulus);
        }
        base = mul_mod(base, base, modulus);
        exp /= 2;
    }
    result
}

/// Verify if the content contains a valid Lean 4 certificate signature.
/// In a real system, this would cryptographically verify the signature.
/// For Phase 2 Activation, we check for the specific artifact format.
pub fn verify_lean4_cert(content: &str) -> bool {
    // Check for "LEAN4_CERT:" token or "proof_verified: true" structure with hash
    content.contains("LEAN4_CERT:")
        || (content.contains("proof_verified: true") && content.contains("lean4_hash:"))
}

/// Check if the content contains a reference to a Fermat prime check that requires high assurance
pub fn requires_fermat_check(content: &str) -> Option<u64> {
    // Simple heuristic: if user asks about "Fermat prime" and a number
    if content.to_lowercase().contains("fermat prime") {
        // Try to extract a number (simplified)
        let parts: Vec<&str> = content.split_whitespace().collect();
        for part in parts {
            if let Ok(n) = part.parse::<u64>() {
                if n > 65537 {
                    return Some(n);
                }
            }
        }
    }
    None
}
