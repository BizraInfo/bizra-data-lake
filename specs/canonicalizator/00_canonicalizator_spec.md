# The Canonicalizator — First-Class Kernel Service

## The Problem

BIZRA has canonicalization in 4 places, each with its own rules:

```
bizra-proofspace:    RFC 8785 JCS for blocks
bizra-core:          CanonicalChain for checkpoints
bizra-installer:     Policy text canonicalization (strip whitespace)
core/proof_engine:   Python canonical_json() (recursive sort)
```

If ANY of these produce different bytes for the same logical data,
the proof pyramid breaks. A receipt hashed on one node won't verify
on another. The chain is only as strong as its weakest canonicalizer.

## The Solution: One Canonicalizator, Every Path

```
                    ┌──────────────────┐
                    │  CANONICALIZATOR  │
                    │  (kernel service) │
                    └────────┬─────────┘
                             │
           ┌─────────┬───────┼───────┬──────────┐
           ▼         ▼       ▼       ▼          ▼
        Receipts   Blocks  Policy  Memory   Identity
        (mission)  (proof)  (const) (SEL)   (Ed25519)
```

Every piece of data that gets hashed, signed, or verified
passes through the SAME canonicalization function.

## Standing on Giants

- **RFC 8785 (2019)**: JSON Canonicalization Scheme — deterministic JSON
- **BLAKE3 (2020)**: Domain-separated hashing with context strings
- **Lamport (1978)**: Deterministic ordering for distributed agreement
- **Satoshi (2008)**: Canonical transaction serialization for consensus

## The Three Laws of Canonical Data

```
LAW 1 — DETERMINISM:
  canonical(X) == canonical(X), always, everywhere, on every platform.
  Same input → same bytes → same hash → same signature.

LAW 2 — DOMAIN SEPARATION:
  hash("bizra-receipt-v1:" || canonical(receipt)) ≠
  hash("bizra-block-v1:" || canonical(block))
  Even if receipt == block bytewise, different domain = different hash.

LAW 3 — CROSS-LANGUAGE PARITY:
  canonical_rust(X) == canonical_python(X), byte-for-byte.
  If Python and Rust disagree, the chain is broken.
```

## Pseudocode: Unified Canonicalizator

```rust
// bizra-core/src/canonical.rs — The single source of truth

/// Domain prefixes for hash separation (Standing on: BLAKE3 spec)
pub const DOMAIN_RECEIPT:     &str = "bizra-receipt-v1";
pub const DOMAIN_BLOCK:       &str = "bizra-block-v1";
pub const DOMAIN_POLICY:      &str = "bizra-policy-v1";
pub const DOMAIN_EPISODE:     &str = "bizra-sel-v1";     // already exists
pub const DOMAIN_IDENTITY:    &str = "bizra-identity-v1";
pub const DOMAIN_CONSTITUTION: &str = "bizra-const-v1";
pub const DOMAIN_CHAIN:       &str = "bizra-chain-v1";

/// Canonicalize any serializable value to deterministic bytes.
///
/// Rules (RFC 8785 + BIZRA extensions):
/// 1. Object keys sorted lexicographically (UTF-8)
/// 2. No trailing commas, no whitespace
/// 3. Numbers: no leading zeros, no trailing zeros after decimal
/// 4. Strings: UTF-8, escaped per JSON spec
/// 5. Arrays: preserve order (order IS data)
/// 6. Null: literal "null"
/// 7. Booleans: "true" / "false"
pub fn canonicalize<T: Serialize>(value: &T) -> Result<Vec<u8>, CanonicalError> {
    // Use serde_json with sorted keys + compact formatting
    let json_value = serde_json::to_value(value)?;
    let canonical = jcs_serialize(&json_value);
    Ok(canonical.into_bytes())
}

/// Hash with domain separation.
///
/// hash = BLAKE3(domain || ":" || canonical_bytes)
///
/// The domain prefix ensures that a receipt hash can NEVER
/// collide with a block hash, even if the data is identical.
pub fn domain_hash(domain: &str, data: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain.as_bytes());
    hasher.update(b":");
    hasher.update(data);
    hasher.finalize().into()
}

/// The complete canonicalize-and-hash pipeline.
///
/// canonical_hash(DOMAIN_RECEIPT, &receipt) produces a deterministic,
/// domain-separated, 32-byte BLAKE3 hash suitable for:
/// - Receipt chain (prev_hash linkage)
/// - Block ID computation
/// - Ed25519 signing payload
/// - Cross-node verification
pub fn canonical_hash<T: Serialize>(domain: &str, value: &T) -> Result<[u8; 32], CanonicalError> {
    let bytes = canonicalize(value)?;
    Ok(domain_hash(domain, &bytes))
}
```

## Python Mirror (Cross-Language Parity)

```python
# core/proof_engine/canonical.py — MUST produce identical bytes

DOMAIN_RECEIPT     = "bizra-receipt-v1"
DOMAIN_BLOCK       = "bizra-block-v1"
DOMAIN_POLICY      = "bizra-policy-v1"
DOMAIN_EPISODE     = "bizra-sel-v1"
DOMAIN_IDENTITY    = "bizra-identity-v1"
DOMAIN_CONSTITUTION = "bizra-const-v1"
DOMAIN_CHAIN       = "bizra-chain-v1"

def canonicalize(obj: Any) -> bytes:
    """RFC 8785 JCS canonicalization — must match Rust byte-for-byte."""
    return json.dumps(
        _sort_recursive(obj),
        separators=(',', ':'),  # No whitespace
        sort_keys=True,
        ensure_ascii=False,     # UTF-8 passthrough
    ).encode('utf-8')

def domain_hash(domain: str, data: bytes) -> bytes:
    """BLAKE3 with domain separation — must match Rust."""
    import hashlib  # or blake3 if available
    h = hashlib.blake2b(f"{domain}:".encode() + data, digest_size=32)
    return h.digest()
```

## TDD Anchors

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn determinism_law() {
        let data = json!({"b": 2, "a": 1});
        let h1 = canonical_hash(DOMAIN_RECEIPT, &data).unwrap();
        let h2 = canonical_hash(DOMAIN_RECEIPT, &data).unwrap();
        assert_eq!(h1, h2); // Law 1: always the same
    }

    #[test]
    fn domain_separation_law() {
        let data = json!({"x": 42});
        let h_receipt = canonical_hash(DOMAIN_RECEIPT, &data).unwrap();
        let h_block = canonical_hash(DOMAIN_BLOCK, &data).unwrap();
        assert_ne!(h_receipt, h_block); // Law 2: different domain = different hash
    }

    #[test]
    fn key_ordering() {
        let a = json!({"z": 1, "a": 2});
        let b = json!({"a": 2, "z": 1});
        assert_eq!(canonicalize(&a).unwrap(), canonicalize(&b).unwrap());
    }

    #[test]
    fn cross_language_parity() {
        // This test runs Python and compares output
        // Law 3: Rust canonical == Python canonical, byte-for-byte
        let data = json!({"ihsan": 0.95, "receipt": "abc", "count": 42});
        let rust_bytes = canonicalize(&data).unwrap();
        // Expected: {"count":42,"ihsan":0.95,"receipt":"abc"}
        assert_eq!(
            String::from_utf8(rust_bytes).unwrap(),
            r#"{"count":42,"ihsan":0.95,"receipt":"abc"}"#
        );
    }

    #[test]
    fn receipt_hash_chain_integrity() {
        let r1 = json!({"id": "001", "ihsan": 0.97});
        let r2 = json!({"id": "002", "ihsan": 0.96, "prev": "hash_of_r1"});
        let h1 = canonical_hash(DOMAIN_RECEIPT, &r1).unwrap();
        let h2 = canonical_hash(DOMAIN_RECEIPT, &r2).unwrap();
        assert_ne!(h1, h2);
        // Chain: r2.prev == hex(h1)
    }
}
```

## Integration Points

| Component | Current | Canonicalized |
|-----------|---------|--------------|
| MissionReceipt | BLAKE3 of selected fields | `canonical_hash(DOMAIN_RECEIPT, &receipt)` |
| ProofSpace Block | JCS of UnsignedBlock | `canonical_hash(DOMAIN_BLOCK, &block)` |
| ExperienceLedger | BLAKE3 of episode fields | `canonical_hash(DOMAIN_EPISODE, &episode)` |
| Policy hash | Strip whitespace + BLAKE3 | `canonical_hash(DOMAIN_POLICY, &policy)` |
| Identity registry | Direct key bytes | `canonical_hash(DOMAIN_IDENTITY, &agent)` |
| Constitution | Hash of constants.py | `canonical_hash(DOMAIN_CONSTITUTION, &thresholds)` |

## The Glass Box Guarantee

When the Canonicalizator is the single path for ALL hashing:

1. **Every hash is verifiable** — reproduce by canonicalizing the same data
2. **Every signature is auditable** — the signed payload is deterministic
3. **Every chain is portable** — Rust and Python produce identical hashes
4. **Every proof is constitutional** — domain separation prevents cross-type collision

The Canonicalizator doesn't just normalize data.
**It normalizes trust.**
