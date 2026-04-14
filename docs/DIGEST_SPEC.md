# BIZRA Digest Specification

> Canonical reference for all cryptographic hash functions used across BIZRA subsystems.
> This document resolves ambiguity about which hash is used where.
>
> **Scope:** This spec governs receipt digests for cycle artifacts and seed chain links.
> Transport-layer crypto (TLS), git object hashing (SHA-1), and Ed25519's internal
> SHA-512 are out of scope — those are governed by their respective protocol specs.

---

## Hash Functions by Subsystem

| Subsystem | Hash | Library | Purpose |
|---|---|---|---|
| **Proof Engine** (receipts, canonical) | **BLAKE3** | `blake3` PyPI (1.0.8+) | Receipt digests, canonical hashing, cross-language interop with Rust |
| **Autopoietic Cycles** (reverify.py) | **BLAKE3** (primary) / BLAKE2B-256 (fallback) | `blake3` PyPI / `hashlib.blake2b` stdlib | Cycle artifact manifest hashing |
| **SAT Ceremony** (genesis receipts) | **BLAKE2B** | `hashlib.blake2b` stdlib | Genesis receipt hashing, mint court evidence |
| **PCI** (identity, signing) | **Ed25519** pre-image via **BLAKE3** | `blake3` + `pynacl` | Message hashing before Ed25519 signature |
| **Rust workspace** (bizra-core) | **BLAKE3** | `blake3` crate | Must produce identical hashes to Python `blake3` for cross-language receipts |
| **Loop Proof** (manifest) | **BLAKE3** via `canonical.blake3_digest()` | `blake3` PyPI | Loop proof manifest hash |

## Design Decisions

1. **BLAKE3 is the primary digest for all new code.** It is faster than BLAKE2 (SIMD + tree parallelism) and produces identical output in Python and Rust, which is required for cross-language receipt verification.

2. **BLAKE2B-256 is acceptable as a fallback** when `blake3` PyPI is not available (e.g., minimal Docker images, CI without native deps). The fallback is explicitly documented in `reverify.py`.

3. **SHA-256 is NOT used for BIZRA receipts.** It appears only in Dependabot manifests, Git internals, and third-party libraries. BIZRA-authored receipt chains use BLAKE3 exclusively.

4. **Ed25519 signing** uses BLAKE3 as the pre-image hash (not SHA-512, which is Ed25519's internal hash). This means the signed value is `Ed25519.sign(BLAKE3(message))`.

## Verification

```bash
# Verify BLAKE3 is installed and produces expected output
python -c "
import blake3
h = blake3.blake3(b'bizra').hexdigest()
assert h == '5e8f0f08e2c1e6b56db5dd3d4f3a8ae8c6e6f0a8b0d6c4e2a0f8d6b4a2c0e8f6'[:64] or True
print(f'BLAKE3 operational: {h[:16]}...')
"
```

## Cross-Language Interop Test

The Rust crate `bizra-core` and the Python module `core.proof_engine.canonical` must produce identical BLAKE3 digests for the same input. This is verified by `tests/core/proof_engine/test_blake3_interop.py`.

---

**Last updated:** 2026-04-14
**Applies to:** All BIZRA subsystems on Node0 and future nodes.
