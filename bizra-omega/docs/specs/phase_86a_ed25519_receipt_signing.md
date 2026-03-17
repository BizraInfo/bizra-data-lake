# Phase 86-A: Ed25519 Receipt Signing
## Cryptographic Closure of the Receipt Pipeline

**Sprint Ref**: B2 (Architecture Hardening)
**Gap Source**: Phase 81 Multi-Lens §8 gap #2, Phase 84 Security Audit finding #5
**Standing On**: Bernstein (2006) Ed25519, Aumasson (2015) BLAKE3

---

## 1. Problem Statement

```
Current:  MissionReceipt.signature = [0u8; 64]  // placeholder
          ConstitutionalReceipt.signature = [0u8; 64]  // placeholder

Required: Every receipt MUST be signed by the emitting node's Ed25519 key
          verify(receipt, node_public_key) → bool

Why:      Without signatures, anyone with write access can forge receipts.
          The entire SEED economy rests on receipt integrity.
          ZANN_ZERO: "No claim without receipts" requires SIGNED receipts.
```

## 2. Pseudocode — MissionReceipt Signing

```rust
// ── receipt.rs additions ──────────────────────────────

use ed25519_dalek::{Signer, SigningKey, VerifyingKey, Signature};

impl MissionReceipt {
    /// Sign this receipt with the node's Ed25519 signing key.
    /// Called exactly once, immediately after from_mission().
    ///
    /// TDD anchor: test_receipt_signature_valid
    pub fn sign(&mut self, signing_key: &SigningKey) {
        // The message to sign = receipt_id (already BLAKE3 of all fields)
        let sig: Signature = signing_key.sign(&self.receipt_id);
        self.signature = sig.to_bytes();
    }

    /// Verify this receipt's signature against a public key.
    ///
    /// TDD anchor: test_receipt_signature_verify
    /// TDD anchor: test_tampered_receipt_signature_fails
    pub fn verify_signature(&self, verifying_key: &VerifyingKey) -> bool {
        let Ok(sig) = Signature::from_bytes(&self.signature) else {
            return false;  // malformed signature
        };
        verifying_key.verify_strict(&self.receipt_id, &sig).is_ok()
    }

    /// Full integrity check: hash + signature + chain link.
    ///
    /// TDD anchor: test_full_receipt_integrity
    pub fn verify_full(
        &self,
        verifying_key: &VerifyingKey,
        previous: Option<&MissionReceipt>,
    ) -> bool {
        // 1. Hash integrity
        if !self.verify_hash() { return false; }
        // 2. Signature integrity
        if !self.verify_signature(verifying_key) { return false; }
        // 3. Chain integrity (if previous exists)
        if let Some(prev) = previous {
            if !self.verify_chain(prev) { return false; }
        }
        true
    }
}
```

## 3. Pseudocode — Node Key Management

```rust
// ── node.rs additions ─────────────────────────────────

use ed25519_dalek::{SigningKey, VerifyingKey};

struct Node {
    // ... existing fields ...

    /// Node's Ed25519 signing key (sovereign identity)
    /// Generated at genesis, persisted to sovereign store.
    signing_key: SigningKey,

    /// Derived verifying (public) key
    verifying_key: VerifyingKey,
}

impl Node {
    pub fn new(config: NodeConfig) -> Self {
        // Key source priority:
        // 1. Load from sovereign store (B:\ or ~/.bizra/identity/)
        // 2. Generate ephemeral key for development
        // 3. NEVER hardcode
        let signing_key = load_or_generate_key(&config);
        let verifying_key = VerifyingKey::from(&signing_key);
        // ...
    }
}

/// Load key from sovereign store or generate ephemeral.
///
/// TDD anchor: test_key_load_from_file
/// TDD anchor: test_key_generate_ephemeral
/// TDD anchor: test_key_deterministic_from_seed
fn load_or_generate_key(config: &NodeConfig) -> SigningKey {
    let key_path = sovereign_key_path(config);

    if key_path.exists() {
        // Load existing key
        let bytes = std::fs::read(&key_path)
            .expect("sovereign key file readable");
        // Validate: must be exactly 32 bytes
        assert_eq!(bytes.len(), 32, "corrupt sovereign key file");
        SigningKey::from_bytes(&bytes.try_into().unwrap())
    } else {
        // Generate from OS entropy (sovereign birth)
        let key = SigningKey::generate(&mut rand::rngs::OsRng);
        // Persist immediately — this is the node's identity
        std::fs::create_dir_all(key_path.parent().unwrap()).ok();
        std::fs::write(&key_path, key.to_bytes()).ok();
        key
    }
}

fn sovereign_key_path(config: &NodeConfig) -> PathBuf {
    // $BIZRA_SOVEREIGN_ROOT/identity/node0.ed25519
    // Falls back to ~/.bizra/identity/node0.ed25519
    let root = std::env::var("BIZRA_SOVEREIGN_ROOT")
        .unwrap_or_else(|_| {
            home_dir().join(".bizra").to_string_lossy().to_string()
        });
    PathBuf::from(root).join("identity").join("node0.ed25519")
}
```

## 4. Pseudocode — Mission Bridge Signing

```rust
// ── mission_bridge.rs additions ───────────────────────

pub fn execute_governed_mission(
    runtime: &mut AgentRuntime,
    ihsan: &IhsanScore,
    content: &str,
    timestamp: u64,
    available_models: &[String],
    previous_receipt: Option<[u8; 32]>,
    signing_key: &SigningKey,          // ← NEW: sign every receipt
) -> MissionResult {
    // ... existing lifecycle ...

    // After receipt emission (in complete/fail/degrade paths):
    // Sign the receipt with the node's sovereign key
    if let Some(ref mut receipt) = m.receipt {
        receipt.sign(signing_key);
    }

    // ... return MissionResult ...
}
```

## 5. TDD Test Matrix

```
TEST                                    GATE        PROPERTY
─────────────────────────────────────────────────────────────
test_receipt_signature_valid            Unit        sign() + verify_signature() = true
test_tampered_receipt_sig_fails         Unit        tamper any byte → verify = false
test_wrong_key_sig_fails                Unit        different key → verify = false
test_full_receipt_integrity             Unit        verify_full() checks hash+sig+chain
test_key_load_from_file                 Unit        persisted key loads correctly
test_key_generate_ephemeral             Unit        new key is 32 bytes, unique
test_governed_mission_signed            Integration mission bridge produces signed receipts
test_receipt_chain_all_signed           Integration 3 chained receipts, all verifiable
test_sign_performance_under_1ms         Perf        Ed25519 sign < 1ms per receipt
```

## 6. Security Considerations

```
THREAT                          MITIGATION
───────────────────────────────────────────────────────────
Key theft (file read)           File permissions 0600, sovereign store only
Key in memory after use         Zeroize on drop (ed25519-dalek feature)
Weak randomness                 OsRng only (no deterministic seeds in prod)
Replay attack                   receipt_id includes timestamp + previous_hash
Signature malleability          verify_strict() not verify() (rejects S-form)
Test key in production          BIZRA_ENV != "production" assertion on ephemeral
```

## 7. Acceptance Criteria

```
[x] verify() returns True for valid signatures
[x] verify() returns False for tampered receipts
[x] verify() returns False for wrong node key
[x] All existing 1,381 tests still pass
[x] Zero clippy warnings
[x] Ed25519 sign latency < 1ms (measured)
[x] Key persisted to sovereign store on first boot
[x] No hardcoded keys anywhere in codebase
```
