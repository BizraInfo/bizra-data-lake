// examples/receipt_signing_demo.rs - Demonstration of receipt signing
//
// Run with: cargo run --example receipt_signing_demo

use meta_alpha_dual_agentic::signing::{canonical_json_bytes, ReceiptSigner};
use serde::Serialize;

#[derive(Serialize)]
struct ExampleReceipt {
    receipt_id: String,
    task: String,
    ihsan_score: f64,
    timestamp: String,
}

fn main() -> anyhow::Result<()> {
    println!("🔐 BIZRA Receipt Signing Demo\n");

    // 1. Load or generate signing key from keystore
    println!("Loading signing key from ~/.bizra/node0/keys/signing.key...");
    let signer = ReceiptSigner::from_keystore()?;
    println!("✅ Loaded key with public key: {}\n", signer.public_key_hex());

    // 2. Create a test receipt
    let receipt = ExampleReceipt {
        receipt_id: "EXEC-20260214-001".to_string(),
        task: "Generate unit tests".to_string(),
        ihsan_score: 0.97,
        timestamp: "2026-02-14T12:00:00Z".to_string(),
    };

    // 3. Serialize to canonical JSON (deterministic)
    println!("Serializing receipt to canonical JSON...");
    let canonical_json = canonical_json_bytes(&receipt);
    let json_str = String::from_utf8_lossy(&canonical_json);
    println!("Canonical JSON: {}\n", json_str);

    // 4. Sign the receipt
    println!("Signing receipt with Ed25519...");
    let signature = signer.sign_receipt(&canonical_json);
    println!("✅ Signature: {}", signature.signature_hex);
    println!("✅ Domain: {}\n", signature.domain);

    // 5. Verify the signature
    println!("Verifying signature...");
    let is_valid = ReceiptSigner::verify_receipt(
        &signature.signer_public_key,
        &canonical_json,
        &signature.signature_hex,
    )?;

    if is_valid {
        println!("✅ Signature verification PASSED\n");
    } else {
        println!("❌ Signature verification FAILED\n");
    }

    // 6. Test tampering detection
    println!("Testing tampering detection...");
    let mut tampered_receipt = receipt;
    tampered_receipt.ihsan_score = 0.50; // Tamper with score
    let tampered_json = canonical_json_bytes(&tampered_receipt);

    let is_valid_tampered = ReceiptSigner::verify_receipt(
        &signature.signer_public_key,
        &tampered_json,
        &signature.signature_hex,
    )?;

    if is_valid_tampered {
        println!("❌ Tampered receipt incorrectly validated (SECURITY BUG!)");
    } else {
        println!("✅ Tampered receipt correctly rejected\n");
    }

    println!("🎯 Receipt signing system operational!");
    println!("📍 Key location: ~/.bizra/node0/keys/");
    println!("🔒 Domain separation: bizra-receipt-v1:");
    println!("📋 Canonical JSON: RFC 8785 JCS (sorted keys, no whitespace)");

    Ok(())
}
