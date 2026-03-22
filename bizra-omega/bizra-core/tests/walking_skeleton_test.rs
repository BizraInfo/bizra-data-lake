//! Walking Skeleton Integration Test
//!
//! THE constitutional liveness proof. If this test passes, the system is alive.
//! If it fails, something fundamental is broken.

use bizra_core::{
    walking_skeleton::{run_skeleton, SkeletonReceipt},
    IHSAN_THRESHOLD, SNR_THRESHOLD,
};

#[test]
fn walking_skeleton_proves_constitutional_liveness() {
    let receipt = run_skeleton().expect("Walking skeleton must succeed");

    // Constitutional gates must pass
    assert!(receipt.constitutional_pass, "Constitutional gate must pass");
    assert!(
        receipt.ihsan_score >= IHSAN_THRESHOLD,
        "Ihsan {:.4} must meet threshold {IHSAN_THRESHOLD}",
        receipt.ihsan_score
    );
    assert!(
        receipt.snr_score >= SNR_THRESHOLD,
        "SNR {:.4} must meet threshold {SNR_THRESHOLD}",
        receipt.snr_score
    );

    // Cryptographic artifacts must be non-trivial
    assert_ne!(
        receipt.genesis_hash, [0u8; 32],
        "Genesis hash must be non-zero"
    );
    assert_ne!(receipt.state_root, [0u8; 32], "State root must be non-zero");
    assert_ne!(
        receipt.evidence_hash, [0u8; 32],
        "Evidence hash must be non-zero"
    );

    // Exactly one cycle completed
    assert_eq!(receipt.cycle_count, 1, "Exactly one cycle must complete");

    // Meta-constitution era must be valid
    assert!(receipt.era_version >= 1, "Era version must be >= 1");

    // Timestamp must be present
    assert!(!receipt.timestamp.is_empty(), "Timestamp must be present");

    // Performance: must complete in under 1 second
    assert!(
        receipt.elapsed_us < 1_000_000,
        "Skeleton path took {}us, must complete in <1s",
        receipt.elapsed_us
    );

    // Print receipt for CI visibility
    println!("=== WALKING SKELETON RECEIPT ===");
    println!(
        "  Genesis:      {}",
        hex::encode(&receipt.genesis_hash[..8])
    );
    println!("  Cycle count:  {}", receipt.cycle_count);
    println!("  Ihsan score:  {:.4}", receipt.ihsan_score);
    println!("  SNR score:    {:.4}", receipt.snr_score);
    println!("  Constitutional: {}", receipt.constitutional_pass);
    println!("  State root:   {}", hex::encode(&receipt.state_root[..8]));
    println!(
        "  Evidence:     {}",
        hex::encode(&receipt.evidence_hash[..8])
    );
    println!("  Era version:  {}", receipt.era_version);
    println!("  Elapsed:      {}us", receipt.elapsed_us);
    println!("  Timestamp:    {}", receipt.timestamp);
    println!("================================");
}

#[test]
fn walking_skeleton_is_deterministic() {
    let r1 = run_skeleton().unwrap();
    let r2 = run_skeleton().unwrap();

    // Core fields must be identical across runs
    assert_eq!(r1.genesis_hash, r2.genesis_hash);
    assert_eq!(r1.cycle_count, r2.cycle_count);
    assert_eq!(r1.state_root, r2.state_root);
    assert_eq!(r1.constitutional_pass, r2.constitutional_pass);
    assert_eq!(r1.era_version, r2.era_version);
    assert!((r1.ihsan_score - r2.ihsan_score).abs() < f64::EPSILON);
    assert!((r1.snr_score - r2.snr_score).abs() < f64::EPSILON);
}

#[test]
fn walking_skeleton_receipt_serializes_to_json() {
    let receipt = run_skeleton().unwrap();
    let json = serde_json::to_string_pretty(&receipt).expect("Receipt must serialize to JSON");

    // Verify it round-trips
    let deserialized: SkeletonReceipt =
        serde_json::from_str(&json).expect("Receipt must deserialize from JSON");

    assert_eq!(receipt.genesis_hash, deserialized.genesis_hash);
    assert_eq!(receipt.state_root, deserialized.state_root);
    assert_eq!(receipt.evidence_hash, deserialized.evidence_hash);
    assert_eq!(receipt.cycle_count, deserialized.cycle_count);
}
