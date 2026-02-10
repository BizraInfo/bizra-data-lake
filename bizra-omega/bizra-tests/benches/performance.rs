//! BIZRA Performance Benchmarks
//!
//! Comprehensive benchmarks for:
//! - Cryptographic operations (sign, verify, hash)
//! - PCI envelope lifecycle
//! - Gate chain validation
//! - Inference tier selection
//! - Gossip protocol throughput
//! - Consensus voting
//!
//! Run: cargo run -p bizra-tests --bin bizra-bench

use std::time::{Duration, Instant};

fn main() {
    run_all_benchmarks();
    estimate_memory_usage();
}
use bizra_core::{
    domain_separated_digest,
    pci::gates::{default_gate_chain, GateContext},
    simd::{blake3_parallel, validate_gates_batch},
    Constitution, NodeIdentity, PCIEnvelope,
};
use bizra_inference::selector::{ModelSelector, TaskComplexity};

/// Benchmark result
#[derive(Clone, Debug)]
pub struct BenchResult {
    pub name: String,
    pub iterations: usize,
    pub total_time: Duration,
    pub avg_time: Duration,
    pub ops_per_sec: f64,
    pub min_time: Duration,
    pub max_time: Duration,
}

impl BenchResult {
    pub fn print(&self) {
        println!(
            "  {:30} {:>10} iters  {:>12.3?} avg  {:>12.0} ops/sec",
            self.name, self.iterations, self.avg_time, self.ops_per_sec
        );
    }
}

/// Run a benchmark
fn bench<F>(name: &str, iterations: usize, mut f: F) -> BenchResult
where
    F: FnMut(),
{
    // Warmup
    for _ in 0..10 {
        f();
    }

    let mut times = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let start = Instant::now();
        f();
        times.push(start.elapsed());
    }

    let total_time: Duration = times.iter().sum();
    let avg_time = total_time / iterations as u32;
    let min_time = *times.iter().min().unwrap();
    let max_time = *times.iter().max().unwrap();
    let ops_per_sec = iterations as f64 / total_time.as_secs_f64();

    BenchResult {
        name: name.into(),
        iterations,
        total_time,
        avg_time,
        ops_per_sec,
        min_time,
        max_time,
    }
}

/// All benchmark suites
pub fn run_all_benchmarks() -> Vec<BenchResult> {
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                    BIZRA PERFORMANCE BENCHMARKS                       ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let mut results = Vec::new();

    // Cryptographic benchmarks
    println!("📊 Cryptographic Operations");
    println!("─────────────────────────────────────────────────────────────────────────");
    results.extend(bench_crypto());

    // PCI benchmarks
    println!("\n📊 PCI Protocol");
    println!("─────────────────────────────────────────────────────────────────────────");
    results.extend(bench_pci());

    // Gate chain benchmarks
    println!("\n📊 Gate Chain Validation");
    println!("─────────────────────────────────────────────────────────────────────────");
    results.extend(bench_gates());

    // Inference benchmarks
    println!("\n📊 Inference Tier Selection");
    println!("─────────────────────────────────────────────────────────────────────────");
    results.extend(bench_inference());

    // SIMD benchmarks
    println!("\n📊 SIMD Acceleration");
    println!("─────────────────────────────────────────────────────────────────────────");
    results.extend(bench_simd());

    // Summary
    println!("\n╔══════════════════════════════════════════════════════════════════════╗");
    println!("║                           SUMMARY                                     ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝\n");

    let total_ops: f64 = results.iter().map(|r| r.ops_per_sec).sum();
    let total_iters: usize = results.iter().map(|r| r.iterations).sum();

    println!("  Total benchmarks:    {}", results.len());
    println!("  Total iterations:    {}", total_iters);
    println!("  Combined throughput: {:.0} ops/sec", total_ops);

    results
}

/// Cryptographic operation benchmarks
fn bench_crypto() -> Vec<BenchResult> {
    let mut results = Vec::new();
    let identity = NodeIdentity::generate();
    let message = b"Benchmark message for BIZRA sovereign operations";

    // Identity generation
    let r = bench("identity_generate", 100, || {
        let _ = NodeIdentity::generate();
    });
    r.print();
    results.push(r);

    // Signing
    let r = bench("ed25519_sign", 1000, || {
        let _ = identity.sign(message);
    });
    r.print();
    results.push(r);

    // Verification
    let signature = identity.sign(message);
    let r = bench("ed25519_verify", 1000, || {
        let _ = NodeIdentity::verify(message, &signature, identity.verifying_key());
    });
    r.print();
    results.push(r);

    // Domain-separated hashing
    let r = bench("blake3_domain_hash", 10000, || {
        let _ = domain_separated_digest(message);
    });
    r.print();
    results.push(r);

    // Raw BLAKE3 for comparison
    let r = bench("blake3_raw", 10000, || {
        let _ = blake3::hash(message);
    });
    r.print();
    results.push(r);

    results
}

/// PCI protocol benchmarks
fn bench_pci() -> Vec<BenchResult> {
    let mut results = Vec::new();
    let identity = NodeIdentity::generate();

    // Envelope creation
    let payload = serde_json::json!({
        "action": "inference",
        "model": "qwen2.5-7b",
        "prompt": "Test prompt"
    });

    let r = bench("envelope_create", 500, || {
        let _ = PCIEnvelope::create(&identity, payload.clone(), 3600, vec![]);
    });
    r.print();
    results.push(r);

    // Envelope verification
    let envelope = PCIEnvelope::create(&identity, payload.clone(), 3600, vec![]).unwrap();

    let r = bench("envelope_verify", 500, || {
        let _ = envelope.verify();
    });
    r.print();
    results.push(r);

    // Full envelope lifecycle
    let r = bench("envelope_lifecycle", 200, || {
        let e = PCIEnvelope::create(&identity, payload.clone(), 3600, vec![]).unwrap();
        let _ = e.verify();
    });
    r.print();
    results.push(r);

    results
}

/// Gate chain benchmarks
fn bench_gates() -> Vec<BenchResult> {
    let mut results = Vec::new();
    let chain = default_gate_chain();
    let constitution = Constitution::default();

    // Valid content
    let valid_ctx = GateContext {
        sender_id: "node_123".into(),
        envelope_id: "pci_abc123".into(),
        content: br#"{"valid": "json", "data": 42}"#.to_vec(),
        constitution: constitution.clone(),
        snr_score: Some(0.90),
        ihsan_score: Some(0.96),
    };

    let r = bench("gate_chain_valid", 5000, || {
        let _ = chain.verify(&valid_ctx);
    });
    r.print();
    results.push(r);

    // Invalid content (fails fast)
    let invalid_ctx = GateContext {
        sender_id: "node_123".into(),
        envelope_id: "pci_abc123".into(),
        content: b"invalid json".to_vec(),
        constitution: constitution.clone(),
        snr_score: Some(0.90),
        ihsan_score: Some(0.96),
    };

    let r = bench("gate_chain_invalid", 5000, || {
        let _ = chain.verify(&invalid_ctx);
    });
    r.print();
    results.push(r);

    // Large content
    let large_content = serde_json::json!({
        "data": "x".repeat(10000),
        "valid": true
    });
    let large_ctx = GateContext {
        sender_id: "node_123".into(),
        envelope_id: "pci_abc123".into(),
        content: serde_json::to_vec(&large_content).unwrap(),
        constitution,
        snr_score: Some(0.90),
        ihsan_score: Some(0.96),
    };

    let r = bench("gate_chain_large_10kb", 1000, || {
        let _ = chain.verify(&large_ctx);
    });
    r.print();
    results.push(r);

    results
}

/// Inference tier selection benchmarks
fn bench_inference() -> Vec<BenchResult> {
    let mut results = Vec::new();
    let selector = ModelSelector::default();

    // Simple prompts
    let r = bench("tier_select_simple", 10000, || {
        let c = TaskComplexity::estimate("What is 2+2?", 50);
        let _ = selector.select_tier(&c);
    });
    r.print();
    results.push(r);

    // Complex prompts
    let complex_prompt = "Explain the mathematical foundations of quantum computing, \
                          including Shor's algorithm and the quantum Fourier transform. \
                          Provide detailed examples and proofs.";
    let r = bench("tier_select_complex", 10000, || {
        let c = TaskComplexity::estimate(complex_prompt, 2000);
        let _ = selector.select_tier(&c);
    });
    r.print();
    results.push(r);

    results
}

/// SIMD acceleration benchmarks
fn bench_simd() -> Vec<BenchResult> {
    let mut results = Vec::new();
    let constitution = Constitution::default();

    // Batch gate validation (branchless)
    let contexts: Vec<GateContext> = (0..100)
        .map(|i| GateContext {
            sender_id: format!("node_{}", i),
            envelope_id: format!("pci_{}", i),
            content: br#"{"valid": "json"}"#.to_vec(),
            constitution: constitution.clone(),
            snr_score: Some(0.90),
            ihsan_score: Some(0.96),
        })
        .collect();

    let r = bench("batch_gates_100", 1000, || {
        let _ = validate_gates_batch(&contexts);
    });
    r.print();
    results.push(r);

    // Parallel hashing
    let messages: Vec<&[u8]> = vec![
        b"message_1",
        b"message_2",
        b"message_3",
        b"message_4",
        b"message_5",
        b"message_6",
        b"message_7",
        b"message_8",
    ];

    let r = bench("parallel_hash_8x", 10000, || {
        let _ = blake3_parallel(&messages);
    });
    r.print();
    results.push(r);

    // Single fast hash for comparison
    let r = bench("blake3_fast_inline", 10000, || {
        let _ = bizra_core::simd::hash::domain_hash_fast(b"benchmark message");
    });
    r.print();
    results.push(r);

    results
}

/// Memory usage estimation
pub fn estimate_memory_usage() {
    use std::mem::size_of;

    println!("\n📊 Memory Usage (struct sizes)");
    println!("─────────────────────────────────────────────────────────────────────────");
    println!("  NodeIdentity:    {} bytes", size_of::<NodeIdentity>());
    println!("  Constitution:    {} bytes", size_of::<Constitution>());
    println!("  GateContext:     {} bytes", size_of::<GateContext>());
    println!("  TaskComplexity:  {} bytes", size_of::<TaskComplexity>());
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)] // Used in test fns, not visible in bench compilation mode
    use super::{estimate_memory_usage, run_all_benchmarks};
    #[allow(unused_imports)] // Used in assertions, not visible in bench compilation mode
    use std::time::Duration;

    #[test]
    fn test_run_benchmarks() {
        let results = run_all_benchmarks();
        assert!(!results.is_empty());

        // Verify performance thresholds
        for r in &results {
            // All operations should complete in reasonable time
            assert!(
                r.avg_time < Duration::from_secs(1),
                "{} too slow: {:?}",
                r.name,
                r.avg_time
            );
        }

        estimate_memory_usage();
    }
}
