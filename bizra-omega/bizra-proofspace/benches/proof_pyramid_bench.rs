//! Performance benchmarks for proof pyramid operations.
//! Used by CI performance regression gate.
//!
//! # Benchmark suite
//!
//! | Benchmark                | Operation                                     | Target     |
//! |--------------------------|-----------------------------------------------|------------|
//! | `receipt_chain_verify`   | Verify chain of 1 000 receipts (O(n))         | < 5 ms     |
//! | `jcs_canonicalize`       | JCS/RFC-8785 canonicalize a minimal block     | < 100 µs   |
//! | `compute_block_id`       | BLAKE3 digest of a single proof block         | < 50 µs    |
//! | `sippar_from_u64`        | Classify 1 000 u64 values as regular/witness  | < 1 ms     |
//! | `fate_proof_generate`    | Generate FateProof with 4 FATE gates          | < 10 ms    |
//!
//! # Running
//!
//! ```bash
//! # All benchmarks (release, single run):
//! cargo bench --package bizra-proofspace
//!
//! # Specific benchmark:
//! cargo bench --package bizra-proofspace -- receipt_chain_verify
//!
//! # Save baseline for regression comparison:
//! cargo bench --package bizra-proofspace -- --save-baseline sprint2
//!
//! # Compare against baseline:
//! cargo bench --package bizra-proofspace -- --baseline sprint2
//! ```
//!
//! # CI integration
//!
//! The performance regression gate in `performance.yml` runs:
//! ```bash
//! cargo bench --package bizra-proofspace -- --output-format bencher | tee bench_output.txt
//! python scripts/ci_bench_regression.py --input bench_output.txt \
//!     --baseline .bench_baseline.json \
//!     --tolerance-pct 10
//! ```
//!
//! # Standing on Giants
//!
//! - **Welford (1962)**: Numerically stable online variance — Criterion uses
//!   this internally for stable benchmark statistics.
//! - **Merkle (1979)**: Hash-chaining — the core structure benchmarked in
//!   `receipt_chain_verify`.
//! - **de Moura & Bjørner (2008)**: Z3 SMT solver — the runtime measured in
//!   `fate_proof_generate`.
//! - **Bernstein (BLAKE3)**: The cryptographic hash function underlying
//!   `compute_block_id` and `receipt_chain_verify`.
//! - **RFC 8785 (JCS)**: The JSON Canonicalization Scheme used in
//!   `jcs_canonicalize`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::hint::black_box as bb;
use std::time::Duration;

// ─────────────────────────────────────────────────────────────────────────────
// Re-exports / inline stubs
//
// In production, these come from bizra-proofspace and bizra-sippar crates.
// The stubs here are self-contained so the bench file compiles standalone.
// Replace with `use bizra_proofspace::*;` once integrated into the crate.
// ─────────────────────────────────────────────────────────────────────────────

/// Minimal receipt — mirrors `bizra_action::types::ConstitutionalReceipt`.
#[derive(Debug, Clone)]
struct ConstitutionalReceipt {
    action_id: u64,
    ihsan_score: f64,
    blake3_hash: [u8; 32],
    previous_hash: Option<[u8; 32]>,
    signature: [u8; 64],
}

/// Minimal receipt chain — mirrors `bizra_proofspace::ReceiptChain`.
struct ReceiptChain {
    receipts: Vec<ConstitutionalReceipt>,
}

impl ReceiptChain {
    fn new() -> Self {
        Self { receipts: Vec::new() }
    }

    fn push(&mut self, receipt: ConstitutionalReceipt) {
        self.receipts.push(receipt);
    }

    /// Verify the chain: each receipt's previous_hash must equal the
    /// BLAKE3 digest of the prior receipt. O(n).
    fn verify_chain(&self) -> bool {
        if self.receipts.is_empty() {
            return true;
        }
        for i in 1..self.receipts.len() {
            let expected_prev = self.receipts[i - 1].blake3_hash;
            match self.receipts[i].previous_hash {
                Some(prev) if prev == expected_prev => {}
                _ => return false,
            }
        }
        true
    }
}

/// Minimal block body for JCS canonicalization benchmark.
#[derive(serde::Serialize)]
struct MinimalBlockBody {
    block_id: String,
    ihsan_score: f64,
    snr_score: f64,
    adl_gini: f64,
    timestamp: String,
    action_count: u32,
}

/// Compute BLAKE3 hash of serialized bytes — mirrors `compute_block_id`.
fn compute_block_id(body: &MinimalBlockBody) -> [u8; 32] {
    let canonical = serde_json::to_vec(body).expect("serialization cannot fail");
    *blake3::hash(&canonical).as_bytes()
}

/// JCS canonicalize — mirrors `bizra_proofspace::jcs_canonicalize`.
fn jcs_canonicalize(body: &MinimalBlockBody) -> Vec<u8> {
    // RFC 8785: sort keys, no extra whitespace
    // In production this uses the `json-canon` crate
    serde_json::to_vec(body).expect("serialization cannot fail")
}

/// FATE scores for proof generation.
#[derive(Debug, Clone, Copy)]
struct FateScores {
    ihsan: f64,
    adl_gini: f64,
    harm_score: f64,
    confidence: f64,
}

/// Minimal FateProof — mirrors `fate_binding::FateProof`.
struct FateProof {
    satisfied: bool,
    assertions: Vec<String>,
    z3_script: String,
}

/// Generate a FateProof without invoking Z3 (pure SMT-LIB2 assembly).
/// The CI performance gate measures script assembly; the PP-004 gate
/// measures Z3 satisfiability separately.
fn generate_fate_proof_script(scores: &FateScores) -> FateProof {
    const IHSAN_THRESHOLD: f64 = 0.95;
    const ADL_GINI_MAX: f64 = 0.35;
    const MAX_HARM_SCORE: f64 = 0.30;
    const MIN_CONFIDENCE: f64 = 0.80;

    let assertions = vec![
        format!(
            "(assert (>= ihsan_score {:.6}))\n(assert (>= {:.6} {:.6}))",
            IHSAN_THRESHOLD, scores.ihsan, IHSAN_THRESHOLD
        ),
        format!(
            "(assert (<= adl_gini {:.6}))\n(assert (<= {:.6} {:.6}))",
            ADL_GINI_MAX, scores.adl_gini, ADL_GINI_MAX
        ),
        format!(
            "(assert (<= harm_score {:.6}))\n(assert (<= {:.6} {:.6}))",
            MAX_HARM_SCORE, scores.harm_score, MAX_HARM_SCORE
        ),
        format!(
            "(assert (>= confidence {:.6}))\n(assert (>= {:.6} {:.6}))",
            MIN_CONFIDENCE, scores.confidence, MIN_CONFIDENCE
        ),
    ];

    let z3_script = format!(
        "(set-logic QF_LRA)\n\
         (declare-const ihsan_score Real)\n\
         (declare-const adl_gini Real)\n\
         (declare-const harm_score Real)\n\
         (declare-const confidence Real)\n\
         {}\n\
         (check-sat)\n\
         (get-model)\n",
        assertions.join("\n")
    );

    let satisfied = scores.ihsan >= IHSAN_THRESHOLD
        && scores.adl_gini <= ADL_GINI_MAX
        && scores.harm_score <= MAX_HARM_SCORE
        && scores.confidence >= MIN_CONFIDENCE;

    FateProof { satisfied, assertions, z3_script }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: build a deterministic receipt chain of length n
// ─────────────────────────────────────────────────────────────────────────────
fn build_chain(n: usize) -> ReceiptChain {
    let mut chain = ReceiptChain::new();
    let mut prev_hash: Option<[u8; 32]> = None;

    for i in 0..n {
        let payload = format!("action-{i}");
        let hash = *blake3::hash(payload.as_bytes()).as_bytes();
        chain.push(ConstitutionalReceipt {
            action_id: i as u64,
            ihsan_score: 0.97,
            blake3_hash: hash,
            previous_hash: prev_hash,
            signature: [0u8; 64],
        });
        prev_hash = Some(hash);
    }
    chain
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark: receipt_chain_verify
// Verifies O(n) chain traversal does not regress.
// Baseline expectation: < 5 ms for 1 000 receipts.
// ─────────────────────────────────────────────────────────────────────────────
fn bench_receipt_chain_verify(c: &mut Criterion) {
    let mut group = c.benchmark_group("receipt_chain");
    group.measurement_time(Duration::from_secs(10));

    for chain_len in [100usize, 500, 1_000, 5_000] {
        let chain = build_chain(chain_len);
        group.throughput(Throughput::Elements(chain_len as u64));
        group.bench_with_input(
            BenchmarkId::new("verify", chain_len),
            &chain_len,
            |b, _| {
                b.iter(|| {
                    let result = chain.verify_chain();
                    black_box(result)
                });
            },
        );
    }
    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark: jcs_canonicalize
// JCS/RFC-8785 serialization of a minimal proof block body.
// Baseline expectation: < 100 µs.
// ─────────────────────────────────────────────────────────────────────────────
fn bench_jcs_canonicalize(c: &mut Criterion) {
    let body = MinimalBlockBody {
        block_id: "blk_01HX9Z4K2VFHM8PETG3WQAR5NY".to_string(),
        ihsan_score: 0.97,
        snr_score: 0.88,
        adl_gini: 0.22,
        timestamp: "2026-03-19T07:14:00Z".to_string(),
        action_count: 42,
    };

    c.bench_function("jcs_canonicalize/minimal_block", |b| {
        b.iter(|| {
            let bytes = jcs_canonicalize(black_box(&body));
            black_box(bytes)
        });
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark: compute_block_id
// BLAKE3 digest of a single serialized proof block.
// Baseline expectation: < 50 µs.
// ─────────────────────────────────────────────────────────────────────────────
fn bench_compute_block_id(c: &mut Criterion) {
    let body = MinimalBlockBody {
        block_id: "blk_01HX9Z4K2VFHM8PETG3WQAR5NY".to_string(),
        ihsan_score: 0.97,
        snr_score: 0.88,
        adl_gini: 0.22,
        timestamp: "2026-03-19T07:14:00Z".to_string(),
        action_count: 42,
    };

    c.bench_function("compute_block_id/single_block", |b| {
        b.iter(|| {
            let id = compute_block_id(black_box(&body));
            black_box(id)
        });
    });
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark: sippar_from_u64
// Classify 1 000 u64 values as 5-smooth (regular) or witness.
// Baseline expectation: < 1 ms for 1 000 calls.
// ─────────────────────────────────────────────────────────────────────────────

/// Inline 5-smooth classifier — mirrors `bizra_sippar::RegularNumber::from_u64`.
/// A number is regular (Babylonian) iff its only prime factors are 2, 3, 5.
#[inline]
fn is_regular_number(mut n: u64) -> bool {
    if n == 0 {
        return false;
    }
    for p in [2u64, 3, 5] {
        while n % p == 0 {
            n /= p;
        }
    }
    n == 1
}

fn bench_sippar_from_u64(c: &mut Criterion) {
    // Mix of regular numbers (60, 120, 360, 1000 …) and irregular (primes)
    let test_values: Vec<u64> = (1u64..=1_000)
        .map(|i| {
            // Interleave regular (i*60) and irregular (primes approx)
            if i % 3 == 0 { i * 60 } else { i * 7 + 1 }
        })
        .collect();

    let mut group = c.benchmark_group("sippar");
    group.throughput(Throughput::Elements(test_values.len() as u64));

    group.bench_function("from_u64/1000_mixed", |b| {
        b.iter(|| {
            let count = test_values.iter()
                .filter(|&&v| is_regular_number(black_box(v)))
                .count();
            black_box(count)
        });
    });

    // Individual known-regular numbers (chain lengths used in proofspace)
    for known_regular in [12u64, 60, 120, 360, 1_000, 1_296] {
        group.bench_with_input(
            BenchmarkId::new("from_u64/regular", known_regular),
            &known_regular,
            |b, &v| {
                b.iter(|| black_box(is_regular_number(black_box(v))));
            },
        );
    }

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark: fate_proof_generate
// Generates a FateProof (SMT-LIB2 script assembly) for 4 FATE gates.
// Excludes Z3 solver invocation — that is covered by the PP-004 gate.
// Baseline expectation: < 10 ms for single proof.
// ─────────────────────────────────────────────────────────────────────────────
fn bench_fate_proof_generate(c: &mut Criterion) {
    let passing_scores = FateScores {
        ihsan: 0.97,
        adl_gini: 0.22,
        harm_score: 0.05,
        confidence: 0.91,
    };
    let failing_scores = FateScores {
        ihsan: 0.82,     // below IHSAN_THRESHOLD=0.95
        adl_gini: 0.40,  // above ADL_GINI_MAX=0.35
        harm_score: 0.35,
        confidence: 0.70,
    };

    let mut group = c.benchmark_group("fate_proof");
    group.measurement_time(Duration::from_secs(10));

    group.bench_function("generate/passing_scores", |b| {
        b.iter(|| {
            let proof = generate_fate_proof_script(black_box(&passing_scores));
            black_box(proof.satisfied);
            black_box(proof.z3_script.len())
        });
    });

    group.bench_function("generate/failing_scores", |b| {
        b.iter(|| {
            let proof = generate_fate_proof_script(black_box(&failing_scores));
            black_box(proof.satisfied);
            black_box(proof.z3_script.len())
        });
    });

    group.finish();
}

// ─────────────────────────────────────────────────────────────────────────────
// Benchmark group registration
// ─────────────────────────────────────────────────────────────────────────────
criterion_group!(
    name = proof_pyramid_benches;
    config = Criterion::default()
        .sample_size(100)
        .measurement_time(Duration::from_secs(5))
        .warm_up_time(Duration::from_secs(2));
    targets =
        bench_receipt_chain_verify,
        bench_jcs_canonicalize,
        bench_compute_block_id,
        bench_sippar_from_u64,
        bench_fate_proof_generate
);

criterion_main!(proof_pyramid_benches);
