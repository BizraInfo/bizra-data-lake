//! SNR Pipeline Performance Benchmarks
//!
//! Target: 47.9M ops/sec sustained throughput

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use bizra_hunter::{
    EntropyCalculator, InvariantCache, MultiAxisEntropy, SNRPipeline,
    CriticalCascade, GateType,
};

/// Benchmark entropy calculation (TRICK 3: Multi-axis SIMD)
fn bench_entropy_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("entropy");

    // Test with various bytecode sizes
    for size in [256, 1024, 4096, 16384].iter() {
        let bytecode: Vec<u8> = (0..*size).map(|i| (i * 7 % 256) as u8).collect();

        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_function(format!("bytecode_{}", size), |b| {
            let mut calc = EntropyCalculator::new();
            b.iter(|| {
                black_box(calc.bytecode_entropy(black_box(&bytecode)))
            });
        });

        group.bench_function(format!("multi_axis_{}", size), |b| {
            let mut calc = EntropyCalculator::new();
            b.iter(|| {
                black_box(calc.calculate_all(black_box(&bytecode)))
            });
        });
    }
    group.finish();
}

/// Benchmark invariant cache (TRICK 2: O(1) deduplication)
fn bench_invariant_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("invariant_cache");

    group.throughput(Throughput::Elements(1));

    // Hash computation
    group.bench_function("compute_hash", |b| {
        let addr = [0xdeu8; 20];
        let bytecode = vec![0x60u8; 1024];
        b.iter(|| {
            black_box(InvariantCache::compute_hash(black_box(&addr), black_box(&bytecode)))
        });
    });

    // Check and insert (cold cache)
    group.bench_function("check_insert_cold", |b| {
        let cache = InvariantCache::new(1_000_000);
        let bytecode = vec![0x60u8; 1024];
        let mut counter = 0u64;
        b.iter(|| {
            let addr = counter.to_le_bytes();
            let mut full_addr = [0u8; 20];
            full_addr[..8].copy_from_slice(&addr);
            counter += 1;
            black_box(cache.check_and_insert(black_box(&full_addr), black_box(&bytecode)))
        });
    });

    // Check and insert (hot cache - duplicates)
    group.bench_function("check_insert_hot", |b| {
        let cache = InvariantCache::new(1_000_000);
        let addr = [0xdeu8; 20];
        let bytecode = vec![0x60u8; 1024];
        cache.check_and_insert(&addr, &bytecode);
        b.iter(|| {
            black_box(cache.check_and_insert(black_box(&addr), black_box(&bytecode)))
        });
    });

    group.finish();
}

/// Benchmark critical cascade (TRICK 7: Fail-safe gates)
fn bench_critical_cascade(c: &mut Criterion) {
    let mut group = c.benchmark_group("cascade");

    group.throughput(Throughput::Elements(1));

    group.bench_function("is_open_check", |b| {
        let cascade = CriticalCascade::new();
        b.iter(|| {
            black_box(cascade.is_open(black_box(GateType::Technical)))
        });
    });

    group.bench_function("all_open_check", |b| {
        let cascade = CriticalCascade::new();
        b.iter(|| {
            black_box(cascade.all_open())
        });
    });

    group.bench_function("record_success", |b| {
        let cascade = CriticalCascade::new();
        b.iter(|| {
            cascade.record_success(black_box(GateType::Technical));
        });
    });

    group.finish();
}

/// Benchmark full Lane 1 processing
fn bench_lane1_processing(c: &mut Criterion) {
    let mut group = c.benchmark_group("lane1");

    for size in [1024, 4096, 16384].iter() {
        let bytecode: Vec<u8> = (0..*size).map(|i| (i * 7 % 256) as u8).collect();

        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_function(format!("process_{}", size), |b| {
            let pipeline: SNRPipeline<65536> = SNRPipeline::new();
            let mut calc = EntropyCalculator::new();
            let mut counter = 0u64;

            b.iter(|| {
                let addr_bytes = counter.to_le_bytes();
                let mut addr = [0u8; 20];
                addr[..8].copy_from_slice(&addr_bytes);
                counter += 1;
                black_box(pipeline.process_lane1(
                    black_box(addr),
                    black_box(&bytecode),
                    black_box(&mut calc),
                ))
            });
        });
    }
    group.finish();
}

/// Benchmark queue operations
fn bench_queue_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("queue");

    group.throughput(Throughput::Elements(1));

    group.bench_function("push_pop_lane1", |b| {
        let pipeline: SNRPipeline<65536> = SNRPipeline::new();
        let result = bizra_hunter::pipeline::HeuristicResult {
            contract_addr: [0u8; 20],
            entropy: MultiAxisEntropy::new(),
            complexity: bizra_hunter::pipeline::Complexity::Medium,
            timestamp: 0,
            bounty_estimate: 1000,
        };

        b.iter(|| {
            pipeline.push_to_lane1(black_box(result));
            black_box(pipeline.pop_from_lane1())
        });
    });

    group.finish();
}

/// Benchmark throughput (ops/sec)
fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");
    group.sample_size(50);

    // Target: 47.9M ops/sec
    let batch_size = 10_000u64;
    group.throughput(Throughput::Elements(batch_size));

    group.bench_function("batch_entropy", |b| {
        let bytecode: Vec<u8> = (0..4096).map(|i| (i * 7 % 256) as u8).collect();
        let mut calc = EntropyCalculator::new();

        b.iter(|| {
            for _ in 0..batch_size {
                black_box(calc.bytecode_entropy(black_box(&bytecode)));
            }
        });
    });

    group.bench_function("batch_hash", |b| {
        let addr = [0xdeu8; 20];
        let bytecode = vec![0x60u8; 1024];

        b.iter(|| {
            for _ in 0..batch_size {
                black_box(InvariantCache::compute_hash(black_box(&addr), black_box(&bytecode)));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_entropy_calculation,
    bench_invariant_cache,
    bench_critical_cascade,
    bench_lane1_processing,
    bench_queue_operations,
    bench_throughput,
);

criterion_main!(benches);
