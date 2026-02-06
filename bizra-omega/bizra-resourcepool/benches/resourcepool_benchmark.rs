//! BIZRA Resource Pool Benchmarks

use bizra_resourcepool::*;
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;
use rust_decimal::Decimal;
use tokio::runtime::Runtime;

fn generate_keypair() -> (SigningKey, ed25519_dalek::VerifyingKey) {
    let signing_key = SigningKey::generate(&mut OsRng);
    let verifying_key = signing_key.verifying_key();
    (signing_key, verifying_key)
}

fn bench_pool_genesis(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    c.bench_function("pool_genesis", |b| {
        b.iter(|| {
            let (_, verifying_key) = generate_keypair();
            let node_id = hex::encode(verifying_key.as_bytes());

            rt.block_on(async {
                black_box(
                    ResourcePool::genesis(node_id, "BenchNode".to_string(), verifying_key)
                        .await
                        .unwrap(),
                )
            })
        })
    });
}

fn bench_gini_calculation(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let (_, verifying_key) = generate_keypair();
    let node_id = hex::encode(verifying_key.as_bytes());

    let pool = rt.block_on(async {
        ResourcePool::genesis(node_id, "BenchNode".to_string(), verifying_key)
            .await
            .unwrap()
    });

    c.bench_function("gini_calculation", |b| {
        b.iter(|| rt.block_on(async { black_box(pool.calculate_gini().await) }))
    });
}

fn bench_zakat_calculation(c: &mut Criterion) {
    let node = PoolNode {
        node_id: "a".repeat(64),
        name: "TestNode".to_string(),
        class: NodeClass::Sovereign,
        status: NodeStatus::Active,
        verifying_key: generate_keypair().1,
        place_id: uuid::Uuid::new_v4(),
        ihsan_score: Decimal::from_str("0.95").unwrap(),
        resources: NodeResources::default(),
        token_balance: 10_000_000, // 10M tokens (above nisab)
        zakat_paid_year: 0,
        last_tax_payment: chrono::Utc::now(),
        pat_agents: Vec::new(),
        registered_at: chrono::Utc::now(),
        last_activity: chrono::Utc::now(),
        node_hash: [0u8; 32],
    };

    c.bench_function("zakat_calculation", |b| {
        b.iter(|| black_box(node.calculate_zakat()))
    });
}

fn bench_harberger_tax_calculation(c: &mut Criterion) {
    let node = PoolNode {
        node_id: "a".repeat(64),
        name: "TestNode".to_string(),
        class: NodeClass::Sovereign,
        status: NodeStatus::Active,
        verifying_key: generate_keypair().1,
        place_id: uuid::Uuid::new_v4(),
        ihsan_score: Decimal::from_str("0.95").unwrap(),
        resources: NodeResources {
            cpu_millicores: 8000,
            gpu_tflops: Decimal::from_str("24.0").unwrap(),
            memory_bytes: 128 * 1024 * 1024 * 1024,
            storage_bytes: 1024 * 1024 * 1024 * 1024,
            network_bps: 10 * 1024 * 1024 * 1024,
            inference_tps: 100,
            self_assessment: 1_000_000,
            availability: Decimal::from_str("0.99").unwrap(),
        },
        token_balance: 10_000_000,
        zakat_paid_year: 0,
        last_tax_payment: chrono::Utc::now(),
        pat_agents: Vec::new(),
        registered_at: chrono::Utc::now(),
        last_activity: chrono::Utc::now(),
        node_hash: [0u8; 32],
    };

    c.bench_function("harberger_tax_calculation", |b| {
        b.iter(|| black_box(node.calculate_harberger_tax()))
    });
}

use rust_decimal::prelude::*;

criterion_group!(
    benches,
    bench_pool_genesis,
    bench_gini_calculation,
    bench_zakat_calculation,
    bench_harberger_tax_calculation,
);
criterion_main!(benches);
