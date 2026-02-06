//! Benchmark suite for TELESCRIPT-BIZRA BRIDGE
//!
//! Measures performance of:
//! - Authority chain verification
//! - Permit verification
//! - FATE gate checks
//! - Gini coefficient calculation
//! - Agent creation and travel

use bizra_telescript::*;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn bench_authority_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("authority_chain");

    for depth in [1, 3, 5, 7] {
        group.bench_with_input(
            BenchmarkId::new("verify_depth", depth),
            &depth,
            |b, &depth| {
                let mut auth = Authority::genesis();
                for i in 0..depth {
                    auth = auth.delegate(&format!("node{}", i)).unwrap();
                }
                b.iter(|| black_box(auth.verify_chain()));
            },
        );
    }

    group.finish();
}

fn bench_permit_verification(c: &mut Criterion) {
    let genesis = Authority::genesis();
    let permit = Permit::new(
        genesis,
        vec![Capability::Go, Capability::Meet, Capability::Compute],
        ResourceLimits::default(),
        3600,
    );

    c.bench_function("permit_verify", |b| {
        b.iter(|| black_box(permit.verify()));
    });
}

fn bench_gini_calculation(c: &mut Criterion) {
    let mut group = c.benchmark_group("gini_coefficient");

    for size in [10, 100, 1000, 10000] {
        let values: Vec<f64> = (0..size).map(|i| i as f64).collect();

        group.bench_with_input(BenchmarkId::new("calculate", size), &values, |b, values| {
            b.iter(|| black_box(TelescriptEngine::calculate_gini(values)));
        });
    }

    group.finish();
}

fn bench_place_fate_check(c: &mut Criterion) {
    let place = Place::new("test://benchmark", None);

    c.bench_function("place_fate_check", |b| {
        b.iter(|| black_box(place.passes_fate()));
    });
}

fn bench_ticket_operations(c: &mut Criterion) {
    let genesis = Authority::genesis();
    let permit = Permit::new(
        genesis,
        vec![Capability::Go],
        ResourceLimits::default(),
        3600,
    );
    let agent = Agent::new("benchmark_agent", permit, vec![1, 2, 3]);
    let from_place = uuid::Uuid::new_v4();
    let to_place = uuid::Uuid::new_v4();

    let mut group = c.benchmark_group("ticket_ops");

    group.bench_function("ticket_issue", |b| {
        b.iter(|| black_box(Ticket::issue(&agent, from_place, to_place, 300).unwrap()));
    });

    let ticket = Ticket::issue(&agent, from_place, to_place, 300).unwrap();
    group.bench_function("ticket_verify", |b| {
        b.iter(|| black_box(ticket.verify()));
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_authority_chain,
    bench_permit_verification,
    bench_gini_calculation,
    bench_place_fate_check,
    bench_ticket_operations,
);

criterion_main!(benches);
