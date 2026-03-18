use criterion::{criterion_group, criterion_main, Criterion};

fn resourcepool_benchmarks(_c: &mut Criterion) {
    // TODO: Add resource pool benchmarks
}

criterion_group!(benches, resourcepool_benchmarks);
criterion_main!(benches);
