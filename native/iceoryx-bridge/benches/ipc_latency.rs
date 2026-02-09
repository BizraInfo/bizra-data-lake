//! IPC Latency Benchmarks for Iceoryx2 Bridge
//!
//! Target: 250ns roundtrip latency
//!
//! Standing on Giants: Criterion • Iceoryx2 • MessagePack

use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};

use iceoryx_bridge::{
    BridgeConfig, BridgeStats, Channel, ChannelTopology, InferenceRequest, InferenceResponse,
    MessageEnvelope,
};

// =============================================================================
// MESSAGE SERIALIZATION BENCHMARKS
// =============================================================================

/// Benchmark MessageEnvelope creation (MessagePack serialization)
fn bench_envelope_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("envelope_create");
    group.throughput(Throughput::Elements(1));

    let request = InferenceRequest {
        id: "bench-001".to_string(),
        prompt: "Explain quantum entanglement in simple terms.".to_string(),
        model_id: "sovereign-7b-v1".to_string(),
        max_tokens: 256,
        temperature: 0.7,
        top_p: 0.9,
    };

    group.bench_function("inference_request", |b| {
        b.iter(|| black_box(MessageEnvelope::inference_request(black_box(&request), "orchestrator")));
    });

    let response = InferenceResponse {
        id: "bench-001".to_string(),
        content: "Quantum entanglement is a phenomenon where two particles become linked.".to_string(),
        model_id: "sovereign-7b-v1".to_string(),
        tokens_generated: 12,
        generation_time_ms: 150,
        ihsan_score: 0.97,
        snr_score: 0.92,
        success: true,
        error: None,
    };

    group.bench_function("inference_response", |b| {
        b.iter(|| {
            black_box(MessageEnvelope::inference_response(
                black_box(&response),
                "sandbox",
            ))
        });
    });

    group.finish();
}

/// Benchmark MessageEnvelope decode (MessagePack deserialization)
fn bench_envelope_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("envelope_decode");
    group.throughput(Throughput::Elements(1));

    let request = InferenceRequest {
        id: "bench-001".to_string(),
        prompt: "Explain quantum entanglement in simple terms.".to_string(),
        model_id: "sovereign-7b-v1".to_string(),
        max_tokens: 256,
        temperature: 0.7,
        top_p: 0.9,
    };

    let envelope = MessageEnvelope::inference_request(&request, "orchestrator");

    group.bench_function("inference_request", |b| {
        b.iter(|| {
            let decoded: InferenceRequest = black_box(&envelope).decode().unwrap();
            black_box(decoded);
        });
    });

    let response = InferenceResponse {
        id: "bench-001".to_string(),
        content: "Quantum entanglement is a phenomenon where two particles become linked.".to_string(),
        model_id: "sovereign-7b-v1".to_string(),
        tokens_generated: 12,
        generation_time_ms: 150,
        ihsan_score: 0.97,
        snr_score: 0.92,
        success: true,
        error: None,
    };

    let envelope = MessageEnvelope::inference_response(&response, "sandbox");

    group.bench_function("inference_response", |b| {
        b.iter(|| {
            let decoded: InferenceResponse = black_box(&envelope).decode().unwrap();
            black_box(decoded);
        });
    });

    group.finish();
}

/// Benchmark full roundtrip: create envelope → serialize → deserialize
fn bench_envelope_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("envelope_roundtrip");
    group.throughput(Throughput::Elements(1));

    // Small payload
    let small_request = InferenceRequest {
        id: "rt-001".to_string(),
        prompt: "Hello".to_string(),
        model_id: "model".to_string(),
        max_tokens: 1,
        temperature: 0.0,
        top_p: 1.0,
    };

    group.bench_function("small_payload", |b| {
        b.iter(|| {
            let envelope = MessageEnvelope::inference_request(black_box(&small_request), "bench");
            let decoded: InferenceRequest = envelope.decode().unwrap();
            black_box(decoded);
        });
    });

    // Large payload (1KB prompt)
    let large_prompt = "a".repeat(1024);
    let large_request = InferenceRequest {
        id: "rt-002".to_string(),
        prompt: large_prompt,
        model_id: "sovereign-70b-v1".to_string(),
        max_tokens: 4096,
        temperature: 0.8,
        top_p: 0.95,
    };

    group.bench_function("large_payload_1kb", |b| {
        b.iter(|| {
            let envelope = MessageEnvelope::inference_request(black_box(&large_request), "bench");
            let decoded: InferenceRequest = envelope.decode().unwrap();
            black_box(decoded);
        });
    });

    // Very large payload (64KB prompt — max inference channel size)
    let xlarge_prompt = "b".repeat(64 * 1024);
    let xlarge_request = InferenceRequest {
        id: "rt-003".to_string(),
        prompt: xlarge_prompt,
        model_id: "sovereign-70b-v1".to_string(),
        max_tokens: 8192,
        temperature: 0.8,
        top_p: 0.95,
    };

    group.bench_function("large_payload_64kb", |b| {
        b.iter(|| {
            let envelope =
                MessageEnvelope::inference_request(black_box(&xlarge_request), "bench");
            let decoded: InferenceRequest = envelope.decode().unwrap();
            black_box(decoded);
        });
    });

    group.finish();
}

// =============================================================================
// CHANNEL METADATA BENCHMARKS
// =============================================================================

/// Benchmark channel service name lookups
fn bench_channel_metadata(c: &mut Criterion) {
    let mut group = c.benchmark_group("channel_metadata");
    group.throughput(Throughput::Elements(1));

    group.bench_function("service_name_lookup", |b| {
        b.iter(|| {
            black_box(Channel::InferenceRequest.service_name());
            black_box(Channel::InferenceResponse.service_name());
            black_box(Channel::GateRequest.service_name());
            black_box(Channel::GateResponse.service_name());
            black_box(Channel::Control.service_name());
            black_box(Channel::Metrics.service_name());
            black_box(Channel::ModelRegistry.service_name());
        });
    });

    group.bench_function("topology_calculation", |b| {
        b.iter(|| black_box(ChannelTopology::sovereign_runtime()));
    });

    group.finish();
}

// =============================================================================
// CONFIG & STATS BENCHMARKS
// =============================================================================

/// Benchmark BridgeConfig and BridgeStats operations
fn bench_config_stats(c: &mut Criterion) {
    let mut group = c.benchmark_group("config_stats");
    group.throughput(Throughput::Elements(1));

    group.bench_function("default_config", |b| {
        b.iter(|| black_box(BridgeConfig::default()));
    });

    group.bench_function("default_stats", |b| {
        b.iter(|| black_box(BridgeStats::default()));
    });

    // JSON serialization of stats (used by get_stats)
    let stats = BridgeStats {
        messages_sent: 1_000_000,
        messages_received: 999_500,
        avg_latency_ns: 180,
        p99_latency_ns: 450,
        errors: 3,
        buffer_utilization: 0.42,
    };

    group.bench_function("stats_json_serialize", |b| {
        b.iter(|| black_box(serde_json::to_string(black_box(&stats)).unwrap()));
    });

    group.finish();
}

// =============================================================================
// THROUGHPUT BENCHMARKS
// =============================================================================

/// Batch envelope creation throughput
fn bench_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("throughput");

    let request = InferenceRequest {
        id: "tput-001".to_string(),
        prompt: "Test prompt for throughput measurement.".to_string(),
        model_id: "sovereign-7b-v1".to_string(),
        max_tokens: 128,
        temperature: 0.7,
        top_p: 0.9,
    };

    let batch_size = 1000u64;
    group.throughput(Throughput::Elements(batch_size));

    group.bench_function("batch_envelope_create_1k", |b| {
        b.iter(|| {
            for _ in 0..batch_size {
                let env = MessageEnvelope::inference_request(black_box(&request), "bench");
                black_box(env);
            }
        });
    });

    let envelope = MessageEnvelope::inference_request(&request, "bench");

    group.bench_function("batch_envelope_decode_1k", |b| {
        b.iter(|| {
            for _ in 0..batch_size {
                let decoded: InferenceRequest = black_box(&envelope).decode().unwrap();
                black_box(decoded);
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_envelope_creation,
    bench_envelope_decode,
    bench_envelope_roundtrip,
    bench_channel_metadata,
    bench_config_stats,
    bench_throughput,
);
criterion_main!(benches);
