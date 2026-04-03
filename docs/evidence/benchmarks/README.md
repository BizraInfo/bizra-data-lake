# Benchmark Evidence Directory

This directory contains benchmark results and evidence for BIZRA quality measurements.

## File Naming Conventions

| Pattern | Description | Example |
|---------|-------------|---------|
| `model_baselines_YYYYMMDD_HHMMSS.json` | Individual model benchmark results | `model_baselines_20260122_143022.json` |
| `comparison_YYYYMMDD_HHMMSS.json` | Mode comparison results (Direct/Routed/BIZRA) | `comparison_20260122_143022.json` |
| `benchmark_TESTSET_YYYYMMDD_HHMMSS.json` | Specific test set results | `benchmark_mmlu_mini_20260122_143022.json` |

## File Structure

### Model Baselines (`model_baselines_*.json`)

```json
{
  "run_id": "baseline_TIMESTAMP",
  "timestamp": "ISO-8601",
  "test_set": "mmlu_mini|hellaswag_mini|bizra_qa",
  "results": [
    {
      "model_name": "deepseek-r1:14b",
      "provider": "ollama",
      "metrics": {
        "accuracy": 0.75,
        "latency_p50_ms": 1200,
        "latency_p95_ms": 2100,
        "tokens_per_second": 45.2,
        "total_questions": 100,
        "correct": 75,
        "errors": 2
      }
    }
  ]
}
```

### Comparison Results (`comparison_*.json`)

```json
{
  "run_id": "compare_TIMESTAMP",
  "timestamp": "ISO-8601",
  "test_set": "mmlu_mini",
  "modes": ["direct", "routed", "bizra"],
  "results": [
    {
      "mode": "direct",
      "model": "deepseek-r1:14b",
      "metrics": { ... }
    },
    {
      "mode": "bizra",
      "metrics": { ... },
      "bizra_metrics": {
        "ihsan_score": 0.97,
        "snr_score": 0.92,
        "sat_consensus_rate": 0.95
      }
    }
  ],
  "comparison_summary": {
    "best_accuracy": {"mode": "bizra", "value": 0.82},
    "accuracy_delta": 0.07,
    "latency_overhead": 1.8
  }
}
```

## Integration

These files are read by:

- `scripts/quality_radar_elite.py` - Benchmark probes
- `scripts/benchmark_dashboard.py` - Dashboard generation
- CI/CD pipeline for regression detection

## Retention Policy

- Keep last 30 days of benchmark results
- Archive older results to `docs/evidence/benchmarks/archive/`
- Critical regression evidence is preserved indefinitely
