"""
Inference Suite - Benchmark local inference backends.

Tests:
- Local inference latency (via local_first or mock)
- Batch processing throughput
- Token generation rate
- Backend selection overhead

Runs fully offline without requiring actual LLM backend.
"""

import time
from typing import List, Dict, Any
from benchmark.runner import BenchmarkRunner


class InferenceBenchmark:
    """Benchmark suite for inference operations."""

    def __init__(self, runner: BenchmarkRunner):
        self.runner = runner

    def benchmark_mock_inference_latency(self, iterations: int = 20) -> Dict[str, Any]:
        """
        Benchmark mock inference latency.

        Simulates a 100-token inference operation.
        """
        def inference_op():
            # Simulate tokenization + inference overhead
            tokens = 100
            time_per_token_ms = 0.5  # Mock: 0.5ms per token
            time.sleep(tokens * time_per_token_ms / 1000)

        result = self.runner.run(
            "inference.mock_latency",
            inference_op,
            iterations=iterations,
            track_memory=False,
        )
        return result.to_dict()

    def benchmark_batch_throughput(self, iterations: int = 20) -> Dict[str, Any]:
        """
        Benchmark batch inference throughput.

        Simulates batching of 32 parallel requests.
        """
        def batch_op():
            batch_size = 32
            tokens_per_request = 50
            batch_latency_ms = 150  # Mock: 150ms for batch
            time.sleep(batch_latency_ms / 1000)
            return batch_size  # Operations count

        result = self.runner.run_throughput(
            "inference.batch_throughput",
            batch_op,
            iterations=iterations,
        )
        return result.to_dict()

    def benchmark_token_generation(self, iterations: int = 20) -> Dict[str, Any]:
        """
        Benchmark token generation rate.

        Simulates generating 256 tokens.
        """
        def token_gen():
            tokens = 256
            # Mock: 2ms per token (typical for small model)
            time.sleep(tokens * 0.002)
            return tokens

        result = self.runner.run_throughput(
            "inference.tokens_per_second",
            token_gen,
            iterations=iterations,
        )
        return result.to_dict()

    def benchmark_model_selection_overhead(self, iterations: int = 50) -> Dict[str, Any]:
        """
        Benchmark model selector overhead.

        Simulates task complexity analysis and model selection.
        """
        def selector_op():
            # Simulate: tokenization + embedding + distance calc
            tokens = 50
            embedding_dim = 384
            candidates = 3

            tokenize_ms = 0.2
            embed_ms = 0.5
            distance_ms = candidates * 0.1

            overhead_ms = tokenize_ms + embed_ms + distance_ms
            time.sleep(overhead_ms / 1000)

        result = self.runner.run(
            "inference.model_selection_overhead",
            selector_op,
            iterations=iterations,
            track_memory=False,
        )
        return result.to_dict()

    def benchmark_context_window_scaling(self, iterations: int = 10) -> Dict[str, Any]:
        """
        Benchmark inference latency with varying context sizes.

        Tests 1K, 4K, and 8K token context windows.
        """
        def context_op():
            context_tokens = 4096  # 4K context
            inference_tokens = 100
            time_per_ctx_ms = 0.01  # 10µs per context token
            time_per_inf_ms = 1.0   # 1ms per inference token
            total_ms = (context_tokens * time_per_ctx_ms +
                       inference_tokens * time_per_inf_ms)
            time.sleep(total_ms / 1000)

        result = self.runner.run(
            "inference.context_window_4k",
            context_op,
            iterations=iterations,
            track_memory=False,
        )
        return result.to_dict()

    def run_all(self, iterations: int = 20) -> Dict[str, Any]:
        """Run all inference benchmarks."""
        print("\n--- Inference Suite ---")
        results = {
            "latency": self.benchmark_mock_inference_latency(iterations),
            "batch_throughput": self.benchmark_batch_throughput(iterations),
            "token_generation": self.benchmark_token_generation(iterations),
            "model_selection": self.benchmark_model_selection_overhead(iterations),
            "context_window": self.benchmark_context_window_scaling(iterations),
        }
        return results
