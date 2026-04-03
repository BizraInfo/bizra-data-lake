#!/usr/bin/env python3
"""
BIZRA Model Baseline Benchmark Suite v1.0
=========================================

Measures individual model performance across multiple dimensions:
- Latency: P50, P95, P99 response times
- Throughput: Tokens/second generation
- Accuracy: Correct answers on standard test sets
- Token Efficiency: Output tokens vs input tokens
- Memory Usage: VRAM consumption (where available)

Models Benchmarked:
- deepseek-r1:14b (Ollama) - Reasoning
- llama3.1:8b (Ollama) - Analysis
- qwen2.5-coder:14b (LM Studio) - Instruction following
- agentflow-planner-7b (LM Studio) - Planning
- mistral:latest (Ollama) - User-facing
- nomic-embed-text (Ollama) - Embeddings

Usage:
    python scripts/model_baseline_benchmark.py [--model MODEL] [--test-set TESTSET] [--iterations N]
    python scripts/model_baseline_benchmark.py --all-models --test-set mmlu_mini
    python scripts/model_baseline_benchmark.py --model deepseek-r1:14b --iterations 50

Exit Codes:
    0 - Benchmark completed successfully
    1 - Benchmark failed (errors in testing)
    2 - Configuration/connection error
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    httpx = None  # type: ignore

try:
    from bizra_kernel.benchmarks.base import (
        BenchmarkResult,
        BenchmarkMetrics,
        BenchmarkReport,
        TestQuestion,
        TestSet,
        ExecutionMode,
        load_test_set,
        list_test_sets,
        BENCHMARK_DATA_PATH,
    )
except ImportError as e:
    print(f"ERROR: Could not import benchmark base: {e}")
    print("Ensure bizra_kernel/benchmarks/base.py exists")
    sys.exit(2)


# ============================================================================
# CONFIGURATION
# ============================================================================

WORKSPACE = Path(__file__).parent.parent

# Model configurations - maps model name to provider and settings
MODEL_CONFIGS = {
    # Ollama models
    "deepseek-r1:14b": {
        "provider": "ollama",
        "description": "DeepSeek R1 - Reasoning specialist",
        "role": "reasoning",
        "timeout_s": 120,
    },
    "deepseek-r1:8b": {
        "provider": "ollama",
        "description": "DeepSeek R1 8B - Compact reasoning",
        "role": "reasoning",
        "timeout_s": 60,
    },
    "llama3.1:8b": {
        "provider": "ollama",
        "description": "Llama 3.1 8B - General analysis",
        "role": "analysis",
        "timeout_s": 60,
    },
    "mistral:latest": {
        "provider": "ollama",
        "description": "Mistral - User-facing interactions",
        "role": "user_facing",
        "timeout_s": 60,
    },
    "qwen2.5:7b": {
        "provider": "ollama",
        "description": "Qwen 2.5 7B - Instruction following",
        "role": "instruction",
        "timeout_s": 60,
    },
    "nomic-embed-text:latest": {
        "provider": "ollama",
        "description": "Nomic Embed - Text embeddings",
        "role": "embeddings",
        "timeout_s": 30,
    },
    # LM Studio models
    "qwen2.5-coder:14b": {
        "provider": "lmstudio",
        "description": "Qwen 2.5 Coder 14B - Code generation",
        "role": "coding",
        "timeout_s": 90,
    },
    "agentflow-planner-7b": {
        "provider": "lmstudio",
        "description": "AgentFlow Planner - Task planning",
        "role": "planning",
        "timeout_s": 60,
    },
}

# Endpoint configuration
OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234/v1")


# ============================================================================
# LLM CLIENT
# ============================================================================

@dataclass
class ModelResponse:
    """Response from an LLM call."""
    text: str
    tokens_input: int
    tokens_output: int
    latency_ms: float
    model: str
    provider: str
    raw: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


async def call_ollama(
    model: str,
    prompt: str,
    system_prompt: str = "",
    timeout_s: float = 60,
) -> ModelResponse:
    """Call Ollama API for completion."""
    url = f"{OLLAMA_URL}/api/chat"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.0,  # Deterministic for benchmarking
            "num_ctx": 4096,
        }
    }

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(url, json=payload)

        latency_ms = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            return ModelResponse(
                text="",
                tokens_input=0,
                tokens_output=0,
                latency_ms=latency_ms,
                model=model,
                provider="ollama",
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        data = resp.json()
        message = data.get("message", {})
        text = message.get("content", "")

        return ModelResponse(
            text=text,
            tokens_input=data.get("prompt_eval_count", 0),
            tokens_output=data.get("eval_count", 0),
            latency_ms=latency_ms,
            model=model,
            provider="ollama",
            raw=data,
        )

    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return ModelResponse(
            text="",
            tokens_input=0,
            tokens_output=0,
            latency_ms=latency_ms,
            model=model,
            provider="ollama",
            error=str(e),
        )


async def call_lmstudio(
    model: str,
    prompt: str,
    system_prompt: str = "",
    timeout_s: float = 60,
) -> ModelResponse:
    """Call LM Studio OpenAI-compatible API."""
    url = f"{LMSTUDIO_URL}/chat/completions"

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
    }

    start = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            resp = await client.post(url, json=payload)

        latency_ms = (time.perf_counter() - start) * 1000

        if resp.status_code != 200:
            return ModelResponse(
                text="",
                tokens_input=0,
                tokens_output=0,
                latency_ms=latency_ms,
                model=model,
                provider="lmstudio",
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )

        data = resp.json()
        choices = data.get("choices", [])
        text = ""
        if choices:
            message = choices[0].get("message", {})
            text = message.get("content", "")

        usage = data.get("usage", {})

        return ModelResponse(
            text=text,
            tokens_input=usage.get("prompt_tokens", 0),
            tokens_output=usage.get("completion_tokens", 0),
            latency_ms=latency_ms,
            model=model,
            provider="lmstudio",
            raw=data,
        )

    except Exception as e:
        latency_ms = (time.perf_counter() - start) * 1000
        return ModelResponse(
            text="",
            tokens_input=0,
            tokens_output=0,
            latency_ms=latency_ms,
            model=model,
            provider="lmstudio",
            error=str(e),
        )


async def call_model(
    model: str,
    prompt: str,
    system_prompt: str = "",
) -> ModelResponse:
    """Call the appropriate provider for the model."""
    config = MODEL_CONFIGS.get(model)
    if not config:
        return ModelResponse(
            text="",
            tokens_input=0,
            tokens_output=0,
            latency_ms=0,
            model=model,
            provider="unknown",
            error=f"Unknown model: {model}",
        )

    provider = config["provider"]
    timeout_s = config.get("timeout_s", 60)

    if provider == "ollama":
        return await call_ollama(model, prompt, system_prompt, timeout_s)
    elif provider == "lmstudio":
        return await call_lmstudio(model, prompt, system_prompt, timeout_s)
    else:
        return ModelResponse(
            text="",
            tokens_input=0,
            tokens_output=0,
            latency_ms=0,
            model=model,
            provider=provider,
            error=f"Unsupported provider: {provider}",
        )


# ============================================================================
# BENCHMARK EVALUATION
# ============================================================================

def format_question(question: TestQuestion) -> str:
    """Format a test question as a prompt."""
    prompt = question.question + "\n\n"

    if question.choices:
        prompt += "Options:\n"
        for choice in question.choices:
            prompt += f"  {choice}\n"

    prompt += "\nAnswer with just the letter (A, B, C, or D):"

    return prompt


def extract_answer(response: str) -> str:
    """Extract the answer letter from a model response."""
    response = response.strip().upper()

    # Try to find a single letter answer
    for letter in ["A", "B", "C", "D"]:
        if response.startswith(letter):
            return letter
        if f"({letter})" in response or f" {letter}:" in response:
            return letter
        if response == letter:
            return letter

    # Look for "The answer is X" patterns
    import re
    match = re.search(r"answer\s*(?:is)?\s*:?\s*([ABCD])", response, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return response[:1].upper() if response else ""


def check_answer(question: TestQuestion, response: str) -> bool:
    """Check if the response matches the correct answer."""
    extracted = extract_answer(response)
    correct = question.correct_answer.strip().upper()

    # Handle both "A" and "A:" style answers
    if ":" in correct:
        correct = correct.split(":")[0].strip()

    return extracted == correct


# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

class ModelBaselineBenchmark:
    """Runs baseline benchmarks for individual models."""

    SYSTEM_PROMPT = """You are a helpful assistant taking a knowledge test.
Answer each question by selecting the best option (A, B, C, or D).
Respond with just the letter of your answer."""

    def __init__(
        self,
        warmup_iterations: int = 3,
        output_dir: Optional[Path] = None,
    ):
        self.warmup_iterations = warmup_iterations
        self.output_dir = output_dir or (WORKSPACE / "docs" / "evidence" / "benchmarks")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    async def check_model_availability(self, model: str) -> bool:
        """Check if a model is available."""
        config = MODEL_CONFIGS.get(model)
        if not config:
            return False

        provider = config["provider"]

        try:
            if provider == "ollama":
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{OLLAMA_URL}/api/tags")
                    if resp.status_code == 200:
                        tags = resp.json().get("models", [])
                        model_names = [t.get("name", "") for t in tags]
                        # Check if model is in available models (exact or partial match)
                        return any(model in name or name in model for name in model_names)
            elif provider == "lmstudio":
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{LMSTUDIO_URL}/models")
                    return resp.status_code == 200
        except Exception:
            pass

        return False

    async def warmup_model(self, model: str) -> None:
        """Warm up the model with a few simple queries."""
        print(f"   Warming up {model}...")

        for i in range(self.warmup_iterations):
            await call_model(
                model=model,
                prompt="What is 2 + 2?",
                system_prompt="Answer briefly.",
            )

    async def benchmark_model(
        self,
        model: str,
        test_set: TestSet,
        max_questions: Optional[int] = None,
    ) -> BenchmarkResult:
        """Benchmark a single model against a test set."""
        config = MODEL_CONFIGS.get(model, {})
        provider = config.get("provider", "unknown")

        print(f"\n{'=' * 60}")
        print(f"Benchmarking: {model}")
        print(f"Provider: {provider}")
        print(f"Test Set: {test_set.name} ({len(test_set.questions)} questions)")
        print("=" * 60)

        # Check availability
        if not await self.check_model_availability(model):
            print(f"   WARNING: Model {model} may not be available")

        # Warmup
        await self.warmup_model(model)

        # Prepare metrics
        metrics = BenchmarkMetrics()
        questions = test_set.questions[:max_questions] if max_questions else test_set.questions
        metrics.total_questions = len(questions)

        errors: List[str] = []
        start_time = time.perf_counter()

        # Run benchmark
        for i, question in enumerate(questions):
            prompt = format_question(question)

            response = await call_model(
                model=model,
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
            )

            if response.error:
                metrics.errors += 1
                if "timeout" in response.error.lower():
                    metrics.timeouts += 1
                errors.append(f"Q{question.id}: {response.error}")
            else:
                metrics.latencies_ms.append(response.latency_ms)
                metrics.tokens_input.append(response.tokens_input)
                metrics.tokens_generated.append(response.tokens_output)

                if check_answer(question, response.text):
                    metrics.correct_answers += 1

            # Progress
            if (i + 1) % 10 == 0:
                acc = metrics.correct_answers / (i + 1 - metrics.errors) if (i + 1 - metrics.errors) > 0 else 0
                print(f"   Progress: {i + 1}/{len(questions)} | Accuracy: {acc:.1%} | Avg Latency: {metrics.mean_latency:.0f}ms")

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Create result
        result = BenchmarkResult(
            model_name=model,
            model_provider=provider,
            execution_mode=ExecutionMode.DIRECT,
            test_set_name=test_set.name,
            metrics=metrics,
            duration_ms=duration_ms,
            config={
                "warmup_iterations": self.warmup_iterations,
                "max_questions": max_questions,
                "model_config": config,
            },
            errors=errors[:10],  # Keep first 10 errors
        )

        # Print summary
        self._print_result_summary(result)

        return result

    def _print_result_summary(self, result: BenchmarkResult) -> None:
        """Print a summary of benchmark results."""
        m = result.metrics

        print(f"\n--- Results for {result.model_name} ---")
        print(f"Accuracy: {m.accuracy:.1%} ({m.correct_answers}/{m.total_questions})")
        print(f"Latency P50: {m.p50_latency:.0f}ms")
        print(f"Latency P95: {m.p95_latency:.0f}ms")
        print(f"Latency P99: {m.p99_latency:.0f}ms")
        print(f"Tokens/sec: {m.tokens_per_second:.1f}")
        print(f"Token Efficiency: {m.token_efficiency:.3f}")
        print(f"Errors: {m.errors} | Timeouts: {m.timeouts}")
        print(f"Total Duration: {result.duration_ms/1000:.1f}s")

    async def run_all_models(
        self,
        test_set: TestSet,
        max_questions: Optional[int] = None,
        models: Optional[List[str]] = None,
    ) -> BenchmarkReport:
        """Run benchmarks for all configured models."""
        report = BenchmarkReport(
            title="BIZRA Model Baseline Benchmarks",
            description=f"Individual model performance on {test_set.name}",
        )
        report.test_sets.append(test_set.name)

        models_to_benchmark = models or list(MODEL_CONFIGS.keys())
        start_time = time.perf_counter()

        print("\n" + "=" * 70)
        print("BIZRA Model Baseline Benchmark Suite")
        print("=" * 70)
        print(f"Models to benchmark: {len(models_to_benchmark)}")
        print(f"Test set: {test_set.name} ({len(test_set.questions)} questions)")
        if max_questions:
            print(f"Max questions per model: {max_questions}")
        print("=" * 70)

        for model in models_to_benchmark:
            try:
                result = await self.benchmark_model(model, test_set, max_questions)
                report.add_result(result)
            except Exception as e:
                print(f"ERROR benchmarking {model}: {e}")
                report.errors.append(f"{model}: {str(e)}")

        report.total_duration_ms = (time.perf_counter() - start_time) * 1000

        # Compute summary
        self._print_comparison_table(report)

        return report

    def _print_comparison_table(self, report: BenchmarkReport) -> None:
        """Print a comparison table of all results."""
        print("\n" + "=" * 80)
        print("COMPARISON TABLE")
        print("=" * 80)
        print(f"{'Model':<25} {'Accuracy':>10} {'P95 (ms)':>10} {'Tok/s':>10} {'Errors':>8}")
        print("-" * 80)

        for result in sorted(report.results, key=lambda r: r.metrics.accuracy, reverse=True):
            m = result.metrics
            print(
                f"{result.model_name:<25} "
                f"{m.accuracy:>9.1%} "
                f"{m.p95_latency:>10.0f} "
                f"{m.tokens_per_second:>10.1f} "
                f"{m.errors:>8}"
            )

        print("=" * 80)

    def save_report(self, report: BenchmarkReport, filename: str = "model_baselines") -> Path:
        """Save the benchmark report to a JSON file."""
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.output_dir / f"{filename}_{timestamp}.json"
        report.save(path)
        print(f"\nReport saved to: {path}")
        return path


# ============================================================================
# CLI
# ============================================================================

def list_available_models() -> None:
    """List all configured models."""
    print("\nAvailable models:")
    print("-" * 60)

    for model, config in MODEL_CONFIGS.items():
        print(f"  {model:<30} [{config['provider']}] - {config['description']}")


async def main():
    parser = argparse.ArgumentParser(
        description="BIZRA Model Baseline Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/model_baseline_benchmark.py --list-models
  python scripts/model_baseline_benchmark.py --list-test-sets
  python scripts/model_baseline_benchmark.py --model mistral:latest --test-set mmlu_mini
  python scripts/model_baseline_benchmark.py --all-models --test-set bizra_qa --max-questions 50
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to benchmark"
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Benchmark all configured models"
    )
    parser.add_argument(
        "--test-set",
        type=str,
        default="mmlu_mini",
        help="Test set to use (default: mmlu_mini)"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum questions per model (default: all)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup iterations (default: 3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename prefix"
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit"
    )
    parser.add_argument(
        "--list-test-sets",
        action="store_true",
        help="List available test sets and exit"
    )

    args = parser.parse_args()

    # Handle list commands
    if args.list_models:
        list_available_models()
        sys.exit(0)

    if args.list_test_sets:
        test_sets = list_test_sets()
        print("\nAvailable test sets:")
        for ts in test_sets:
            print(f"  - {ts}")
        sys.exit(0)

    # Validate arguments
    if not args.model and not args.all_models:
        print("ERROR: Specify --model MODEL or --all-models")
        parser.print_help()
        sys.exit(2)

    # Check httpx availability (needed for HTTP calls)
    if not HTTPX_AVAILABLE:
        print("ERROR: httpx required. Install with: pip install httpx")
        sys.exit(2)

    # Load test set
    try:
        test_set = load_test_set(args.test_set)
    except FileNotFoundError:
        print(f"ERROR: Test set not found: {args.test_set}")
        print(f"Available: {list_test_sets()}")
        sys.exit(2)

    # Create benchmark runner
    benchmark = ModelBaselineBenchmark(warmup_iterations=args.warmup)

    # Run benchmarks
    if args.all_models:
        report = await benchmark.run_all_models(
            test_set=test_set,
            max_questions=args.max_questions,
        )
    else:
        result = await benchmark.benchmark_model(
            model=args.model,
            test_set=test_set,
            max_questions=args.max_questions,
        )
        report = BenchmarkReport(
            title=f"BIZRA Model Baseline: {args.model}",
            description=f"Baseline benchmark for {args.model}",
        )
        report.add_result(result)
        report.test_sets.append(test_set.name)

    # Save report
    filename = args.output or "model_baselines"
    benchmark.save_report(report, filename)

    # Exit code based on success
    if report.errors:
        print(f"\nCompleted with {len(report.errors)} errors")
        sys.exit(1)
    else:
        print("\nBenchmark completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
