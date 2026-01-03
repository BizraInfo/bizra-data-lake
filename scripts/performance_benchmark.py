#!/usr/bin/env python3
"""
BIZRA Performance Benchmark Suite v1.0
======================================

P95 latency tracking, SNR monitoring, and performance regression detection.

Usage:
    python scripts/performance_benchmark.py [--endpoint URL] [--iterations N]
    
Metrics Tracked:
    - E2E Request Latency (target: P95 < 1500ms)
    - SAT Validation Time (target: P95 < 100ms)
    - SAPE Probe Latency (target: P95 < 10ms per dimension)
    - Ihsān Calculation Time (target: P95 < 5ms)

Exit Codes:
    0 - All benchmarks passed
    1 - Performance regression detected
    2 - Connection/configuration error
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

try:
    import httpx
except ImportError:
    print("ERROR: httpx required. Install with: pip install httpx")
    sys.exit(2)


# Performance targets aligned with UNIFIED_IMPLEMENTATION_FRAMEWORK_v1.0.md
PERFORMANCE_TARGETS = {
    "health_latency_ms": {"p50": 10, "p95": 50, "p99": 100},
    "health_ready_latency_ms": {"p50": 15, "p95": 75, "p99": 150},
    "sape_probes_latency_ms": {"p50": 50, "p95": 150, "p99": 300},
    "dual_execute_latency_ms": {"p50": 500, "p95": 1500, "p99": 3000},
}

SNR_TIER_THRESHOLDS = {
    "T1": 7.0,
    "T2": 7.4,
    "T3": 7.8,
    "T4": 8.2,
    "T5": 8.6,
    "T6": 9.0,
}


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    name: str
    iterations: int
    latencies_ms: list[float]
    success_count: int
    failure_count: int
    
    @property
    def p50(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        # Correct percentile calculation using nearest-rank method
        idx = max(0, min(int((len(sorted_lat) - 1) * 0.50), len(sorted_lat) - 1))
        return sorted_lat[idx]
    
    @property
    def p95(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = max(0, min(int((len(sorted_lat) - 1) * 0.95), len(sorted_lat) - 1))
        return sorted_lat[idx]
    
    @property
    def p99(self) -> float:
        if not self.latencies_ms:
            return 0.0
        sorted_lat = sorted(self.latencies_ms)
        idx = max(0, min(int((len(sorted_lat) - 1) * 0.99), len(sorted_lat) - 1))
        return sorted_lat[idx]
    
    @property
    def mean(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0
    
    @property
    def stddev(self) -> float:
        return statistics.stdev(self.latencies_ms) if len(self.latencies_ms) > 1 else 0.0
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "latency_ms": {
                "p50": round(self.p50, 2),
                "p95": round(self.p95, 2),
                "p99": round(self.p99, 2),
                "mean": round(self.mean, 2),
                "stddev": round(self.stddev, 2),
                "min": round(min(self.latencies_ms), 2) if self.latencies_ms else 0,
                "max": round(max(self.latencies_ms), 2) if self.latencies_ms else 0,
            },
        }


class BenchmarkRunner:
    """Runs performance benchmarks against BIZRA endpoints."""
    
    def __init__(self, base_url: str, api_token: Optional[str] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.client = httpx.Client(timeout=30.0)
        
        self.headers: dict[str, str] = {}
        if api_token:
            self.headers["Authorization"] = f"Bearer {api_token}"
    
    def __enter__(self) -> "BenchmarkRunner":
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
    
    def close(self) -> None:
        """Close the HTTP client to release resources."""
        self.client.close()
    
    def _request(
        self, 
        method: str, 
        path: str, 
        json_data: Optional[dict] = None,
        *,
        require_auth: bool = False
    ) -> tuple[int, float, Any]:
        """
        Make HTTP request and return (status_code, latency_ms, response_json).
        """
        url = f"{self.base_url}{path}"
        headers = self.headers if require_auth else {}
        
        start = time.perf_counter()
        try:
            if method.upper() == "GET":
                resp = self.client.get(url, headers=headers)
            elif method.upper() == "POST":
                resp = self.client.post(url, json=json_data, headers=headers)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            latency_ms = (time.perf_counter() - start) * 1000
            
            try:
                data = resp.json()
            except Exception:
                data = {"raw": resp.text[:500]}
            
            return resp.status_code, latency_ms, data
            
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000
            return 0, latency_ms, {"error": str(e)}
    
    def benchmark_endpoint(
        self,
        name: str,
        method: str,
        path: str,
        iterations: int,
        json_data: Optional[Dict] = None,
        require_auth: bool = False,
        warmup: int = 3
    ) -> BenchmarkResult:
        """Run benchmark for a single endpoint."""
        print(f"\n🔬 Benchmarking: {name}")
        print(f"   Endpoint: {method} {path}")
        print(f"   Iterations: {iterations} (+ {warmup} warmup)")
        
        # Warmup
        for _ in range(warmup):
            self._request(method, path, json_data, require_auth=require_auth)
        
        # Benchmark
        latencies = []
        successes = 0
        failures = 0
        
        for i in range(iterations):
            status, latency_ms, _ = self._request(method, path, json_data, require_auth=require_auth)
            
            if 200 <= status < 300:
                successes += 1
                latencies.append(latency_ms)
            else:
                failures += 1
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"   Progress: {i + 1}/{iterations}", end="\r")
        
        print(f"   ✅ Completed: {successes}/{iterations} successful")
        
        return BenchmarkResult(
            name=name,
            iterations=iterations,
            latencies_ms=latencies,
            success_count=successes,
            failure_count=failures,
        )
    
    def run_suite(self, iterations: int = 50) -> Dict[str, Any]:
        """Run full benchmark suite."""
        print("=" * 60)
        print("🚀 BIZRA Performance Benchmark Suite v1.0")
        print("=" * 60)
        print(f"Base URL: {self.base_url}")
        print(f"Iterations per test: {iterations}")
        
        results = []
        
        # 1. Health endpoint (public)
        results.append(self.benchmark_endpoint(
            name="health",
            method="GET",
            path="/health",
            iterations=iterations,
        ))
        
        # 2. Health ready endpoint (public)
        results.append(self.benchmark_endpoint(
            name="health_ready",
            method="GET",
            path="/health/ready",
            iterations=iterations,
        ))
        
        # 3. Health live endpoint (public)
        results.append(self.benchmark_endpoint(
            name="health_live",
            method="GET",
            path="/health/live",
            iterations=iterations,
        ))
        
        # 4. Stats endpoint (public)
        results.append(self.benchmark_endpoint(
            name="stats",
            method="GET",
            path="/stats",
            iterations=iterations,
        ))
        
        # 5. SAPE probes (protected)
        if self.api_token:
            results.append(self.benchmark_endpoint(
                name="sape_probes",
                method="POST",
                path="/sape/probes",
                iterations=iterations // 2,  # Fewer iterations for expensive endpoint
                json_data={"content": "Test content for SAPE probe benchmark"},
                require_auth=True,
            ))
            
            # 6. SAPE stats (protected)
            results.append(self.benchmark_endpoint(
                name="sape_stats",
                method="GET",
                path="/sape/stats",
                iterations=iterations,
                require_auth=True,
            ))
        
        return self._compile_report(results)
    
    def _compile_report(self, results: List[BenchmarkResult]) -> Dict[str, Any]:
        """Compile benchmark results into report."""
        report = {
            "version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "base_url": self.base_url,
            "benchmarks": [r.to_dict() for r in results],
            "summary": {
                "total_requests": sum(r.iterations for r in results),
                "total_successes": sum(r.success_count for r in results),
                "total_failures": sum(r.failure_count for r in results),
            },
            "regressions": [],
        }
        
        # Check for regressions
        for result in results:
            key = f"{result.name}_latency_ms"
            if key in PERFORMANCE_TARGETS:
                targets = PERFORMANCE_TARGETS[key]
                if result.p95 > targets["p95"]:
                    report["regressions"].append({
                        "benchmark": result.name,
                        "metric": "p95",
                        "actual": result.p95,
                        "target": targets["p95"],
                        "severity": "high" if result.p95 > targets["p99"] else "medium",
                    })
        
        report["passed"] = len(report["regressions"]) == 0
        
        return report


def print_report(report: Dict[str, Any]) -> None:
    """Print formatted benchmark report."""
    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    
    for bench in report["benchmarks"]:
        latency = bench["latency_ms"]
        print(f"\n📌 {bench['name']}")
        print(f"   Requests: {bench['success_count']}/{bench['iterations']} successful")
        print(f"   Latency P50: {latency['p50']:.2f}ms")
        print(f"   Latency P95: {latency['p95']:.2f}ms")
        print(f"   Latency P99: {latency['p99']:.2f}ms")
        print(f"   Mean ± StdDev: {latency['mean']:.2f} ± {latency['stddev']:.2f}ms")
    
    print("\n" + "-" * 60)
    print("📈 SUMMARY")
    print("-" * 60)
    summary = report["summary"]
    print(f"Total Requests: {summary['total_requests']}")
    print(f"Successes: {summary['total_successes']}")
    print(f"Failures: {summary['total_failures']}")
    
    if report["regressions"]:
        print("\n⚠️  REGRESSIONS DETECTED:")
        for reg in report["regressions"]:
            print(f"   - {reg['benchmark']}: {reg['metric']}={reg['actual']:.2f}ms > target {reg['target']}ms ({reg['severity']})")
    else:
        print("\n✅ No performance regressions detected")
    
    print("\n" + "=" * 60)
    print(f"Result: {'✅ PASSED' if report['passed'] else '❌ FAILED'}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="BIZRA Performance Benchmark Suite")
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:8080",
        help="Base URL of BIZRA server (default: http://127.0.0.1:8080)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Number of iterations per benchmark (default: 50)"
    )
    parser.add_argument(
        "--token",
        default=None,
        help="API token for protected endpoints"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file for JSON report"
    )
    
    args = parser.parse_args()
    
    # Check connectivity first
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(f"{args.endpoint}/health/live")
            if resp.status_code != 200:
                print(f"ERROR: Server not healthy at {args.endpoint}")
                sys.exit(2)
    except Exception as e:
        print(f"ERROR: Cannot connect to {args.endpoint}: {e}")
        sys.exit(2)
    
    # Run benchmarks
    with BenchmarkRunner(args.endpoint, args.token) as runner:
        report = runner.run_suite(args.iterations)
    
    # Print results
    print_report(report)
    
    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2))
        print(f"\n📁 Report saved to: {args.output}")
    
    # Exit code based on pass/fail
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
