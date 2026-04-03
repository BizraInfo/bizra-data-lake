"""
BIZRA Performance Suite - Phase 9 Benchmark Optimization

This suite provides comprehensive benchmarking capabilities including:
- Federation scenarios across multiple nodes
- Chaos engineering with MTTR measurement
- CI/CD integration for continuous benchmarking
"""

import time
import statistics
import threading
import subprocess
import json
import os
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from benchmark_util import BIZRABenchmark


class FederationBenchmark:
    """Benchmarks for federated node operations across multiple nodes."""

    def __init__(self, node_count: int = 3):
        self.node_count = node_count
        self.nodes = [f"node_{i}" for i in range(node_count)]
        self.benchmark = BIZRABenchmark()

    def benchmark_cross_node_latency(self, iterations: int = 100) -> Dict:
        """Measure latency for operations spanning multiple nodes."""
        print(f"[*] Benchmarking cross-node latency ({iterations} iterations, {self.node_count} nodes)...")

        latencies = []

        for _ in range(iterations):
            start_time = time.perf_counter_ns()

            # Simulate cross-node operation
            with ThreadPoolExecutor(max_workers=self.node_count) as executor:
                futures = [executor.submit(self._simulate_node_operation, node) for node in self.nodes]
                results = [future.result() for future in as_completed(futures)]

            end_time = time.perf_counter_ns()
            latency_ms = (end_time - start_time) / 1_000_000
            latencies.append(latency_ms)

        avg_latency = statistics.mean(latencies)
        p95_latency = statistics.quantiles(latencies, n=20)[18]  # 95th percentile

        result = {
            "avg_latency_ms": avg_latency,
            "p95_latency_ms": p95_latency,
            "node_count": self.node_count,
            "iterations": iterations
        }

        print(f"[+] Cross-node latency: {avg_latency:.2f}ms avg, {p95_latency:.2f}ms p95")
        return result

    def benchmark_federation_consensus(self, rounds: int = 50) -> Dict:
        """Benchmark consensus performance in federated setup."""
        print(f"[*] Benchmarking federation consensus ({rounds} rounds, {self.node_count} nodes)...")

        consensus_times = []

        for round_num in range(rounds):
            start_time = time.perf_counter_ns()

            # Simulate consensus round across nodes
            with ThreadPoolExecutor(max_workers=self.node_count) as executor:
                futures = [executor.submit(self._simulate_consensus_vote, node, round_num) for node in self.nodes]
                votes = [future.result() for future in as_completed(futures)]

            # Simulate consensus resolution
            consensus_reached = len(set(votes)) == 1  # All votes same
            end_time = time.perf_counter_ns()

            if consensus_reached:
                consensus_time_ms = (end_time - start_time) / 1_000_000
                consensus_times.append(consensus_time_ms)

        if consensus_times:
            avg_consensus_time = statistics.mean(consensus_times)
            success_rate = len(consensus_times) / rounds * 100

            result = {
                "avg_consensus_time_ms": avg_consensus_time,
                "success_rate_percent": success_rate,
                "rounds": rounds,
                "node_count": self.node_count
            }

            print(f"[+] Consensus: {avg_consensus_time:.2f}ms avg, {success_rate:.1f}% success rate")
            return result
        else:
            return {"error": "No consensus rounds completed successfully"}

    def _simulate_node_operation(self, node_id: str) -> bool:
        """Simulate a single node operation."""
        # Placeholder for real network latency and processing
        time.sleep(0.001)  # 1ms placeholder operation
        return True

    def _simulate_consensus_vote(self, node_id: str, round_num: int) -> str:
        """Simulate a consensus vote from a node."""
        # Placeholder for real voting with occasional delays
        time.sleep(0.0005)  # 0.5ms placeholder vote
        return f"vote_{round_num % 3}"  # Simulate different vote options


class ChaosEngineeringBenchmark:
    """Chaos engineering benchmarks with MTTR measurement."""

    def __init__(self):
        self.benchmark = BIZRABenchmark()
        self.failure_scenarios = [
            "node_crash",
            "network_partition",
            "resource_exhaustion",
            "corrupted_state"
        ]

    def benchmark_mttr(self, scenario: str, iterations: int = 10) -> Dict:
        """Measure Mean Time To Recovery for a specific failure scenario."""
        if scenario not in self.failure_scenarios:
            raise ValueError(f"Unknown scenario: {scenario}")

        print(f"[*] Benchmarking MTTR for {scenario} ({iterations} iterations)...")

        recovery_times = []

        for i in range(iterations):
            print(f"  Iteration {i+1}/{iterations}")

            # Inject failure
            failure_start = time.perf_counter_ns()
            self._inject_failure(scenario)

            # Measure recovery time
            recovery_start = time.perf_counter_ns()
            recovered = self._attempt_recovery(scenario)
            recovery_end = time.perf_counter_ns()

            if recovered:
                mttr_ms = (recovery_end - recovery_start) / 1_000_000
                recovery_times.append(mttr_ms)
                print(f"    Recovered in {mttr_ms:.2f}ms")
            else:
                print("    Recovery failed")

        if recovery_times:
            avg_mttr = statistics.mean(recovery_times)
            p95_mttr = statistics.quantiles(recovery_times, n=20)[18] if len(recovery_times) >= 20 else max(recovery_times)

            result = {
                "scenario": scenario,
                "avg_mttr_ms": avg_mttr,
                "p95_mttr_ms": p95_mttr,
                "successful_recoveries": len(recovery_times),
                "total_iterations": iterations,
                "success_rate_percent": len(recovery_times) / iterations * 100
            }

            print(f"[+] MTTR for {scenario}: {avg_mttr:.2f}ms avg, {len(recovery_times)}/{iterations} successful")
            return result
        else:
            return {"error": f"No successful recoveries for {scenario}"}

    def benchmark_chaos_resilience(self, duration_sec: int = 300) -> Dict:
        """Run continuous chaos testing for specified duration."""
        print(f"[*] Running chaos resilience test ({duration_sec}s)...")

        start_time = time.time()
        end_time = start_time + duration_sec

        incidents = []
        total_downtime = 0

        while time.time() < end_time:
            # Randomly select and inject failure
            scenario = self.failure_scenarios[int(time.time()) % len(self.failure_scenarios)]

            incident_start = time.perf_counter_ns()
            self._inject_failure(scenario)

            # Attempt recovery
            recovery_start = time.perf_counter_ns()
            recovered = self._attempt_recovery(scenario)
            recovery_end = time.perf_counter_ns()

            if recovered:
                downtime_ms = (recovery_end - recovery_start) / 1_000_000
                total_downtime += downtime_ms

                incident = {
                    "scenario": scenario,
                    "timestamp": time.time(),
                    "downtime_ms": downtime_ms
                }
                incidents.append(incident)

            # Wait before next incident
            time.sleep(10)  # 10 second intervals

        total_uptime = duration_sec * 1000 - total_downtime  # Convert to ms
        availability_percent = (total_uptime / (duration_sec * 1000)) * 100

        result = {
            "duration_sec": duration_sec,
            "total_incidents": len(incidents),
            "total_downtime_ms": total_downtime,
            "availability_percent": availability_percent,
            "incidents": incidents
        }

        print(f"[+] Chaos test complete: {availability_percent:.2f}% availability, {len(incidents)} incidents")
        return result

    def _inject_failure(self, scenario: str):
        """Inject a specific failure scenario."""
        # Simulate failure injection
        time.sleep(0.1)  # Simulate injection time

    def _attempt_recovery(self, scenario: str) -> bool:
        """Attempt to recover from a failure scenario."""
        # Simulate recovery process
        recovery_time = {
            "node_crash": 2.0,
            "network_partition": 1.5,
            "resource_exhaustion": 3.0,
            "corrupted_state": 5.0
        }.get(scenario, 1.0)

        time.sleep(recovery_time / 1000)  # Convert to seconds
        return True  # Assume recovery succeeds for benchmarking


class CIBenchmarkIntegration:
    """CI/CD integration for continuous benchmarking."""

    def __init__(self, results_dir: str = ".benchmarks/results"):
        self.results_dir = results_dir
        self.ensure_results_dir()

    def ensure_results_dir(self):
        """Ensure the results directory exists."""
        os.makedirs(self.results_dir, exist_ok=True)

    def run_ci_benchmarks(self) -> Dict:
        """Run the complete benchmark suite for CI/CD."""
        print("[*] Running CI benchmark suite...")

        results = {
            "timestamp": time.time(),
            "federation": {},
            "chaos": {},
            "system": {}
        }

        # Federation benchmarks
        fed_bench = FederationBenchmark()
        results["federation"]["cross_node_latency"] = fed_bench.benchmark_cross_node_latency()
        results["federation"]["consensus"] = fed_bench.benchmark_federation_consensus()

        # Chaos engineering benchmarks
        chaos_bench = ChaosEngineeringBenchmark()
        for scenario in chaos_bench.failure_scenarios:
            results["chaos"][scenario] = chaos_bench.benchmark_mttr(scenario, iterations=5)

        # System benchmarks
        system_bench = BIZRABenchmark()
        results["system"]["logic_gate"] = system_bench.benchmark_logic_gate()
        system_bench.simulate_blockgraph_tps()

        # Save results
        self.save_results(results)

        # Generate report
        self.generate_report(results)

        print("[+] CI benchmark suite completed")
        return results

    def save_results(self, results: Dict):
        """Save benchmark results to file."""
        timestamp = int(results["timestamp"])
        filename = f"benchmark_results_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)

        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"[+] Results saved to {filepath}")

    def generate_report(self, results: Dict):
        """Generate a human-readable benchmark report."""
        report_lines = [
            "# BIZRA Performance Benchmark Report",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(results['timestamp']))}",
            "",
            "## Federation Benchmarks",
        ]

        # Federation results
        fed = results.get("federation", {})
        if "cross_node_latency" in fed:
            latency = fed["cross_node_latency"]
            report_lines.extend([
                f"- Cross-node latency: {latency.get('avg_latency_ms', 0):.2f}ms avg, {latency.get('p95_latency_ms', 0):.2f}ms p95",
                f"- Node count: {latency.get('node_count', 0)}"
            ])

        if "consensus" in fed:
            consensus = fed["consensus"]
            if "avg_consensus_time_ms" in consensus:
                report_lines.extend([
                    f"- Consensus time: {consensus['avg_consensus_time_ms']:.2f}ms avg",
                    f"- Success rate: {consensus.get('success_rate_percent', 0):.1f}%"
                ])

        report_lines.extend([
            "",
            "## Chaos Engineering Benchmarks"
        ])

        # Chaos results
        chaos = results.get("chaos", {})
        for scenario, data in chaos.items():
            if "avg_mttr_ms" in data:
                report_lines.append(f"- {scenario}: {data['avg_mttr_ms']:.2f}ms MTTR ({data.get('success_rate_percent', 0):.1f}% success)")

        report_lines.extend([
            "",
            "## System Benchmarks"
        ])

        # System results
        system = results.get("system", {})
        if "logic_gate" in system:
            logic = system["logic_gate"]
            report_lines.append(f"- Logic gate: {logic.get('avg', 0):.4f}ms avg, {logic.get('p99', 0):.4f}ms p99")

        # Save report
        timestamp = int(results["timestamp"])
        report_path = os.path.join(self.results_dir, f"benchmark_report_{timestamp}.md")
        with open(report_path, 'w') as f:
            f.write("\n".join(report_lines))

        print(f"[+] Report generated: {report_path}")

    def compare_with_baseline(self, current_results: Dict, baseline_file: Optional[str] = None) -> Dict:
        """Compare current results with baseline."""
        if not baseline_file:
            # Find latest baseline
            baseline_files = [f for f in os.listdir(self.results_dir) if f.startswith("benchmark_results_") and f.endswith(".json")]
            if baseline_files:
                baseline_file = sorted(baseline_files)[-1]

        if not baseline_file or not os.path.exists(os.path.join(self.results_dir, baseline_file)):
            return {"error": "No baseline found for comparison"}

        with open(os.path.join(self.results_dir, baseline_file), 'r') as f:
            baseline = json.load(f)

        comparison = {
            "current_timestamp": current_results["timestamp"],
            "baseline_timestamp": baseline["timestamp"],
            "federation_changes": {},
            "chaos_changes": {},
            "system_changes": {}
        }

        # Compare federation benchmarks
        for key in ["cross_node_latency", "consensus"]:
            if key in current_results.get("federation", {}) and key in baseline.get("federation", {}):
                current_val = current_results["federation"][key]
                baseline_val = baseline["federation"][key]

                if "avg_latency_ms" in current_val and "avg_latency_ms" in baseline_val:
                    change = ((current_val["avg_latency_ms"] - baseline_val["avg_latency_ms"]) / baseline_val["avg_latency_ms"]) * 100
                    comparison["federation_changes"][f"{key}_latency"] = f"{change:+.2f}%"

        # Compare chaos benchmarks
        for scenario in current_results.get("chaos", {}):
            if scenario in baseline.get("chaos", {}):
                current_mttr = current_results["chaos"][scenario].get("avg_mttr_ms")
                baseline_mttr = baseline["chaos"][scenario].get("avg_mttr_ms")

                if current_mttr and baseline_mttr:
                    change = ((current_mttr - baseline_mttr) / baseline_mttr) * 100
                    comparison["chaos_changes"][scenario] = f"{change:+.2f}%"

        return comparison


def main():
    """Main entry point for running the performance suite."""
    import argparse

    parser = argparse.ArgumentParser(description="BIZRA Performance Suite")
    parser.add_argument("--federation", action="store_true", help="Run federation benchmarks")
    parser.add_argument("--chaos", action="store_true", help="Run chaos engineering benchmarks")
    parser.add_argument("--ci", action="store_true", help="Run complete CI benchmark suite")
    parser.add_argument("--nodes", type=int, default=3, help="Number of nodes for federation tests")
    parser.add_argument("--iterations", type=int, default=100, help="Number of iterations for benchmarks")

    args = parser.parse_args()

    if args.ci:
        ci_integration = CIBenchmarkIntegration()
        results = ci_integration.run_ci_benchmarks()
        print("CI benchmarks completed successfully")
        return

    if args.federation:
        fed_bench = FederationBenchmark(node_count=args.nodes)
        fed_bench.benchmark_cross_node_latency(iterations=args.iterations)
        fed_bench.benchmark_federation_consensus(rounds=args.iterations)

    if args.chaos:
        chaos_bench = ChaosEngineeringBenchmark()
        for scenario in chaos_bench.failure_scenarios:
            chaos_bench.benchmark_mttr(scenario, iterations=args.iterations // 10)  # Fewer iterations for chaos


if __name__ == "__main__":
    if os.getenv("BIZRA_BENCHMARK_MODE") != "1":
        raise SystemExit("Performance suite is disabled for production release")
    main()
