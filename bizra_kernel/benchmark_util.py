import time
import statistics
import os
import json
from .ihsan_gate import IhsanGate

class BIZRABenchmark:
    """
    State-of-the-Art Performance Verification Utility.
    """
    
    def __init__(self):
        self.gate = IhsanGate()

    def benchmark_logic_gate(self, iterations=1000):
        print(f"[*] Benchmarking IhsanGate Logic Floor ({iterations} iterations)...")
        latencies = []
        mission = {"task_id": "BENCH", "truthfulness": 1.0, "dignity": 1.0, "fairness": 1.0, "sustainability": 1.0}
        
        # Warm-up
        for _ in range(10): self.gate.verify_mission(mission)
        
        for _ in range(iterations):
            start = time.perf_counter_ns()
            self.gate.verify_mission(mission)
            end = time.perf_counter_ns()
            latencies.append((end - start) / 1_000_000) # Convert to ms
            
        avg = statistics.mean(latencies)
        p99 = statistics.quantiles(latencies, n=100)[98]
        
        print(f"[+] Result: Average Latency = {avg:.4f}ms | P99 = {p99:.4f}ms")
        return {"avg": avg, "p99": p99}

    def load_blockgraph_benchmark(self, path: str | None = None) -> dict:
        """
        Load BlockGraph benchmark results from a production run.
        """
        if path is None:
            path = os.getenv("BIZRA_BLOCKGRAPH_BENCHMARK_PATH")
        if not path:
            raise RuntimeError("BIZRA_BLOCKGRAPH_BENCHMARK_PATH is required")
        if not os.path.exists(path):
            raise RuntimeError(f"BlockGraph benchmark file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise RuntimeError("BlockGraph benchmark must be a JSON object")
        return data

if __name__ == "__main__":
    bench = BIZRABenchmark()
    bench.benchmark_logic_gate()
    print("-" * 50)
    bench.load_blockgraph_benchmark()
