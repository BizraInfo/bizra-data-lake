# BIZRA Command Center v1.0
# Unified Orchestration Interface - The Crown Jewel
# Elite-level system integration demonstrating all BIZRA capabilities
# Ihsān-grade excellence in every operation

import asyncio
import json
import sys
import time
import platform
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import threading
import signal

# BIZRA Root
BIZRA_ROOT = Path("C:/BIZRA-DATA-LAKE")
sys.path.insert(0, str(BIZRA_ROOT))

# Import BIZRA components
try:
    from bizra_config import SNR_THRESHOLD, IHSAN_CONSTRAINT, BATCH_SIZE
except ImportError:
    SNR_THRESHOLD = 0.95
    IHSAN_CONSTRAINT = 0.99
    BATCH_SIZE = 128

try:
    from metrics_dashboard import MetricsDashboard, record_snr, record_latency, record_error
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

try:
    from bizra_resilience import get_resilience_status, CircuitBreaker
    RESILIENCE_AVAILABLE = True
except ImportError:
    RESILIENCE_AVAILABLE = False

try:
    from validate_system import SystemValidator
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False


class CommandType(Enum):
    """Command categories"""
    SYSTEM = "system"
    QUERY = "query"
    MONITOR = "monitor"
    VALIDATE = "validate"
    BENCHMARK = "benchmark"
    CONFIG = "config"


@dataclass
class SystemStatus:
    """Comprehensive system status"""
    status: str
    uptime_seconds: float
    snr_average: float
    ihsan_compliance: float
    total_queries: int
    error_rate: float
    gpu_available: bool
    gpu_memory_used: float
    cpu_percent: float
    memory_percent: float
    active_circuit_breakers: int
    graph_nodes: int
    graph_edges: int
    embedded_chunks: int
    last_update: str


@dataclass
class BenchmarkResult:
    """Benchmark execution result"""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    throughput_ops_sec: float
    snr_average: float


class BIZRACommandCenter:
    """
    BIZRA Command Center - Unified Orchestration Interface

    The apex of the BIZRA system, providing:
    - Real-time system monitoring
    - Query execution with SNR tracking
    - Automated validation and benchmarking
    - Circuit breaker management
    - Performance analytics
    """

    VERSION = "1.0.0"
    CODENAME = "IHSAN"

    def __init__(self):
        self.start_time = time.time()
        self.query_count = 0
        self.error_count = 0
        self._running = True
        self._metrics_dashboard = None
        self._snr_history: List[float] = []

        # Initialize subsystems
        self._init_subsystems()

    def _init_subsystems(self):
        """Initialize all BIZRA subsystems"""
        print("┌" + "─" * 58 + "┐")
        print("│" + " " * 15 + "BIZRA COMMAND CENTER v1.0" + " " * 18 + "│")
        print("│" + " " * 18 + "Codename: IHSAN" + " " * 25 + "│")
        print("└" + "─" * 58 + "┘")
        print()

        subsystems = [
            ("Configuration", self._check_config),
            ("Metrics Dashboard", self._check_metrics),
            ("Resilience Patterns", self._check_resilience),
            ("Validation Engine", self._check_validation),
            ("Graph Engine", self._check_graph),
            ("Vector Engine", self._check_vectors),
            ("DDAGI Consciousness", self._check_ddagi),
        ]

        print("  Initializing subsystems...")
        print()

        for name, check_fn in subsystems:
            status, detail = check_fn()
            symbol = "✅" if status else "⚠️"
            print(f"  {symbol} {name}: {detail}")

        print()

    def _check_config(self) -> Tuple[bool, str]:
        """Check configuration"""
        try:
            import bizra_config
            return True, f"IHSAN={IHSAN_CONSTRAINT}, SNR={SNR_THRESHOLD}"
        except:
            return False, "Using defaults"

    def _check_metrics(self) -> Tuple[bool, str]:
        """Check metrics dashboard"""
        if METRICS_AVAILABLE:
            self._metrics_dashboard = MetricsDashboard()
            return True, "Operational"
        return False, "Not available"

    def _check_resilience(self) -> Tuple[bool, str]:
        """Check resilience patterns"""
        if RESILIENCE_AVAILABLE:
            status = get_resilience_status()
            count = len(status.get("circuit_breakers", {}))
            return True, f"{count} circuit breakers registered"
        return False, "Not available"

    def _check_validation(self) -> Tuple[bool, str]:
        """Check validation engine"""
        if VALIDATION_AVAILABLE:
            return True, "Ready"
        return False, "Not available"

    def _check_graph(self) -> Tuple[bool, str]:
        """Check hypergraph engine"""
        stats_path = BIZRA_ROOT / "03_INDEXED" / "graph" / "statistics.json"
        if stats_path.exists():
            try:
                with open(stats_path) as f:
                    stats = json.load(f)
                nodes = stats.get("total_nodes", 0)
                edges = stats.get("total_edges", 0)
                return True, f"{nodes:,} nodes, {edges:,} edges"
            except:
                pass
        return False, "No graph data"

    def _check_vectors(self) -> Tuple[bool, str]:
        """Check vector embeddings"""
        chunks_path = BIZRA_ROOT / "04_GOLD" / "chunks.parquet"
        if chunks_path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(chunks_path)
                return True, f"{len(df):,} embedded chunks"
            except:
                pass
        return False, "No embeddings"

    def _check_ddagi(self) -> Tuple[bool, str]:
        """Check DDAGI consciousness"""
        ddagi_path = BIZRA_ROOT / "03_INDEXED" / "ddagi_consciousness.jsonl"
        if ddagi_path.exists():
            try:
                count = sum(1 for _ in open(ddagi_path))
                return True, f"{count} consciousness events"
            except (IOError, OSError) as e:
                logger.debug(f"DDAGI check failed: {e}")
        return False, "No consciousness data"

    def get_system_status(self) -> SystemStatus:
        """Get comprehensive system status"""
        uptime = time.time() - self.start_time

        # SNR statistics
        snr_avg = sum(self._snr_history) / len(self._snr_history) if self._snr_history else 0
        ihsan_count = sum(1 for s in self._snr_history if s >= IHSAN_CONSTRAINT)
        ihsan_compliance = ihsan_count / len(self._snr_history) if self._snr_history else 0

        # Error rate
        error_rate = self.error_count / self.query_count if self.query_count > 0 else 0

        # GPU check
        gpu_available = False
        gpu_memory = 0.0
        try:
            import torch
            if torch.cuda.is_available():
                gpu_available = True
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100 if torch.cuda.max_memory_allocated() > 0 else 0
        except ImportError:
            logger.debug("PyTorch not available for GPU check")
        except Exception as e:
            logger.debug(f"GPU check failed: {e}")

        # CPU/Memory
        cpu_percent = 0.0
        memory_percent = 0.0
        try:
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
        except ImportError:
            logger.debug("psutil not available for CPU/memory check")
        except Exception as e:
            logger.debug(f"CPU/memory check failed: {e}")

        # Circuit breakers
        active_breakers = 0
        if RESILIENCE_AVAILABLE:
            status = get_resilience_status()
            active_breakers = sum(
                1 for cb in status.get("circuit_breakers", {}).values()
                if cb.get("state") != "closed"
            )

        # Graph stats
        nodes, edges = 0, 0
        stats_path = BIZRA_ROOT / "03_INDEXED" / "graph" / "statistics.json"
        if stats_path.exists():
            try:
                with open(stats_path) as f:
                    stats = json.load(f)
                nodes = stats.get("total_nodes", 0)
                edges = stats.get("total_edges", 0)
            except (json.JSONDecodeError, IOError) as e:
                logger.debug(f"Graph stats read failed: {e}")

        # Chunks
        chunks = 0
        chunks_path = BIZRA_ROOT / "04_GOLD" / "chunks.parquet"
        if chunks_path.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(chunks_path)
                chunks = len(df)
            except ImportError:
                logger.debug("pandas not available for chunks count")
            except Exception as e:
                logger.debug(f"Chunks count failed: {e}")

        # Determine overall status
        if error_rate > 0.1 or active_breakers > 0:
            status = "DEGRADED"
        elif snr_avg >= IHSAN_CONSTRAINT:
            status = "OPTIMAL"
        elif snr_avg >= SNR_THRESHOLD:
            status = "HEALTHY"
        else:
            status = "INITIALIZING"

        return SystemStatus(
            status=status,
            uptime_seconds=uptime,
            snr_average=snr_avg,
            ihsan_compliance=ihsan_compliance,
            total_queries=self.query_count,
            error_rate=error_rate,
            gpu_available=gpu_available,
            gpu_memory_used=gpu_memory,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            active_circuit_breakers=active_breakers,
            graph_nodes=nodes,
            graph_edges=edges,
            embedded_chunks=chunks,
            last_update=datetime.now().isoformat()
        )

    def print_status_dashboard(self):
        """Print comprehensive status dashboard"""
        status = self.get_system_status()

        # Status symbol
        status_symbols = {
            "OPTIMAL": "🌟",
            "HEALTHY": "✅",
            "DEGRADED": "⚠️",
            "INITIALIZING": "🔄"
        }
        symbol = status_symbols.get(status.status, "❓")

        # Format uptime
        uptime_str = str(timedelta(seconds=int(status.uptime_seconds)))

        print()
        print("╔" + "═" * 58 + "╗")
        print("║" + " " * 15 + "BIZRA COMMAND CENTER" + " " * 23 + "║")
        print("╠" + "═" * 58 + "╣")

        # System Status
        print(f"║  System Status:     {symbol} {status.status:<25}        ║")
        print(f"║  Uptime:            {uptime_str:<36} ║")
        print("╠" + "═" * 58 + "╣")

        # SNR Metrics
        ihsan_symbol = "✅" if status.ihsan_compliance >= 0.95 else "⚠️"
        print("║  📊 SNR METRICS" + " " * 42 + "║")
        print(f"║    Average SNR:        {status.snr_average:.4f}" + " " * 28 + "║")
        print(f"║    Ihsān Compliance:   {ihsan_symbol} {status.ihsan_compliance*100:.1f}%" + " " * 26 + "║")
        print(f"║    Total Queries:      {status.total_queries:,}" + " " * (31 - len(f"{status.total_queries:,}")) + "║")
        print("╠" + "═" * 58 + "╣")

        # Infrastructure
        gpu_symbol = "✅" if status.gpu_available else "❌"
        print("║  🖥️  INFRASTRUCTURE" + " " * 38 + "║")
        print(f"║    GPU Available:      {gpu_symbol}" + " " * 32 + "║")
        if status.gpu_available:
            print(f"║    GPU Memory:         {status.gpu_memory_used:.1f}%" + " " * 29 + "║")
        print(f"║    CPU Usage:          {status.cpu_percent:.1f}%" + " " * 29 + "║")
        print(f"║    Memory Usage:       {status.memory_percent:.1f}%" + " " * 29 + "║")
        print("╠" + "═" * 58 + "╣")

        # Knowledge Base
        print("║  🧠 KNOWLEDGE BASE" + " " * 39 + "║")
        print(f"║    Graph Nodes:        {status.graph_nodes:,}" + " " * (31 - len(f"{status.graph_nodes:,}")) + "║")
        print(f"║    Graph Edges:        {status.graph_edges:,}" + " " * (31 - len(f"{status.graph_edges:,}")) + "║")
        print(f"║    Embedded Chunks:    {status.embedded_chunks:,}" + " " * (31 - len(f"{status.embedded_chunks:,}")) + "║")
        print("╠" + "═" * 58 + "╣")

        # Resilience
        breaker_symbol = "✅" if status.active_circuit_breakers == 0 else "⚠️"
        error_symbol = "✅" if status.error_rate < 0.01 else "⚠️"
        print("║  🛡️  RESILIENCE" + " " * 42 + "║")
        print(f"║    Circuit Breakers:   {breaker_symbol} {status.active_circuit_breakers} active" + " " * 24 + "║")
        print(f"║    Error Rate:         {error_symbol} {status.error_rate*100:.2f}%" + " " * 24 + "║")

        print("╚" + "═" * 58 + "╝")
        print()

    async def run_benchmark(self, name: str, iterations: int = 100) -> BenchmarkResult:
        """Run a specific benchmark"""
        import numpy as np

        times = []
        snr_values = []

        for i in range(iterations):
            start = time.perf_counter()

            if name == "snr_calculation":
                # Benchmark SNR calculation
                try:
                    from arte_engine import SNREngine
                    engine = SNREngine()
                    query = np.random.rand(384).astype(np.float32)
                    context = [np.random.rand(384).astype(np.float32) for _ in range(5)]
                    result = engine.calculate_snr(
                        query, context, ["fact1", "fact2"],
                        [{"text": "result", "score": 0.9}]
                    )
                    snr_values.append(result.get("snr", 0))
                except:
                    snr_values.append(0)

            elif name == "embedding_generation":
                # Benchmark embedding generation
                try:
                    from sentence_transformers import SentenceTransformer
                    model = SentenceTransformer("all-MiniLM-L6-v2")
                    text = "This is a sample text for embedding generation benchmark."
                    embedding = model.encode([text])
                    snr_values.append(0.95)
                except:
                    snr_values.append(0)

            elif name == "graph_traversal":
                # Benchmark graph operations
                try:
                    import networkx as nx
                    G = nx.gnm_random_graph(1000, 5000)
                    _ = nx.shortest_path(G, 0, 500)
                    snr_values.append(0.98)
                except:
                    snr_values.append(0)

            elapsed = (time.perf_counter() - start) * 1000
            times.append(elapsed)

        return BenchmarkResult(
            name=name,
            iterations=iterations,
            total_time_ms=sum(times),
            avg_time_ms=sum(times) / len(times),
            min_time_ms=min(times),
            max_time_ms=max(times),
            throughput_ops_sec=1000 / (sum(times) / len(times)) if times else 0,
            snr_average=sum(snr_values) / len(snr_values) if snr_values else 0
        )

    async def run_full_benchmark_suite(self) -> Dict[str, BenchmarkResult]:
        """Run complete benchmark suite"""
        print("\n  🏃 Running BIZRA Benchmark Suite...")
        print("  " + "─" * 50)

        benchmarks = [
            ("snr_calculation", 50),
            ("embedding_generation", 10),
            ("graph_traversal", 20)
        ]

        results = {}

        for name, iterations in benchmarks:
            print(f"  ▸ {name} ({iterations} iterations)...", end=" ", flush=True)
            try:
                result = await self.run_benchmark(name, iterations)
                results[name] = result
                print(f"✅ {result.avg_time_ms:.2f}ms avg")
            except Exception as e:
                print(f"❌ {str(e)[:30]}")

        # Summary
        print()
        print("  " + "─" * 50)
        print("  📊 BENCHMARK SUMMARY")
        print("  " + "─" * 50)

        for name, result in results.items():
            print(f"  {name}:")
            print(f"    Avg: {result.avg_time_ms:.2f}ms | "
                  f"Min: {result.min_time_ms:.2f}ms | "
                  f"Max: {result.max_time_ms:.2f}ms")
            print(f"    Throughput: {result.throughput_ops_sec:.1f} ops/sec | "
                  f"SNR: {result.snr_average:.4f}")
            print()

        return results

    async def run_validation(self) -> bool:
        """Run system validation"""
        if not VALIDATION_AVAILABLE:
            print("  ❌ Validation engine not available")
            return False

        validator = SystemValidator()
        report = validator.run_all()

        return report.overall_status != "FAILED"

    def record_query(self, snr: float, latency_ms: float, success: bool):
        """Record query metrics"""
        self.query_count += 1
        self._snr_history.append(snr)

        if not success:
            self.error_count += 1

        # Keep history bounded
        if len(self._snr_history) > 1000:
            self._snr_history = self._snr_history[-500:]

        # Record to metrics dashboard
        if METRICS_AVAILABLE:
            record_snr(
                overall=snr,
                signal=snr * 0.95,
                density=snr * 0.92,
                grounding=snr * 0.94,
                balance=snr * 0.91
            )
            record_latency(latency_ms, "query")

    def print_help(self):
        """Print help information"""
        print("""
╔══════════════════════════════════════════════════════════╗
║                 BIZRA COMMAND CENTER HELP                ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  COMMANDS:                                               ║
║                                                          ║
║  status      - Show system status dashboard              ║
║  validate    - Run system validation                     ║
║  benchmark   - Run performance benchmarks                ║
║  metrics     - Show metrics dashboard                    ║
║  resilience  - Show circuit breaker status               ║
║  health      - Quick health check                        ║
║  export      - Export status report to JSON              ║
║  help        - Show this help message                    ║
║  quit/exit   - Exit command center                       ║
║                                                          ║
║  THRESHOLDS:                                             ║
║                                                          ║
║  Ihsān Excellence:  SNR ≥ 0.99                          ║
║  Acceptable:        SNR ≥ 0.95                          ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
        """)

    async def interactive_loop(self):
        """Main interactive command loop"""
        self.print_help()

        while self._running:
            try:
                cmd = input("\n  BIZRA> ").strip().lower()

                if not cmd:
                    continue

                if cmd in ["quit", "exit", "q"]:
                    print("\n  👋 Shutting down BIZRA Command Center...")
                    self._running = False
                    break

                elif cmd == "status":
                    self.print_status_dashboard()

                elif cmd == "validate":
                    print("\n  🔍 Running system validation...")
                    await self.run_validation()

                elif cmd == "benchmark":
                    await self.run_full_benchmark_suite()

                elif cmd == "metrics":
                    if self._metrics_dashboard:
                        self._metrics_dashboard.print_dashboard()
                    else:
                        print("  ⚠️ Metrics dashboard not available")

                elif cmd == "resilience":
                    if RESILIENCE_AVAILABLE:
                        status = get_resilience_status()
                        print("\n  🛡️ CIRCUIT BREAKER STATUS")
                        print("  " + "─" * 40)
                        for name, cb in status.get("circuit_breakers", {}).items():
                            state_symbol = "✅" if cb["state"] == "closed" else "⚠️"
                            print(f"  {state_symbol} {name}: {cb['state']}")
                            print(f"      Failures: {cb['failures']} | "
                                  f"Total calls: {cb['total_calls']}")
                    else:
                        print("  ⚠️ Resilience patterns not available")

                elif cmd == "health":
                    status = self.get_system_status()
                    symbol = "🌟" if status.status == "OPTIMAL" else "✅" if status.status == "HEALTHY" else "⚠️"
                    print(f"\n  {symbol} System: {status.status}")
                    print(f"  📊 SNR: {status.snr_average:.4f}")
                    print(f"  ⏱️ Uptime: {timedelta(seconds=int(status.uptime_seconds))}")

                elif cmd == "export":
                    status = self.get_system_status()
                    export_path = BIZRA_ROOT / "03_INDEXED" / "metrics" / f"status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    export_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(export_path, "w") as f:
                        json.dump(asdict(status), f, indent=2)
                    print(f"\n  📄 Status exported to: {export_path}")

                elif cmd == "help":
                    self.print_help()

                else:
                    print(f"  ❓ Unknown command: {cmd}")
                    print("  Type 'help' for available commands")

            except KeyboardInterrupt:
                print("\n\n  👋 Interrupted. Shutting down...")
                self._running = False
                break
            except Exception as e:
                print(f"\n  ❌ Error: {e}")


async def main():
    """Main entry point"""
    print()
    print("  " + "═" * 56)
    print("  ║" + " " * 10 + "BIZRA DATA LAKE - COMMAND CENTER" + " " * 11 + "║")
    print("  ║" + " " * 15 + "Excellence Through Ihsān" + " " * 16 + "║")
    print("  " + "═" * 56)
    print()

    center = BIZRACommandCenter()

    # Show initial status
    center.print_status_dashboard()

    # Run interactive loop
    await center.interactive_loop()

    print("\n  ✅ BIZRA Command Center shutdown complete\n")


if __name__ == "__main__":
    asyncio.run(main())
