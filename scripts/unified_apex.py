#!/usr/bin/env python3
"""
UNIFIED SOVEREIGN APEX — Peak Masterpiece + PAT Integration
============================================================

The ultimate autonomous engine combining:
- Peak Masterpiece Orchestrator (GoT + SNR Apex + Interdisciplinary)
- PAT Team (7-agent Personal Agentic Team)
- True Spearpoint (Benchmark Dominance Loop)
- Constitutional/Ihsān Gates

Usage:
    python scripts/unified_apex.py --query "Your question here"
    python scripts/unified_apex.py --benchmark    # Run CLEAR benchmark
    python scripts/unified_apex.py --demo         # Demo mode
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import engines directly (avoid heavy core imports)
import importlib.util
spec = importlib.util.spec_from_file_location(
    "peak_masterpiece", 
    PROJECT_ROOT / "core" / "apex" / "peak_masterpiece.py"
)
peak_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(peak_module)

PeakMasterpieceOrchestrator = peak_module.PeakMasterpieceOrchestrator
GIANTS_REGISTRY = peak_module.GIANTS_REGISTRY
PEAK_SNR_FLOOR = peak_module.PEAK_SNR_FLOOR
PEAK_SNR_TARGET = peak_module.PEAK_SNR_TARGET


async def run_query(query: str, verbose: bool = True) -> dict:
    """Execute a single query through the Peak Masterpiece."""
    orchestrator = PeakMasterpieceOrchestrator()
    result = await orchestrator.execute(query)
    
    return {
        "query": result.query,
        "snr_score": result.snr_score,
        "snr_db": result.snr_db,
        "clear_overall": result.clear_score.overall(),
        "clear_breakdown": {
            "cost": result.clear_score.cost,
            "latency": result.clear_score.latency,
            "efficacy": result.clear_score.efficacy,
            "assurance": result.clear_score.assurance,
            "reliability": result.clear_score.reliability,
        },
        "ihsan_compliant": result.ihsan_compliant,
        "giants_cited": result.giants_cited,
        "disciplines_used": len(result.disciplines_used),
        "thought_nodes": len(result.thought_path),
        "execution_time_ms": result.execution_time_ms,
        "proof_id": result.proof_id,
    }


async def run_benchmark() -> dict:
    """Run comprehensive CLEAR benchmark."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║            UNIFIED SOVEREIGN APEX — CLEAR BENCHMARK                          ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    print()

    orchestrator = PeakMasterpieceOrchestrator()

    # Benchmark queries across different domains
    benchmark_queries = [
        # Analytical
        ("Explain the mathematical foundations of cryptographic hash functions.", "analytical"),
        # Creative
        ("Design an innovative approach to sustainable urban transportation.", "creative"),
        # Critical
        ("Evaluate the ethical implications of autonomous weapon systems.", "critical"),
        # Synthesis
        ("How do quantum mechanics and consciousness relate?", "synthesis"),
        # Technical
        ("Implement a lock-free concurrent data structure.", "technical"),
    ]

    results = []
    total_snr = 0.0
    total_clear = 0.0
    ihsan_passes = 0

    for query, domain in benchmark_queries:
        print(f"\n🔄 Benchmarking [{domain.upper()}]: {query[:50]}...")
        result = await orchestrator.execute(query)
        
        total_snr += result.snr_score
        total_clear += result.clear_score.overall()
        if result.ihsan_compliant:
            ihsan_passes += 1

        results.append({
            "domain": domain,
            "query": query[:50] + "...",
            "snr": result.snr_score,
            "clear": result.clear_score.overall(),
            "ihsan": result.ihsan_compliant,
            "time_ms": result.execution_time_ms,
        })
        print(f"   SNR: {result.snr_score:.4f} | CLEAR: {result.clear_score.overall():.2f} | Ihsān: {'✓' if result.ihsan_compliant else '✗'}")

    # Summary
    n = len(benchmark_queries)
    avg_snr = total_snr / n
    avg_clear = total_clear / n
    pass_rate = ihsan_passes / n * 100

    print("\n" + "═" * 78)
    print("                         BENCHMARK RESULTS")
    print("═" * 78)
    
    def bar(v, w=30):
        f = int(v * w)
        return "█" * f + "░" * (w - f)

    print(f"\n  Queries Executed: {n}")
    print(f"\n  ┌────────────────────────────────────────────────────────────┐")
    print(f"  │  Average SNR:      {bar(avg_snr)} {avg_snr:.4f}   │")
    print(f"  │  Average CLEAR:    {bar(avg_clear)} {avg_clear:.4f}   │")
    print(f"  │  Ihsān Pass Rate:  {bar(pass_rate/100)} {pass_rate:.1f}%     │")
    print(f"  └────────────────────────────────────────────────────────────┘")

    status = "✅ OPERATIONAL" if pass_rate >= 80 else "⚠️  NEEDS IMPROVEMENT"
    print(f"\n  STATUS: {status}")
    print("═" * 78)

    stats = orchestrator.stats()
    
    return {
        "benchmark_time": datetime.now(timezone.utc).isoformat(),
        "queries_executed": n,
        "average_snr": avg_snr,
        "average_clear": avg_clear,
        "ihsan_pass_rate": pass_rate,
        "results": results,
        "orchestrator_stats": stats,
    }


async def run_demo():
    """Run demonstration of unified capabilities."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                                              ║
║   ██╗   ██╗███╗   ██╗██╗███████╗██╗███████╗██████╗      █████╗ ██████╗ ███████╗██╗  ██╗                      ║
║   ██║   ██║████╗  ██║██║██╔════╝██║██╔════╝██╔══██╗    ██╔══██╗██╔══██╗██╔════╝╚██╗██╔╝                      ║
║   ██║   ██║██╔██╗ ██║██║█████╗  ██║█████╗  ██║  ██║    ███████║██████╔╝█████╗   ╚███╔╝                       ║
║   ██║   ██║██║╚██╗██║██║██╔══╝  ██║██╔══╝  ██║  ██║    ██╔══██║██╔═══╝ ██╔══╝   ██╔██╗                       ║
║   ╚██████╔╝██║ ╚████║██║██║     ██║███████╗██████╔╝    ██║  ██║██║     ███████╗██╔╝ ██╗                      ║
║    ╚═════╝ ╚═╝  ╚═══╝╚═╝╚═╝     ╚═╝╚══════╝╚═════╝     ╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝                      ║
║                                                                                                              ║
║   THE ULTIMATE AUTONOMOUS ENGINE                                                                             ║
║   ═══════════════════════════════                                                                            ║
║                                                                                                              ║
║   EMBODIES:                                                                                                  ║
║   ├── Interdisciplinary Thinking (47 disciplines synthesized)                                                ║
║   ├── Graph-of-Thoughts (Non-linear multi-hypothesis reasoning)                                              ║
║   ├── SNR Highest Score Autonomous Engine (Target: 0.99)                                                     ║
║   ├── Standing on Giants Protocol (Formal attribution tracking)                                              ║
║   └── Ihsān Excellence (Constitutional AI with ethical constraints)                                          ║
║                                                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════════════════════════════════════╝
    """)

    print("\n📚 STANDING ON THE SHOULDERS OF GIANTS:")
    print("─" * 60)
    for key, giant in list(GIANTS_REGISTRY.items())[:8]:
        print(f"   • {giant.name} ({giant.year})")
        print(f"     └── {giant.contribution}")
    print()

    # Run benchmark
    benchmark_result = await run_benchmark()

    # Save results
    output_path = PROJECT_ROOT / "sovereign_state" / "unified_apex_benchmark.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(benchmark_result, indent=2))
    print(f"\n📄 Results saved to: {output_path}")

    return benchmark_result


async def main():
    parser = argparse.ArgumentParser(
        description="Unified Sovereign Apex — Peak Masterpiece Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/unified_apex.py --demo              # Full demonstration
  python scripts/unified_apex.py --benchmark         # CLEAR benchmark
  python scripts/unified_apex.py --query "Question"  # Single query
        """,
    )
    parser.add_argument("--query", "-q", type=str, help="Single query to execute")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Run CLEAR benchmark")
    parser.add_argument("--demo", "-d", action="store_true", help="Run full demonstration")
    parser.add_argument("--output", "-o", type=Path, help="Output path for JSON results")

    args = parser.parse_args()

    if args.demo:
        result = await run_demo()
    elif args.benchmark:
        result = await run_benchmark()
    elif args.query:
        result = await run_query(args.query)
        print(json.dumps(result, indent=2))
    else:
        # Default to demo
        result = await run_demo()

    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
        print(f"\n📄 Results saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
