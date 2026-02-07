"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ███████╗ ██████╗ ██╗   ██╗███████╗██████╗ ███████╗██╗ ██████╗ ███╗   ██╗  ║
║   ██╔════╝██╔═══██╗██║   ██║██╔════╝██╔══██╗██╔════╝██║██╔════╝ ████╗  ██║  ║
║   ███████╗██║   ██║██║   ██║█████╗  ██████╔╝█████╗  ██║██║  ███╗██╔██╗ ██║  ║
║   ╚════██║██║   ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗██╔══╝  ██║██║   ██║██║╚██╗██║  ║
║   ███████║╚██████╔╝ ╚████╔╝ ███████╗██║  ██║███████╗██║╚██████╔╝██║ ╚████║  ║
║   ╚══════╝ ╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝  ║
║                                                                              ║
║                    BIZRA SOVEREIGN ENGINE v1.0                               ║
║                      Unified Entry Point                                     ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   Usage:                                                                     ║
║   ══════                                                                     ║
║                                                                              ║
║   Interactive REPL:                                                          ║
║     python -m core.sovereign                                                 ║
║                                                                              ║
║   Query Mode:                                                                ║
║     python -m core.sovereign query "What is sovereignty?"                    ║
║                                                                              ║
║   API Server:                                                                ║
║     python -m core.sovereign serve --port 8080                               ║
║                                                                              ║
║   Status Check:                                                              ║
║     python -m core.sovereign status                                          ║
║                                                                              ║
║   Run Tests:                                                                 ║
║     python -m core.sovereign test                                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import argparse
import asyncio
import sys
from typing import Optional

# Banner
BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██████╗ ██╗███████╗██████╗  █████╗     ███████╗ ██████╗ ██╗   ██╗         ║
║   ██╔══██╗██║╚══███╔╝██╔══██╗██╔══██╗    ██╔════╝██╔═══██╗██║   ██║         ║
║   ██████╔╝██║  ███╔╝ ██████╔╝███████║    ███████╗██║   ██║██║   ██║         ║
║   ██╔══██╗██║ ███╔╝  ██╔══██╗██╔══██║    ╚════██║██║   ██║╚██╗ ██╔╝         ║
║   ██████╔╝██║███████╗██║  ██║██║  ██║    ███████║╚██████╔╝ ╚████╔╝          ║
║   ╚═════╝ ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝    ╚══════╝ ╚═════╝   ╚═══╝           ║
║                                                                              ║
║                    SOVEREIGN AUTONOMOUS ENGINE v1.0                          ║
║            Graph-of-Thoughts • SNR Maximization • Ihsān Gate                 ║
║                                                                              ║
║   "Every inference carries proof. Every decision passes the gate."           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


def print_banner():
    """Print the startup banner."""
    print(BANNER)


async def run_repl():
    """Run interactive REPL mode."""
    from ..inference.local_first import LocalBackend, get_local_first_backend
    from .runtime import RuntimeConfig, SovereignRuntime

    print_banner()

    # Detect best local backend (zero-token operation)
    best_backend = await get_local_first_backend()
    if best_backend == LocalBackend.NONE:
        print("WARNING: No local inference backends detected.")
        print(
            "Configure LM Studio (192.168.56.1:1234), Ollama (localhost:11434), or llama.cpp"
        )
    else:
        print(f"Local-first mode: Using {best_backend.value}")

    print("Interactive mode. Type 'exit' or 'quit' to leave.")
    print("Type 'status' for system status, 'help' for commands.")
    print("-" * 70)

    config = RuntimeConfig(autonomous_enabled=False)

    async with SovereignRuntime.create(config) as runtime:
        while True:
            try:
                query = input("\n🔮 sovereign> ").strip()

                if not query:
                    continue

                if query.lower() in ("exit", "quit", "q"):
                    print("Farewell. May your inferences be sovereign.")
                    break

                if query.lower() == "help":
                    print("""
Commands:
  status    - Show system status
  metrics   - Show performance metrics
  clear     - Clear screen
  exit      - Exit REPL

Or type any query to get a sovereign response.
                    """)
                    continue

                if query.lower() == "status":
                    status = runtime.status()
                    ident = status['identity']
                    node_label = ident.get('node_name', ident['node_id'])
                    print(f"\nNode: {node_label} ({ident['node_id']})")
                    if ident.get('pat_agents'):
                        print(f"PAT: {ident['pat_agents']} agents | SAT: {ident['sat_agents']} agents")
                    print(
                        f"Health: {status['health']['status']} ({status['health']['score']})"
                    )
                    continue

                if query.lower() == "metrics":
                    metrics = runtime.metrics.to_dict()
                    print(
                        f"\nQueries: {metrics['queries']['total']} (success: {metrics['queries']['success_rate']})"
                    )
                    print(f"Avg Time: {metrics['timing']['avg_query_ms']}ms")
                    print(f"Cache Hit Rate: {metrics['cache']['hit_rate']}")
                    continue

                if query.lower() == "clear":
                    print("\033[2J\033[H", end="")
                    print_banner()
                    continue

                # Process query
                result = await runtime.query(query)

                print(f"\n{'─' * 60}")
                if result.success:
                    print(f"Answer: {result.response}")
                    print(f"{'─' * 60}")
                    print(
                        f"SNR: {result.snr_score:.3f} | Ihsān: {result.ihsan_score:.3f}"
                    )
                    print(
                        f"Time: {result.processing_time_ms:.1f}ms | Depth: {result.reasoning_depth}"
                    )
                else:
                    print(f"Error: {result.error}")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.")
            except EOFError:
                break


async def run_query(query: str, json_output: bool = False):
    """Run a single query."""
    from .runtime import RuntimeConfig, SovereignRuntime

    config = RuntimeConfig(autonomous_enabled=False)

    async with SovereignRuntime.create(config) as runtime:
        result = await runtime.query(query)

        if json_output:
            import json

            print(
                json.dumps(
                    {
                        "success": result.success,
                        "response": result.response,
                        "snr": result.snr_score,
                        "ihsan": result.ihsan_score,
                        "time_ms": result.processing_time_ms,
                        "reasoning_depth": result.reasoning_depth,
                    },
                    indent=2,
                )
            )
        else:
            if result.success:
                print(f"\nAnswer: {result.response}")
                print(f"\nSNR: {result.snr_score:.3f}")
                print(f"Ihsān: {result.ihsan_score:.3f}")
                print(f"Time: {result.processing_time_ms:.1f}ms")
            else:
                print(f"Error: {result.error}")
                sys.exit(1)


async def run_server(host: str, port: int, api_keys: Optional[list] = None):
    """Run API server."""
    from .api import serve

    await serve(host, port, api_keys)


async def run_status(json_output: bool = False):
    """Show system status."""
    from ..inference.local_first import LocalFirstDetector
    from .runtime import RuntimeConfig, SovereignRuntime

    config = RuntimeConfig(autonomous_enabled=False)

    async with SovereignRuntime.create(config) as runtime:
        status = runtime.status()
        backends = await LocalFirstDetector.detect_available()

        if json_output:
            import json

            status["backends"] = [
                {
                    "name": b.backend.value,
                    "available": b.available,
                    "latency_ms": b.latency_ms,
                    "reason": b.reason,
                }
                for b in backends
            ]
            print(json.dumps(status, indent=2))
        else:
            print_banner()
            print("System Status")
            print("=" * 60)
            ident = status['identity']
            print(f"Node ID:    {ident['node_id']}")
            if ident.get('node_name'):
                print(f"Node Name:  {ident['node_name']}")
                print(f"Location:   {ident.get('location', 'unknown')}")
                print(f"Genesis:    {ident.get('genesis_hash', 'none')}")
                print(f"PAT Team:   {ident.get('pat_agents', 0)} agents")
                print(f"SAT Team:   {ident.get('sat_agents', 0)} agents")
            print(f"Version:    {ident['version']}")
            print(f"Mode:       {status['state']['mode']}")
            print("-" * 60)
            health = status.get('health', {})
            print(f"Health:     {health.get('status', 'unknown')}")
            print(f"Score:      {health.get('score', 'N/A')}")
            print(f"SNR:        {health.get('snr', 'N/A')}")
            print(f"Ihsan:      {health.get('ihsan', 'N/A')}")
            print("-" * 60)
            mem = status.get('memory', {})
            if mem.get('running'):
                print(f"Memory:     Auto-save active ({mem.get('save_count', 0)} saves)")
                print(f"Providers:  {', '.join(mem.get('providers', []))}")
            else:
                print("Memory:     Auto-save inactive")
            print("-" * 60)
            print("Local Backends (Zero-Token Operation):")
            for b in backends:
                status_str = "READY" if b.available else "offline"
                print(
                    f"  {b.backend.value:12s} {status_str:8s} {b.latency_ms:6.1f}ms  {b.reason}"
                )
            print("=" * 60)


def run_tests():
    """Run integration tests."""
    try:
        from .tests.test_integration import run_all_tests
    except ImportError:
        print("Test module not found. Run tests via pytest instead:")
        print("  pytest tests/ -v")
        sys.exit(1)

    sys.exit(run_all_tests())


async def run_doctor(verbose: bool = False, json_output: bool = False):
    """Run system health check."""
    from .doctor import run_doctor as doctor_check

    sys.exit(await doctor_check(verbose, json_output))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BIZRA Sovereign Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m core.sovereign                          # Interactive REPL
  python -m core.sovereign query "Your question"    # Single query
  python -m core.sovereign serve --port 8080        # Start API server
  python -m core.sovereign status                   # Check status
  python -m core.sovereign test                     # Run tests
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Query command
    query_parser = subparsers.add_parser("query", help="Run a single query")
    query_parser.add_argument("text", help="Query text")
    query_parser.add_argument("--json", action="store_true", help="JSON output")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Run API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8080, help="Port to bind")
    serve_parser.add_argument("--api-key", action="append", help="API keys")

    # Status command
    status_parser = subparsers.add_parser("status", help="Show system status")
    status_parser.add_argument("--json", action="store_true", help="JSON output")

    # Test command
    subparsers.add_parser("test", help="Run integration tests")

    # Doctor command
    doctor_parser = subparsers.add_parser("doctor", help="Run system health check")
    doctor_parser.add_argument(
        "-v", "--verbose", action="store_true", help="Verbose output"
    )
    doctor_parser.add_argument("--json", action="store_true", help="JSON output")

    # Version command
    subparsers.add_parser("version", help="Show version")

    # Global flags
    parser.add_argument("--version", "-V", action="store_true", help="Show version")
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Quiet mode (no banner)"
    )

    args = parser.parse_args()

    # Handle global version flag
    if getattr(args, "version", False):
        print("BIZRA Sovereign Engine v1.0.0")
        print("Codename: Genesis")
        print("Standing on Giants: Shannon • Lamport • Vaswani • Anthropic")
        sys.exit(0)

    # Route to command
    if args.command == "query":
        asyncio.run(run_query(args.text, args.json))
    elif args.command == "serve":
        asyncio.run(run_server(args.host, args.port, args.api_key))
    elif args.command == "status":
        asyncio.run(run_status(args.json))
    elif args.command == "test":
        run_tests()
    elif args.command == "doctor":
        asyncio.run(run_doctor(args.verbose, args.json))
    elif args.command == "version":
        print("BIZRA Sovereign Engine v1.0.0")
        print("Codename: Genesis")
        print("Standing on Giants: Shannon • Lamport • Vaswani • Anthropic")
    else:
        # Default: interactive REPL
        if not getattr(args, "quiet", False):
            asyncio.run(run_repl())
        else:
            print("Error: --quiet requires a command")
            sys.exit(1)


if __name__ == "__main__":
    main()
