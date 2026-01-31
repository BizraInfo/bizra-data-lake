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
    from .runtime import SovereignRuntime, RuntimeConfig

    print_banner()
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
                    print(f"\nNode: {status['identity']['node_name']}")
                    print(f"Health: {status['health']['status']} ({status['health']['score']})")
                    print(f"SNR: {status['health']['snr']}")
                    print(f"Ihsān: {status['health']['ihsan']}")
                    continue

                if query.lower() == "metrics":
                    metrics = runtime.metrics.to_dict()
                    print(f"\nQueries: {metrics['queries']['total']} (success: {metrics['queries']['success_rate']})")
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
                    print(f"Answer: {result.answer}")
                    print(f"{'─' * 60}")
                    print(f"Confidence: {result.confidence:.1%} | SNR: {result.snr_score:.3f} | Ihsān: {result.ihsan_score:.3f}")
                    print(f"Time: {result.total_time_ms:.1f}ms | Verdict: {result.guardian_verdict}")
                else:
                    print(f"Error: {result.error}")

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type 'exit' to quit.")
            except EOFError:
                break


async def run_query(query: str, json_output: bool = False):
    """Run a single query."""
    from .runtime import SovereignRuntime, RuntimeConfig

    config = RuntimeConfig(autonomous_enabled=False)

    async with SovereignRuntime.create(config) as runtime:
        result = await runtime.query(query)

        if json_output:
            import json
            print(json.dumps({
                "success": result.success,
                "answer": result.answer,
                "confidence": result.confidence,
                "snr": result.snr_score,
                "ihsan": result.ihsan_score,
                "time_ms": result.total_time_ms,
                "verdict": result.guardian_verdict,
            }, indent=2))
        else:
            if result.success:
                print(f"\nAnswer: {result.answer}")
                print(f"\nConfidence: {result.confidence:.1%}")
                print(f"SNR: {result.snr_score:.3f}")
                print(f"Ihsān: {result.ihsan_score:.3f}")
                print(f"Time: {result.total_time_ms:.1f}ms")
            else:
                print(f"Error: {result.error}")
                sys.exit(1)


async def run_server(host: str, port: int, api_keys: Optional[list] = None):
    """Run API server."""
    from .api import serve
    await serve(host, port, api_keys)


async def run_status(json_output: bool = False):
    """Show system status."""
    from .runtime import SovereignRuntime, RuntimeConfig

    config = RuntimeConfig(autonomous_enabled=False)

    async with SovereignRuntime.create(config) as runtime:
        status = runtime.status()

        if json_output:
            import json
            print(json.dumps(status, indent=2))
        else:
            print_banner()
            print("System Status")
            print("=" * 60)
            print(f"Node ID:    {status['identity']['node_id']}")
            print(f"Node Name:  {status['identity']['node_name']}")
            print(f"Version:    {status['identity']['version']}")
            print(f"Mode:       {status['state']['mode']}")
            print("-" * 60)
            print(f"Health:     {status['health']['status']}")
            print(f"Score:      {status['health']['score']}")
            print(f"SNR:        {status['health']['snr']}")
            print(f"Ihsān:      {status['health']['ihsan']}")
            print("=" * 60)


def run_tests():
    """Run integration tests."""
    from .tests.test_integration import run_all_tests
    sys.exit(run_all_tests())


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
        """
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

    # Version command
    subparsers.add_parser("version", help="Show version")

    args = parser.parse_args()

    # Route to command
    if args.command == "query":
        asyncio.run(run_query(args.text, args.json))
    elif args.command == "serve":
        asyncio.run(run_server(args.host, args.port, args.api_key))
    elif args.command == "status":
        asyncio.run(run_status(args.json))
    elif args.command == "test":
        run_tests()
    elif args.command == "version":
        print("BIZRA Sovereign Engine v1.0.0")
        print("Codename: Genesis")
    else:
        # Default: interactive REPL
        asyncio.run(run_repl())


if __name__ == "__main__":
    main()
