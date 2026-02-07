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
  onboard   - Create sovereign identity (if not yet onboarded)
  dashboard - View node identity and agents
  impact    - Sovereignty progression and UERS scores
  status    - Show system status
  metrics   - Show performance metrics
  clear     - Clear screen
  exit      - Exit REPL

Or type any query to get a sovereign response.
                    """)
                    continue

                if query.lower() == "onboard":
                    run_onboard()
                    continue

                if query.lower() == "dashboard":
                    run_dashboard()
                    continue

                if query.lower() in ("impact", "progress"):
                    run_impact()
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


def run_onboard(name: Optional[str] = None, node_dir: Optional[str] = None, json_output: bool = False):
    """Run node onboarding wizard."""
    from pathlib import Path

    from ..pat.onboarding import OnboardingWizard

    dir_path = Path(node_dir) if node_dir else None
    wizard = OnboardingWizard(node_dir=dir_path)

    if json_output:
        import json as json_mod

        # Non-interactive mode
        existing = wizard.load_existing_credentials()
        if existing is not None:
            print(json_mod.dumps({"status": "already_onboarded", **existing.to_dict()}, indent=2))
            return

        try:
            credentials = wizard.onboard(name=name)
            print(json_mod.dumps({"status": "success", **credentials.to_dict()}, indent=2))
        except (RuntimeError, FileExistsError) as e:
            print(json_mod.dumps({"status": "error", "error": str(e)}, indent=2))
            sys.exit(1)
    else:
        # Interactive mode
        wizard.run_interactive()


def run_dashboard(node_dir: Optional[str] = None, json_output: bool = False):
    """Show node identity dashboard."""
    import json as json_mod
    from pathlib import Path

    from ..pat.onboarding import OnboardingWizard

    dir_path = Path(node_dir) if node_dir else None
    wizard = OnboardingWizard(node_dir=dir_path)

    credentials = wizard.load_existing_credentials()
    if credentials is None:
        if json_output:
            print(json_mod.dumps({"status": "not_onboarded"}, indent=2))
        else:
            print("\n  No node identity found.")
            print("  Run 'bizra onboard' to create your sovereign identity.\n")
        sys.exit(1)

    if json_output:
        # Full JSON output
        identity_data = {}
        if wizard.identity_file.exists():
            identity_data = json_mod.loads(wizard.identity_file.read_text())
        agents_data = {}
        if wizard.agents_file.exists():
            agents_data = json_mod.loads(wizard.agents_file.read_text())
        print(json_mod.dumps({
            "credentials": credentials.to_dict(),
            "identity": identity_data,
            "agents": agents_data,
        }, indent=2))
        return

    # Terminal dashboard
    tier = credentials.sovereignty_tier.upper()
    tier_bars = {
        "SEED": "[#---]",
        "SPROUT": "[##--]",
        "TREE": "[###-]",
        "FOREST": "[####]",
    }
    bar = tier_bars.get(tier, "[----]")

    print()
    print("=" * 60)
    print("  BIZRA NODE DASHBOARD")
    print("=" * 60)
    print()
    print(f"  Node ID:          {credentials.node_id}")
    print(f"  Sovereignty:      {tier} {bar}  ({credentials.sovereignty_score:.2f})")
    print(f"  Created:          {credentials.created_at[:19]}")
    print(f"  Public Key:       {credentials.public_key[:16]}...{credentials.public_key[-8:]}")
    print()
    print("  PERSONAL AGENTIC TEAM (PAT)")
    print("  " + "-" * 40)
    for agent_id in credentials.pat_agent_ids:
        parts = agent_id.split("-")
        agent_type = parts[2] if len(parts) >= 3 else "???"
        idx = parts[3] if len(parts) >= 4 else "?"
        type_names = {
            "WRK": "Worker",
            "RSC": "Researcher",
            "GRD": "Guardian",
            "SYN": "Synthesizer",
            "VAL": "Validator",
            "CRD": "Coordinator",
            "EXC": "Executor",
        }
        type_name = type_names.get(agent_type, agent_type)
        print(f"    {agent_id}  {type_name}")
    print()
    print("  SYSTEM AGENTIC TEAM (SAT)")
    print("  " + "-" * 40)
    for agent_id in credentials.sat_agent_ids:
        parts = agent_id.split("-")
        agent_type = parts[2] if len(parts) >= 3 else "???"
        type_names = {
            "WRK": "Worker",
            "RSC": "Researcher",
            "GRD": "Guardian",
            "SYN": "Synthesizer",
            "VAL": "Validator",
            "CRD": "Coordinator",
            "EXC": "Executor",
        }
        type_name = type_names.get(agent_type, agent_type)
        print(f"    {agent_id}  {type_name}")
    print()
    print(f"  Data: {wizard.node_dir}")
    print("=" * 60)
    print()


def run_impact(node_dir: Optional[str] = None, json_output: bool = False):
    """Show sovereignty progression and impact history."""
    import json as json_mod
    from pathlib import Path

    from ..pat.impact_tracker import ImpactTracker
    from ..pat.onboarding import OnboardingWizard

    dir_path = Path(node_dir) if node_dir else None
    wizard = OnboardingWizard(node_dir=dir_path)

    credentials = wizard.load_existing_credentials()
    if credentials is None:
        if json_output:
            print(json_mod.dumps({"status": "not_onboarded"}, indent=2))
        else:
            print("\n  No node identity found.")
            print("  Run 'bizra onboard' to create your sovereign identity.\n")
        sys.exit(1)

    tracker = ImpactTracker(
        node_id=credentials.node_id,
        state_dir=dir_path or Path.home() / ".bizra-node",
    )

    progress = tracker.get_progress()
    tier_info = tracker.get_tier_progress()

    if json_output:
        print(json_mod.dumps({
            "progress": progress.to_dict(),
            "tier_progress": tier_info,
        }, indent=2))
        return

    # Terminal display
    tier = tier_info["current_tier"].upper()
    pct = tier_info["progress_percent"]
    bar_len = 20
    filled = int(bar_len * pct / 100)
    bar = "#" * filled + "-" * (bar_len - filled)

    print()
    print("=" * 60)
    print("  BIZRA SOVEREIGNTY PROGRESSION")
    print("=" * 60)
    print()
    print(f"  Node:             {progress.node_id}")
    print(f"  Tier:             {tier}")
    print(f"  Score:            {progress.sovereignty_score:.4f}")
    print(f"  Progress:         [{bar}] {pct}%")
    if tier_info["next_tier"]:
        print(f"  Next Tier:        {tier_info['next_tier'].upper()} at {tier_info['next_threshold']:.2f}")
    print()
    print(f"  Total Bloom:      {progress.total_bloom:.2f}")
    print(f"  Impact Events:    {progress.total_events}")
    print(f"  Streak:           {progress.streak_days} days")
    print()

    # UERS breakdown
    uers = progress.uers_aggregate
    print("  UERS DIMENSIONS")
    print("  " + "-" * 40)
    for dim, score in [
        ("Utility", uers.utility),
        ("Efficiency", uers.efficiency),
        ("Resilience", uers.resilience),
        ("Sustainability", uers.sustainability),
        ("Ethics", uers.ethics),
    ]:
        dim_bar_len = 15
        dim_filled = int(dim_bar_len * min(1.0, score))
        dim_bar = "#" * dim_filled + "-" * (dim_bar_len - dim_filled)
        print(f"    {dim:18s} [{dim_bar}] {score:.3f}")
    print()

    # Achievements
    if progress.achievements:
        print("  ACHIEVEMENTS")
        print("  " + "-" * 40)
        for ach in progress.achievements:
            print(f"    {ach}")
    else:
        print("  No achievements yet. Keep contributing!")
    print()
    print("=" * 60)
    print()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BIZRA Sovereign Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m core.sovereign                          # Interactive REPL
  python -m core.sovereign onboard                  # Create sovereign identity
  python -m core.sovereign dashboard                # View node identity
  python -m core.sovereign impact                   # Sovereignty progression
  python -m core.sovereign query "Your question"    # Single query
  python -m core.sovereign serve --port 8080        # Start API server
  python -m core.sovereign gateway telegram          # Run Telegram gateway
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

    # Onboard command
    onboard_parser = subparsers.add_parser("onboard", help="Create sovereign identity")
    onboard_parser.add_argument("--name", help="Display name (optional)")
    onboard_parser.add_argument("--node-dir", help="Node data directory")
    onboard_parser.add_argument("--json", action="store_true", help="JSON output")

    # Dashboard command
    dashboard_parser = subparsers.add_parser("dashboard", help="View node identity")
    dashboard_parser.add_argument("--node-dir", help="Node data directory")
    dashboard_parser.add_argument("--json", action="store_true", help="JSON output")

    # Impact / Progress command
    impact_parser = subparsers.add_parser("impact", help="Sovereignty progression")
    impact_parser.add_argument("--node-dir", help="Node data directory")
    impact_parser.add_argument("--json", action="store_true", help="JSON output")

    # Gateway command
    gateway_parser = subparsers.add_parser("gateway", help="Run messaging gateway")
    gateway_parser.add_argument(
        "platform", choices=["telegram"], help="Messaging platform"
    )
    gateway_parser.add_argument("--token", help="Bot token (or use env var)")
    gateway_parser.add_argument("--webhook", help="Webhook URL (for production)")
    gateway_parser.add_argument("--node-dir", help="Node data directory")

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
    elif args.command == "onboard":
        run_onboard(args.name, args.node_dir, args.json)
    elif args.command == "dashboard":
        run_dashboard(args.node_dir, args.json)
    elif args.command == "impact":
        run_impact(args.node_dir, args.json)
    elif args.command == "gateway":
        from ..pat.gateway import run_telegram_gateway

        node_id = ""
        if getattr(args, "node_dir", None):
            from ..pat.onboarding import get_node_credentials
            from pathlib import Path

            creds = get_node_credentials(node_dir=Path(args.node_dir))
            if creds:
                node_id = creds.node_id

        if args.platform == "telegram":
            asyncio.run(
                run_telegram_gateway(
                    token=args.token or "",
                    node_id=node_id,
                    webhook_url=args.webhook or "",
                )
            )
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
