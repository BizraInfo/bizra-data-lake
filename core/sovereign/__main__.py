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


def print_banner() -> None:
    """Print the startup banner."""
    print(BANNER)


def _handle_profile_command(runtime: object) -> None:
    """Interactive profile editor — teach your PAT who you are."""
    uc = getattr(runtime, "_user_context", None)
    if not uc:
        print("User context not initialized.")
        return

    profile = uc.profile
    if profile.is_populated():
        print(f"\n{'=' * 50}")
        print("YOUR PROFILE")
        print(f"{'=' * 50}")
        print(profile.summary_for_prompt())
        print(f"{'=' * 50}")
        print("\nTo update, type 'profile set' and follow prompts.")
    else:
        print("\nYour profile is empty. Let's set it up.")
        print("Your PAT team needs to know who they serve.\n")

    update = input("Update profile? (y/n): ").strip().lower()
    if update != "y":
        return

    fields = [
        ("name", "Your name"),
        ("bio", "Brief bio (who are you, what do you do)"),
        ("mission", "Your mission (why does this matter to you)"),
        ("active_focus", "What are you working on right now"),
    ]

    for field_name, prompt_text in fields:
        current = getattr(profile, field_name, "")
        display = f" [{current}]" if current else ""
        value = input(f"  {prompt_text}{display}: ").strip()
        if value:
            setattr(profile, field_name, value)

    list_fields = [
        ("expertise", "Your expertise areas (comma-separated)"),
        ("values", "Your core values (comma-separated)"),
        ("pain_points", "Your biggest pain points (comma-separated)"),
        ("goals_short", "Goals for next 1-3 months (comma-separated)"),
        ("goals_long", "Goals for next 1-3 years (comma-separated)"),
        ("dreams", "Your life vision/dreams (comma-separated)"),
    ]

    for field_name, prompt_text in list_fields:
        current = getattr(profile, field_name, [])
        display = f" [{', '.join(current)}]" if current else ""
        value = input(f"  {prompt_text}{display}: ").strip()
        if value:
            setattr(profile, field_name, [v.strip() for v in value.split(",")])

    uc.save()
    print("\nProfile saved. Your PAT team now knows who they serve.")


def _handle_memory_command(runtime: object) -> None:
    """Show conversation memory stats."""
    uc = getattr(runtime, "_user_context", None)
    if not uc:
        print("User context not initialized.")
        return

    turns = uc.conversation.get_turn_count()
    print(f"\nConversation Memory: {turns} turns")

    if turns > 0:
        recent = uc.conversation.get_recent_context(max_turns=5)
        print(f"\nLast {min(5, turns)} exchanges:")
        print(f"{'─' * 50}")
        print(recent)
        print(f"{'─' * 50}")

    if uc.profile.is_populated():
        print(f"Profile: {uc.profile.name}")
    else:
        print("Profile: Not yet populated (use 'profile' command)")


def _handle_wallet_command() -> None:
    """Display token balances for all known accounts."""
    from core.token.ledger import TokenLedger
    from core.token.types import TokenType

    ledger = TokenLedger()
    print("=== BIZRA TOKEN WALLET ===")
    print()

    accounts = ledger.list_accounts()
    if not accounts:
        print("No accounts found. Run 'sovereign onboard' to create identity.")
        return

    for account in sorted(accounts):
        print(f"  {account}:")
        for token_type in TokenType:
            balance = ledger.get_balance(account, token_type)
            if balance.balance > 0 or balance.staked > 0:
                print(f"    {token_type.value}: {balance.balance:,.2f}")
                if balance.staked > 0:
                    print(f"    {token_type.value} (staked): {balance.staked:,.2f}")
        print()

    print(f"  Total accounts: {len(accounts)}")


def _handle_tokens_command() -> None:
    """Display token supply statistics and chain validity."""
    from core.token.ledger import TokenLedger
    from core.token.types import SEED_SUPPLY_CAP_PER_YEAR, TokenType

    ledger = TokenLedger()

    print("=== BIZRA TOKEN SUPPLY ===")
    print()

    for token_type in TokenType:
        supply = ledger.get_total_supply(token_type)
        if supply > 0:
            print(f"  {token_type.value}: {supply:,.2f}")

    print()
    print(f"  Yearly SEED cap: {SEED_SUPPLY_CAP_PER_YEAR:,.0f}")

    # Chain integrity
    valid, tx_count, error = ledger.verify_chain()
    status = "VALID" if valid else f"INVALID ({error})"
    print(f"  Chain status: {status}")
    print(f"  {tx_count} transactions")


async def _handle_import_command(runtime: object) -> None:
    """Import external data into Living Memory."""
    from .data_import import run_import_wizard

    await run_import_wizard(runtime)


async def run_repl() -> None:
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
  profile   - View/set your personal profile
  import    - Import your data (chat history, notes, documents)
  status    - Show system status
  metrics   - Show performance metrics
  memory    - Show conversation memory stats
  hunter    - Run vulnerability scanner (bounty pipeline)
  clear     - Clear screen
  exit      - Exit REPL

Or type any query to get a sovereign response.
Your PAT team remembers the conversation and learns who you are.
                    """)
                    continue

                if query.lower() == "onboard":
                    run_onboard()
                    continue

                if query.lower() == "profile":
                    _handle_profile_command(runtime)
                    continue

                if query.lower() == "memory":
                    _handle_memory_command(runtime)
                    continue

                if query.lower() in ("import", "load"):
                    await _handle_import_command(runtime)
                    continue

                if query.lower() == "dashboard":
                    run_dashboard()
                    continue

                if query.lower() in ("impact", "progress"):
                    run_impact()
                    continue

                if query.lower() == "status":
                    status = runtime.status()
                    ident = status["identity"]
                    node_label = ident.get("node_name", ident["node_id"])
                    print(f"\nNode: {node_label} ({ident['node_id']})")
                    if ident.get("pat_agents"):
                        print(
                            f"PAT: {ident['pat_agents']} agents | SAT: {ident['sat_agents']} agents"
                        )
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

                # Process query through PAT pipeline
                result = await runtime.query(query)

                print(f"\n{'─' * 60}")
                if result.success:
                    # Show which PAT agent responded
                    uc = getattr(runtime, "_user_context", None)
                    if uc and uc.conversation.get_turn_count() > 0:
                        last_turns = list(uc.conversation._turns)
                        if last_turns and last_turns[-1].agent_role:
                            print(f"[PAT {last_turns[-1].agent_role.upper()}]")
                    print(f"{result.response}")
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


async def run_query(query: str, json_output: bool = False) -> None:
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


async def run_server(host: str, port: int, api_keys: Optional[list] = None) -> None:
    """Run API server."""
    from .api import serve

    await serve(host, port, api_keys)


async def run_status(json_output: bool = False) -> None:
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
            ident = status["identity"]
            print(f"Node ID:    {ident['node_id']}")
            if ident.get("node_name"):
                print(f"Node Name:  {ident['node_name']}")
                print(f"Location:   {ident.get('location', 'unknown')}")
                print(f"Genesis:    {ident.get('genesis_hash', 'none')}")
                print(f"PAT Team:   {ident.get('pat_agents', 0)} agents")
                print(f"SAT Team:   {ident.get('sat_agents', 0)} agents")
            print(f"Version:    {ident['version']}")
            print(f"Mode:       {status['state']['mode']}")
            print("-" * 60)
            health = status.get("health", {})
            print(f"Health:     {health.get('status', 'unknown')}")
            print(f"Score:      {health.get('score', 'N/A')}")
            print(f"SNR:        {health.get('snr', 'N/A')}")
            print(f"Ihsan:      {health.get('ihsan', 'N/A')}")
            print("-" * 60)
            mem = status.get("memory", {})
            if mem.get("running"):
                print(
                    f"Memory:     Auto-save active ({mem.get('save_count', 0)} saves)"
                )
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


def run_tests() -> None:
    """Run integration tests."""
    try:
        from .tests.test_integration import run_all_tests
    except ImportError:
        print("Test module not found. Run tests via pytest instead:")
        print("  pytest tests/ -v")
        sys.exit(1)

    sys.exit(run_all_tests())


async def run_doctor(verbose: bool = False, json_output: bool = False) -> None:
    """Run system health check."""
    from .doctor import run_doctor as doctor_check

    sys.exit(await doctor_check(verbose, json_output))


def run_onboard(
    name: Optional[str] = None,
    node_dir: Optional[str] = None,
    json_output: bool = False,
) -> None:
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
            print(
                json_mod.dumps(
                    {"status": "already_onboarded", **existing.to_dict()}, indent=2
                )
            )
            return

        try:
            credentials = wizard.onboard(name=name)
            print(
                json_mod.dumps({"status": "success", **credentials.to_dict()}, indent=2)
            )
        except (RuntimeError, FileExistsError) as e:
            print(json_mod.dumps({"status": "error", "error": str(e)}, indent=2))
            sys.exit(1)
    else:
        # Interactive mode
        wizard.run_interactive()


def run_dashboard(node_dir: Optional[str] = None, json_output: bool = False) -> None:
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
        print(
            json_mod.dumps(
                {
                    "credentials": credentials.to_dict(),
                    "identity": identity_data,
                    "agents": agents_data,
                },
                indent=2,
            )
        )
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
    print(
        f"  Public Key:       {credentials.public_key[:16]}...{credentials.public_key[-8:]}"
    )
    print()
    print("  PERSONAL AGENTIC TEAM (PAT)")
    print("  " + "-" * 40)
    for agent_id in credentials.pat_agent_ids:
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


def run_impact(node_dir: Optional[str] = None, json_output: bool = False) -> None:
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
        print(
            json_mod.dumps(
                {
                    "progress": progress.to_dict(),
                    "tier_progress": tier_info,
                },
                indent=2,
            )
        )
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
        print(
            f"  Next Tier:        {tier_info['next_tier'].upper()} at {tier_info['next_threshold']:.2f}"
        )
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


def main() -> None:
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
  python -m core.sovereign bridge start              # Start desktop bridge (AHK)
  python -m core.sovereign bridge ping               # Ping running bridge
  python -m core.sovereign bridge status             # Bridge status
  python -m core.sovereign hunter scan 0x...        # Scan contract for vulns
  python -m core.sovereign hunter report scan.json  # Generate Immunefi report
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Query command
    query_parser = subparsers.add_parser("query", help="Run a single query")
    query_parser.add_argument("text", help="Query text")
    query_parser.add_argument("--json", action="store_true", help="JSON output")

    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Run API server")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
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

    # Hunter command
    hunter_parser = subparsers.add_parser("hunter", help="Bounty vulnerability scanner")
    hunter_sub = hunter_parser.add_subparsers(
        dest="hunter_command", help="Hunter action"
    )

    # hunter scan
    scan_parser = hunter_sub.add_parser(
        "scan", help="Scan contract for vulnerabilities"
    )
    scan_parser.add_argument("address", help="Contract address (0x...)")
    scan_parser.add_argument(
        "--chain", default="ethereum", help="Blockchain (default: ethereum)"
    )
    scan_parser.add_argument("--name", help="Protocol/contract name")
    scan_parser.add_argument("--bytecode", help="Path to bytecode file (hex or binary)")
    scan_parser.add_argument("--source", help="Path to Solidity source file")
    scan_parser.add_argument("--abi", help="Path to ABI JSON file")
    scan_parser.add_argument(
        "--rpc", help="JSON-RPC URL for bytecode fetch (e.g. https://eth.llamarpc.com)"
    )
    scan_parser.add_argument("--output-dir", help="Output directory for results")
    scan_parser.add_argument("--json", action="store_true", help="JSON output")

    # hunter report
    report_parser = hunter_sub.add_parser(
        "report", help="Generate platform report from scan"
    )
    report_parser.add_argument("scan_file", help="Path to scan result JSON")
    report_parser.add_argument(
        "--platform", default="immunefi", choices=["immunefi"], help="Target platform"
    )
    report_parser.add_argument("--output", help="Output file path")

    # hunter list
    list_parser = hunter_sub.add_parser("list", help="List previous scan results")
    list_parser.add_argument("--output-dir", help="Scan output directory")

    # Bridge command (Desktop Bridge — AHK ↔ TCP JSON-RPC)
    bridge_parser = subparsers.add_parser(
        "bridge", help="Desktop bridge (AHK ↔ TCP JSON-RPC)"
    )
    bridge_sub = bridge_parser.add_subparsers(
        dest="bridge_command", help="Bridge action"
    )
    bridge_start_parser = bridge_sub.add_parser("start", help="Start desktop bridge")
    bridge_start_parser.add_argument(
        "--port", type=int, default=9742, help="Bridge port (default: 9742)"
    )
    bridge_sub.add_parser("ping", help="Ping the running bridge")
    bridge_sub.add_parser("status", help="Get bridge status")

    # Wallet command
    subparsers.add_parser("wallet", help="View token wallet balances")

    # Tokens command
    subparsers.add_parser("tokens", help="View token supply and stats")

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
            from pathlib import Path

            from ..pat.onboarding import get_node_credentials

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
    elif args.command == "hunter":
        from .hunter_cli import run_hunter_list, run_hunter_report, run_hunter_scan

        if args.hunter_command == "scan":
            sys.exit(
                asyncio.run(
                    run_hunter_scan(
                        address=args.address,
                        chain=args.chain,
                        name=args.name,
                        bytecode_file=args.bytecode,
                        source_file=args.source,
                        abi_file=args.abi,
                        rpc_url=args.rpc,
                        json_output=args.json,
                        output_dir=getattr(args, "output_dir", None),
                    )
                )
            )
        elif args.hunter_command == "report":
            sys.exit(
                asyncio.run(
                    run_hunter_report(
                        scan_file=args.scan_file,
                        platform=args.platform,
                        output_file=args.output,
                    )
                )
            )
        elif args.hunter_command == "list":
            sys.exit(asyncio.run(run_hunter_list(getattr(args, "output_dir", None))))
        else:
            hunter_parser.print_help()
    elif args.command == "bridge":
        if args.bridge_command == "start":
            from core.bridges.desktop_bridge import DesktopBridge

            async def _run_bridge():
                import signal as sig

                bridge = DesktopBridge(host="127.0.0.1", port=args.port)
                stop = asyncio.Event()

                def _stop():
                    stop.set()

                loop = asyncio.get_running_loop()
                for s in (sig.SIGINT, sig.SIGTERM):
                    try:
                        loop.add_signal_handler(s, _stop)
                    except NotImplementedError:
                        pass

                await bridge.start()
                print(f"Desktop Bridge listening on 127.0.0.1:{args.port}")
                print("Press Ctrl+C to stop.")
                await stop.wait()
                await bridge.stop()

            asyncio.run(_run_bridge())
        elif args.bridge_command in ("ping", "status"):
            import json as json_mod
            import os
            import socket
            import time
            import uuid

            method = args.bridge_command
            token = os.getenv("BIZRA_BRIDGE_TOKEN", "").strip()
            if not token:
                print("Missing bridge auth token: set BIZRA_BRIDGE_TOKEN")
                sys.exit(1)
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(5.0)
                s.connect(("127.0.0.1", 9742))
                headers = {
                    "X-BIZRA-TOKEN": token,
                    "X-BIZRA-TS": int(time.time() * 1000),
                    "X-BIZRA-NONCE": uuid.uuid4().hex,
                }
                payload = (
                    json_mod.dumps(
                        {
                            "jsonrpc": "2.0",
                            "method": method,
                            "id": 1,
                            "headers": headers,
                        }
                    ).encode()
                    + b"\n"
                )
                s.sendall(payload)
                data = s.recv(65536).decode()
                s.close()
                resp = json_mod.loads(data)
                if "result" in resp:
                    print(json_mod.dumps(resp["result"], indent=2))
                else:
                    print(json_mod.dumps(resp, indent=2))
            except ConnectionRefusedError:
                print("Bridge not running on 127.0.0.1:9742")
                print("Start with: python -m core.sovereign bridge start")
                sys.exit(1)
            except Exception as e:
                print(f"Error: {e}")
                sys.exit(1)
        else:
            bridge_parser.print_help()
    elif args.command == "wallet":
        _handle_wallet_command()
    elif args.command == "tokens":
        _handle_tokens_command()
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
