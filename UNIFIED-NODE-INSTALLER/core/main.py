"""
BIZRA Unified Node - Main Entry Point
======================================

This is the heart of the BIZRA node.
It integrates:
- PAT Engine (Personal AI Think Tank)
- Network Node (Resource sharing, tokens)
- Dashboard Server (Web interface)
- Content Feed (Free virtual world)

Start command: python -m bizra.main
"""

import asyncio
import json
import signal
import sys
from pathlib import Path
from typing import Dict, Any

# Local imports (will be in bizra package)
# from .pat_engine import create_pat, PATOrchestrator
# from .network_node import create_network_node, NetworkNode, ContentFeed

# For standalone testing, use relative imports
try:
    from pat_engine import create_pat, PATOrchestrator
    from network_node import create_network_node, NetworkNode, ContentFeed
except ImportError:
    # Fallback for module structure
    pass


# ============================================================================
# CONFIGURATION
# ============================================================================

BIZRA_HOME = Path.home() / ".bizra"
CONFIG_FILE = BIZRA_HOME / "config.json"
DATA_DIR = BIZRA_HOME / "data"
DASHBOARD_PORT = 8888


# ============================================================================
# BIZRA NODE
# ============================================================================

class BizraNode:
    """
    The unified BIZRA node - everything in one package.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.running = False

        # Extract profile
        profile = config.get("profile", {})
        system = config.get("system", {})

        # Initialize components
        self.pat: PATOrchestrator = None
        self.network: NetworkNode = None
        self.feed: ContentFeed = None

        self._setup_components(profile, system)

    def _setup_components(self, profile: Dict, system: Dict):
        """Initialize all node components."""
        print("\n[BIZRA] Initializing node components...")

        # 1. Personal AI Think Tank
        print("  -> Setting up PAT (Personal AI Think Tank)...")
        self.pat = create_pat(profile)

        # 2. Network Node
        print("  -> Setting up Network Node...")
        resource_config = profile.get("resource_allocation", {
            "cpu_cores": 2,
            "ram_gb": 4,
            "storage_gb": 25,
            "gpu_enabled": False,
        })
        username = profile.get("username", "bizra_user")
        self.network = create_network_node(username, DATA_DIR, resource_config)

        # 3. Content Feed
        print("  -> Setting up Content Feed...")
        self.feed = ContentFeed(self.network)

        print("[BIZRA] All components initialized.")

    async def start(self):
        """Start the BIZRA node."""
        print("\n[BIZRA] Starting node...")
        self.running = True

        # Start network node
        await self.network.start()

        # Start dashboard (would be FastAPI/uvicorn in production)
        asyncio.create_task(self._run_dashboard())

        # Start main loop
        await self._main_loop()

    async def stop(self):
        """Stop the BIZRA node gracefully."""
        print("\n[BIZRA] Shutting down...")
        self.running = False
        await self.network.stop()
        print("[BIZRA] Node stopped.")

    async def _main_loop(self):
        """Main node loop."""
        print(f"\n[BIZRA] Node running!")
        print(f"  Dashboard: http://localhost:{DASHBOARD_PORT}")
        print(f"  Wallet: {self.network.wallet_address}")
        print("\nPress Ctrl+C to stop.\n")

        while self.running:
            try:
                await asyncio.sleep(1)
            except asyncio.CancelledError:
                break

    async def _run_dashboard(self):
        """Run the web dashboard."""
        # In production, this would be a FastAPI app
        # For now, just indicate it's running
        print(f"[Dashboard] Would run on http://localhost:{DASHBOARD_PORT}")

        # Placeholder - in production use:
        # import uvicorn
        # from .dashboard import app
        # uvicorn.run(app, host="0.0.0.0", port=DASHBOARD_PORT)

    async def process_chat(self, message: str) -> str:
        """Process a chat message through the PAT."""
        result = await self.pat.process_request(message)
        return json.dumps(result, indent=2)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive node status."""
        return {
            "node": {
                "running": self.running,
                "version": "0.1.0-alpha",
            },
            "pat": self.pat.get_team_summary(),
            "network": self.network.get_status(),
            "config": {
                "username": self.config.get("profile", {}).get("username"),
                "goals": self.config.get("profile", {}).get("goals", []),
            },
        }


# ============================================================================
# CLI INTERFACE
# ============================================================================

async def interactive_mode(node: BizraNode):
    """Run in interactive CLI mode."""
    print("\n" + "="*60)
    print("BIZRA Interactive Mode")
    print("="*60)
    print("Type your message to talk to your PAT.")
    print("Commands: /status, /balance, /feed, /quit")
    print("="*60 + "\n")

    while node.running:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("You: ")
            )

            if not user_input.strip():
                continue

            # Handle commands
            if user_input.startswith("/"):
                cmd = user_input[1:].lower().strip()

                if cmd == "quit" or cmd == "exit":
                    await node.stop()
                    break
                elif cmd == "status":
                    print("\n" + json.dumps(node.get_status(), indent=2) + "\n")
                elif cmd == "balance":
                    balance = node.network.ledger.get_balance(node.network.wallet_address)
                    print(f"\n  BZT Balance: {balance['bzt']:.4f}")
                    print(f"  Contribution Score: {balance['contribution_score']:.4f}\n")
                elif cmd == "feed":
                    posts = node.feed.get_feed()
                    if posts:
                        for p in posts[:5]:
                            print(f"\n  [{p.content_type}] {p.data[:100]}...")
                    else:
                        print("\n  No posts yet. Be the first!\n")
                elif cmd == "post":
                    content = input("  Content: ")
                    post = node.feed.create_post("text", content)
                    print(f"  Posted! ID: {post.content_id}\n")
                else:
                    print(f"  Unknown command: /{cmd}")
                    print("  Available: /status, /balance, /feed, /post, /quit\n")
            else:
                # Send to PAT
                print("\nPAT: Thinking...\n")
                response = await node.process_chat(user_input)
                print(f"PAT Response:\n{response}\n")

        except EOFError:
            break
        except KeyboardInterrupt:
            await node.stop()
            break


# ============================================================================
# MAIN
# ============================================================================

def load_config() -> Dict[str, Any]:
    """Load configuration from file."""
    if CONFIG_FILE.exists():
        return json.loads(CONFIG_FILE.read_text())

    # Default config for testing
    return {
        "version": "0.1.0-alpha",
        "profile": {
            "username": "test_user",
            "goals": ["productivity", "learning"],
            "domains": ["technology", "business"],
            "work_style": "collaborative",
            "privacy_level": "balanced",
            "resource_allocation": {
                "cpu_cores": 2,
                "ram_gb": 4,
                "storage_gb": 25,
                "gpu_enabled": False,
            },
        },
        "system": {
            "tier": "NORMAL",
        },
    }


async def main():
    """Main entry point."""
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║                                                          ║
    ║     ██████╗ ██╗███████╗██████╗  █████╗                  ║
    ║     ██╔══██╗██║╚══███╔╝██╔══██╗██╔══██╗                 ║
    ║     ██████╔╝██║  ███╔╝ ██████╔╝███████║                 ║
    ║     ██╔══██╗██║ ███╔╝  ██╔══██╗██╔══██║                 ║
    ║     ██████╔╝██║███████╗██║  ██║██║  ██║                 ║
    ║     ╚═════╝ ╚═╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝                 ║
    ║                                                          ║
    ║              UNIFIED NODE v0.1.0-alpha                   ║
    ║        Decentralized AI - Free Virtual World             ║
    ║                                                          ║
    ╚══════════════════════════════════════════════════════════╝
    """)

    # Load config
    config = load_config()

    # Create node
    node = BizraNode(config)

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        asyncio.create_task(node.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, signal_handler)
        except NotImplementedError:
            # Windows doesn't support add_signal_handler
            pass

    # Start node
    try:
        # Run both the node and interactive mode
        await asyncio.gather(
            node.start(),
            interactive_mode(node),
        )
    except KeyboardInterrupt:
        await node.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
