#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   ███╗   ██╗ ██████╗ ██████╗ ███████╗     ██████╗                            ║
║   ████╗  ██║██╔═══██╗██╔══██╗██╔════╝    ██╔═████╗                           ║
║   ██╔██╗ ██║██║   ██║██║  ██║█████╗      ██║██╔██║                           ║
║   ██║╚██╗██║██║   ██║██║  ██║██╔══╝      ████╔╝██║                           ║
║   ██║ ╚████║╚██████╔╝██████╔╝███████╗    ╚██████╔╝                           ║
║   ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝     ╚═════╝                            ║
║                                                                              ║
║   BIZRA Node0 — Local Sovereign AI Home Base                                 ║
║   PAT Team Proactive Coworker Activation                                     ║
╚══════════════════════════════════════════════════════════════════════════════╝

This script activates your local Node0 with:
- Proactive Execution Kernel (PEK)
- PAT Team (7 agents)
- Multi-model LLM routing
- 24/7 autonomous operation

Usage:
    # Token auto-loaded from .env (LM_API_TOKEN or LM_STUDIO_API_KEY)
    python scripts/node0_activate.py              # Start full node
    python scripts/node0_activate.py --status     # Check status
    python scripts/node0_activate.py --mission "task"  # Assign mission
"""

import argparse
import asyncio
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv

    # Load .env from project root (supports both LM_STUDIO_API_KEY and LM_API_TOKEN)
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass


def _resolve_lm_token() -> str:
    """Resolve LM Studio auth token from environment.

    Checks LM_API_TOKEN first, falls back to LM_STUDIO_API_KEY.
    Mirrors the unification logic in core/integration/constants.py.
    """
    return os.getenv("LM_API_TOKEN") or os.getenv("LM_STUDIO_API_KEY") or ""

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)-12s │ %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("Node0")

# Suppress noisy logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ════════════════════════════════════════════════════════════════════════════════
# PAT AGENT DEFINITIONS
# ════════════════════════════════════════════════════════════════════════════════

PAT_AGENTS = {
    "strategist": {
        "name": "Strategist",
        "role": "Strategic planning and long-term thinking",
        "giants": "Sun Tzu, John Boyd, Michael Porter",
        "model_purpose": "reasoning",
    },
    "researcher": {
        "name": "Researcher",
        "role": "Deep investigation and evidence gathering",
        "giants": "Vannevar Bush, Claude Shannon, Douglas Engelbart",
        "model_purpose": "reasoning",
    },
    "analyst": {
        "name": "Analyst",
        "role": "Data analysis and pattern recognition",
        "giants": "Herbert Simon, Daniel Kahneman, Judea Pearl",
        "model_purpose": "reasoning",
    },
    "creator": {
        "name": "Creator",
        "role": "Content creation and design",
        "giants": "Leonardo da Vinci, Steve Jobs, Dieter Rams",
        "model_purpose": "general",
    },
    "executor": {
        "name": "Executor",
        "role": "Task execution and automation",
        "giants": "Frederick Taylor, W. Edwards Deming",
        "model_purpose": "agentic",
    },
    "guardian": {
        "name": "Guardian",
        "role": "Ethical oversight and security",
        "giants": "Al-Ghazali, John Rawls, Anthropic",
        "model_purpose": "reasoning",
    },
    "coordinator": {
        "name": "Coordinator",
        "role": "Team synthesis and integration",
        "giants": "Norbert Wiener, Peter Senge",
        "model_purpose": "reasoning",
    },
}


# ════════════════════════════════════════════════════════════════════════════════
# NODE0 PROACTIVE KERNEL
# ════════════════════════════════════════════════════════════════════════════════

class Node0ProactiveKernel:
    """
    The Proactive Execution Kernel for Node0.

    Loop: SENSE → PREDICT → SCORE → VERIFY → EXECUTE → PROVE → LEARN
    """

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.token = _resolve_lm_token()
        self.base_url = "http://192.168.56.1:1234"

        # State
        self._running = False
        self._cycle_count = 0
        self._missions: List[Dict] = []
        self._completed: List[Dict] = []
        self._metrics = {
            "cycles": 0,
            "missions_completed": 0,
            "tokens_used": 0,
            "ihsan_score": 0.0,
        }

        # Cycle timing
        self.cycle_interval = self.config.get("cycle_interval", 30.0)  # 30 seconds
        self.ihsan_threshold = 0.95

    async def start(self):
        """Start the proactive kernel."""
        self._running = True
        logger.info("═" * 60)
        logger.info("NODE0 PROACTIVE KERNEL ACTIVATED")
        logger.info("═" * 60)
        logger.info("  Mode: proactive_partner")
        logger.info(f"  Cycle Interval: {self.cycle_interval}s")
        logger.info(f"  Ihsān Threshold: {self.ihsan_threshold}")
        logger.info(f"  PAT Agents: {len(PAT_AGENTS)}")
        logger.info("═" * 60)

        await self._run_loop()

    async def stop(self):
        """Stop the kernel."""
        self._running = False
        logger.info("Node0 kernel stopping...")

    async def add_mission(self, description: str, priority: str = "normal"):
        """Add a mission for PAT team."""
        mission = {
            "id": f"mission-{len(self._missions)+1:03d}",
            "description": description,
            "priority": priority,
            "status": "pending",
            "created": datetime.now(timezone.utc).isoformat(),
            "assigned_agents": [],
            "result": None,
        }
        self._missions.append(mission)
        logger.info(f"📋 Mission added: {mission['id']} - {description[:50]}...")
        return mission

    async def _run_loop(self):
        """Main proactive loop."""
        while self._running:
            self._cycle_count += 1
            cycle_start = time.perf_counter()

            logger.info(f"─── Cycle {self._cycle_count} ───")

            try:
                # 1. SENSE - Check for pending missions
                pending = [m for m in self._missions if m["status"] == "pending"]

                if pending:
                    mission = pending[0]
                    logger.info(f"📌 Processing: {mission['id']}")

                    # 2. ASSIGN - Select agents
                    agents = self._select_agents(mission)
                    mission["assigned_agents"] = agents
                    mission["status"] = "in_progress"

                    # 3. EXECUTE - Run PAT team
                    result = await self._execute_mission(mission, agents)

                    # 4. VERIFY - Check Ihsān compliance
                    ihsan_ok = result.get("ihsan_score", 0) >= self.ihsan_threshold

                    # 5. PROVE - Record result
                    mission["result"] = result
                    mission["status"] = "completed" if ihsan_ok else "needs_review"
                    mission["completed"] = datetime.now(timezone.utc).isoformat()

                    self._completed.append(mission)
                    self._missions.remove(mission)
                    self._metrics["missions_completed"] += 1

                    logger.info(f"✓ Mission {mission['id']}: {'PASS' if ihsan_ok else 'REVIEW'}")
                else:
                    # Idle - proactive monitoring
                    logger.info("  ○ Idle - monitoring for opportunities")

                # 6. LEARN - Update metrics
                self._metrics["cycles"] = self._cycle_count

            except Exception as e:
                logger.error(f"Cycle error: {e}")

            # Sleep until next cycle
            elapsed = time.perf_counter() - cycle_start
            sleep_time = max(1.0, self.cycle_interval - elapsed)

            # Show countdown for long sleeps
            if sleep_time > 5:
                logger.info(f"  Next cycle in {sleep_time:.0f}s (Ctrl+C to stop)")

            await asyncio.sleep(sleep_time)

    def _select_agents(self, mission: Dict) -> List[str]:
        """Select appropriate agents for mission."""
        desc = mission["description"].lower()

        agents = ["coordinator"]  # Always include coordinator

        if any(w in desc for w in ["plan", "strategy", "approach"]):
            agents.append("strategist")
        if any(w in desc for w in ["research", "investigate", "find"]):
            agents.append("researcher")
        if any(w in desc for w in ["analyze", "data", "pattern"]):
            agents.append("analyst")
        if any(w in desc for w in ["create", "design", "build"]):
            agents.append("creator")
        if any(w in desc for w in ["security", "safe", "risk", "ethic"]):
            agents.append("guardian")

        # Default to strategist + guardian + coordinator
        if len(agents) == 1:
            agents = ["strategist", "guardian", "coordinator"]

        return agents

    async def _execute_mission(self, mission: Dict, agents: List[str]) -> Dict:
        """Execute mission with PAT team."""
        import httpx

        results = []
        total_tokens = 0

        for agent_id in agents:
            agent = PAT_AGENTS[agent_id]

            system_prompt = f"""You are the PAT {agent['name']}. Your role is {agent['role']}.
Standing on Giants: {agent['giants']}.
Be concise (2-3 paragraphs). Focus on actionable insights."""

            logger.info(f"    🤖 {agent['name']}...")

            headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

            try:
                async with httpx.AsyncClient(headers=headers, timeout=180.0) as client:
                    resp = await client.post(f"{self.base_url}/v1/chat/completions", json={
                        "model": "deepseek/deepseek-r1-0528-qwen3-8b",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Mission: {mission['description']}"},
                        ],
                        "max_tokens": 600,
                    })

                    if resp.status_code == 200:
                        data = resp.json()
                        content = data["choices"][0]["message"].get("content", "")
                        tokens = data.get("usage", {}).get("total_tokens", 0)
                        total_tokens += tokens

                        results.append({
                            "agent": agent_id,
                            "name": agent["name"],
                            "content": content,
                            "tokens": tokens,
                            "success": True,
                        })
                    else:
                        results.append({
                            "agent": agent_id,
                            "success": False,
                            "error": f"HTTP {resp.status_code}",
                        })
            except Exception as e:
                results.append({
                    "agent": agent_id,
                    "success": False,
                    "error": str(e),
                })

        # Calculate Ihsān score
        successful = sum(1 for r in results if r.get("success"))
        ihsan_score = successful / len(results) if results else 0

        self._metrics["tokens_used"] += total_tokens
        self._metrics["ihsan_score"] = ihsan_score

        return {
            "agents": results,
            "total_tokens": total_tokens,
            "ihsan_score": ihsan_score,
            "success_count": successful,
            "total_count": len(results),
        }

    def get_status(self) -> Dict:
        """Get current kernel status."""
        return {
            "running": self._running,
            "cycles": self._cycle_count,
            "pending_missions": len(self._missions),
            "completed_missions": len(self._completed),
            "metrics": self._metrics,
        }


# ════════════════════════════════════════════════════════════════════════════════
# NODE0 ORCHESTRATOR
# ════════════════════════════════════════════════════════════════════════════════

class Node0Orchestrator:
    """
    Main Node0 orchestrator - coordinates all subsystems.
    """

    def __init__(self):
        self.kernel = Node0ProactiveKernel()
        self._shutdown_event = asyncio.Event()

    async def start(self):
        """Start Node0."""
        self._print_banner()

        # Check LM Studio connection
        if not await self._check_connection():
            logger.error("Cannot connect to LM Studio. Exiting.")
            return

        # Setup signal handlers
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._handle_shutdown)
            except NotImplementedError:
                pass  # Windows doesn't support add_signal_handler

        # Start kernel
        kernel_task = asyncio.create_task(self.kernel.start())

        # Wait for shutdown
        await self._shutdown_event.wait()

        # Cleanup
        await self.kernel.stop()
        kernel_task.cancel()

        logger.info("Node0 shutdown complete.")

    async def _check_connection(self) -> bool:
        """Check LM Studio connection."""
        import httpx

        token = _resolve_lm_token()
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        try:
            async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
                resp = await client.get("http://192.168.56.1:1234/v1/models")
                if resp.status_code == 200:
                    models = resp.json().get("data", [])
                    loaded = [m for m in models if m.get("loaded")]
                    logger.info(f"✓ LM Studio connected: {len(models)} models, {len(loaded)} loaded")
                    return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")

        return False

    def _handle_shutdown(self):
        """Handle shutdown signal."""
        logger.info("\nShutdown signal received...")
        self._shutdown_event.set()

    def _print_banner(self):
        """Print Node0 banner."""
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║   ███╗   ██╗ ██████╗ ██████╗ ███████╗     ██████╗                            ║
║   ████╗  ██║██╔═══██╗██╔══██╗██╔════╝    ██╔═████╗                           ║
║   ██╔██╗ ██║██║   ██║██║  ██║█████╗      ██║██╔██║                           ║
║   ██║╚██╗██║██║   ██║██║  ██║██╔══╝      ████╔╝██║                           ║
║   ██║ ╚████║╚██████╔╝██████╔╝███████╗    ╚██████╔╝                           ║
║   ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝ ╚══════╝     ╚═════╝                            ║
║                                                                              ║
║   B I Z R A   S O V E R E I G N   A I   -   H O M E   B A S E                ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  PAT Team: 7 Agents | Mode: Proactive Partner | Ihsān: 0.95                  ║
║  Model: DeepSeek R1 8B | Backend: LM Studio                                  ║
║  لا نفترض — We do not assume. إحسان — Excellence in all things.              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)


# ════════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════════

async def cmd_start(args):
    """Start Node0."""
    orchestrator = Node0Orchestrator()
    await orchestrator.start()


async def cmd_status(args):
    """Check system status."""
    import httpx

    print("\n" + "═" * 60)
    print("NODE0 STATUS CHECK")
    print("═" * 60)

    token = _resolve_lm_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    # Check LM Studio
    try:
        async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
            resp = await client.get("http://192.168.56.1:1234/v1/models")
            if resp.status_code == 200:
                models = resp.json().get("data", [])
                loaded = [m for m in models if m.get("loaded")]
                print("  LM Studio:    ✓ Connected")
                print(f"  Models:       {len(models)} available, {len(loaded)} loaded")
                for m in loaded:
                    print(f"    → {m['id']}")
            else:
                print(f"  LM Studio:    ✗ Error {resp.status_code}")
    except Exception as e:
        print(f"  LM Studio:    ✗ {e}")

    print()
    print(f"  Token:        {'✓ Set' if token else '✗ Not set'}")
    print("  PAT Agents:   7 configured")
    print("  Mode:         proactive_partner")
    print("═" * 60 + "\n")


async def cmd_mission(args):
    """Run a single mission."""
    kernel = Node0ProactiveKernel({"cycle_interval": 5.0})

    print("\n" + "═" * 60)
    print("NODE0 MISSION EXECUTION")
    print("═" * 60)
    print(f"Mission: {args.task[:60]}...")
    print("═" * 60 + "\n")

    mission = await kernel.add_mission(args.task)

    # Execute immediately
    agents = kernel._select_agents(mission)
    print(f"Assigned agents: {', '.join(agents)}\n")

    result = await kernel._execute_mission(mission, agents)

    # Display results
    print("\n" + "═" * 60)
    print("MISSION RESULTS")
    print("═" * 60)

    for r in result["agents"]:
        if r.get("success"):
            print(f"\n┌─ {r['name'].upper()} ─")
            for line in r.get("content", "").split("\n")[:10]:
                print(f"│  {line}")
            print(f"└─ ({r.get('tokens', 0)} tokens)")

    print("\n" + "─" * 60)
    print(f"Total Tokens: {result['total_tokens']}")
    print(f"Ihsān Score:  {result['ihsan_score']:.2%}")
    print(f"Status:       {'✓ PASS' if result['ihsan_score'] >= 0.95 else '⚠ REVIEW'}")
    print("═" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="BIZRA Node0 — Local Sovereign AI Home Base",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command")

    # Start command
    subparsers.add_parser("start", help="Start Node0 proactive kernel")

    # Status command
    subparsers.add_parser("status", help="Check system status")

    # Mission command
    p_mission = subparsers.add_parser("mission", help="Execute a mission")
    p_mission.add_argument("task", help="Mission description")

    args = parser.parse_args()

    if not args.command:
        # Default to start
        args.command = "start"

    commands = {
        "start": cmd_start,
        "status": cmd_status,
        "mission": cmd_mission,
    }

    try:
        asyncio.run(commands[args.command](args))
    except KeyboardInterrupt:
        print("\nShutdown requested.")


if __name__ == "__main__":
    main()
