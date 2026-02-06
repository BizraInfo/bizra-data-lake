#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ██╗      █████╗ ██╗   ██╗███╗   ██╗ ██████╗██╗  ██╗                       ║
║   ██║     ██╔══██╗██║   ██║████╗  ██║██╔════╝██║  ██║                       ║
║   ██║     ███████║██║   ██║██╔██╗ ██║██║     ███████║                       ║
║   ██║     ██╔══██║██║   ██║██║╚██╗██║██║     ██╔══██║                       ║
║   ███████╗██║  ██║╚██████╔╝██║ ╚████║╚██████╗██║  ██║                       ║
║   ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝╚═╝  ╚═╝                       ║
║                                                                              ║
║                    SOVEREIGN ENGINE PRODUCTION LAUNCHER                      ║
║         Full Integration: Runtime + Bridge + Metrics + Autonomy              ║
║                                                                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   This is the production entry point that:                                   ║
║   1. Initializes SovereignRuntime                                            ║
║   2. Connects SovereignBridge (inference, federation, memory, A2A)           ║
║   3. Starts MetricsCollector                                                 ║
║   4. Wires AutonomousLoop with real observers/analyzers                      ║
║   5. Optionally starts API server                                            ║
║                                                                              ║
║   Usage:                                                                     ║
║     python -m core.sovereign.launch                      # Full stack        ║
║     python -m core.sovereign.launch --no-api             # Without API       ║
║     python -m core.sovereign.launch --mode AUTONOMOUS    # Full autonomy     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
from pathlib import Path
from typing import Any, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)-24s │ %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sovereign.launch")

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
║   "Every inference carries proof. Every decision passes the gate.            ║
║    Every node is sovereign. Every human is a seed."                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""


class SovereignLauncher:
    """
    Production launcher for the Sovereign Engine.

    Orchestrates:
    - SovereignRuntime (core query processing)
    - SovereignBridge (subsystem integration)
    - MetricsCollector (real-time monitoring)
    - AutonomousLoop (OODA decision cycle)
    - API Server (external access)
    """

    def __init__(
        self,
        node_id: str = "",
        state_dir: str = "./sovereign_state",
        api_port: int = 8080,
        enable_api: bool = True,
        enable_autonomy: bool = True,
        snr_threshold: float = 0.95,
        ihsan_threshold: float = 0.95,
    ):
        self.node_id = node_id
        self.state_dir = Path(state_dir)
        self.api_port = api_port
        self.enable_api = enable_api
        self.enable_autonomy = enable_autonomy
        self.snr_threshold = snr_threshold
        self.ihsan_threshold = ihsan_threshold

        # Components (initialized in start())
        self.runtime = None
        self.bridge = None
        self.metrics = None
        self.api_server = None

        # State
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start all components."""
        logger.info("=" * 70)
        logger.info("SOVEREIGN ENGINE STARTING")
        logger.info("=" * 70)

        # Ensure state directory
        self.state_dir.mkdir(parents=True, exist_ok=True)

        # 1. Initialize Runtime
        logger.info("[1/5] Initializing Runtime...")
        from .runtime import RuntimeConfig, SovereignRuntime

        config = RuntimeConfig(
            node_id=self.node_id,
            state_dir=self.state_dir,
            snr_threshold=self.snr_threshold,
            ihsan_threshold=self.ihsan_threshold,
            autonomous_enabled=self.enable_autonomy,
        )
        self.runtime = SovereignRuntime(config)
        await self.runtime.initialize()
        logger.info(f"  ✓ Runtime initialized: {config.node_id}")

        # 2. Connect Bridge
        logger.info("[2/5] Connecting Bridge...")
        from .bridge import SovereignBridge

        self.bridge = SovereignBridge(
            node_id=config.node_id,
            state_dir=self.state_dir,
        )
        bridge_status = await self.bridge.connect_all()
        logger.info(f"  ✓ Bridge connected: {bridge_status}")

        # 3. Start Metrics Collector
        logger.info("[3/5] Starting Metrics...")
        from .metrics import MetricsCollector

        self.metrics = MetricsCollector(collection_interval=1.0)
        await self.metrics.start()
        logger.info("  ✓ Metrics collector started")

        # 4. Wire Autonomy
        if self.enable_autonomy:
            logger.info("[4/5] Wiring Autonomy...")
            await self._wire_autonomy()
            logger.info("  ✓ Autonomous loop wired")
        else:
            logger.info("[4/5] Autonomy disabled")

        # 5. Start API Server
        if self.enable_api:
            logger.info("[5/5] Starting API Server...")
            from .api import SovereignAPIServer

            self.api_server = SovereignAPIServer(
                runtime=self.runtime,
                port=self.api_port,
            )
            await self.api_server.start()
            logger.info(f"  ✓ API server on port {self.api_port}")
        else:
            logger.info("[5/5] API disabled")

        # Setup signal handlers
        self._setup_signals()

        self._running = True
        logger.info("=" * 70)
        logger.info("SOVEREIGN ENGINE READY")
        logger.info("=" * 70)
        logger.info("")
        logger.info(f"  Node ID:     {config.node_id}")
        logger.info(f"  Mode:        {config.mode.name}")
        logger.info(f"  SNR Target:  {self.snr_threshold}")
        logger.info(f"  Ihsān Gate:  {self.ihsan_threshold}")
        if self.enable_api:
            logger.info(f"  API:         http://0.0.0.0:{self.api_port}")
        logger.info("")
        logger.info("Press Ctrl+C to shutdown gracefully")
        logger.info("=" * 70)

    async def _wire_autonomy(self) -> None:
        """Wire AutonomousLoop with real observers and analyzers."""
        from .metrics import create_autonomy_analyzer, create_autonomy_observer

        # Get autonomy loop from runtime
        autonomy = self.runtime._autonomous_loop
        if autonomy is None:
            return

        # Register metrics observer
        observer = create_autonomy_observer(self.metrics)
        autonomy.register_observer(observer)

        # Register metrics analyzer
        analyzer = create_autonomy_analyzer(self.metrics)
        autonomy.register_analyzer(analyzer)

        # Register action executors
        await self._register_executors(autonomy)

    async def _register_executors(self, autonomy) -> None:
        """Register decision executors."""

        async def reduce_memory(decision) -> bool:
            """Clear caches to reduce memory."""
            logger.info("Executing: reduce_memory")
            if self.runtime:
                self.runtime._cache.clear()
            return True

        async def throttle_inference(decision) -> bool:
            """Reduce inference rate for cooling."""
            logger.info("Executing: throttle_inference")
            # Would adjust inference rate limits
            return True

        async def switch_inference_tier(decision) -> bool:
            """Switch to a different inference tier."""
            logger.info("Executing: switch_inference_tier")
            # Would trigger tier switch in bridge
            return True

        async def boost_snr(decision) -> bool:
            """Boost SNR through optimization."""
            logger.info("Executing: boost_snr")
            # Would trigger SNR optimization
            return True

        autonomy.register_executor("reduce_memory", reduce_memory)
        autonomy.register_executor("throttle_inference", throttle_inference)
        autonomy.register_executor("switch_inference_tier", switch_inference_tier)
        autonomy.register_executor("boost_snr", boost_snr)

    def _setup_signals(self) -> None:
        """Setup graceful shutdown handlers."""
        try:
            loop = asyncio.get_running_loop()
            for sig in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.stop()))
        except (NotImplementedError, RuntimeError):
            pass

    async def stop(self) -> None:
        """Stop all components gracefully."""
        if not self._running:
            return

        logger.info("")
        logger.info("=" * 70)
        logger.info("SOVEREIGN ENGINE SHUTTING DOWN")
        logger.info("=" * 70)

        self._running = False

        # Stop in reverse order
        if self.api_server:
            logger.info("Stopping API server...")
            await self.api_server.stop()

        if self.metrics:
            logger.info("Stopping metrics collector...")
            await self.metrics.stop()

        if self.bridge:
            logger.info("Disconnecting bridge...")
            await self.bridge.disconnect_all()

        if self.runtime:
            logger.info("Shutting down runtime...")
            await self.runtime.shutdown()

        logger.info("=" * 70)
        logger.info("SOVEREIGN ENGINE STOPPED")
        logger.info("=" * 70)

        self._shutdown_event.set()

    async def wait(self) -> None:
        """Wait until shutdown."""
        await self._shutdown_event.wait()

    async def run(self) -> None:
        """Start and run until shutdown."""
        await self.start()
        await self.wait()

    def status(self) -> Dict[str, Any]:
        """Get comprehensive status."""
        status = {
            "running": self._running,
            "node_id": self.node_id,
            "components": {},
        }

        if self.runtime:
            status["runtime"] = self.runtime.status()

        if self.bridge:
            status["bridge"] = self.bridge.status()

        if self.metrics:
            status["metrics"] = self.metrics.status()

        return status


# =============================================================================
# CLI
# =============================================================================


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="BIZRA Sovereign Engine Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m core.sovereign.launch                    # Full stack
  python -m core.sovereign.launch --no-api           # Without API
  python -m core.sovereign.launch --port 9000        # Custom port
  python -m core.sovereign.launch --no-autonomy      # Manual mode
        """,
    )

    parser.add_argument(
        "--node-id", default="", help="Node identifier (auto-generated if empty)"
    )
    parser.add_argument(
        "--state-dir", default="./sovereign_state", help="State directory path"
    )
    parser.add_argument("--port", type=int, default=8080, help="API server port")
    parser.add_argument("--no-api", action="store_true", help="Disable API server")
    parser.add_argument(
        "--no-autonomy", action="store_true", help="Disable autonomous loop"
    )
    parser.add_argument("--snr", type=float, default=0.95, help="SNR threshold")
    parser.add_argument("--ihsan", type=float, default=0.95, help="Ihsān threshold")
    parser.add_argument("--quiet", action="store_true", help="Suppress banner")

    args = parser.parse_args()

    if not args.quiet:
        print(BANNER)

    launcher = SovereignLauncher(
        node_id=args.node_id,
        state_dir=args.state_dir,
        api_port=args.port,
        enable_api=not args.no_api,
        enable_autonomy=not args.no_autonomy,
        snr_threshold=args.snr,
        ihsan_threshold=args.ihsan,
    )

    await launcher.run()


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = ["SovereignLauncher", "main"]


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    asyncio.run(main())
