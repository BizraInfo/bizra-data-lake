"""
Runtime CLI — Command-Line Interface for Sovereign Runtime
==========================================================
Provides CLI interface for interacting with the Sovereign Runtime.

Usage:
    python -m core.sovereign.runtime query "What is sovereignty?"
    python -m core.sovereign.runtime status
    python -m core.sovereign.runtime --mode AUTONOMOUS run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from .runtime_core import SovereignRuntime
from .runtime_types import RuntimeConfig, RuntimeMode

logger = logging.getLogger("sovereign.cli")


async def cli_main() -> None:
    """Command-line interface for the Sovereign Runtime."""
    parser = argparse.ArgumentParser(
        description="BIZRA Sovereign Runtime CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m core.sovereign.runtime query "What is sovereignty?"
  python -m core.sovereign.runtime status
  python -m core.sovereign.runtime --mode AUTONOMOUS run
        """,
    )

    parser.add_argument(
        "command",
        choices=["query", "status", "run", "version"],
        help="Command to execute",
    )
    parser.add_argument("args", nargs="*", help="Command arguments")
    parser.add_argument(
        "--mode",
        choices=["MINIMAL", "STANDARD", "AUTONOMOUS", "DEBUG"],
        default="STANDARD",
        help="Runtime mode",
    )
    parser.add_argument("--snr", type=float, default=0.95, help="SNR threshold")
    parser.add_argument("--json", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    # Configure
    config = RuntimeConfig(
        mode=RuntimeMode[args.mode],
        snr_threshold=args.snr,
        enable_autonomous_loop=(args.mode == "AUTONOMOUS"),
    )

    if args.command == "version":
        print("Sovereign Runtime v1.0.0")
        return

    async with SovereignRuntime.create(config) as runtime:
        if args.command == "query":
            await _handle_query(runtime, args)
        elif args.command == "status":
            await _handle_status(runtime, args)
        elif args.command == "run":
            await _handle_run(runtime)


async def _handle_query(runtime: SovereignRuntime, args) -> None:
    """Handle query command."""
    query_text = " ".join(args.args) if args.args else "Hello, Sovereign"
    result = await runtime.query(query_text)

    if args.json:
        print(
            json.dumps(
                {
                    "success": result.success,
                    "response": result.response,
                    "snr": result.snr_score,
                    "ihsan": result.ihsan_score,
                    "time_ms": result.processing_time_ms,
                },
                indent=2,
            )
        )
    else:
        print(f"\n{'─' * 60}")
        print(f"Query: {query_text}")
        print(f"{'─' * 60}")
        print(f"Response: {result.response}")
        print(f"{'─' * 60}")
        print(f"SNR: {result.snr_score:.3f}")
        print(f"Ihsan: {result.ihsan_score:.3f}")
        print(f"Time: {result.processing_time_ms:.1f}ms")
        print(f"{'─' * 60}\n")


async def _handle_status(runtime: SovereignRuntime, args) -> None:
    """Handle status command."""
    status = runtime.status()
    if args.json:
        print(json.dumps(status, indent=2))
    else:
        print(f"\n{'═' * 60}")
        print("SOVEREIGN RUNTIME STATUS")
        print(f"{'═' * 60}")
        print(f"Node ID: {status['identity']['node_id']}")
        print(f"Mode: {status['state']['mode']}")
        print(f"Health: {status['health']['status']} ({status['health']['score']:.3f})")
        print(f"{'═' * 60}\n")


async def _handle_run(runtime: SovereignRuntime) -> None:
    """Handle run command."""
    print("Sovereign Runtime running. Press Ctrl+C to stop.")
    await runtime.wait_for_shutdown()


def main() -> None:
    """Entry point for CLI."""
    asyncio.run(cli_main())


__all__ = [
    "cli_main",
    "main",
]
