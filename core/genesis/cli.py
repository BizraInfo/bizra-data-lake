"""
BIZRA Genesis CLI — Command-Line Interface for Node Bootstrap
================================================================

Provides the `genesis` subcommand for the Sovereign Engine CLI.
Maps CLI flags to GenesisConfig and runs the GenesisOrchestrator.

Usage:
    python -m core.sovereign genesis --identity-genesis --pat-7 --sat-5
    python -m core.sovereign genesis --identity-genesis --hardware-scan \\
        --pat-7 --sat-5 --hda-bridge --mobile-pair "Z Fold 6:SM-F956B" \\
        --guild-join agriculture --quest-accept 001-sustainable-water \\
        --ihsan-target 0.999

Standing on Giants:
- Thompson & Ritchie (1973): Unix CLI conventions
- GNU (1987): Long-form argument standards
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from .orchestrator import GenesisOrchestrator
from .types import GenesisConfig


def build_genesis_parser(subparsers: Any) -> argparse.ArgumentParser:
    """
    Add the 'genesis' subcommand to the Sovereign Engine CLI.

    Args:
        subparsers: argparse subparsers object from the main parser

    Returns:
        The genesis subparser for further customization
    """
    genesis_parser = subparsers.add_parser(
        "genesis",
        help="Bootstrap a new BIZRA node (one-command genesis)",
        description="One-command BIZRA node genesis: identity, hardware, "
        "PAT/SAT, tokens, guild, quest, and Ihsan targeting.",
    )

    # Identity flags
    genesis_parser.add_argument(
        "--identity-genesis",
        action="store_true",
        help="Mint genesis identity (Node0 = Block0)",
    )

    # Hardware
    genesis_parser.add_argument(
        "--hardware-scan",
        action="store_true",
        help="Scan and fingerprint local hardware",
    )

    # Agent counts (shorthand flags)
    genesis_parser.add_argument(
        "--pat-7",
        action="store_true",
        dest="pat_7",
        help="Instantiate 7 PAT (Personal Agentic Team) agents",
    )
    genesis_parser.add_argument(
        "--sat-5",
        action="store_true",
        dest="sat_5",
        help="Instantiate 5 SAT (System Agentic Team) agents",
    )
    genesis_parser.add_argument(
        "--sat-49",
        action="store_true",
        dest="sat_49",
        help="Instantiate full SAT-49 operating profile",
    )
    genesis_parser.add_argument(
        "--pat-count",
        type=int,
        default=None,
        help="Custom PAT agent count (default: 7)",
    )
    genesis_parser.add_argument(
        "--sat-count",
        type=int,
        default=None,
        help="Custom SAT agent count (default: 5)",
    )
    genesis_parser.add_argument(
        "--sat-mode",
        choices=["mini5", "full49"],
        default=None,
        help="SAT operating profile (mini5 for User Zero, full49 for parity mode)",
    )

    # Bridge
    genesis_parser.add_argument(
        "--hda-bridge",
        action="store_true",
        help="Initialize AutoHotkey-Rust IPC bridge",
    )

    # Mobile
    genesis_parser.add_argument(
        "--mobile-pair",
        type=str,
        default=None,
        metavar="DEVICE",
        help='Mobile device to pair (e.g., "Z Fold 6:SM-F956B")',
    )

    # Social
    genesis_parser.add_argument(
        "--guild-join",
        type=str,
        default=None,
        metavar="GUILD",
        help='Guild to join (e.g., "agriculture")',
    )
    genesis_parser.add_argument(
        "--quest-accept",
        type=str,
        default=None,
        metavar="QUEST",
        help='Quest to accept (e.g., "001-sustainable-water")',
    )

    # Constitutional
    genesis_parser.add_argument(
        "--ihsan-target",
        type=float,
        default=0.999,
        help="Ihsan excellence target (default: 0.999)",
    )
    strict_group = genesis_parser.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict-bootstrap",
        dest="strict_bootstrap",
        action="store_true",
        help="Fail-closed bootstrap: reject any deferred/stub step",
    )
    strict_group.add_argument(
        "--allow-degraded",
        dest="strict_bootstrap",
        action="store_false",
        help="Allow degraded bootstrap steps (diagnostic mode only)",
    )
    genesis_parser.set_defaults(strict_bootstrap=True)

    # Output
    genesis_parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )

    # Architect name
    genesis_parser.add_argument(
        "--architect",
        type=str,
        default="MoMo",
        help="Genesis architect name (default: MoMo)",
    )

    return genesis_parser


def handle_genesis(args: argparse.Namespace) -> None:
    """
    Handle the 'genesis' subcommand.

    Parses CLI args into GenesisConfig, runs the orchestrator,
    and prints formatted output.
    """
    # Resolve PAT/SAT counts
    pat_count = 7 if getattr(args, "pat_7", False) else (args.pat_count or 7)
    sat_mode = args.sat_mode
    if getattr(args, "sat_49", False):
        sat_mode = "full49"
        sat_count = 49
    elif getattr(args, "sat_5", False):
        sat_mode = "mini5"
        sat_count = 5
    else:
        sat_count = args.sat_count or 5
        if sat_mode is None:
            sat_mode = "full49" if sat_count >= 49 else "mini5"
    if sat_mode == "full49":
        sat_count = max(sat_count, 49)
    elif sat_mode == "mini5" and sat_count == 5:
        pass  # default mini5 — no override
    # else: honour explicit --sat-count as-is

    config = GenesisConfig(
        identity_genesis=args.identity_genesis,
        hardware_scan=args.hardware_scan,
        pat_count=pat_count,
        sat_count=sat_count,
        sat_mode=sat_mode,
        hda_bridge=args.hda_bridge,
        mobile_pair=args.mobile_pair,
        guild_join=args.guild_join,
        quest_accept=args.quest_accept,
        ihsan_target=args.ihsan_target,
        strict_bootstrap=getattr(args, "strict_bootstrap", True),
        allow_degraded=not getattr(args, "strict_bootstrap", True),
        architect_name=getattr(args, "architect", "MoMo"),
    )

    orchestrator = GenesisOrchestrator(config)
    result = orchestrator.run()

    if getattr(args, "json", False):
        import json

        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(orchestrator.format_output(result))

    # Exit with appropriate code
    sys.exit(0 if result.success else 1)
