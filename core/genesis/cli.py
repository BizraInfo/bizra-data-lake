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
from pathlib import Path
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


def build_activate_parser(subparsers: Any) -> argparse.ArgumentParser:
    """Add the 'activate' subcommand — full genesis activation pipeline.

    Usage:
        python -m core.sovereign activate --seed-file ~/.bizra/seed.bin
        python -m core.sovereign activate --seed-phrase "my secret phrase"
        python -m core.sovereign activate --skip-orchestrator --skip-breath
    """
    activate_parser = subparsers.add_parser(
        "activate",
        help="Full genesis activation: ceremony + orchestrator + heartbeat",
        description=(
            "Run the complete BIZRA genesis activation pipeline.\n"
            "Wires: ceremony (cryptographic root) -> orchestrator (12-step bootstrap) "
            "-> heartbeat (boot + first breath) -> evidence (activation receipt)."
        ),
    )

    # Seed source (mutually exclusive)
    seed_group = activate_parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to 32+ byte seed file",
    )
    seed_group.add_argument(
        "--seed-phrase",
        type=str,
        default=None,
        metavar="PHRASE",
        help="Passphrase to derive seed via BLAKE3",
    )

    # Output directory
    activate_parser.add_argument(
        "--data-dir",
        type=str,
        default="sovereign_state/genesis",
        help="Output directory for genesis artifacts (default: sovereign_state/genesis)",
    )

    # Pipeline control
    activate_parser.add_argument(
        "--skip-orchestrator",
        action="store_true",
        help="Skip 12-step orchestrator (ceremony + heartbeat only)",
    )
    activate_parser.add_argument(
        "--skip-breath",
        action="store_true",
        help="Skip first breath (boot only, no Helix3 tick)",
    )

    # Output format
    activate_parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )
    activate_parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing activation artifacts instead of creating new ones",
    )

    return activate_parser


def handle_activate(args: argparse.Namespace) -> None:
    """Handle the 'activate' subcommand — full genesis activation pipeline."""
    from core.genesis.activation import GenesisActivation
    from core.proof_engine.canonical import blake3_digest

    # Resolve seed
    seed: bytes | None = None

    if args.seed_file:
        seed_path = Path(args.seed_file).expanduser()
        if not seed_path.exists():
            print(f"Error: seed file not found: {seed_path}", file=sys.stderr)
            sys.exit(1)
        seed = seed_path.read_bytes()
        if len(seed) < 16:
            print("Error: seed file must be at least 16 bytes", file=sys.stderr)
            sys.exit(1)
    elif args.seed_phrase:
        seed = blake3_digest(args.seed_phrase.encode("utf-8"))
    else:
        # Generate ephemeral seed (for development/testing)
        import os

        seed = os.urandom(32)
        print("Warning: using ephemeral random seed (not reproducible)", file=sys.stderr)
        print("For reproducible activation, use --seed-file or --seed-phrase", file=sys.stderr)

    data_dir = Path(args.data_dir)

    activation = GenesisActivation(
        node_seed=seed,
        data_dir=data_dir,
        skip_orchestrator=getattr(args, "skip_orchestrator", False),
        skip_breath=getattr(args, "skip_breath", False),
    )

    # Verify mode
    if getattr(args, "verify", False):
        valid, reasons = activation.verify()
        if getattr(args, "json", False):
            import json

            print(json.dumps({"valid": valid, "reasons": reasons}, indent=2))
        else:
            if valid:
                print("Activation artifacts verified: all checks passed")
            else:
                print("Activation verification FAILED:")
                for r in reasons:
                    print(f"  - {r}")
        sys.exit(0 if valid else 1)

    # Activate
    result = activation.activate()

    if getattr(args, "json", False):
        import json

        print(json.dumps(result.as_dict(), indent=2, default=str))
    else:
        print(f"Node ID:          {result.node_id}")
        print(f"Genesis Hash:     {result.genesis_hash[:16]}...")
        print(f"Activation Hash:  {result.activation_hash[:16]}...")
        print(f"Evidence Valid:   {result.evidence_chain_valid}")
        print(f"Duration:         {result.duration_ms:.1f}ms")
        print(f"Artifacts:        {data_dir}")
        if result.orchestrator_reason_codes:
            print(f"Reason Codes:     {', '.join(result.orchestrator_reason_codes)}")

    sys.exit(0 if result.evidence_chain_valid else 1)
