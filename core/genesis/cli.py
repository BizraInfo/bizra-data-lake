"""
Genesis CLI — Command-Line Interface for Node Bootstrap
========================================================

Provides the `genesis` subcommand for the sovereign CLI:

  python -m core.sovereign genesis [options]

Flags:
  --all               Run all genesis steps
  --identity-genesis  Mint new node identity
  --hardware-scan     Fingerprint hardware covenant
  --hda-bridge        HDA attestation (stub)
  --mobile-pair STR   Pair mobile companion ("Name:Model")
  --guild STR         Join an impact guild (e.g. "agriculture")
  --quest STR         Accept a quest (e.g. "001-sustainable-water")
  --ihsan FLOAT       Constitutional Ihsan target (default: 0.999)
  --node-dir PATH     Node data directory
  --json              JSON output mode

Standing on Giants:
- McIlroy (1978, Unix philosophy): Do one thing and do it well
- Deming (1950): Quality-gated CLI — each flag maps to one step
"""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser, Namespace
from typing import Optional

from .orchestrator import GenesisOrchestrator
from .types import GenesisConfig


def add_genesis_parser(subparsers: object) -> None:
    """
    Add the 'genesis' subcommand to an argparse subparsers group.

    Args:
        subparsers: The subparsers group from argparse.ArgumentParser
    """
    p: ArgumentParser = subparsers.add_parser(  # type: ignore[attr-defined]
        "genesis",
        help="Bootstrap a sovereign BIZRA node",
        description=(
            "Execute the full 8-step genesis protocol to bootstrap a sovereign node.\n"
            "Use --all to run all steps, or specify individual steps."
        ),
    )

    p.add_argument(
        "--all",
        action="store_true",
        dest="all_steps",
        help="Run all genesis steps (identity + hardware + urp + guild + quest + ihsan)",
    )
    p.add_argument(
        "--identity-genesis",
        action="store_true",
        dest="identity_genesis",
        help="Mint a new node identity (PAT-7 + SAT-5 agents)",
    )
    p.add_argument(
        "--hardware-scan",
        action="store_true",
        dest="hardware_scan",
        help="Fingerprint hardware covenant (CPU/GPU/RAM)",
    )
    p.add_argument(
        "--hda-bridge",
        action="store_true",
        dest="hda_bridge",
        help="Hardware Data Attestation bridge (stub, TPM/ZKP future)",
    )
    p.add_argument(
        "--mobile-pair",
        metavar="DEVICE",
        dest="mobile_pair",
        help="Pair mobile companion device, e.g. 'Z Fold 6:SM-F956B'",
    )
    p.add_argument(
        "--guild",
        metavar="GUILD_ID",
        dest="guild_join",
        help="Join an impact guild, e.g. 'agriculture', 'healthcare', 'energy'",
    )
    p.add_argument(
        "--quest",
        metavar="QUEST_ID",
        dest="quest_accept",
        help="Accept a quest, e.g. '001-sustainable-water'",
    )
    p.add_argument(
        "--ihsan",
        metavar="FLOAT",
        dest="ihsan_target",
        type=float,
        default=0.999,
        help="Constitutional Ihsan target (default: 0.999)",
    )
    p.add_argument(
        "--node-dir",
        metavar="PATH",
        dest="node_dir",
        help="Node data directory",
    )
    p.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON (machine-readable)",
    )


def build_config(args: Namespace) -> GenesisConfig:
    """Build a GenesisConfig from parsed CLI args."""
    all_steps = getattr(args, "all_steps", False)

    return GenesisConfig(
        identity_genesis=all_steps or getattr(args, "identity_genesis", False),
        hardware_scan=all_steps or getattr(args, "hardware_scan", False),
        hda_bridge=getattr(args, "hda_bridge", False),
        mobile_pair=getattr(args, "mobile_pair", None),
        guild_join=getattr(args, "guild_join", None)
        or ("agriculture" if all_steps else None),
        quest_accept=getattr(args, "quest_accept", None)
        or ("001-sustainable-water" if all_steps else None),
        ihsan_target=getattr(args, "ihsan_target", 0.999),
        node_dir=getattr(args, "node_dir", None),
        json_output=getattr(args, "json_output", False),
    )


def run_genesis(args: Namespace) -> None:
    """
    Execute genesis and render output.

    Args:
        args: Parsed CLI arguments from argparse
    """
    config = build_config(args)

    # Require at least one step or --all
    has_any = (
        config.identity_genesis
        or config.hardware_scan
        or config.hda_bridge
        or config.mobile_pair
        or config.guild_join
        or config.quest_accept
    )
    if not has_any:
        _print_genesis_help()
        sys.exit(0)

    orchestrator = GenesisOrchestrator()
    result = orchestrator.run(config)

    if config.json_output:
        print(json.dumps(result.to_dict(), indent=2))
        sys.exit(0 if result.success else 1)

    # Human-readable output
    _print_genesis_result(result, config)
    sys.exit(0 if result.success else 1)


def _print_genesis_result(result: object, config: GenesisConfig) -> None:
    """Render genesis result to stdout in human-readable form."""
    from .types import GenesisResult

    r: GenesisResult = result  # type: ignore[assignment]

    print()
    print("=" * 70)
    print("  BIZRA GENESIS PROTOCOL")
    print("=" * 70)

    if r.node_id:
        print(f"\n  Node ID:       {r.node_id}")
    if r.genesis_hash:
        print(f"  Genesis Hash:  {r.genesis_hash[:16]}...{r.genesis_hash[-8:]}")
    print(f"  Duration:      {r.total_duration_ms:.0f}ms")
    print()

    # Step table
    print("  STEP RESULTS")
    print("  " + "-" * 50)
    status_icon = {
        "success": "✓",
        "failed": "✗",
        "skipped": "○",
        "pending": "…",
    }
    for step in r.steps:
        icon = status_icon.get(step.status, "?")
        name_padded = step.name.replace("_", " ").title().ljust(22)
        timing = f"{step.duration_ms:.0f}ms".rjust(7)
        detail = ""
        if step.status == "failed":
            detail = f"  ERROR: {step.error[:40]}"
        elif step.status == "skipped":
            detail = "  (disabled)"
        elif step.details:
            # Show first meaningful detail value
            key_vals = [
                f"{k}={v}" for k, v in step.details.items()
                if k not in ("strict", "action_on_mismatch", "commitment")
                and str(v)[:20]
            ]
            detail = "  " + " | ".join(key_vals[:2])
        print(f"    {icon} {name_padded} {timing}{detail}")

    print()
    overall = "COMPLETE" if r.success else (
        "PARTIAL" if r.success_count > 0 else "FAILED"
    )
    print(f"  Status:  {overall}  ({r.success_count}/{r.step_count} steps OK)")

    if overall == "COMPLETE":
        print()
        print("  Standing on Giants. Your node is sovereign.")
        print("  BIZRA — بذرة — Seed. You are the beginning.")

    print("=" * 70)
    print()


def _print_genesis_help() -> None:
    """Print a concise usage message when no flags are given."""
    print()
    print("  Usage: python -m core.sovereign genesis [flags]")
    print()
    print("  Flags:")
    print("    --all              Run all steps (recommended)")
    print("    --identity-genesis Mint node identity")
    print("    --hardware-scan    Fingerprint hardware covenant")
    print("    --guild GUILD_ID   Join impact guild")
    print("    --quest QUEST_ID   Accept impact quest")
    print("    --json             JSON output")
    print()
    print("  Examples:")
    print("    python -m core.sovereign genesis --all")
    print("    python -m core.sovereign genesis --guild agriculture --quest 001-sustainable-water")
    print("    python -m core.sovereign genesis --all --json")
    print()
