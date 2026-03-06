"""
Sovereignty CLI Entry Point
════════════════════════════

Run with: python -m core.constitutional <command> [args]

Commands:
    init    [name]           Generate keypair, sign the Covenant
    work    "description"    Do verified work, earn SEED
    attest  <peer_id>        Vouch for another node's work
    status                   See your sovereign state
    ledger  [n]              Show last N events
    reset                    Delete node (irreversible)

Phase 67.04 — Sovereign Instantiation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from core.constitutional.cli import (
    attest_peer,
    get_status,
    init_node,
    load_node_state,
    process_work,
)
from core.constitutional.fixed_point import fp_float

# ═══════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════

DEFAULT_STATE_DIR = Path.home() / ".bizra"

GOLD = "\033[38;2;201;169;98m"
DIM = "\033[2m"
BOLD = "\033[1m"
GREEN = "\033[38;2;52;211;153m"
RED = "\033[38;2;248;113;113m"
PURPLE = "\033[38;2;167;139;250m"
RESET = "\033[0m"


def _sep():
    print(f"  {GOLD}{'─' * 40}{RESET}")


# ═══════════════════════════════════════════════════════════════════
# Command Handlers
# ═══════════════════════════════════════════════════════════════════


def cmd_init(args: list[str]) -> None:
    name = args[0] if args else f"node_{id(object()) % 100000}"
    print(f"\n  {DIM}Initializing sovereign node...{RESET}\n")

    result = init_node(name, DEFAULT_STATE_DIR)
    if not result.success:
        print(f"  {RED}Error:{RESET} {result.error}\n")
        return

    print(f"  {GREEN}✓{RESET} Node established")
    _sep()
    print(f"  {BOLD}Name:{RESET}     {name}")
    print(f"  {BOLD}ID:{RESET}       {result.node_id[:24]}...")
    print(f"  {BOLD}Covenant:{RESET} {result.covenant_hash[:20]}... {GREEN}✓{RESET}")
    print(f"  {BOLD}Status:{RESET}   {GREEN}SOVEREIGN{RESET}")
    _sep()
    print(f'\n  {DIM}Run: python -m core.constitutional work "description"{RESET}\n')


def cmd_work(args: list[str]) -> None:
    description = " ".join(args) if args else ""
    if not description:
        print(f'\n  {RED}Usage: work "description of your contribution"{RESET}\n')
        return

    print(f"\n  {DIM}Processing action receipt...{RESET}\n")

    result = process_work(description, DEFAULT_STATE_DIR)
    if not result.success:
        if not result.intent_passed:
            print(f"  {RED}✗{RESET} Intent Gate: {RED}FAILED{RESET}")
        elif result.ihsan_score > 0:
            print(
                f"  {RED}✗{RESET} Ihsan Score: {RED}{result.ihsan_score:.4f}{RESET} (floor: 0.95)"
            )
        print(f"  {DIM}{result.error}{RESET}\n")
        return

    print(f"  {GREEN}✓{RESET} Intent Gate: {GREEN}PASSED{RESET}")
    print(f"  {GREEN}✓{RESET} Ihsan Score: {GREEN}{result.ihsan_score:.4f}{RESET}")
    print(f"  {GREEN}✓{RESET} Throttle: {result.throttle:.2f}")
    print()

    state = load_node_state(DEFAULT_STATE_DIR)
    _sep()
    print(f"  {BOLD}Receipt:{RESET}  #{state.total_actions:04d}")
    print(f"  {BOLD}Hash:{RESET}     {result.receipt_hash[:24]}...")
    print(f"  {BOLD}Minted:{RESET}   {GREEN}+{result.seed_minted:.6f} SEED{RESET}")
    print(
        f"  {BOLD}Balance:{RESET}  {fp_float(state.seed_balance):.6f} SEED | {fp_float(state.bloom_balance):.6f} BLOOM"
    )
    _sep()
    print()


def cmd_attest(args: list[str]) -> None:
    if not args:
        print(f"\n  {RED}Usage: attest <peer_id>{RESET}\n")
        return

    peer_id = args[0]
    result = attest_peer(peer_id, DEFAULT_STATE_DIR)
    if not result.success:
        print(f"\n  {RED}Error:{RESET} {result.error}\n")
        return

    print(f"\n  {GREEN}✓{RESET} Attestation created")
    _sep()
    print(
        f"  {BOLD}Attested:{RESET}  {peer_id[:24]}{'...' if len(peer_id) > 24 else ''}"
    )
    print(f"  {BOLD}Hash:{RESET}      {result.attestation_hash[:24]}...")
    _sep()
    print()


def cmd_status(args: list[str]) -> None:
    result = get_status(DEFAULT_STATE_DIR)
    if not result.success:
        print(f"\n  {RED}Error:{RESET} {result.error}\n")
        return

    print(f"\n  {GOLD}SOVEREIGN STATUS{RESET}")
    _sep()
    print(f"  {BOLD}Name:{RESET}         {result.name}")
    print(f"  {BOLD}Node:{RESET}         {result.node_id[:20]}...")
    print(
        f"  {BOLD}Covenant:{RESET}     {result.covenant_hash[:20]}... {GREEN}✓{RESET}"
    )
    print(f"  {BOLD}Age:{RESET}          {result.age_days:.0f} days")
    print()
    print(
        f"  {BOLD}SEED:{RESET}         {GREEN}{fp_float(result.seed_balance):.6f}{RESET}"
    )
    print(
        f"  {BOLD}BLOOM:{RESET}        {PURPLE}{fp_float(result.bloom_balance):.6f}{RESET}"
    )
    print(f"  {BOLD}Actions:{RESET}      {result.total_actions}")
    if result.avg_ihsan > 0:
        print(f"  {BOLD}Avg Ihsan:{RESET}    {GOLD}{result.avg_ihsan:.4f}{RESET}")
    print()
    print(f"  {BOLD}Asabiyyah:{RESET}    {PURPLE}{result.asabiyyah_score:.4f}{RESET}")
    print(
        f"  {BOLD}Attested:{RESET}     {result.attestations_given} given / {result.attestations_received} received"
    )
    print(f"  {BOLD}Peers:{RESET}        {result.peers_count} known")
    _sep()
    print()


def cmd_ledger(args: list[str]) -> None:
    ledger_file = DEFAULT_STATE_DIR / "ledger.jsonl"
    if not ledger_file.exists():
        print(f"\n  {DIM}Ledger is empty.{RESET}\n")
        return

    events = [
        json.loads(line)
        for line in ledger_file.read_text().strip().split("\n")
        if line.strip()
    ]
    n = int(args[0]) if args else 10
    recent = events[-n:]

    print(
        f"\n  {GOLD}EVENT LEDGER{RESET} ({len(events)} total, showing last {len(recent)})"
    )
    _sep()
    for ev in recent:
        t = ev.get("type", "?")
        if t == "genesis":
            print(f"  {GOLD}◆{RESET} GENESIS — Node established")
        elif t == "action":
            ihsan = ev.get("ihsan_score", 0)
            minted = ev.get("seed_minted", 0)
            desc = ev.get("description", "")[:40]
            print(
                f"  {GREEN}●{RESET} ACTION — Ihsan:{ihsan:.4f} +{minted:.4f}S {DIM}{desc}{RESET}"
            )
        elif t == "attestation":
            peer = ev.get("attestee", "?")[:16]
            print(f"  {PURPLE}○{RESET} ATTEST — Vouched for {peer}...")
    _sep()
    print()


def cmd_reset(args: list[str]) -> None:
    import shutil

    if not DEFAULT_STATE_DIR.exists():
        print(f"\n  {DIM}Nothing to reset.{RESET}\n")
        return
    confirm = input(f"  {RED}Type 'I AM SOVEREIGN' to confirm:{RESET} ")
    if confirm.strip() != "I AM SOVEREIGN":
        print(f"  {DIM}Reset cancelled.{RESET}\n")
        return
    shutil.rmtree(DEFAULT_STATE_DIR)
    print(f"  {GREEN}✓{RESET} Node reset.\n")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

COMMANDS = {
    "init": cmd_init,
    "work": cmd_work,
    "attest": cmd_attest,
    "status": cmd_status,
    "ledger": cmd_ledger,
    "reset": cmd_reset,
}


def main() -> None:
    if len(sys.argv) < 2:
        print(f"""
{GOLD}  BIZRA Sovereignty CLI{RESET}
  {DIM}Phase 67 — Constitutional Kernel{RESET}

  {BOLD}python -m core.constitutional init{RESET}    [name]
  {BOLD}python -m core.constitutional work{RESET}    "description"
  {BOLD}python -m core.constitutional attest{RESET}  <peer_id>
  {BOLD}python -m core.constitutional status{RESET}

  {DIM}Additional:{RESET}
  {BOLD}python -m core.constitutional ledger{RESET}  [n]
  {BOLD}python -m core.constitutional reset{RESET}

  {DIM}15 algorithms · 7 invariants · zero compromise{RESET}
""")
        return

    cmd = sys.argv[1].lower()
    args = sys.argv[2:]

    if cmd in COMMANDS:
        COMMANDS[cmd](args)
    else:
        print(f"\n  {RED}Unknown command: {cmd}{RESET}")
        print(f"  {DIM}Available: {', '.join(COMMANDS.keys())}{RESET}\n")


if __name__ == "__main__":
    main()
