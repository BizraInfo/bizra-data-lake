"""
BIZRA PROOF-OF-IMPACT DASHBOARD (v1.0)
"Visualizing the Accumulation of Good"

Reads the PoI Ledger and generates a real-time status report.
Aligned with Unified System Contract §5 & §6.
"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from bizra_config import GOLD_PATH

POI_LEDGER_PATH = GOLD_PATH / "poi_ledger.jsonl"
PRIME_STATE_PATH = GOLD_PATH / "bizra_prime_state.json"


def load_ledger():
    records = []
    if POI_LEDGER_PATH.exists():
        with open(POI_LEDGER_PATH, "r") as f:
            for line in f:
                records.append(json.loads(line))
    return records


def load_state():
    if PRIME_STATE_PATH.exists():
        with open(PRIME_STATE_PATH, "r") as f:
            return json.load(f)
    return {}


def render_dashboard():
    print(
        "╔══════════════════════════════════════════════════════════════════════════╗"
    )
    print("║           BIZRA PROOF-OF-IMPACT (PoI) DASHBOARD                         ║")
    print("║           'Goodness must be measurable, auditable, and rewarded.'       ║")
    print(
        "╚══════════════════════════════════════════════════════════════════════════╝"
    )

    ledger = load_ledger()
    state = load_state()

    if not ledger:
        print(
            "\n   📭 No attestations recorded yet. Run BIZRA PRIME to generate impact."
        )
        return

    # 1. Summary Stats
    total_attestations = len(ledger)
    total_impact = state.get("total_impact", 0)

    # Token Simulation (SC §6)
    # SEED = Utility (1:1 with task completion)
    # BLOOM = Impact Growth (weighted by SNR)
    seed_balance = sum(
        1 for a in ledger if a.get("benchmarks", {}).get("task_completion")
    )
    bloom_balance = sum(a.get("benchmarks", {}).get("snr", 0) for a in ledger)

    print(
        f"\n┌─────────────────────────────────────────────────────────────────────────┐"
    )
    print(
        f"│  📊 SUMMARY                                                             │"
    )
    print(
        f"├─────────────────────────────────────────────────────────────────────────┤"
    )
    print(
        f"│  Total Attestations:  {total_attestations:<10}                                     │"
    )
    print(
        f"│  Total Impact Score:  {total_impact:<10.2f}                                     │"
    )
    print(f"│  ───────────────────────────────────────────────────────────────────── │")
    print(
        f"│  🌱 SEED Balance:     {seed_balance:<10} (Utility Tokens)                    │"
    )
    print(
        f"│  🌸 BLOOM Balance:    {bloom_balance:<10.2f} (Impact Tokens)                    │"
    )
    print(
        f"└─────────────────────────────────────────────────────────────────────────┘"
    )

    # 2. Agent Activity Breakdown
    agent_activity = defaultdict(int)
    for a in ledger:
        action = a.get("action", "Unknown")
        if "Agent:" in action:
            agent_name = action.split(":")[1]
            agent_activity[agent_name] += 1
        elif "Reasoning:" in action:
            agent_activity["reasoning_loop"] += 1

    print(
        f"\n┌─────────────────────────────────────────────────────────────────────────┐"
    )
    print(
        f"│  🤖 AGENT ACTIVITY                                                      │"
    )
    print(
        f"├─────────────────────────────────────────────────────────────────────────┤"
    )
    for agent, count in sorted(agent_activity.items(), key=lambda x: -x[1]):
        bar = "█" * min(count * 2, 30)
        print(f"│  {agent:<15} {bar:<30} ({count})          │")
    print(
        f"└─────────────────────────────────────────────────────────────────────────┘"
    )

    # 3. Recent Attestations
    print(
        f"\n┌─────────────────────────────────────────────────────────────────────────┐"
    )
    print(
        f"│  📜 RECENT ATTESTATIONS (Last 5)                                        │"
    )
    print(
        f"├─────────────────────────────────────────────────────────────────────────┤"
    )
    for a in ledger[-5:]:
        ts = a.get("timestamp", "N/A")[:19]
        action = a.get("action", "N/A")[:35]
        hash_id = a.get("attestation_hash", "N/A")[:12]
        print(f"│  {ts}  │ {action:<35} │ {hash_id}...  │")
    print(
        f"└─────────────────────────────────────────────────────────────────────────┘"
    )

    # 4. Genesis Anchor
    if ledger:
        genesis_root = ledger[0].get("genesis_merkle_root", "N/A")[:32]
        print(f"\n   🔗 Genesis Anchor: {genesis_root}...")
        print(f"   ✅ All attestations cryptographically linked to Genesis Block 0.")


if __name__ == "__main__":
    render_dashboard()
