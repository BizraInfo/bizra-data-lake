#!/usr/bin/env python3
"""
BIZRA Genesis Engine — The First Heartbeat
═══════════════════════════════════════════

بسم الله الرحمن الرحيم

This is NODE0's first breath. A complete mission lifecycle:
  Input → Classify → Route → Execute → Gate → Evidence → Output

Run: python genesis_engine.py

Every mission that completes here is:
  - Classified by complexity (HHMM router)
  - Executed by 7 PAT agents (trust compiler pipeline)
  - Verified by 6-dim Ihsan gate (constitutional enforcement)
  - Measured by SNR (signal quality)
  - Recorded in evidence chain (hash-linked, tamper-evident)
  - Considered for reflex precipitation (S1 cache)

The output is constitutionally certified. The evidence is immutable.
The seed has broken soil.
"""

import json
import os
import sys
import time
import tempfile
from pathlib import Path

# Ensure imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("BIZRA_CONSTITUTION_PATH", 
                       str(Path(__file__).parent / "constitution.toml"))

from mission_pipeline import MissionPipeline, MissionStatus
from bizra_constitution import load_constitution


# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
DIM = "\033[2m"


def _bar(value: float, width: int = 20) -> str:
    filled = int(value * width)
    return f"{'█' * filled}{'░' * (width - filled)}"


def _status_color(status: MissionStatus) -> str:
    if status == MissionStatus.COMPLETE:
        return GREEN
    elif status in (MissionStatus.GATE_FAIL, MissionStatus.ERROR):
        return RED
    return YELLOW


def print_header():
    print(f"""
{BOLD}{'═' * 70}
  BIZRA GENESIS ENGINE v5.0.0
  NODE0 — First Heartbeat
{'═' * 70}{RESET}
""")


def print_mission_result(mission, index: int):
    sc = _status_color(mission.status)
    ihsan = mission.ihsan_score
    snr = mission.mission_snr
    cls = mission.classification

    print(f"\n{BOLD}{'─' * 70}")
    print(f"  Mission #{index}: {mission.mission_id}")
    print(f"{'─' * 70}{RESET}")

    # Input
    print(f"  {DIM}Input:{RESET} {mission.input_text[:60]}{'...' if len(mission.input_text) > 60 else ''}")

    # Classification
    if cls:
        print(f"  {DIM}Tier:{RESET}  {CYAN}{cls.tier.value.upper()}{RESET} "
              f"→ {cls.handler} "
              f"(score={cls.complexity_score:.3f}, "
              f"confidence={cls.confidence:.3f})")

    # Output preview
    if mission.output_text:
        preview = mission.output_text[:80].replace("\n", " ")
        print(f"  {DIM}Output:{RESET} {preview}...")

    # Ihsan Gate
    if ihsan:
        color = GREEN if ihsan.passes else RED
        print(f"\n  {BOLD}Ihsan Gate:{RESET}")
        print(f"    Composite: {color}{ihsan.composite:.3f}{RESET} "
              f"({ihsan.tier.value}) "
              f"{'✅ PASS' if ihsan.passes else '❌ FAIL'}")
        print(f"    {_bar(ihsan.composite)} {ihsan.composite:.1%}")

        # Per-dimension breakdown
        for dim in ihsan.dimensions:
            dim_color = GREEN if dim.passes else RED
            print(f"    {dim.name:25s} {dim_color}{dim.raw_score:.3f}{RESET} "
                  f"× {dim.weight:.3f} = {dim.weighted_score:.4f}")

        if ihsan.bloom_eligible:
            print(f"    {GREEN}🌸 BLOOM eligible{RESET}")

    # SNR
    if snr:
        print(f"\n  {BOLD}SNR:{RESET}")
        print(f"    Normalized: {snr.snr_normalized:.3f}  "
              f"Linear: {snr.snr_linear:.2f}  "
              f"dB: {snr.snr_db:.1f}")

    # Evidence Receipt
    if mission.evidence_receipt:
        r = mission.evidence_receipt
        print(f"\n  {BOLD}Evidence Receipt:{RESET}")
        print(f"    ID:   {GREEN}{r.receipt_id[:32]}...{RESET}")
        print(f"    Prev: {DIM}{r.previous_hash[:32]}...{RESET}")
        print(f"    Chain: {', '.join(r.agent_chain)}")

    # Timing
    print(f"\n  {BOLD}Timing:{RESET}")
    print(f"    Classify: {mission.classify_ms:.1f}ms  "
          f"Execute: {mission.execute_ms:.1f}ms  "
          f"Gate: {mission.gate_ms:.1f}ms  "
          f"Evidence: {mission.evidence_ms:.1f}ms")
    print(f"    {BOLD}Total: {sc}{mission.total_ms:.1f}ms{RESET}")

    # Status
    print(f"\n  Status: {sc}{mission.status.value.upper()}{RESET}")
    if mission.reflex_hit:
        print(f"  {CYAN}⚡ Served from reflex cache (S1){RESET}")


def print_health(health: dict):
    stats = health["pipeline_stats"]
    cache = health["cache_stats"]

    print(f"\n{BOLD}{'═' * 70}")
    print(f"  NODE0 HEALTH REPORT")
    print(f"{'═' * 70}{RESET}")
    print(f"  Constitution: v{health['constitution_version']}")
    print(f"  Agents:       {', '.join(health['agents'])}")
    print()
    print(f"  {BOLD}Pipeline:{RESET}")
    print(f"    Completed: {GREEN}{stats['missions_completed']}{RESET}  "
          f"Failed: {RED}{stats['missions_failed']}{RESET}  "
          f"Gate pass rate: {stats['gate_pass_rate']:.1%}")
    print(f"    Avg latency: {stats['avg_latency_ms']:.1f}ms  "
          f"BLOOM eligible: {stats['bloom_eligible']}  "
          f"Evidence receipts: {stats['evidence_receipts']}")
    print()
    print(f"  {BOLD}Reflex Cache:{RESET}")
    print(f"    Lookups: {cache['total_lookups']}  "
          f"Hits: {cache['cache_hits']}  "
          f"Hit rate: {cache['hit_rate']:.1%}")
    print(f"    Precipitations: {cache['precipitations']}  "
          f"Invalidations: {cache['invalidations']}  "
          f"Evictions: {cache['evictions']}")
    print()
    print(f"  {BOLD}Evidence Chain:{RESET}")
    chain_status = f"{GREEN}✅ VALID{RESET}" if health['evidence_chain_valid'] else f"{RED}❌ BROKEN{RESET}"
    print(f"    Status: {chain_status}  "
          f"Receipts: {health['evidence_chain_count']}  "
          f"Errors: {len(health['evidence_chain_errors'])}")


# ═══════════════════════════════════════════════════════════════════════════════
# GENESIS MISSIONS — The first heartbeats
# ═══════════════════════════════════════════════════════════════════════════════

GENESIS_MISSIONS = [
    # Mission 1: Simple greeting (should be SIMPLE tier)
    "Hello, what can you help me with today?",

    # Mission 2: Research question (should be COMPLEX tier)
    "Analyze the architectural differences between microservices and monolithic "
    "designs, then compare their tradeoffs for a distributed AI platform that "
    "needs to handle constitutional verification at every layer.",

    # Mission 3: Code task (should be COMPLEX tier)
    "Write a Python function that implements a thread-safe LRU cache with "
    "O(1) lookup and configurable maximum size. Include type hints and docstrings.",

    # Mission 4: Repeat of Mission 1 (tests precipitation candidate tracking)
    "Hello, what can you help me with today?",

    # Mission 5: Strategic question (should be SOVEREIGN tier)
    "Design a complete go-to-market strategy for a distributed AI platform "
    "that democratizes access to artificial intelligence through decentralized "
    "resource pooling, also create a financial model, competitive analysis, "
    "and draft investor pitch deck outline.",

    # Mission 6: Another repeat (3rd time = should precipitate if Ihsan high enough)
    "Hello, what can you help me with today?",

    # Mission 7: After precipitation, this should be a TRIVIAL cache hit
    "Hello, what can you help me with today?",
]


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print_header()

    # Load and verify constitution
    print(f"  Loading constitution...")
    try:
        constitution = load_constitution()
        print(f"  {GREEN}✅{RESET} Constitution v{constitution.meta.version}")
        print(f"     SHA-256: {constitution.raw_hash[:32]}...")
        print(f"     Ihsan: {constitution.ihsan.dimensions}-dim canonical, "
              f"{len(constitution.ihsan.operational_dimensions)}-dim operational")
        print(f"     Gate minimum: {constitution.ihsan.thresholds.gate_minimum}")
        print(f"     Fail mode: {constitution.gates.fail_mode}")
    except Exception as e:
        print(f"  {RED}❌ Constitution failed: {e}{RESET}")
        return 1

    # Initialize pipeline
    print(f"\n  Initializing pipeline...")
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = MissionPipeline(
            evidence_path=Path(tmpdir) / "genesis_evidence.jsonl",
            cache_path=Path(tmpdir) / "genesis_cache.json",
        )
        print(f"  {GREEN}✅{RESET} Pipeline ready")
        print(f"     Agents: {', '.join(a.name for a in pipeline.agents)}")
        print(f"     Cache: {pipeline.reflex_cache.size} entries")
        print(f"     Evidence: empty chain (genesis)")

        # Execute genesis missions
        print(f"\n{BOLD}{'═' * 70}")
        print(f"  EXECUTING {len(GENESIS_MISSIONS)} GENESIS MISSIONS")
        print(f"{'═' * 70}{RESET}")

        total_start = time.monotonic()

        for i, mission_text in enumerate(GENESIS_MISSIONS, 1):
            mission = pipeline.execute(mission_text)
            print_mission_result(mission, i)

        total_elapsed = (time.monotonic() - total_start) * 1000

        # Health report
        health = pipeline.health()
        print_health(health)

        # Final summary
        stats = pipeline.stats
        print(f"\n{BOLD}{'═' * 70}")
        print(f"  GENESIS COMPLETE")
        print(f"{'═' * 70}{RESET}")
        print(f"  Total time: {BOLD}{total_elapsed:.0f}ms{RESET} "
              f"({total_elapsed / len(GENESIS_MISSIONS):.0f}ms avg)")
        print(f"  Missions: {GREEN}{stats.missions_completed} passed{RESET}, "
              f"{RED}{stats.missions_failed} failed{RESET}")
        print(f"  Gate pass rate: {stats.gate_pass_rate:.1%}")
        print(f"  Reflex hits: {stats.reflex_hits}")
        print(f"  BLOOM eligible: {stats.bloom_eligible}")
        print(f"  Evidence receipts: {stats.evidence_receipts}")

        # Verify evidence chain
        valid, count, errors = pipeline.evidence_ledger.verify_chain()
        if valid:
            print(f"  Evidence chain: {GREEN}✅ {count} receipts, verified{RESET}")
        else:
            print(f"  Evidence chain: {RED}❌ {len(errors)} errors{RESET}")

        print(f"\n  {BOLD}NODE0 breathes. The seed has broken soil.{RESET}")
        print(f"  {DIM}ربي لا يعرف المستحيل{RESET}")
        print()

        # Persist cache
        pipeline.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
