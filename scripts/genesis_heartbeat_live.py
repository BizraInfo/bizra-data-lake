"""
Genesis Engine v5 — Live LLM Heartbeat via Ollama phi3:mini.

Executes the full 7-agent PAT trust compiler with REAL inference,
proving the constitutional pipeline works end-to-end:

  Classify (HHMM) -> Execute (7 PAT agents + LLM) -> Gate (Ihsan 6D)
  -> SNR (normalize) -> Evidence (hash-chain) -> Precipitate (reflex)

Standing on Giants:
  Shannon (SNR monotonicity) . Al-Ghazali (Ihsan 6D gate)
  Lamport (evidence chain)   . Besta (GoT reasoning)

Usage:
  python scripts/genesis_heartbeat_live.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure bizra-constitution is importable
_ROOT = Path(__file__).resolve().parent.parent
_CONST_PKG = _ROOT / "bizra-constitution"
if _CONST_PKG.is_dir():
    sys.path.insert(0, str(_CONST_PKG))

import httpx
from mission_pipeline import MissionPipeline, MissionStatus


# ── Ollama LLM function ─────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "phi3:mini"


def ollama_llm(prompt: str) -> str:
    """Call Ollama phi3:mini for real inference."""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.7,
            "num_predict": 256,
        },
    }
    resp = httpx.post(OLLAMA_URL, json=payload, timeout=120.0)
    resp.raise_for_status()
    return resp.json()["response"].strip()


# ── Genesis Missions ─────────────────────────────────────────────────────────

GENESIS_MISSIONS = [
    "What is the principle of Ihsan in system design?",
    "Explain how SNR monotonicity guarantees quality improvement over time.",
    "How does a hash-chained evidence ledger provide auditability?",
    "Describe the role of the Reflex Cache in reducing latency.",
    "Why is Byzantine fault tolerance important for distributed AI systems?",
]


def main() -> None:
    print("=" * 72)
    print("  BIZRA Genesis Engine v5 — Live LLM Heartbeat")
    print(f"  Model: {OLLAMA_MODEL} via Ollama (localhost:11434)")
    print("=" * 72)
    print()

    # Verify Ollama is reachable
    try:
        test_resp = ollama_llm("Say 'ready' in one word.")
        print(f"  Ollama warmup: '{test_resp[:60]}...'")
    except Exception as e:
        print(f"  ERROR: Ollama not reachable: {e}")
        sys.exit(1)

    # Initialize pipeline with real LLM
    evidence_path = _ROOT / "04_GOLD" / "genesis_heartbeat_evidence.jsonl"
    cache_path = _ROOT / "04_GOLD" / "genesis_heartbeat_cache.json"

    pipeline = MissionPipeline(
        evidence_path=evidence_path,
        cache_path=cache_path,
        llm_fn=ollama_llm,
    )

    print(f"\n  Pipeline: {len(pipeline.agents)} PAT agents")
    print(f"  Gate minimum: {pipeline.ihsan_gate.gate_minimum}")
    print(f"  Evidence path: {evidence_path}")
    print()

    # Execute missions
    total_start = time.perf_counter()
    results = []

    for i, mission_text in enumerate(GENESIS_MISSIONS, 1):
        print(f"  [{i}/{len(GENESIS_MISSIONS)}] {mission_text[:55]}...")
        t0 = time.perf_counter()
        m = pipeline.execute(mission_text)
        elapsed = time.perf_counter() - t0

        results.append(m)

        status_icon = "PASS" if m.status == MissionStatus.COMPLETE else "FAIL"
        ihsan_str = f"{m.ihsan_score.composite:.3f}" if m.ihsan_score else "N/A"
        snr_str = f"{m.mission_snr.snr_normalized:.3f}" if m.mission_snr else "N/A"
        gate_str = "PASS" if (m.ihsan_score and m.ihsan_score.passes) else "FAIL"
        reflex_str = "HIT" if m.reflex_hit else "MISS"

        print(f"         Status: {status_icon} | Ihsan: {ihsan_str} | "
              f"Gate: {gate_str} | SNR: {snr_str} | "
              f"Reflex: {reflex_str} | {elapsed:.1f}s")

        # Show first 120 chars of LLM output
        output_preview = m.output_text[:120].replace("\n", " ")
        print(f"         Output: {output_preview}...")
        print()

    total_elapsed = time.perf_counter() - total_start

    # Evidence chain verification
    valid, count, errors = pipeline.evidence_ledger.verify_chain()

    # Health report
    health = pipeline.health()

    # Summary
    print("=" * 72)
    print("  HEARTBEAT SUMMARY")
    print("=" * 72)

    completed = sum(1 for m in results if m.status == MissionStatus.COMPLETE)
    gate_passes = sum(1 for m in results if m.ihsan_score and m.ihsan_score.passes)
    avg_ihsan = sum(m.ihsan_score.composite for m in results if m.ihsan_score) / len(results)
    avg_snr = sum(m.mission_snr.snr_normalized for m in results if m.mission_snr) / len(results)

    print(f"  Missions:     {completed}/{len(results)} COMPLETE")
    print(f"  Gate passes:  {gate_passes}/{len(results)}")
    print(f"  Avg Ihsan:    {avg_ihsan:.4f}")
    print(f"  Avg SNR:      {avg_snr:.4f}")
    print(f"  Evidence:     {count} receipts, chain {'VALID' if valid else 'BROKEN'}")
    print(f"  Total time:   {total_elapsed:.1f}s")
    print(f"  Avg/mission:  {total_elapsed/len(results):.1f}s")
    print(f"  Constitution: {health.get('constitution_version', 'N/A')}")
    print()

    if errors:
        print("  EVIDENCE ERRORS:")
        for err in errors:
            print(f"    - {err}")
        print()

    # Write summary artifact
    summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": OLLAMA_MODEL,
        "missions_completed": completed,
        "missions_total": len(results),
        "gate_passes": gate_passes,
        "avg_ihsan": round(avg_ihsan, 4),
        "avg_snr": round(avg_snr, 4),
        "evidence_chain_valid": valid,
        "evidence_count": count,
        "total_seconds": round(total_elapsed, 2),
        "constitution_version": health.get("constitution_version", "unknown"),
    }

    summary_path = _ROOT / "04_GOLD" / "genesis_heartbeat_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  Summary written: {summary_path}")

    # Shutdown
    pipeline.shutdown()
    print("  Pipeline shutdown complete.")
    print()
    print("  Standing on Giants: Shannon . Al-Ghazali . Lamport . Besta")
    print("=" * 72)


if __name__ == "__main__":
    main()
