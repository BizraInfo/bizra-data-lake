"""
Phase 62 D5: Wire Adapter Live Integration Test
════════════════════════════════════════════════

Executes the GenesisWire adapter with REAL Ollama phi3:mini inference
to prove the complete v6 production stack works end-to-end.

Pipeline: User Input -> GenesisWire -> ProductionPipeline
  -> Identity (Ed25519 keypair, HD agent keys)
  -> HHMM Router (complexity classification)
  -> Reflex Cache (O(1) lookup)
  -> Ollama Provider (phi3:mini inference, circuit breaker)
  -> PAT Pipeline (7 agents)
  -> Ihsan Gate (6-dim tensor evaluation)
  -> SNR Measurement (canonical normalization)
  -> Evidence Receipt (hash-chained, Integrator-signed)
  -> WireResult.to_event_bus_payload() (Rust bus format)
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Ensure bizra-constitution/ is importable
ROOT = Path(__file__).resolve().parent.parent
CONSTITUTION = ROOT / "bizra-constitution"
sys.path.insert(0, str(CONSTITUTION))

from node0_wire import GenesisWire, WireResult, wire_genesis_engine


def main():
    print("Wire Live Integration Test")
    print("=" * 60)

    # ── 1. Verify Ollama reachability ──
    import urllib.request

    try:
        resp = urllib.request.urlopen("http://localhost:11434/api/tags", timeout=5)
        data = json.loads(resp.read())
        models = [m["name"] for m in data.get("models", [])]
        phi3_available = any("phi3" in m for m in models)
        assert phi3_available, f"phi3:mini not found in {models}"
        print(f"  Ollama: phi3:mini available ({len(models)} models)")
    except Exception as e:
        print(f"  FATAL: Ollama not reachable — {e}")
        sys.exit(1)

    # ── 2. Create wire adapter with real Ollama ──
    data_dir = ROOT / "04_GOLD" / "wire_live_test"
    data_dir.mkdir(parents=True, exist_ok=True)

    wire = wire_genesis_engine(
        data_dir=data_dir,
        ollama_url="http://localhost:11434",
        model_chain=["phi3:mini"],
    )
    assert wire is not None, "wire_genesis_engine returned None"
    assert wire._initialized, "Wire not initialized"
    assert wire._pipeline is not None, "Pipeline is None"

    node_id = wire._pipeline.identity.node_id
    total_agents = wire._pipeline.identity.total_agents
    print(f"  Identity: {node_id[:16]}... ({total_agents} agents)")
    assert total_agents == 12, f"Expected 12 agents, got {total_agents}"

    # ── 3. Execute 3 missions through full stack ──
    missions = [
        "What is the principle of Ihsan in AI ethics?",
        "Explain hash-chained evidence for audit trails.",
        "How does a circuit breaker improve system resilience?",
    ]

    results: list[WireResult] = []
    print()

    # Sprint A: baseline chain count for delta assertions
    _baseline_health = wire.health() if wire else {}
    _baseline_chain_count = (_baseline_health.get("pipeline_health", {}) or {}).get(
        "evidence_chain_count", 0
    )

    for i, text in enumerate(missions, 1):
        t0 = time.time()
        result = wire.execute(text)
        elapsed = time.time() - t0

        assert result is not None, f"Mission {i} returned None"
        assert result.success, f"Mission {i} failed: {result}"
        assert (
            result.ihsan_composite >= 0.0
        ), f"Ihsan negative: {result.ihsan_composite}"
        assert result.snr_normalized > 0, f"SNR zero: {result.snr_normalized}"
        assert result.signed, f"Mission {i} not signed"
        assert len(result.node_id) == 64, f"Bad node_id length: {len(result.node_id)}"
        assert result.evidence_receipt_id is not None, "No evidence receipt"
        assert (
            len(result.output) > 50
        ), f"Output too short ({len(result.output)} chars) — may be template"

        label = text[:35] + "..."
        print(f"  [{i}/{len(missions)}] {label}")
        print(
            f"    Ihsan: {result.ihsan_composite:.3f} | "
            f"SNR: {result.snr_normalized:.3f} | "
            f"Signed: {'YES' if result.signed else 'NO'} | "
            f"~{elapsed:.1f}s"
        )
        results.append(result)

    # ── 4. Verify event bus payload format ──
    print()
    print("  Payload format verification:")
    for i, result in enumerate(results, 1):
        payload = result.to_event_bus_payload()
        assert payload["type"] == "mission_complete", f"Bad type: {payload['type']}"
        assert "ihsan" in payload, "Missing ihsan"
        assert "composite" in payload["ihsan"], "Missing ihsan.composite"
        assert "dimensions" in payload["ihsan"], "Missing ihsan.dimensions"
        assert "snr" in payload, "Missing snr"
        assert "evidence" in payload, "Missing evidence"
        assert payload["evidence"]["signed"] is True, "Evidence not signed"
        assert "agent_trace" in payload, "Missing agent_trace"
        print(f"    [{i}] event bus payload: OK")

    # ── 5. Verify evidence chain integrity ──
    health = wire.health()
    pipeline_health = health["pipeline_health"]
    chain_valid = pipeline_health["evidence_chain_valid"]
    chain_count = pipeline_health["evidence_chain_count"]
    print()
    print(
        f"  Evidence chain: {chain_count} receipts, "
        f"{'VALID' if chain_valid else 'INVALID'}"
    )
    assert chain_valid, "Evidence chain integrity FAILED"
    # Sprint A: delta-based assertion (replaces absolute count)
    _delta_chain = chain_count - _baseline_chain_count
    assert _delta_chain == len(
        missions
    ), f"Expected +{len(missions)}, got +{_delta_chain}"

    # ── 6. Verify wire metrics ──
    assert health["total_missions"] == 3, f"total_missions: {health['total_missions']}"
    assert (
        health["genesis_missions"] == 3
    ), f"genesis_missions: {health['genesis_missions']}"
    assert (
        health["fallback_missions"] == 0
    ), f"fallback_missions: {health['fallback_missions']}"
    assert health["genesis_rate"] == 1.0, f"genesis_rate: {health['genesis_rate']}"
    assert health["avg_latency_ms"] > 0, f"avg_latency: {health['avg_latency_ms']}"
    print(
        f"  Wire metrics: {health['total_missions']}/{health['total_missions']} genesis "
        f"(100%), avg {health['avg_latency_ms']:.0f}ms"
    )

    # ── 7. Verify identity ──
    assert len(node_id) == 64, f"Bad node_id: {node_id}"
    integrator = wire._pipeline.identity.get_agent("Integrator")
    assert integrator is not None, "Integrator agent not found"
    assert integrator.agent_type == "pat", f"Bad agent type: {integrator.agent_type}"

    # ── 8. Write summary artifact ──
    avg_ihsan = sum(r.ihsan_composite for r in results) / len(results)
    avg_snr = sum(r.snr_normalized for r in results) / len(results)

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test": "wire_live_integration",
        "phase": "62_D5",
        "model": "phi3:mini",
        "missions": 3,
        "all_signed": True,
        "evidence_chain_valid": True,
        "avg_ihsan": round(avg_ihsan, 4),
        "avg_snr": round(avg_snr, 4),
        "avg_latency_ms": round(health["avg_latency_ms"], 1),
        "node_id": node_id[:16] + "...",
    }

    summary_path = ROOT / "04_GOLD" / "wire_live_test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary -> {summary_path.relative_to(ROOT)}")

    wire.shutdown()
    print("\nWire live integration: COMPLETE")


if __name__ == "__main__":
    main()
