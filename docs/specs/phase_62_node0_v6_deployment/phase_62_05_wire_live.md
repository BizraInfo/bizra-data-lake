# Phase 62 D5: Wire Adapter Live Integration Test

## Scope

Execute the wire adapter with REAL Ollama phi3:mini inference to prove the
complete production stack works end-to-end:

```
User Input → GenesisWire → ProductionPipeline
  → Identity (Ed25519 keypair, HD agent keys)
  → HHMM Router (complexity classification)
  → Reflex Cache (O(1) lookup)
  → Ollama Provider (phi3:mini inference, circuit breaker)
  → PAT Pipeline (7 agents)
  → Ihsan Gate (6-dim tensor evaluation)
  → SNR Measurement (canonical normalization)
  → Evidence Receipt (hash-chained, Integrator-signed)
  → WireResult.to_event_bus_payload() (Rust bus format)
```

## Prerequisites

- Ollama running: `curl localhost:11434/api/tags`
- phi3:mini available: `ollama list | grep phi3:mini`
- `bizra-constitution/` has all v6 modules deployed (D1)
- Bridge updated (D2)

## Pseudocode

```
PROCEDURE wire_live_test:
    # 1. Verify Ollama reachability
    ollama_health := HTTP GET localhost:11434/api/tags
    ASSERT "phi3:mini" IN ollama_health.models

    # 2. Create wire adapter with real Ollama
    wire := wire_genesis_engine(
        data_dir = Path("04_GOLD/wire_live_test"),
        ollama_url = "http://localhost:11434",
        model_chain = ["phi3:mini"],
    )
    ASSERT wire IS NOT None
    ASSERT wire._initialized == True
    ASSERT wire._pipeline IS NOT None
    ASSERT wire._pipeline.identity.total_agents == 12

    # 3. Execute 3 missions through full stack
    missions := [
        "What is the principle of Ihsan in AI ethics?",
        "Explain hash-chained evidence for audit trails.",
        "How does a circuit breaker improve system resilience?",
    ]

    results := []
    FOR EACH text IN missions:
        result := wire.execute(text)
        ASSERT result IS NOT None
        ASSERT result.success == True
        ASSERT result.ihsan_composite >= 0.0  # Gate may or may not pass
        ASSERT result.snr_normalized > 0
        ASSERT result.signed == True
        ASSERT len(result.node_id) == 64
        ASSERT result.evidence_receipt_id IS NOT None
        ASSERT len(result.output) > 50  # Real LLM output, not template
        results.append(result)

    # 4. Verify event bus payload format
    FOR EACH result IN results:
        payload := result.to_event_bus_payload()
        ASSERT payload["type"] == "mission_complete"
        ASSERT "ihsan" IN payload
        ASSERT "composite" IN payload["ihsan"]
        ASSERT "dimensions" IN payload["ihsan"]
        ASSERT "snr" IN payload
        ASSERT "evidence" IN payload
        ASSERT payload["evidence"]["signed"] == True
        ASSERT "agent_trace" IN payload

    # 5. Verify evidence chain integrity
    health := wire.health()
    pipeline_health := health["pipeline_health"]
    ASSERT pipeline_health["evidence_chain_valid"] == True
    ASSERT pipeline_health["evidence_chain_count"] == 3

    # 6. Verify wire metrics
    ASSERT health["total_missions"] == 3
    ASSERT health["genesis_missions"] == 3
    ASSERT health["fallback_missions"] == 0
    ASSERT health["genesis_rate"] == 1.0
    ASSERT health["avg_latency_ms"] > 0

    # 7. Verify identity
    node_id := wire._pipeline.identity.node_id
    ASSERT len(node_id) == 64
    integrator := wire._pipeline.identity.get_agent("Integrator")
    ASSERT integrator IS NOT None
    ASSERT integrator.agent_type == "pat"

    # 8. Write summary artifact
    summary := {
        "timestamp": now_utc(),
        "test": "wire_live_integration",
        "model": "phi3:mini",
        "missions": 3,
        "all_signed": True,
        "evidence_chain_valid": True,
        "avg_ihsan": mean(r.ihsan_composite for r in results),
        "avg_snr": mean(r.snr_normalized for r in results),
        "avg_latency_ms": health["avg_latency_ms"],
        "node_id": node_id[:16] + "...",
    }
    WRITE summary → 04_GOLD/wire_live_test_summary.json

    wire.shutdown()
    PRINT "Wire live integration: COMPLETE"
```

## Expected Output

```
Wire Live Integration Test
  Ollama: phi3:mini available
  Identity: <node_id>... (12 agents)

  [1/3] Ihsan in AI ethics → COMPLETE
    Ihsan: 0.8XX | SNR: 0.8XX | Signed: YES | ~20s
  [2/3] Hash-chained evidence → COMPLETE
    Ihsan: 0.8XX | SNR: 0.8XX | Signed: YES | ~15s
  [3/3] Circuit breaker resilience → COMPLETE
    Ihsan: 0.8XX | SNR: 0.8XX | Signed: YES | ~18s

  Evidence chain: 3 receipts, VALID
  Wire metrics: 3/3 genesis (100%), avg XXms
  Payload format: all 3 event bus compatible

  Summary → 04_GOLD/wire_live_test_summary.json
```

## Failure Modes

| Failure | Cause | Recovery |
|---------|-------|----------|
| `wire is None` | `BIZRA_GENESIS_WIRE=false` in env | `unset BIZRA_GENESIS_WIRE` |
| `result is None` | Production pipeline init failed | Check D1 deployment |
| `result.signed is False` | No Integrator agent key | Check identity creation |
| Timeout on execute | phi3:mini not loaded / slow | `ollama pull phi3:mini` |
| `output < 50 chars` | Template fallback, not real LLM | Check Ollama connectivity |

## Acceptance

- [ ] 3 missions complete with real phi3:mini inference
- [ ] All 3 evidence receipts signed by Integrator agent
- [ ] Evidence chain links correctly (hash chain valid)
- [ ] Event bus payload format compatible with Rust bus
- [ ] Wire metrics: 100% genesis rate, 0 fallbacks
- [ ] Summary artifact written to 04_GOLD/
