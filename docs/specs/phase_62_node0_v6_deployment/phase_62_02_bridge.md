# Phase 62 D2: Update Constitutional Engine Bridge

## Scope

Extend `core/bridges/constitutional_engine.py` to import the 6 new v6
components alongside the existing 7 v5 components. Maintain the same
try/except graceful fallback pattern.

## Current Bridge State (v5)

```python
# 7 component groups, each with HAS_* flag:
HAS_CONSTITUTIONAL_GATE     # ihsan_gate → ConstitutionalIhsanGate
HAS_MISSION_PIPELINE         # mission_pipeline → MissionPipeline, Mission, ...
HAS_HHMM_ROUTER             # hhmm_router → HhmmRouter, ComplexityTier, ...
HAS_REFLEX_CACHE             # reflex_cache → ReflexCache, ReflexEntry, ...
HAS_EVIDENCE_RECEIPT         # evidence_receipt → EvidenceReceipt, Ledger
HAS_SNR                      # snr → normalize_snr, compute_sape_composite, ...
HAS_CONSTITUTION             # bizra_constitution → load_constitution, Constitution

GENESIS_ENGINE_AVAILABLE = all([...7 flags...])
```

## Target Bridge State (v6)

Add 4 new component groups (identity, ollama, production, wire):

```python
# NEW: Identity Genesis
HAS_IDENTITY_GENESIS         # identity_genesis → create_identity, NodeIdentity, AgentKey
# NEW: Ollama Provider
HAS_OLLAMA_PROVIDER          # ollama_provider → OllamaProvider, InferenceResult, CircuitBreaker
# NEW: Production Pipeline
HAS_PRODUCTION_PIPELINE      # production_pipeline → ProductionPipeline, create_node0
# NEW: Wire Adapter
HAS_GENESIS_WIRE             # node0_wire → GenesisWire, WireResult, wire_genesis_engine

# Updated master flag
GENESIS_ENGINE_AVAILABLE = all([...7 v5 flags...])  # Unchanged
NODE0_PRODUCTION_AVAILABLE = all([
    GENESIS_ENGINE_AVAILABLE,
    HAS_IDENTITY_GENESIS,
    HAS_OLLAMA_PROVIDER,
    HAS_PRODUCTION_PIPELINE,
    HAS_GENESIS_WIRE,
])
```

## Pseudocode

```
# In core/bridges/constitutional_engine.py, APPEND after existing sections:

# ═══ IDENTITY GENESIS (Ed25519 + HD keys) ═══
TRY:
    FROM identity_genesis IMPORT create_identity, NodeIdentity, AgentKey
    FROM identity_genesis IMPORT save_identity, load_public_record
    HAS_IDENTITY_GENESIS := True
EXCEPT ImportError:
    create_identity := None
    NodeIdentity := None
    AgentKey := None
    save_identity := None
    load_public_record := None
    HAS_IDENTITY_GENESIS := False

# ═══ OLLAMA PROVIDER (circuit breaker) ═══
TRY:
    FROM ollama_provider IMPORT OllamaProvider, InferenceResult
    FROM ollama_provider IMPORT CircuitBreaker, CircuitState, ModelMetrics
    HAS_OLLAMA_PROVIDER := True
EXCEPT ImportError:
    OllamaProvider := None
    InferenceResult := None
    CircuitBreaker := None
    CircuitState := None
    ModelMetrics := None
    HAS_OLLAMA_PROVIDER := False

# ═══ PRODUCTION PIPELINE (signed evidence) ═══
TRY:
    FROM production_pipeline IMPORT ProductionPipeline, create_node0
    HAS_PRODUCTION_PIPELINE := True
EXCEPT ImportError:
    ProductionPipeline := None
    create_node0 := None
    HAS_PRODUCTION_PIPELINE := False

# ═══ WIRE ADAPTER (MissionOrchestrator bridge) ═══
TRY:
    FROM node0_wire IMPORT GenesisWire, WireResult, wire_genesis_engine
    HAS_GENESIS_WIRE := True
EXCEPT ImportError:
    GenesisWire := None
    WireResult := None
    wire_genesis_engine := None
    HAS_GENESIS_WIRE := False

# ═══ UPDATED AVAILABILITY ═══
NODE0_PRODUCTION_AVAILABLE: bool = all([
    GENESIS_ENGINE_AVAILABLE,
    HAS_IDENTITY_GENESIS,
    HAS_OLLAMA_PROVIDER,
    HAS_PRODUCTION_PIPELINE,
    HAS_GENESIS_WIRE,
])

# Update availability_report() to include v6 components
def availability_report() -> dict:
    return {
        "genesis_engine_available": GENESIS_ENGINE_AVAILABLE,
        "node0_production_available": NODE0_PRODUCTION_AVAILABLE,
        "components": {
            # ... existing 7 ...
            "identity_genesis": HAS_IDENTITY_GENESIS,
            "ollama_provider": HAS_OLLAMA_PROVIDER,
            "production_pipeline": HAS_PRODUCTION_PIPELINE,
            "genesis_wire": HAS_GENESIS_WIRE,
        },
        "package_path": str(_CONSTITUTION_PKG),
    }
```

## Design Decisions

1. **Two-tier availability**: `GENESIS_ENGINE_AVAILABLE` (library mode, no
   external deps) and `NODE0_PRODUCTION_AVAILABLE` (server mode, needs
   identity + ollama). This preserves the v5 contract.

2. **No aliasing needed**: Unlike v5 where IhsanGate/ComplexityTier/
   EvidenceLedger needed aliasing to avoid core/ collisions, the v6 modules
   have unique names (NodeIdentity, OllamaProvider, GenesisWire).

3. **node0_server.py NOT imported in bridge**: The server module imports
   FastAPI at module level. Importing it in the bridge would make FastAPI a
   hard dependency of core/. Instead, the server is run directly:
   `python bizra-constitution/node0_server.py`

## TDD Anchors

```python
def test_bridge_imports_identity():
    from core.bridges.constitutional_engine import HAS_IDENTITY_GENESIS
    assert HAS_IDENTITY_GENESIS is True

def test_bridge_imports_ollama():
    from core.bridges.constitutional_engine import HAS_OLLAMA_PROVIDER
    assert HAS_OLLAMA_PROVIDER is True

def test_bridge_imports_production():
    from core.bridges.constitutional_engine import HAS_PRODUCTION_PIPELINE
    assert HAS_PRODUCTION_PIPELINE is True

def test_bridge_imports_wire():
    from core.bridges.constitutional_engine import HAS_GENESIS_WIRE
    assert HAS_GENESIS_WIRE is True

def test_node0_production_available():
    from core.bridges.constitutional_engine import NODE0_PRODUCTION_AVAILABLE
    assert NODE0_PRODUCTION_AVAILABLE is True

def test_availability_report_has_v6_components():
    from core.bridges.constitutional_engine import availability_report
    report = availability_report()
    assert "node0_production_available" in report
    assert "identity_genesis" in report["components"]
    assert "ollama_provider" in report["components"]
    assert "production_pipeline" in report["components"]
    assert "genesis_wire" in report["components"]
```

## Acceptance

- [ ] Bridge imports all 11 component groups (7 v5 + 4 v6)
- [ ] `GENESIS_ENGINE_AVAILABLE` unchanged (backward compat)
- [ ] `NODE0_PRODUCTION_AVAILABLE` reports True when all present
- [ ] `availability_report()` includes v6 components
- [ ] `node0_server` NOT imported (avoid FastAPI hard dep)
