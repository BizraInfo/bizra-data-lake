"""
Constitutional Engine Bridge — Imports from bizra-constitution/ with graceful fallback.

╔══════════════════════════════════════════════════════════════════════════════╗
║   Bridge between core/ and the Genesis Engine v6 constitutional package.     ║
║   All imports are try/except — core/ operates normally without the package.  ║
║                                                                              ║
║   v5: 7 component groups (library mode — no external deps)                   ║
║   v6: +4 component groups (production mode — identity + Ollama + server)     ║
╚══════════════════════════════════════════════════════════════════════════════╝

Source: bizra-constitution/ (constitution.toml v5.0.0-GENESIS)
Pattern: Same as core/snr_protocol.py bridge to bizra_constitution.snr

Standing on Giants: Lamport (state convergence) · Shannon (SNR monotonicity)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# ═══════════════════════════════════════════════════════════════════════════════
# PATH SETUP — ensure bizra-constitution/ is importable
# ═══════════════════════════════════════════════════════════════════════════════

_CONSTITUTION_PKG = Path(__file__).resolve().parent.parent.parent / "bizra-constitution"
if _CONSTITUTION_PKG.is_dir() and str(_CONSTITUTION_PKG) not in sys.path:
    sys.path.insert(0, str(_CONSTITUTION_PKG))


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTITUTIONAL IHSAN GATE (6-dim operational projection)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from ihsan_gate import IhsanGate as ConstitutionalIhsanGate
    from ihsan_gate import IhsanScore, IhsanTier as ConstitutionalIhsanTier
    HAS_CONSTITUTIONAL_GATE = True
except ImportError:
    ConstitutionalIhsanGate = None  # type: ignore[assignment,misc]
    IhsanScore = None  # type: ignore[assignment,misc]
    ConstitutionalIhsanTier = None  # type: ignore[assignment,misc]
    HAS_CONSTITUTIONAL_GATE = False


# ═══════════════════════════════════════════════════════════════════════════════
# MISSION PIPELINE (7-agent PAT trust compiler)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from mission_pipeline import (
        MissionPipeline,
        Mission,
        MissionStatus,
        PatAgent,
    )
    HAS_MISSION_PIPELINE = True
except ImportError:
    MissionPipeline = None  # type: ignore[assignment,misc]
    Mission = None  # type: ignore[assignment,misc]
    MissionStatus = None  # type: ignore[assignment,misc]
    PatAgent = None  # type: ignore[assignment,misc]
    HAS_MISSION_PIPELINE = False


# ═══════════════════════════════════════════════════════════════════════════════
# HHMM ROUTER (complexity classification + action bus)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from hhmm_router import (
        HhmmRouter,
        ComplexityTier as ConstitutionalComplexityTier,
        ClassificationResult,
        ActionBus,
        MissionTicket,
    )
    HAS_HHMM_ROUTER = True
except ImportError:
    HhmmRouter = None  # type: ignore[assignment,misc]
    ConstitutionalComplexityTier = None  # type: ignore[assignment,misc]
    ClassificationResult = None  # type: ignore[assignment,misc]
    ActionBus = None  # type: ignore[assignment,misc]
    MissionTicket = None  # type: ignore[assignment,misc]
    HAS_HHMM_ROUTER = False


# ═══════════════════════════════════════════════════════════════════════════════
# REFLEX CACHE (O(1) HashMap with precipitation)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from reflex_cache import ReflexCache, ReflexEntry, CacheStats
    HAS_REFLEX_CACHE = True
except ImportError:
    ReflexCache = None  # type: ignore[assignment,misc]
    ReflexEntry = None  # type: ignore[assignment,misc]
    CacheStats = None  # type: ignore[assignment,misc]
    HAS_REFLEX_CACHE = False


# ═══════════════════════════════════════════════════════════════════════════════
# EVIDENCE RECEIPT (hash-chained ledger)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from evidence_receipt import (
        EvidenceReceipt,
        EvidenceLedger as ConstitutionalEvidenceLedger,
    )
    HAS_EVIDENCE_RECEIPT = True
except ImportError:
    EvidenceReceipt = None  # type: ignore[assignment,misc]
    ConstitutionalEvidenceLedger = None  # type: ignore[assignment,misc]
    HAS_EVIDENCE_RECEIPT = False


# ═══════════════════════════════════════════════════════════════════════════════
# SNR NORMALIZATION (canonical function)
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from snr import normalize_snr, compute_sape_composite, SapeScore, MissionSNR
    HAS_SNR = True
except ImportError:
    normalize_snr = None  # type: ignore[assignment]
    compute_sape_composite = None  # type: ignore[assignment]
    SapeScore = None  # type: ignore[assignment,misc]
    MissionSNR = None  # type: ignore[assignment,misc]
    HAS_SNR = False


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTITUTION PARSER
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from bizra_constitution import load_constitution, Constitution
    HAS_CONSTITUTION = True
except ImportError:
    load_constitution = None  # type: ignore[assignment]
    Constitution = None  # type: ignore[assignment,misc]
    HAS_CONSTITUTION = False


# ═══════════════════════════════════════════════════════════════════════════════
# IDENTITY GENESIS (Ed25519 + HD agent keys) — v6
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from identity_genesis import create_identity, NodeIdentity, AgentKey
    from identity_genesis import save_identity, load_public_record
    HAS_IDENTITY_GENESIS = True
except ImportError:
    create_identity = None  # type: ignore[assignment]
    NodeIdentity = None  # type: ignore[assignment,misc]
    AgentKey = None  # type: ignore[assignment,misc]
    save_identity = None  # type: ignore[assignment]
    load_public_record = None  # type: ignore[assignment]
    HAS_IDENTITY_GENESIS = False


# ═══════════════════════════════════════════════════════════════════════════════
# OLLAMA PROVIDER (circuit breaker + model fallback) — v6
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from ollama_provider import OllamaProvider, InferenceResult
    from ollama_provider import CircuitBreaker, CircuitState, ModelMetrics
    HAS_OLLAMA_PROVIDER = True
except ImportError:
    OllamaProvider = None  # type: ignore[assignment,misc]
    InferenceResult = None  # type: ignore[assignment,misc]
    CircuitBreaker = None  # type: ignore[assignment,misc]
    CircuitState = None  # type: ignore[assignment,misc]
    ModelMetrics = None  # type: ignore[assignment,misc]
    HAS_OLLAMA_PROVIDER = False


# ═══════════════════════════════════════════════════════════════════════════════
# PRODUCTION PIPELINE (signed evidence + real identity) — v6
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from production_pipeline import ProductionPipeline, create_node0
    HAS_PRODUCTION_PIPELINE = True
except ImportError:
    ProductionPipeline = None  # type: ignore[assignment,misc]
    create_node0 = None  # type: ignore[assignment]
    HAS_PRODUCTION_PIPELINE = False


# ═══════════════════════════════════════════════════════════════════════════════
# WIRE ADAPTER (MissionOrchestrator bridge) — v6
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from node0_wire import GenesisWire, WireResult, wire_genesis_engine
    HAS_GENESIS_WIRE = True
except ImportError:
    GenesisWire = None  # type: ignore[assignment,misc]
    WireResult = None  # type: ignore[assignment,misc]
    wire_genesis_engine = None  # type: ignore[assignment]
    HAS_GENESIS_WIRE = False


# ═══════════════════════════════════════════════════════════════════════════════
# AVAILABILITY SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

# v5: Library mode — no external dependencies
GENESIS_ENGINE_AVAILABLE: bool = all([
    HAS_CONSTITUTIONAL_GATE,
    HAS_MISSION_PIPELINE,
    HAS_HHMM_ROUTER,
    HAS_REFLEX_CACHE,
    HAS_EVIDENCE_RECEIPT,
    HAS_SNR,
    HAS_CONSTITUTION,
])

# v6: Production mode — identity + Ollama + server
NODE0_PRODUCTION_AVAILABLE: bool = all([
    GENESIS_ENGINE_AVAILABLE,
    HAS_IDENTITY_GENESIS,
    HAS_OLLAMA_PROVIDER,
    HAS_PRODUCTION_PIPELINE,
    HAS_GENESIS_WIRE,
])


def availability_report() -> dict[str, Any]:
    """Return availability status of all constitutional engine components."""
    return {
        "genesis_engine_available": GENESIS_ENGINE_AVAILABLE,
        "node0_production_available": NODE0_PRODUCTION_AVAILABLE,
        "components": {
            # v5 components
            "constitutional_gate": HAS_CONSTITUTIONAL_GATE,
            "mission_pipeline": HAS_MISSION_PIPELINE,
            "hhmm_router": HAS_HHMM_ROUTER,
            "reflex_cache": HAS_REFLEX_CACHE,
            "evidence_receipt": HAS_EVIDENCE_RECEIPT,
            "snr": HAS_SNR,
            "constitution_parser": HAS_CONSTITUTION,
            # v6 components
            "identity_genesis": HAS_IDENTITY_GENESIS,
            "ollama_provider": HAS_OLLAMA_PROVIDER,
            "production_pipeline": HAS_PRODUCTION_PIPELINE,
            "genesis_wire": HAS_GENESIS_WIRE,
        },
        "package_path": str(_CONSTITUTION_PKG),
    }
