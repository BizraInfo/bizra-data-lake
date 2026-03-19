"""
BIZRA Node0 Integration Wire — Bridge → Live Pipeline
═══════════════════════════════════════════════════════

This module closes the gap identified in the Phase 2 diagnostic:
  "Genesis Engine v5 deployed as standalone constitutional package,
   bridged into core/ but not yet wired into the live Node0 mission pipeline."

The wire connects:
  core/integration/bridge.py (7/7 components)
    ↓
  bizra-constitution/ (Genesis Engine v5)
    ↓
  core/mission_orchestrator.py (live pipeline)

Integration pattern: Adapter.
  - Does NOT modify existing Node0 code
  - Wraps ProductionPipeline to speak the existing MissionOrchestrator interface
  - Falls back gracefully if any Genesis component fails
  - Can be enabled/disabled via environment variable

Usage in existing Node0 codebase:
    # In core/mission_orchestrator.py
    from core.integration.node0_wire import wire_genesis_engine

    # During initialization
    genesis_pipeline = wire_genesis_engine()
    if genesis_pipeline:
        # Route missions through constitutional pipeline
        result = genesis_pipeline.execute(user_input)
    else:
        # Fallback to existing orchestrator
        result = legacy_orchestrate(user_input)

Constitution ref: deployment.phase_2 (Brain_Body, v3.1.0-BRIDGE)
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("bizra.wire")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Environment variable to enable/disable Genesis Engine wiring
WIRE_ENABLED_ENV = "BIZRA_GENESIS_WIRE"
WIRE_DEFAULT = True

# Default paths relative to BIZRA-DATA-LAKE
DEFAULT_DATA_DIR = "sovereign_state/genesis"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EVIDENCE_PATH = "sovereign_state/evidence_ledger.jsonl"
DEFAULT_CACHE_PATH = "sovereign_state/reflex_cache.json"


# ═══════════════════════════════════════════════════════════════════════════════
# WIRE RESULT — Standardized output for MissionOrchestrator
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class WireResult:
    """
    Standardized result that the existing MissionOrchestrator can consume.

    Maps from Genesis Engine's Mission object to the format expected by
    the existing Node0 event bus and response pipeline.
    """

    success: bool
    output: str
    ihsan_composite: float
    ihsan_dimensions: dict[str, float]
    snr_normalized: float
    bloom_eligible: bool
    tier: str
    agent_trace: list[str]
    evidence_receipt_id: str | None
    signed: bool
    node_id: str | None
    latency_ms: float
    classification_confidence: float
    reflex_hit: bool
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_event_bus_payload(self) -> dict[str, Any]:
        """Format for the Rust event bus (12 subscribers)."""
        return {
            "type": "mission_complete",
            "output": self.output,
            "ihsan": {
                "composite": self.ihsan_composite,
                "dimensions": self.ihsan_dimensions,
                "bloom_eligible": self.bloom_eligible,
            },
            "snr": self.snr_normalized,
            "classification": {
                "tier": self.tier,
                "confidence": self.classification_confidence,
                "reflex_hit": self.reflex_hit,
            },
            "evidence": {
                "receipt_id": self.evidence_receipt_id,
                "signed": self.signed,
                "node_id": self.node_id,
            },
            "timing": {
                "total_ms": self.latency_ms,
            },
            "agent_trace": self.agent_trace,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GENESIS WIRE — The Integration Adapter
# ═══════════════════════════════════════════════════════════════════════════════


class GenesisWire:
    """
    Adapter between the Genesis Engine (constitutional pipeline) and
    the existing Node0 MissionOrchestrator.

    Handles:
    1. Lazy initialization (don't import genesis modules until needed)
    2. Graceful fallback (if genesis fails, return None → legacy path)
    3. Result translation (Mission → WireResult → event bus payload)
    4. Health monitoring (track genesis pipeline health)
    """

    def __init__(
        self,
        data_dir: Path | None = None,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        model_chain: list[str] | None = None,
        on_fallback: Callable[[str, Exception], None] | None = None,
    ):
        self._data_dir = data_dir or Path(DEFAULT_DATA_DIR)
        self._ollama_url = ollama_url
        self._model_chain = model_chain
        self._on_fallback = on_fallback

        self._pipeline = None
        self._initialized = False
        self._init_error: str | None = None

        # Metrics
        self._total_missions = 0
        self._genesis_missions = 0
        self._fallback_missions = 0
        self._total_latency_ms = 0.0

    def initialize(self) -> bool:
        """
        Lazily initialize the Genesis Engine pipeline.

        Returns True if successful, False if initialization fails.
        On failure, the wire operates in passthrough mode (returns None
        for all execute() calls, triggering the legacy fallback path).
        """
        if self._initialized:
            return self._pipeline is not None

        try:
            # Import genesis modules (may not be on sys.path yet)
            from production_pipeline import create_node0

            self._pipeline = create_node0(
                data_dir=self._data_dir,
                ollama_url=self._ollama_url,
                model_chain=self._model_chain,
            )
            self._initialized = True
            logger.info(
                f"Genesis Wire initialized: "
                f"node={self._pipeline.identity.node_id[:16]}... "
                f"agents={self._pipeline.identity.total_agents}"
            )
            return True

        except ImportError as e:
            self._init_error = f"Genesis modules not found: {e}"
            logger.warning(f"Genesis Wire init failed: {self._init_error}")
            self._initialized = True  # Mark as attempted
            return False

        except Exception as e:
            self._init_error = f"Genesis init error: {e}"
            logger.error(f"Genesis Wire init failed: {self._init_error}")
            self._initialized = True
            return False

    def execute(self, input_text: str) -> WireResult | None:
        """
        Execute a mission through the Genesis Engine.

        Returns:
            WireResult if Genesis Engine handled it successfully.
            None if Genesis Engine is unavailable (caller should use legacy path).
        """
        self._total_missions += 1

        # Lazy init
        if not self._initialized:
            self.initialize()

        if self._pipeline is None:
            self._fallback_missions += 1
            return None

        try:
            start = time.monotonic()
            mission = self._pipeline.execute(input_text)
            elapsed_ms = (time.monotonic() - start) * 1000

            self._genesis_missions += 1
            self._total_latency_ms += elapsed_ms

            # Translate Mission → WireResult
            receipt = mission.evidence_receipt
            receipt_meta = receipt.metadata if receipt else {}

            # Extract Ihsan dimensions dict from IhsanScore object
            ihsan_obj = mission.ihsan_score
            ihsan_composite = ihsan_obj.composite if ihsan_obj else 0.0
            ihsan_dims = ihsan_obj.as_tensor_dict() if ihsan_obj else {}
            bloom = mission.bloom_eligible

            # Extract SNR from MissionSNR object
            snr_val = mission.mission_snr.snr_normalized if mission.mission_snr else 0.0

            return WireResult(
                success=True,
                output=mission.output_text,
                ihsan_composite=ihsan_composite,
                ihsan_dimensions=ihsan_dims,
                snr_normalized=snr_val,
                bloom_eligible=bloom,
                tier=(
                    mission.classification.tier.value
                    if mission.classification
                    else "unknown"
                ),
                agent_trace=[a.get("agent", "?") for a in mission.agent_trace],
                evidence_receipt_id=receipt.receipt_id if receipt else None,
                signed="signature_hex" in receipt_meta,
                node_id=receipt_meta.get("node_id"),
                latency_ms=round(elapsed_ms, 2),
                classification_confidence=(
                    mission.classification.confidence if mission.classification else 0.0
                ),
                reflex_hit=mission.reflex_hit,
            )

        except Exception as e:
            self._fallback_missions += 1
            if self._on_fallback:
                self._on_fallback(input_text, e)
            logger.warning(f"Genesis execution failed, falling back: {e}")
            return None

    def health(self) -> dict[str, Any]:
        """Wire health report for Node0 status endpoint."""
        return {
            "wire_enabled": self._pipeline is not None,
            "initialized": self._initialized,
            "init_error": self._init_error,
            "total_missions": self._total_missions,
            "genesis_missions": self._genesis_missions,
            "fallback_missions": self._fallback_missions,
            "genesis_rate": (
                round(self._genesis_missions / self._total_missions, 4)
                if self._total_missions > 0
                else 0.0
            ),
            "avg_latency_ms": (
                round(self._total_latency_ms / self._genesis_missions, 2)
                if self._genesis_missions > 0
                else 0.0
            ),
            "pipeline_health": (self._pipeline.health() if self._pipeline else None),
        }

    def shutdown(self):
        """Graceful shutdown — persist cache, close resources."""
        if self._pipeline:
            if hasattr(self._pipeline, "shutdown"):
                self._pipeline.shutdown()
            logger.info("Genesis Wire shutdown complete")


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION — One-line integration
# ═══════════════════════════════════════════════════════════════════════════════


def wire_genesis_engine(
    data_dir: Path | None = None,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model_chain: list[str] | None = None,
) -> GenesisWire | None:
    """
    Wire the Genesis Engine into the live Node0 pipeline.

    This is the ONE function the existing MissionOrchestrator calls.
    Returns None if Genesis wiring is disabled via environment variable.

    Usage in existing codebase:
        wire = wire_genesis_engine()
        if wire:
            result = wire.execute(user_input)
            if result:
                # Constitutional pipeline handled it
                return result.to_event_bus_payload()
        # else: legacy path

    Args:
        data_dir: Directory for genesis state files.
        ollama_url: Ollama server URL.
        model_chain: Model fallback chain.

    Returns:
        GenesisWire instance if enabled, None if disabled.
    """
    # Check environment variable
    enabled = os.environ.get(WIRE_ENABLED_ENV, "").lower()
    if enabled == "false" or enabled == "0":
        logger.info("Genesis Wire disabled via environment variable")
        return None

    wire = GenesisWire(
        data_dir=data_dir,
        ollama_url=ollama_url,
        model_chain=model_chain,
    )

    # Attempt initialization (lazy, non-blocking)
    wire.initialize()

    return wire
