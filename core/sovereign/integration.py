"""
BIZRA Sovereign LLM Integration Module — Genesis Strict Synthesis v2.2.2

This module provides the unified entry point for the Sovereign LLM ecosystem,
connecting all components:
- Capability Cards (model credentials)
- Gate Chain (constitutional enforcement)
- Model Registry (accepted models)
- Inference Backend (llama.cpp sandbox)
- Federation Layer (optional P2P)
- Constitutional Gate (Z3-proven runtime admission)

Module Structure (SPARC refinement):
- integration_types.py    — Data classes, enums, configs
- constitutional_gate.py  — Z3 proof-based admission controller
- integration_runtime.py  — SovereignRuntime implementation
- integration.py          — Public facade (this file)

"We do not assume. We verify with formal proofs."

Standing on Giants: Shannon + Lamport + Vaswani + Anthropic
"""

from __future__ import annotations

import logging
from typing import Optional

# Import unified thresholds from authoritative source
try:
    from core.integration.constants import (
        UNIFIED_IHSAN_THRESHOLD,
        UNIFIED_SNR_THRESHOLD,
    )
except ImportError:
    UNIFIED_IHSAN_THRESHOLD = 0.95
    UNIFIED_SNR_THRESHOLD = 0.85

# =============================================================================
# PUBLIC API - Re-export from modular components
# =============================================================================

# Capability Cards (re-export)
from .capability_card import (
    CapabilityCard,
    CardIssuer,
    ModelTier,
    TaskType,
    create_capability_card,
    verify_capability_card,
)

# Constitutional Gate
from .constitutional_gate import (
    Z3_CERT_DOMAIN_PREFIX,
    ConstitutionalGate,
)

# Sovereign Runtime
from .integration_runtime import SovereignRuntime

# Types, enums, data classes
from .integration_types import (  # Enums; Z3 Certificates; Configuration; Inference
    AdmissionResult,
    AdmissionStatus,
    InferenceRequest,
    InferenceResult,
    NetworkMode,
    SovereignConfig,
    Z3Certificate,
)

# Model License Gate (re-export)
from .model_license_gate import (
    GateChain,
    InMemoryRegistry,
    ModelLicenseGate,
    create_gate_chain,
)

logger = logging.getLogger(__name__)

# Re-export thresholds for backward compatibility
IHSAN_THRESHOLD = UNIFIED_IHSAN_THRESHOLD
SNR_THRESHOLD = UNIFIED_SNR_THRESHOLD


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


async def create_sovereign_runtime(
    config: Optional[SovereignConfig] = None,
) -> SovereignRuntime:
    """Create and start a sovereign runtime."""
    runtime = SovereignRuntime(config)
    await runtime.start()
    return runtime


def print_banner():
    """Print the BIZRA banner."""
    print("""
╔══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║    "بذرة — Every seed is welcome that bears good fruit."         ║
║                                                                   ║
║    BIZRA Sovereign LLM Ecosystem v2.2.0                          ║
║                                                                   ║
║    Ihsan (Excellence) >= 0.95  — Z3 SMT verified                 ║
║    SNR (Signal Quality) >= 0.85 — Shannon enforced               ║
║                                                                   ║
║    "We do not assume. We verify with formal proofs."             ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
""")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "AdmissionStatus",
    "NetworkMode",
    "ModelTier",
    "TaskType",
    # Z3 Certificates
    "Z3Certificate",
    "AdmissionResult",
    # Configuration
    "SovereignConfig",
    # Inference
    "InferenceRequest",
    "InferenceResult",
    # Constitutional Gate
    "ConstitutionalGate",
    "Z3_CERT_DOMAIN_PREFIX",
    # Capability Cards
    "CapabilityCard",
    "CardIssuer",
    "create_capability_card",
    "verify_capability_card",
    # Model License Gate
    "ModelLicenseGate",
    "InMemoryRegistry",
    "GateChain",
    "create_gate_chain",
    # Runtime
    "SovereignRuntime",
    "create_sovereign_runtime",
    # Thresholds
    "IHSAN_THRESHOLD",
    "SNR_THRESHOLD",
    # Utility
    "print_banner",
]
