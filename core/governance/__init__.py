"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA GOVERNANCE — Constitutional Gates & Autonomy                         ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Ethical validation, constitutional enforcement, and autonomy management.   ║
║                                                                              ║
║   Components:                                                                ║
║   - constitutional_gate: Z3-proven synthesis admission                       ║
║   - autonomy_matrix: Multi-level autonomous operation control                ║
║   - model_license_gate: Model capability validation chain                    ║
║   - ihsan_projector: 8-dimensional excellence scoring                        ║
║   - key_registry: Trusted public key management                              ║
║                                                                              ║
║   Constitutional Constraint: All decisions must achieve Ihsan >= 0.95        ║
║                                                                              ║
║   Standing on Giants:                                                        ║
║   - Al-Ghazali (1095): Ihsan Ethics                                          ║
║   - Anthropic (2022): Constitutional AI                                      ║
║   - de Moura & Bjorner (2008): Z3 SMT Solver                                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Created: 2026-02-05 | SAPE Sovereign Module Decomposition
Migrated: 2026-02-05 | Files now in dedicated governance package
"""

# Direct imports from local files (not from sovereign for clean separation)
from .constitutional_gate import (
    ConstitutionalGate,
    AdmissionStatus,
    AdmissionResult,
)
from .autonomy_matrix import (
    AutonomyMatrix,
    AutonomyLevel,
    AutonomyDecision,
)
from .model_license_gate import (
    ModelLicenseGate,
)
from .ihsan_projector import (
    IhsanProjector,
)
from .ihsan_vector import (
    IhsanVector,
    IhsanDimension,
)
from .capability_card import (
    CapabilityCard,
    ModelCapabilities,
    CardIssuer,
    ModelTier,
    TaskType,
)
from .autonomy import (
    AutonomousLoop,
    DecisionGate,
)
from .key_registry import (
    TrustedKeyRegistry,
    RegisteredKey,
    KeyStatus,
    get_key_registry,
)

__all__ = [
    # Constitutional Gate
    "ConstitutionalGate",
    "AdmissionStatus",
    "AdmissionResult",
    # Autonomy Matrix
    "AutonomyMatrix",
    "AutonomyLevel",
    "AutonomyDecision",
    # Model License Gate
    "ModelLicenseGate",
    # Ihsan
    "IhsanProjector",
    "IhsanVector",
    "IhsanDimension",
    # Capability Card
    "CapabilityCard",
    "ModelCapabilities",
    "CardIssuer",
    "ModelTier",
    "TaskType",
    # Autonomy Engine
    "AutonomousLoop",
    "DecisionGate",
    # Key Registry
    "TrustedKeyRegistry",
    "RegisteredKey",
    "KeyStatus",
    "get_key_registry",
]
