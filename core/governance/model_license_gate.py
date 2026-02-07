"""Re-export from canonical location: core.sovereign.model_license_gate"""

# Canonical implementation is in core/sovereign/ (uses centralized constants)
from core.sovereign.model_license_gate import *  # noqa: F401,F403
from core.sovereign.model_license_gate import (
    GateChain,
    GateResult,
    InMemoryRegistry,
    LicenseCheckResult,
    ModelLicenseGate,
    ModelRegistry,
    create_gate_chain,
)
