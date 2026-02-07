"""Re-export from canonical location: core.sovereign.constitutional_gate"""

# Canonical implementation is in core/sovereign/ (uses centralized constants)
from core.sovereign.constitutional_gate import *  # noqa: F401,F403
from core.sovereign.constitutional_gate import (
    Z3_CERT_DOMAIN_PREFIX,
    ConstitutionalGate,
)

# Also re-export types imported by the canonical module (not in __all__)
from core.sovereign.integration_types import (  # noqa: F401
    AdmissionResult,
    AdmissionStatus,
    Z3Certificate,
)
