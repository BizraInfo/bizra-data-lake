"""Re-export from canonical location: core.sovereign.ihsan_projector"""

# Canonical implementation is in core/sovereign/ (uses centralized constants)
from core.sovereign.ihsan_projector import *  # noqa: F401,F403
from core.sovereign.ihsan_projector import (
    IhsanDimension,
    IhsanProjector,
    IhsanVector,
    ProjectorConfig,
    create_ihsan_from_scores,
    project_ihsan_to_ntu,
)
