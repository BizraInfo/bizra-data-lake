"""Re-export from canonical location: core.sovereign.proactive_integration"""
# Canonical implementation is in core/sovereign/ (uses centralized constants)
from core.sovereign.proactive_integration import *  # noqa: F401,F403
from core.sovereign.proactive_integration import (
    EntityConfig,
    EntityCycleResult,
    EntityMode,
    ProactiveSovereignEntity,
    create_proactive_entity,
)
