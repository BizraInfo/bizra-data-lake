"""
BIZRA Config System — 3-Scope YAML Configuration
════════════════════════════════════════════════

Federation > Operator > Node local. Deep merge with constitutional validation.

Standing on Giants:
- Fielding (2000): REST constraints as config
- 12-Factor App (2011): Config in the environment
"""

from core.config.loader import ConfigLoader
from core.config.schema import (
    BizraConfig,
    InferenceConfig,
    OrchestratorConfig,
    PolicyConfig,
)

__all__ = [
    "BizraConfig",
    "ConfigLoader",
    "InferenceConfig",
    "OrchestratorConfig",
    "PolicyConfig",
]
