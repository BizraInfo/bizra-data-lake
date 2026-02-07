"""Re-export from canonical location: core.bridges.local_inference_bridge"""

# Backwards compatibility — canonical implementation is in core/bridges/
from core.bridges.local_inference_bridge import *  # noqa: F401,F403
from core.bridges.local_inference_bridge import (
    InferenceRequest,
    InferenceResponse,
    LocalInferenceBridge,
)
