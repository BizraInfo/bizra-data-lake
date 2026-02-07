"""Re-export from canonical location: core.sovereign.ihsan_vector"""
# Canonical implementation is in core/sovereign/ (uses centralized constants)
from core.sovereign.ihsan_vector import *  # noqa: F401,F403
from core.sovereign.ihsan_vector import (
    CANONICAL_WEIGHTS,
    CONTEXT_THRESHOLDS,
    DimensionId,
    ExecutionContext,
    IhsanDimension,
    IhsanReceipt,
    IhsanVector,
    ThresholdResult,
    create_verifier,
    passes_production,
    quick_ihsan,
)
