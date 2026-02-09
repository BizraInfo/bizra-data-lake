"""
Proactive Execution Kernel (PEK)
================================
Minimal DDAGI OS glue layer that enforces one proactive execution loop:

    SENSE -> PREDICT -> SCORE -> VERIFY -> EXECUTE -> PROVE -> LEARN

PEK v0.2 integrates:
- OpportunityPipeline (intervention flow)
- ProactiveScheduler (Chronos execution lane)
- InferenceGateway and LivingMemory (ambient sensing)
- Optional Z3 FATE gate (formal verification)
- Intervention budget and proof event stream
"""

from .kernel import (
    PEKProofBlock,
    PEKProposal,
    ProactiveExecutionKernel,
    ProactiveExecutionKernelConfig,
)

__all__ = [
    "PEKProofBlock",
    "PEKProposal",
    "ProactiveExecutionKernel",
    "ProactiveExecutionKernelConfig",
]
