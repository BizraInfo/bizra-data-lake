"""BIZRA platform normalizers package."""

try:
    from .engine import (
        AutonomousSNRGoTEngine,
        GIANTS_PROTOCOL,
        GiantPrinciple,
        SignalEdge,
        SignalNode,
        StereoscopicReport,
    )
    from .genesis_gate import GenesisGateConfig, GenesisGateVerdict, evaluate_genesis_gate
except ImportError:
    # Support direct path execution where package-relative imports are unavailable.
    from engine import (  # type: ignore[no-redef]
        AutonomousSNRGoTEngine,
        GIANTS_PROTOCOL,
        GiantPrinciple,
        SignalEdge,
        SignalNode,
        StereoscopicReport,
    )
    from genesis_gate import (  # type: ignore[no-redef]
        GenesisGateConfig,
        GenesisGateVerdict,
        evaluate_genesis_gate,
    )

__all__ = [
    "AutonomousSNRGoTEngine",
    "GIANTS_PROTOCOL",
    "GiantPrinciple",
    "SignalNode",
    "SignalEdge",
    "StereoscopicReport",
    "GenesisGateConfig",
    "GenesisGateVerdict",
    "evaluate_genesis_gate",
]
