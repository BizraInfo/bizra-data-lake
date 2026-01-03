"""
BIZRA Kernel — Core Intelligence & Ethical Governance
======================================================
This package contains the core cognitive and ethical components of the 
BIZRA Sovereign Organism.
"""

# Legacy Core Components
from .sape_engine import SAPEEngine, ElevatedPattern
from .symbolic_harness import SymbolicHarness
from .abstraction_elevator import AbstractionElevator
from .tension_studio import TensionStudio
from .kernel import (
    SystemProtocolKernel, 
    KernelConfig, 
    ExecutionResult, 
    get_kernel, 
    reset_kernel
)
from .ihsan_vector import (
    IhsanVector, 
    IhsanDimension, 
    IhsanConstitution, 
    constitution, 
    threshold_for
)
from .session_manager import SessionManager, Session, SessionState
from .verifier import MultiStageVerifier, VerificationResult, ProbeType, ProbeResult
from .snr_tracker import SNRTracker, SNRMetrics, estimate_useful_tokens
from .lexicon_ledger import (
    LexiconLedger, Term, TruthLabel, TermStatus, LedgerOperation,
    LexiconReceipt, DNA_SIGNATURE
)

# Peak Masterpiece Components (Cognitive Permanence)
from .memory_system import CognitivePermanence
from .sovereign_engine import SovereignEngine
from .ihsan_gate import IhsanGate
from .recursive_node import RecursiveNode
from .benchmark_util import BIZRABenchmark
from .proposal_agent import ProposalAgent
from .omni_awareness import OmniAwareness
from .state_ledger import StateLedger
from .consensus_engine import ConsensusEngine
from .genesis_broadcast import GenesisBroadcast
from .model_hub import SovereignModelHub

__all__ = [
    # Core
    "SAPEEngine", "ElevatedPattern",
    "SymbolicHarness", "AbstractionElevator", "TensionStudio",
    "SystemProtocolKernel", "KernelConfig", "ExecutionResult", "get_kernel", "reset_kernel",
    "IhsanVector", "IhsanDimension", "IhsanConstitution", "constitution", "threshold_for",
    "SessionManager", "Session", "SessionState",
    "MultiStageVerifier", "VerificationResult", "ProbeType", "ProbeResult",
    "SNRTracker", "SNRMetrics", "estimate_useful_tokens",
    "LexiconLedger", "Term", "TruthLabel", "TermStatus", "LedgerOperation",
    "LexiconReceipt", "DNA_SIGNATURE",
    
    # Masterpiece
    "CognitivePermanence",
    "SovereignEngine",
    "IhsanGate",
    "RecursiveNode",
    "BIZRABenchmark",
    "ProposalAgent",
    "OmniAwareness",
    "StateLedger",
    "ConsensusEngine",
    "GenesisBroadcast",
    "SovereignModelHub",
]
