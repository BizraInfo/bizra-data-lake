"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   PROOF-CARRYING EXECUTION ENGINE                                            ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║   The 4 Deliverables of Elite Implementation:                                ║
║   ══════════════════════════════════════════                                 ║
║   1. DETERMINISTIC CANONICALIZATION — same input → same bytes → same hash   ║
║   2. SNR SCORING AS VERIFIABLE FUNCTION — computed scalar with trace        ║
║   3. SIGNED REJECTION RECEIPTS — fail-closed, audit trail for denials       ║
║   4. BENCH-AS-RECEIPT HARNESS — performance claims as sealed evidence       ║
║                                                                              ║
║   The 7-Layer Architecture:                                                  ║
║   ═════════════════════════                                                  ║
║   L1: Input Canon         L5: Compression/Transport                         ║
║   L2: Epigenome Weighting L6: Epistemic Graph Commit                        ║
║   L3: Constitution Gate   L7: Receipt + PoI Settlement                      ║
║   L4: SNR Engine                                                             ║
║                                                                              ║
║   The 6 Gates (fail-closed chain):                                           ║
║   ════════════════════════════════                                           ║
║   SchemaGate → ProvenanceGate → SNRGate → ConstraintGate → SafetyGate →     ║
║   CommitGate                                                                 ║
║                                                                              ║
║   "Stop adding features. Lock the invariants."                               ║
║                                                                              ║
║   BIZRA Proof Engine v1.0.0                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from typing import TYPE_CHECKING

# Version
PROOF_ENGINE_VERSION = "1.0.0"

# The 6 Gates
GATE_CHAIN = [
    "schema",      # Input validation
    "provenance",  # Source verification
    "snr",         # Signal-to-noise gate
    "constraint",  # Z3 + Ihsān constraints
    "safety",      # Constitutional safety check
    "commit",      # Final commit gate
]

# The 9 KPIs
PROOF_KPIS = {
    "determinism_rate": {"target": 1.0, "unit": "ratio"},
    "p99_latency_us": {"target": 1000, "unit": "microseconds"},
    "throughput_ops": {"target": 10000, "unit": "ops/sec"},
    "hot_path_allocs": {"target": 0, "unit": "count"},
    "receipt_verify_rate": {"target": 1.0, "unit": "ratio"},
    "contradiction_handling": {"target": 1.0, "unit": "ratio"},
    "solver_cache_hit": {"target": 0.9, "unit": "ratio"},
    "graph_commit_p99_us": {"target": 500, "unit": "microseconds"},
    "bench_hash_match": {"target": 1.0, "unit": "ratio"},
}

# Receipt status codes
RECEIPT_STATUS = {
    "accepted": 0,
    "rejected": 1,
    "amber_restricted": 2,
    "pending": 3,
}

# SNR Policy defaults
DEFAULT_SNR_POLICY = {
    "snr_min": 0.95,
    "contradiction_penalty": 1.0,
    "unverifiable_penalty": 0.5,
    "provenance_weight": 0.3,
    "constraint_weight": 0.4,
    "prediction_weight": 0.3,
}

# Lazy imports
if TYPE_CHECKING:
    from .canonical import CanonQuery, canonical_json
    from .snr import SNREngine, SNRTrace, SNRPolicy
    from .receipt import Receipt, ReceiptStatus, SovereignSigner
    from .bench import BenchReceipt, bench_to_receipt
    from .gates import GateChain, GateResult


def __getattr__(name: str):
    if name == "CanonQuery":
        from .canonical import CanonQuery
        return CanonQuery
    elif name == "CanonPolicy":
        from .canonical import CanonPolicy
        return CanonPolicy
    elif name == "canonical_bytes":
        from .canonical import canonical_bytes
        return canonical_bytes
    elif name == "blake3_digest":
        from .canonical import blake3_digest
        return blake3_digest
    elif name == "SNREngine":
        from .snr import SNREngine
        return SNREngine
    elif name == "SNRPolicy":
        from .snr import SNRPolicy
        return SNRPolicy
    elif name == "SNRTrace":
        from .snr import SNRTrace
        return SNRTrace
    elif name == "SNRInput":
        from .snr import SNRInput
        return SNRInput
    elif name == "Receipt":
        from .receipt import Receipt
        return Receipt
    elif name == "ReceiptStatus":
        from .receipt import ReceiptStatus
        return ReceiptStatus
    elif name == "ReceiptBuilder":
        from .receipt import ReceiptBuilder
        return ReceiptBuilder
    elif name == "SimpleSigner":
        from .receipt import SimpleSigner
        return SimpleSigner
    elif name == "GateChain":
        from .gates import GateChain
        return GateChain
    elif name == "GateResult":
        from .gates import GateResult
        return GateResult
    elif name == "GateChainResult":
        from .gates import GateChainResult
        return GateChainResult
    elif name == "BenchReceipt":
        from .bench import BenchReceipt
        return BenchReceipt
    elif name == "BenchHarness":
        from .bench import BenchHarness
        return BenchHarness
    elif name == "BenchResult":
        from .bench import BenchResult
        return BenchResult
    elif name == "bench_to_receipt":
        from .bench import bench_to_receipt
        return bench_to_receipt
    raise AttributeError(f"module 'core.proof_engine' has no attribute '{name}'")


__all__ = [
    # Constants
    "PROOF_ENGINE_VERSION",
    "GATE_CHAIN",
    "PROOF_KPIS",
    "RECEIPT_STATUS",
    "DEFAULT_SNR_POLICY",
    # Canonical
    "CanonQuery",
    "CanonPolicy",
    "canonical_bytes",
    "blake3_digest",
    # SNR
    "SNREngine",
    "SNRPolicy",
    "SNRTrace",
    "SNRInput",
    # Receipt
    "Receipt",
    "ReceiptStatus",
    "ReceiptBuilder",
    "SimpleSigner",
    # Gates
    "GateChain",
    "GateResult",
    "GateChainResult",
    # Bench
    "BenchReceipt",
    "BenchHarness",
    "BenchResult",
    "bench_to_receipt",
]
