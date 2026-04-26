"""Stdlib-only schemas for the Omnidirectional Hyper-dimensional Audit Engine.

No third-party dependencies. All dataclasses are serializable to dict via
`dataclasses.asdict` and to JSON via `json.dumps`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import List, Optional


@dataclass
class EvidenceItem:
    item_id: str                 # stable id: e.g. "E0001"
    path: str                    # repo-relative path
    sha256: str
    size_bytes: int
    modified_ts: str             # ISO8601 UTC
    type: str                    # "markdown" | "rust" | "python" | "json" | "lock" | "manifest" | ...
    purpose_guess: str           # heuristic one-line description
    evidence_class: str          # "DOCTRINE" | "MANIFEST" | "CODE" | "ARTIFACT" | "CONFIG" | "CANON_PACK" | ...


@dataclass
class Claim:
    claim_id: str
    text: str                    # verbatim or redacted excerpt
    source: str                  # repo path OR URL
    line: Optional[int] = None
    category: str = "UNKNOWN"    # "NUMERIC" | "SECURITY" | "COST" | "READINESS" | "IDENTITY" | ...
    classification: str = "PROOF_REQUIRED"  # BRAND_SAFE | PROOF_REQUIRED | NEEDS_REWRITE | INTERNAL_ONLY | PROHIBITED
    rationale: str = ""


@dataclass
class Finding:
    finding_id: str
    domain: str                  # "SECURITY" | "ARCHITECTURE" | "PERFORMANCE" | ...
    subsystem: str
    summary: str
    evidence_paths: List[str] = field(default_factory=list)
    severity: str = "MEDIUM"     # LOW | MEDIUM | HIGH | CRITICAL
    confidence: float = 0.6      # 0..1
    signal_score: float = 0.5
    noise_score: float = 0.5
    actionable: bool = True
    owner: str = "operator"
    next_action: str = ""


@dataclass
class Risk:
    risk_id: str
    finding_id: str              # link to finding that surfaced this risk
    description: str
    impact: str                  # LOW | MEDIUM | HIGH | CRITICAL
    likelihood: str              # LOW | MEDIUM | HIGH
    mitigation_ids: List[str] = field(default_factory=list)


@dataclass
class Mitigation:
    mitigation_id: str
    description: str
    effort: str                  # XS | S | M | L | XL
    blocks: List[str] = field(default_factory=list)


@dataclass
class Kpi:
    kpi_id: str
    label: str
    target: str
    measured: Optional[str] = None
    source: str = ""
    classification: str = "TARGET"  # TARGET | MEASURED | SIMULATED | UNVERIFIED


@dataclass
class Gate:
    gate_id: str
    tier: str                    # A | B | C | D | E
    label: str
    status: str                  # PASS | FAIL | BLOCKED | NOT_TESTED
    evidence_path: str = ""
    owner: str = "operator"
    next_action: str = ""


@dataclass
class GraphNode:
    node_id: str
    kind: str                    # "file" | "claim" | "finding" | "risk" | "mitigation" | "kpi" | "gate"
    label: str
    attributes: dict = field(default_factory=dict)


@dataclass
class GraphEdge:
    src: str
    dst: str
    relation: str                # supports | contradicts | requires | blocks | mitigates | duplicates


def to_jsonable(obj) -> dict:
    """Convert any dataclass (or list of dataclasses) to a JSON-safe dict/list."""
    if isinstance(obj, list):
        return [to_jsonable(x) for x in obj]
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj
