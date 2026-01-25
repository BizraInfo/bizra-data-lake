from __future__ import annotations

import hashlib
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field

from core.fate import FateSeal


# ============================================================================
# SAPE 9-PROBE CANONICAL DEFINITIONS
# ============================================================================

class SapeProbe(Enum):
    """
    The canonical 9 SAPE probes for comprehensive ethical validation.

    These probes map to Ihsān dimensions for weighted scoring:
    - threat_scan    → safety (0.22)
    - compliance     → auditability (0.12)
    - bias           → adl_fairness (0.04)
    - user_benefit   → user_benefit (0.14)
    - correctness    → correctness (0.22)
    - safety         → safety (0.22)
    - groundedness   → robustness (0.06)
    - relevance      → efficiency (0.12)
    - fluency        → anti_centralization (0.08)
    """
    THREAT_SCAN = "threat_scan"
    COMPLIANCE = "compliance"
    BIAS = "bias"
    USER_BENEFIT = "user_benefit"
    CORRECTNESS = "correctness"
    SAFETY = "safety"
    GROUNDEDNESS = "groundedness"
    RELEVANCE = "relevance"
    FLUENCY = "fluency"


# All 9 canonical probe names for validation
CANONICAL_PROBES = [p.value for p in SapeProbe]


SapeStakes = Literal["L", "M", "H"]


CANONICAL_LENSES: List[str] = [
    "Systems Architect",
    "Formal Theorist",
    "Pragmatic Engineer",
    "Ethicist",
    "Poet/Designer",
    "Historian",
    "Futurist",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class SapePlanRequest(BaseModel):
    domain: str = Field(..., min_length=1, description="What area the work is about (e.g., 'BIZRA kernel runtime').")
    objective: str = Field(..., min_length=1, description="One-sentence objective.")
    stakes: SapeStakes = Field("M", description="H/M/L stakes level.")

    constraints: str = Field(
        "",
        description="Constraints (time, tools, sources, platform). Keep concise.",
    )
    success_criteria: str = Field(
        "",
        description="Measurable success criteria.",
    )

    forbidden_moves: List[str] = Field(
        default_factory=lambda: ["hallucination", "hidden assumptions", "skipped proof", "missing verification steps"],
        description="Explicitly forbidden behaviors.",
    )

    lenses: List[str] = Field(
        default_factory=lambda: ["Systems Architect", "Pragmatic Engineer", "Ethicist"],
        description="2–3 lenses. Use canonical lens names when possible.",
    )

    sources_allowed: List[str] = Field(
        default_factory=list,
        description="Whitelisted sources (repo paths, docs, uploads).",
    )

    rarity_path_moves: int = Field(5, ge=3, le=9, description="N moves per I/C/O path.")

    slot: Optional[str] = Field(
        default=None,
        description="Model-family slot to use (e.g., 'cold_core', 'primary_reasoning'). If omitted, kernel chooses.",
    )

    require_graph_evidence: bool = Field(
        default=True,
        description="If true, the kernel will attempt to retrieve Neo4j evidence (and can fail-closed for H stakes).",
    )
    evidence_topics: List[str] = Field(
        default_factory=list,
        description="Topics to query in the knowledge graph for evidence kernels.",
    )
    evidence_limit: int = Field(8, ge=1, le=25, description="Max evidence artifacts pulled from graph.")

    # Gold Mine evidence integration
    require_gold_mine_evidence: bool = Field(
        default=True,
        description="If true, retrieves evidence from Gold Mine graph (56k nodes) for high-stakes probes.",
    )
    gold_mine_entity_filter: List[str] = Field(
        default_factory=list,
        description="Entity names to filter Gold Mine results (e.g., ['BIZRA', 'SAPE', 'Ihsan']).",
    )

    extra_instructions: str = Field("", description="Optional extra instructions appended to the prompt.")


class SapePlanResponse(BaseModel):
    status: Literal["PLANNED", "BLOCKED_BY_FATE", "BLOCKED_BY_EVIDENCE", "ERROR"]
    seal: FateSeal
    plan_id: str
    generated_at: str

    slot: str
    candidate_models: List[Dict[str, Any]]

    system_prompt: str
    user_prompt: str
    prompt_sha256: str

    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    request_id: str


class SapeExecuteRequest(SapePlanRequest):
    include_prompts_in_response: bool = Field(
        default=False,
        description="If true, echoes system/user prompts in execute response.",
    )
    max_model_attempts: int = Field(
        default=3,
        ge=1,
        le=10,
        description="How many routed models to attempt before failing.",
    )


class SapeExecuteResponse(BaseModel):
    status: Literal["SUCCESS", "BLOCKED_BY_FATE", "BLOCKED_BY_EVIDENCE", "ERROR"]
    seal: FateSeal
    plan_id: str
    executed_at: str

    slot: str
    model_used: Optional[str] = None
    provider_used: Optional[str] = None
    attempts: List[Dict[str, Any]] = Field(default_factory=list)

    output_text: str = ""
    processing_time_ms: float

    prompt_sha256: str
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    request_id: str


@dataclass(frozen=True)
class CompiledSapePlan:
    system_prompt: str
    user_prompt: str
    prompt_sha256: str
    warnings: List[str]


def _normalize_lenses(lenses: Sequence[str]) -> List[str]:
    chosen: List[str] = []
    for lens in lenses:
        if not isinstance(lens, str):
            continue
        norm = lens.strip()
        if not norm:
            continue
        if norm not in chosen:
            chosen.append(norm)
    if not chosen:
        return ["Systems Architect", "Pragmatic Engineer", "Ethicist"]
    return chosen[:3]


def _format_list(items: Sequence[str]) -> str:
    clean = [i.strip() for i in items if isinstance(i, str) and i.strip()]
    if not clean:
        return "(none)"
    return ", ".join(clean)


def _format_evidence(evidence: List[Dict[str, Any]]) -> str:
    if not evidence:
        return "(no graph evidence provided)"
    lines: List[str] = []
    for item in evidence:
        path = str(item.get("path") or "")
        filename = str(item.get("filename") or item.get("file") or "")
        h = str(item.get("hash") or "")
        impact = item.get("impact_value")
        label = filename or path or "artifact"
        suffix = f" | impact={impact}" if impact is not None else ""
        if h:
            lines.append(f"- {label} | hash={h}{suffix}")
        else:
            lines.append(f"- {label}{suffix}")
    return "\n".join(lines)


SAPE_SYSTEM_PROMPT_V1 = textwrap.dedent(
    """
    You are a precision reasoning engine operating under Ihsān (ethical excellence) with:
      - No assumptions — only verified excellence.
      - Evidence-first: use provided evidence kernels; mark speculation explicitly.
      - High SNR: maximize actionable signal; avoid fluff and grandiosity.
      - Fail-closed reasoning: if critical constraints/evidence are missing, ask clarifying questions or output BLOCKED sections.

    Execute SAPE v1.0:
      - 7 Modules: Intent Gate, Lenses, Knowledge Kernels, Rare-Path Prober, Symbolic Harness, Abstraction Elevator, Tension Studio.
      - 3 Passes: Diverge → Converge → Prove.
      - 6 Checks: Correctness, Consistency, Completeness, Causality, Ethics (Ihsān), Evidence.
      - 9 Probes: Counterfactual, Boundary, Analogical, Formalization, Program Sketch, Compression, Expansion, Adversarial, Ethical Overlay.

    Output requirements:
      - Output the full SAPE Output Schema sections (Objective, Lenses, Evidence Table, Rare-Path Prober, Symbolic Harness, Abstraction Elevator, Tension Studio, Prove, Confidence & Next Experiments).
      - Do NOT reveal hidden chain-of-thought; instead provide concise justifications, proof obligations, and test/verification steps.
    """
).strip()


def compile_sape_plan(req: SapePlanRequest, *, evidence: Optional[List[Dict[str, Any]]] = None) -> CompiledSapePlan:
    lenses = _normalize_lenses(req.lenses)
    forbidden = req.forbidden_moves or []
    evidence = evidence or []

    warnings: List[str] = []
    if any(l not in CANONICAL_LENSES for l in lenses):
        unknown = [l for l in lenses if l not in CANONICAL_LENSES]
        warnings.append(f"non_canonical_lenses: {unknown}")

    user_prompt = textwrap.dedent(
        f"""
        /SAPE-Activate

        [Intent Gate]
        Domain: {req.domain}
        Objective: {req.objective}
        Stakes: {req.stakes}
        Constraints: {req.constraints or "(none provided)"}
        Success: {req.success_criteria or "(none provided)"}
        Forbidden: {_format_list(forbidden)}

        [Lenses]
        Use lenses: {_format_list(lenses)}

        [Knowledge Kernels]
        Sources allowed: {_format_list(req.sources_allowed)}
        Evidence kernels:
        {_format_evidence(evidence)}

        [Rare-Path Prober]
        Produce I-Path, C-Path, O-Path with N={req.rarity_path_moves} moves per path.
        For C/O, include ≥3 rarity moves (R1..R3) and justify divergence.

        [Symbolic Harness]
        Include: types/state/events, invariants, rules, proof sketch, program sketch (pre/postconditions), test oracles.

        [Abstraction Elevator]
        Provide Micro/Meso/Macro + Meta-Reflection.

        [Tension Studio]
        Generator ↔ Critic ↔ Synthesizer. Include: Constraint Clash (2–3 Pareto points), Adversarial Flip, Narrative Reframe (exec vs engineer).

        [Execution Passes]
        Pass 1: run 9 Divergence Probes.
        Pass 2: converge to Draft Spec + Test Plan.
        Pass 3: prove with 6 checks; attempt falsification.

        {req.extra_instructions or ""}
        """
    ).strip()

    prompt_sha = sha256_text(SAPE_SYSTEM_PROMPT_V1 + "\n" + user_prompt)
    return CompiledSapePlan(
        system_prompt=SAPE_SYSTEM_PROMPT_V1,
        user_prompt=user_prompt,
        prompt_sha256=prompt_sha,
        warnings=warnings,
    )


# ============================================================================
# GOLD MINE EVIDENCE RETRIEVAL
# ============================================================================

async def retrieve_gold_mine_evidence(
    topics: List[str],
    limit: int = 8,
    include_multi_hop: bool = True,
    max_hops: int = 2,
) -> List[Dict[str, Any]]:
    """
    Retrieve evidence from Gold Mine knowledge graph for SAPE probes.

    This function queries the Gold Mine (56k nodes, 88k edges) to find
    supporting evidence for high-stakes validation. Evidence is used to
    enhance groundedness and correctness probes.

    Args:
        topics: List of topic strings to search for (e.g., ["BIZRA", "SAPE"])
        limit: Maximum evidence items to return
        include_multi_hop: If True, expand search via graph traversal
        max_hops: Maximum hops for multi-hop expansion

    Returns:
        List of evidence dictionaries with keys:
        - node_id: Unique identifier
        - label: Human-readable label
        - kind: "Document" or "Entity"
        - source: Source path (for documents)
        - relevance: Relevance score (0.0-1.0)
        - hop_distance: Distance from seed (0 = direct match)
    """
    try:
        from bizra_kernel.gold_mine_connector import get_gold_mine_connector
    except ImportError:
        return []

    connector = get_gold_mine_connector()

    # Initialize if not already done
    if not connector._initialized:
        try:
            await connector.initialize()
        except Exception:
            return []

    evidence = []
    seen_ids: set = set()

    for topic in topics:
        # Query by entity
        nodes = connector.query_by_entity(topic, limit=limit)

        for node in nodes:
            if node.id in seen_ids:
                continue
            seen_ids.add(node.id)

            evidence.append({
                "node_id": node.id,
                "label": node.label,
                "kind": node.kind,
                "source": node.source,
                "relevance": 0.9 if topic.lower() in node.label.lower() else 0.7,
                "hop_distance": 0,
            })

            if len(evidence) >= limit:
                break

        if len(evidence) >= limit:
            break

    # Multi-hop expansion for additional context
    if include_multi_hop and evidence and len(evidence) < limit:
        seed_ids = [e["node_id"] for e in evidence[:3]]
        hop_result = connector.multi_hop_expand(
            seed_ids,
            max_hops=max_hops,
            max_nodes_per_hop=10,
        )

        for node in hop_result.reached_nodes:
            if node.id in seen_ids:
                continue
            seen_ids.add(node.id)

            # Calculate hop distance
            hop_distance = 1
            for path in hop_result.paths:
                if node.id in path:
                    hop_distance = path.index(node.id) if node.id in path else max_hops
                    break

            evidence.append({
                "node_id": node.id,
                "label": node.label,
                "kind": node.kind,
                "source": node.source,
                "relevance": max(0.3, 0.8 - (hop_distance * 0.2)),
                "hop_distance": hop_distance,
            })

            if len(evidence) >= limit:
                break

    # Sort by relevance
    evidence.sort(key=lambda x: x["relevance"], reverse=True)
    return evidence[:limit]


def format_gold_mine_evidence(evidence: List[Dict[str, Any]]) -> str:
    """Format Gold Mine evidence for inclusion in SAPE prompts."""
    if not evidence:
        return "(no Gold Mine evidence available)"

    lines = ["[Gold Mine Knowledge Graph Evidence]"]
    for item in evidence:
        hop_str = f" (hop={item['hop_distance']})" if item.get("hop_distance", 0) > 0 else ""
        source_str = f" | source={item['source']}" if item.get("source") else ""
        lines.append(
            f"- [{item['kind']}] {item['label']} "
            f"(relevance={item['relevance']:.2f}{hop_str}{source_str})"
        )

    return "\n".join(lines)
