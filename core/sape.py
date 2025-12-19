from __future__ import annotations

import hashlib
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field

from core.fate import FateSeal


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
