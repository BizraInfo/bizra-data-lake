"""Giants Protocol scoring engine.

Turns external open-source "giants" patterns into a prioritized local backlog
using SAPE/Ihsan/SNR-aligned gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class Gate:
    snr_min: float
    ihsan_min: float


@dataclass(frozen=True)
class Giant:
    id: str
    name: str
    repo: str
    strengths: Tuple[str, ...]
    risk_flags: Tuple[str, ...]
    adoption_weight: float
    integration_cost: float
    ethical_alignment: float


@dataclass(frozen=True)
class OpportunityTemplate:
    id: str
    title: str
    description: str
    target_modules: Tuple[str, ...]
    requires: Tuple[str, ...]
    novelty: float
    impact: float
    complexity: float


@dataclass(frozen=True)
class OpportunityScore:
    opportunity_id: str
    title: str
    snr_score: float
    ihsan_score: float
    priority_score: float
    status: str
    rationale: Tuple[str, ...]
    target_modules: Tuple[str, ...]
    required_giants: Tuple[str, ...]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def load_registry(path: str | Path) -> Dict[str, object]:
    registry_path = Path(path)
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    return payload


def _parse_gate(raw: Dict[str, float]) -> Gate:
    return Gate(
        snr_min=float(raw.get("snr_min", 0.82)),
        ihsan_min=float(raw.get("ihsan_min", 0.88)),
    )


def _parse_giants(raw_giants: Iterable[Dict[str, object]]) -> Dict[str, Giant]:
    giants: Dict[str, Giant] = {}
    for item in raw_giants:
        giant = Giant(
            id=str(item["id"]),
            name=str(item["name"]),
            repo=str(item["repo"]),
            strengths=tuple(item.get("strengths", [])),
            risk_flags=tuple(item.get("risk_flags", [])),
            adoption_weight=float(item.get("adoption_weight", 0.5)),
            integration_cost=float(item.get("integration_cost", 0.5)),
            ethical_alignment=float(item.get("ethical_alignment", 0.9)),
        )
        giants[giant.id] = giant
    return giants


def _parse_opportunities(
    raw_ops: Iterable[Dict[str, object]],
) -> List[OpportunityTemplate]:
    ops: List[OpportunityTemplate] = []
    for item in raw_ops:
        ops.append(
            OpportunityTemplate(
                id=str(item["id"]),
                title=str(item["title"]),
                description=str(item["description"]),
                target_modules=tuple(item.get("target_modules", [])),
                requires=tuple(item.get("requires", [])),
                novelty=float(item.get("novelty", 0.5)),
                impact=float(item.get("impact", 0.5)),
                complexity=float(item.get("complexity", 0.5)),
            )
        )
    return ops


def _risk_penalty(giants: Sequence[Giant]) -> float:
    if not giants:
        return 0.5
    return _clamp(sum(min(len(g.risk_flags), 5) / 5.0 for g in giants) / len(giants))


def score_opportunity(
    template: OpportunityTemplate,
    giant_index: Dict[str, Giant],
    scout_gate: Gate,
    production_gate: Gate,
) -> OpportunityScore:
    missing = [gid for gid in template.requires if gid not in giant_index]
    if missing:
        raise ValueError(
            f"Opportunity '{template.id}' references unknown giants: {', '.join(missing)}"
        )

    required = [giant_index[gid] for gid in template.requires]
    adoption = sum(g.adoption_weight for g in required) / max(len(required), 1)
    integration_cost = sum(g.integration_cost for g in required) / max(len(required), 1)
    ethics = sum(g.ethical_alignment for g in required) / max(len(required), 1)
    risk_penalty = _risk_penalty(required)

    snr_score = _clamp(
        0.45 * template.impact
        + 0.20 * template.novelty
        + 0.20 * adoption
        + 0.15 * (1.0 - integration_cost)
    )
    ihsan_score = _clamp(
        0.50 * ethics + 0.25 * (1.0 - risk_penalty) + 0.25 * (1.0 - template.complexity)
    )
    priority_score = _clamp(
        0.50 * snr_score + 0.35 * ihsan_score + 0.15 * (1.0 - template.complexity)
    )

    rationale: List[str] = []
    if snr_score < production_gate.snr_min:
        rationale.append("Raise benchmarked throughput/quality before production gate.")
    if ihsan_score < production_gate.ihsan_min:
        rationale.append("Strengthen safety, governance, and evidence traceability.")
    if integration_cost > 0.6:
        rationale.append("Break into smaller adapters to reduce integration drag.")

    if (
        snr_score >= production_gate.snr_min
        and ihsan_score >= production_gate.ihsan_min
    ):
        status = "production-ready"
    elif snr_score >= scout_gate.snr_min and ihsan_score >= scout_gate.ihsan_min:
        status = "pilot-ready"
    else:
        status = "research-only"

    if not rationale:
        rationale.append("Meets production quality gates.")

    return OpportunityScore(
        opportunity_id=template.id,
        title=template.title,
        snr_score=round(snr_score, 4),
        ihsan_score=round(ihsan_score, 4),
        priority_score=round(priority_score, 4),
        status=status,
        rationale=tuple(rationale),
        target_modules=template.target_modules,
        required_giants=template.requires,
    )


def build_backlog(path: str | Path, top_n: int = 5) -> Dict[str, object]:
    payload = load_registry(path)
    meta = payload.get("meta", {})
    gates = payload.get("gates", {})

    scout_gate = _parse_gate(gates.get("scout", {}))
    production_gate = _parse_gate(gates.get("production", {}))
    giant_index = _parse_giants(payload.get("giants", []))
    opportunities = _parse_opportunities(payload.get("opportunity_templates", []))

    scored = [
        score_opportunity(t, giant_index, scout_gate, production_gate)
        for t in opportunities
    ]
    scored.sort(key=lambda s: s.priority_score, reverse=True)

    return {
        "meta": meta,
        "gates": {
            "scout": asdict(scout_gate),
            "production": asdict(production_gate),
        },
        "top": [s.to_dict() for s in scored[: max(top_n, 0)]],
        "all": [s.to_dict() for s in scored],
    }


def render_markdown(backlog: Dict[str, object]) -> str:
    lines: List[str] = []
    lines.append("# Giants Protocol Backlog")
    lines.append("")

    meta = backlog.get("meta", {})
    lines.append(f"- profile: `{meta.get('profile', 'unknown')}`")
    lines.append(f"- version: `{meta.get('version', 'unknown')}`")
    lines.append("")

    for item in backlog.get("top", []):
        lines.append(f"## {item['title']} ({item['status']})")
        lines.append(
            f"- score: priority `{item['priority_score']}` | snr `{item['snr_score']}` | ihsan `{item['ihsan_score']}`"
        )
        lines.append(f"- giants: `{', '.join(item['required_giants'])}`")
        lines.append(f"- target modules: `{', '.join(item['target_modules'])}`")
        for reason in item.get("rationale", []):
            lines.append(f"- action: {reason}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"
