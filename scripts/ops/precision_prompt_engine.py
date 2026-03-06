"""
Precision-Guided Prompt Construction Engine.

Builds a multi-tier prompt pack to:
1. Probe rarely fired circuits (deep attention, low-probability paths)
2. Unlock latent symbolic-neural hybrids
3. Trigger higher-order abstraction and meta-reflection
4. Surface logic-creative tension spaces

Outputs:
- Composite prompt (ready to run in LLMs)
- Tier-level prompt sections
- Graph-of-Thought topology
- SNR diagnostics for prompt quality
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ENGINE_VERSION = "1.0.0"


@dataclass(frozen=True)
class PromptRequest:
    intent: str
    context: dict[str, Any]
    symbolic_neural: bool
    creativity: float
    rigor: float


@dataclass(frozen=True)
class EngineConfig:
    ihsan_floor: float
    snr_target: float
    constraints: list[str]
    giants_protocol: list[str]
    snr_signal_weights: dict[str, float]
    snr_noise_weights: dict[str, float]


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _normalize_weights(raw: dict[str, Any], defaults: dict[str, float]) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for key, default in defaults.items():
        value = raw.get(key, default)
        try:
            parsed[key] = max(0.0, float(value))
        except (TypeError, ValueError):
            parsed[key] = default

    total = sum(parsed.values())
    if total <= 0.0:
        return defaults.copy()
    return {k: v / total for k, v in parsed.items()}


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_config(path: Path) -> EngineConfig:
    payload = _load_yaml(path)

    defaults_signal = {
        "specificity": 0.30,
        "context_density": 0.20,
        "tier_coverage": 0.25,
        "constraint_binding": 0.15,
        "evidence_binding": 0.10,
    }
    defaults_noise = {
        "ambiguity": 0.35,
        "sparse_context": 0.25,
        "underconstrained": 0.20,
        "coverage_gap": 0.20,
    }

    signal_weights = _normalize_weights(
        (payload.get("snr_weights") or {}).get("signal") or {},
        defaults_signal,
    )
    noise_weights = _normalize_weights(
        (payload.get("snr_weights") or {}).get("noise") or {},
        defaults_noise,
    )

    ihsan_floor = _clamp(float(payload.get("ihsan_floor", 0.95)), 0.0, 1.0)
    snr_target = _clamp(float(payload.get("snr_target", 0.90)), 0.0, 1.0)
    constraints = [str(x) for x in payload.get("constraints") or []]
    if not constraints:
        constraints = [
            "Bind every non-trivial claim to explicit evidence or assumptions.",
            "Separate facts, hypotheses, and open questions.",
            "Reject unsafe or unverifiable actions.",
        ]

    giants = [str(x) for x in payload.get("giants_protocol") or []]
    if not giants:
        giants = [
            "Shannon:SNR maximization",
            "Boyd:OODA fast loops",
            "Lamport:deterministic ordering",
            "Deming:PDCA quality ratchet",
            "Al-Ghazali:Ihsan as hard floor",
        ]

    return EngineConfig(
        ihsan_floor=ihsan_floor,
        snr_target=snr_target,
        constraints=constraints,
        giants_protocol=giants,
        snr_signal_weights=signal_weights,
        snr_noise_weights=noise_weights,
    )


def _format_context(context: dict[str, Any]) -> str:
    if not context:
        return "{}"
    return json.dumps(context, indent=2, ensure_ascii=False, sort_keys=True)


def _section_rare_circuits(request: PromptRequest) -> str:
    return (
        "Tier-1 | Rare-Circuit Probe\n"
        "- Generate 3 low-probability interpretations of the intent.\n"
        "- For each interpretation, produce one disconfirming test.\n"
        "- Execute a two-pass deep-attention block:\n"
        "  Pass A: isolate hidden assumptions and silent constraints.\n"
        "  Pass B: re-rank hypotheses by falsifiability and expected impact.\n"
        "- Output a shortlist of 2 candidate solution paths with risk tags."
    )


def _section_symbolic_neural(request: PromptRequest) -> str:
    if not request.symbolic_neural:
        return (
            "Tier-2 | Symbolic-Neural Hybrid (disabled)\n"
            "- Skip symbolic-neural fusion and use strict analytical path."
        )
    return (
        "Tier-2 | Symbolic-Neural Hybrid\n"
        "- Build a symbolic frame: constraints, invariants, and objective function.\n"
        "- Build a neural frame: latent patterns, analogies, and semantic priors.\n"
        "- Reconcile both frames into a single executable policy.\n"
        "- Explicitly list where symbolic logic overruled intuition and vice versa."
    )


def _section_abstraction(request: PromptRequest) -> str:
    return (
        "Tier-3 | Higher-Order Abstraction\n"
        "- Lift the problem through 3 abstraction layers:\n"
        "  L0 concrete task -> L1 system pattern -> L2 governing principle.\n"
        "- Perform meta-reflection: what would invalidate the current frame?\n"
        "- Run a dialectic: thesis, antithesis, synthesis.\n"
        "- Return one robust synthesis and one fallback synthesis."
    )


def _section_tension(request: PromptRequest) -> str:
    creativity = f"{request.creativity:.2f}"
    rigor = f"{request.rigor:.2f}"
    return (
        "Tier-4 | Logic-Creative Tension Space\n"
        f"- Creativity coefficient: {creativity}\n"
        f"- Rigor coefficient: {rigor}\n"
        "- Produce 2 unconventional designs that still satisfy hard constraints.\n"
        "- Stress-test each design with adversarial counterexamples.\n"
        "- Keep only options that survive verification and preserve reversibility."
    )


def _build_graph_of_thought(request: PromptRequest) -> dict[str, Any]:
    nodes = [
        {"id": "intent", "label": "Intent Parse"},
        {"id": "rare_probe", "label": "Rare-Circuit Probe"},
        {"id": "symbolic_neural", "label": "Symbolic-Neural Fusion"},
        {"id": "abstraction", "label": "Higher-Order Abstraction"},
        {"id": "tension", "label": "Logic-Creative Tension"},
        {"id": "decision", "label": "Decision + Verification"},
    ]
    edges = [
        {"from": "intent", "to": "rare_probe"},
        {"from": "rare_probe", "to": "symbolic_neural"},
        {"from": "symbolic_neural", "to": "abstraction"},
        {"from": "abstraction", "to": "tension"},
        {"from": "tension", "to": "decision"},
    ]
    if not request.symbolic_neural:
        edges.append({"from": "rare_probe", "to": "abstraction"})
    return {"nodes": nodes, "edges": edges}


def _token_count(text: str) -> int:
    return len([w for w in text.split() if w.strip()])


def _compute_snr(
    request: PromptRequest,
    config: EngineConfig,
    sections: list[str],
) -> dict[str, Any]:
    intent_tokens = _token_count(request.intent)
    context_keys = len(request.context.keys())
    constraints_count = len(config.constraints)
    tier_coverage = 1.0 if request.symbolic_neural else 0.75
    evidence_binding = (
        1.0
        if any(k in request.context for k in ("sources", "evidence", "citations", "refs"))
        else 0.6
    )

    signal_components = {
        "specificity": _clamp(intent_tokens / 18.0, 0.0, 1.0),
        "context_density": _clamp(context_keys / 6.0, 0.0, 1.0),
        "tier_coverage": tier_coverage,
        "constraint_binding": _clamp(constraints_count / 8.0, 0.0, 1.0),
        "evidence_binding": evidence_binding,
    }

    noise_components = {
        "ambiguity": 1.0 - signal_components["specificity"],
        "sparse_context": 1.0 - signal_components["context_density"],
        "underconstrained": max(0.0, 0.7 - signal_components["constraint_binding"]),
        "coverage_gap": 1.0 - signal_components["tier_coverage"],
    }

    signal = sum(
        config.snr_signal_weights.get(k, 0.0) * v for k, v in signal_components.items()
    )
    noise = sum(
        config.snr_noise_weights.get(k, 0.0) * v for k, v in noise_components.items()
    )
    raw = signal / (noise + 0.10)
    normalized = raw / (1.0 + raw)

    return {
        "signal": round(signal, 4),
        "noise": round(noise, 4),
        "raw": round(raw, 4),
        "normalized": round(normalized, 4),
        "target": config.snr_target,
        "meets_target": normalized >= config.snr_target,
        "signal_components": signal_components,
        "noise_components": noise_components,
        "tiers_emitted": len(sections),
    }


def _derive_snr_tuning_actions(snr: dict[str, Any]) -> list[str]:
    signal_components = snr.get("signal_components") or {}
    noise_components = snr.get("noise_components") or {}
    actions: list[str] = []

    if float(signal_components.get("specificity", 0.0)) < 0.70:
        actions.append(
            "Increase intent specificity: include objective, constraints, and acceptance criteria."
        )
    if float(signal_components.get("context_density", 0.0)) < 0.70:
        actions.append(
            "Add richer context: known facts, environment limits, and expected outputs."
        )
    if float(signal_components.get("evidence_binding", 0.0)) < 0.80:
        actions.append(
            "Attach explicit evidence references (sources/citations) for key claims."
        )
    if float(noise_components.get("underconstrained", 0.0)) > 0.0:
        actions.append("Add hard constraints to reduce underconstrained solution space.")
    if float(noise_components.get("ambiguity", 0.0)) > 0.35:
        actions.append("Clarify ambiguous terms and remove vague language.")
    if float(noise_components.get("coverage_gap", 0.0)) > 0.0:
        actions.append("Enable all tiers to maximize reasoning coverage.")

    if not actions:
        actions.append("SNR is stable; keep constraints and evidence bindings unchanged.")
    return actions


def _build_composite_prompt(
    request: PromptRequest,
    config: EngineConfig,
    sections: list[str],
) -> str:
    constraints_block = "\n".join(f"- {item}" for item in config.constraints)
    giants_block = "\n".join(f"- {item}" for item in config.giants_protocol)
    tiers_block = "\n\n".join(sections)

    return (
        "SYSTEM CONTRACT\n"
        f"- Ihsan floor: {config.ihsan_floor:.2f}\n"
        "- Truthfulness: Separate facts vs assumptions.\n"
        "- Safety: Refuse unsafe, unverifiable, or harmful actions.\n"
        "\n"
        "STANDING ON THE SHOULDER OF GIANTS PROTOCOL\n"
        f"{giants_block}\n"
        "\n"
        "HARD CONSTRAINTS\n"
        f"{constraints_block}\n"
        "\n"
        "USER INTENT\n"
        f"{request.intent}\n"
        "\n"
        "CONTEXT SNAPSHOT\n"
        f"{_format_context(request.context)}\n"
        "\n"
        "EXECUTION TIERS\n"
        f"{tiers_block}\n"
        "\n"
        "OUTPUT CONTRACT\n"
        "- Section 1: Assumptions (explicit)\n"
        "- Section 2: Competing hypotheses + falsification checks\n"
        "- Section 3: Final synthesis (actionable)\n"
        "- Section 4: Risks + rollback plan\n"
        "- Section 5: SNR self-score + confidence\n"
        "- Section 6: Next best action in <= 3 steps"
    )


def build_prompt_artifact(request: PromptRequest, config: EngineConfig) -> dict[str, Any]:
    sections = [
        _section_rare_circuits(request),
        _section_symbolic_neural(request),
        _section_abstraction(request),
        _section_tension(request),
    ]

    got = _build_graph_of_thought(request)
    snr = _compute_snr(request, config, sections)
    snr_tuning_actions = _derive_snr_tuning_actions(snr)
    prompt = _build_composite_prompt(request, config, sections)

    return {
        "engine_version": ENGINE_VERSION,
        "request": {
            "intent": request.intent,
            "context": request.context,
            "symbolic_neural": request.symbolic_neural,
            "creativity": request.creativity,
            "rigor": request.rigor,
        },
        "graph_of_thought": got,
        "snr": snr,
        "snr_tuning_actions": snr_tuning_actions,
        "tiers": sections,
        "composite_prompt": prompt,
    }


def _load_context(context_arg: str | None, context_file: Path | None) -> dict[str, Any]:
    if context_file is not None:
        payload = json.loads(context_file.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    if context_arg is None or not context_arg.strip():
        return {}
    payload = json.loads(context_arg)
    return payload if isinstance(payload, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build precision-guided multi-tier prompt artifacts."
    )
    parser.add_argument("--intent", required=True, help="Primary user intent.")
    parser.add_argument(
        "--context",
        default="{}",
        help="Optional JSON object string for context (ignored if --context-file is used).",
    )
    parser.add_argument(
        "--context-file",
        type=Path,
        default=None,
        help="Optional JSON file path containing context object.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/precision_prompt_engine.yaml"),
        help="Optional YAML config path.",
    )
    parser.add_argument(
        "--symbolic-neural",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable symbolic-neural hybrid tier.",
    )
    parser.add_argument(
        "--creativity",
        type=float,
        default=0.62,
        help="Creativity coefficient [0.0,1.0].",
    )
    parser.add_argument(
        "--rigor",
        type=float,
        default=0.88,
        help="Rigor coefficient [0.0,1.0].",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output file for JSON artifact.",
    )
    args = parser.parse_args()

    context = _load_context(args.context, args.context_file)
    config = load_config(args.config)
    request = PromptRequest(
        intent=args.intent.strip(),
        context=context,
        symbolic_neural=bool(args.symbolic_neural),
        creativity=_clamp(float(args.creativity), 0.0, 1.0),
        rigor=_clamp(float(args.rigor), 0.0, 1.0),
    )
    artifact = build_prompt_artifact(request, config)

    output = json.dumps(artifact, indent=2, ensure_ascii=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
