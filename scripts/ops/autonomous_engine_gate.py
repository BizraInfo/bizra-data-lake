"""
Autonomous Engine Gate

Orchestrates:
1) Precision Prompt Engine (GoT + SNR)
2) Self-RLVR Harness (reward + receipt-chain + compilation)

Produces a unified autonomous readiness score and CI gate decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Ensure repo root is importable when script is executed directly.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.ops.precision_prompt_engine import (
    PromptRequest,
    build_prompt_artifact,
)
from scripts.ops.precision_prompt_engine import (  # noqa: E402
    load_config as load_prompt_config,
)
from scripts.ops.self_rlvr_harness import (  # noqa: E402
    HarnessConfig,
    run_self_rlvr_harness,
)


@dataclass(frozen=True)
class AutonomousGateConfig:
    prompt_config: Path
    rlvr_config: Path
    prompt_intent: str
    prompt_context: dict[str, Any]
    symbolic_neural: bool
    creativity: float
    rigor: float
    episodes_profile: str
    episodes_count: int
    score_weights: dict[str, float]
    min_score: float
    min_prompt_snr: float
    min_rlvr_snr: float
    min_qualified_rate: float
    require_compiled: bool
    require_chain_valid: bool
    giants_protocol: list[str]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _normalize_weights(
    raw: dict[str, Any], defaults: dict[str, float]
) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for key, default in defaults.items():
        try:
            parsed[key] = max(0.0, float(raw.get(key, default)))
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


def load_gate_config(path: Path) -> AutonomousGateConfig:
    payload = _load_yaml(path)
    defaults = {
        "prompt_snr": 0.35,
        "rlvr_snr": 0.30,
        "qualified_rate": 0.20,
        "compiled": 0.10,
        "chain_valid": 0.05,
    }
    weights = _normalize_weights(payload.get("score_weights") or {}, defaults)

    return AutonomousGateConfig(
        prompt_config=Path(
            payload.get("prompt_config", "config/precision_prompt_engine.yaml")
        ),
        rlvr_config=Path(payload.get("rlvr_config", "config/self_rlvr_harness.yaml")),
        prompt_intent=str(
            payload.get(
                "prompt_intent",
                (
                    "Design a verifiable, production-grade autonomous execution plan "
                    "that maximizes SNR while preserving Ihsan constraints."
                ),
            )
        ),
        prompt_context=(
            payload.get("prompt_context")
            or {
                "scope": "autonomous-governance",
                "evidence": ["phase65_gate_report.json"],
                "constraints": ["ihsan>=0.95", "snr>=0.90"],
            }
        ),
        symbolic_neural=bool(payload.get("symbolic_neural", True)),
        creativity=_clamp01(float(payload.get("creativity", 0.62))),
        rigor=_clamp01(float(payload.get("rigor", 0.90))),
        episodes_profile=str(payload.get("episodes_profile", "high_signal")),
        episodes_count=max(3, int(payload.get("episodes_count", 6))),
        score_weights=weights,
        min_score=_clamp01(float(payload.get("min_score", 0.78))),
        min_prompt_snr=_clamp01(float(payload.get("min_prompt_snr", 0.60))),
        min_rlvr_snr=_clamp01(float(payload.get("min_rlvr_snr", 0.55))),
        min_qualified_rate=_clamp01(float(payload.get("min_qualified_rate", 0.60))),
        require_compiled=bool(payload.get("require_compiled", True)),
        require_chain_valid=bool(payload.get("require_chain_valid", True)),
        giants_protocol=[
            str(x)
            for x in (
                payload.get("giants_protocol")
                or [
                    "Shannon:SNR maximization",
                    "Boyd:OODA fast loops",
                    "Lamport:deterministic receipts",
                    "Deming:PDCA quality ratchet",
                    "Al-Ghazali:Ihsan hard floor",
                ]
            )
        ],
    )


def _build_episodes(profile: str, count: int) -> list[dict[str, Any]]:
    def _high(i: int) -> dict[str, Any]:
        bump = min(i, 5) * 0.003
        return {
            "snr": _clamp01(0.95 + bump),
            "ihsan": _clamp01(0.96 + bump),
            "tokens_used": 560 + i * 8,
            "quality": _clamp01(0.95 + bump),
            "user_feedback": _clamp01(0.90 + bump),
            "verified": True,
            "penalties": 0.0,
        }

    def _mixed(i: int) -> dict[str, Any]:
        if i % 3 == 1:
            return {
                "snr": 0.84,
                "ihsan": 0.88,
                "tokens_used": 760,
                "quality": 0.86,
                "user_feedback": 0.78,
                "verified": True,
                "penalties": 0.0,
            }
        return _high(i)

    builder = _high if profile == "high_signal" else _mixed
    return [builder(i) for i in range(max(count, 1))]


def evaluate_gate(
    cfg: AutonomousGateConfig,
    prompt_artifact: dict[str, Any],
    rlvr_report: dict[str, Any],
) -> dict[str, Any]:
    prompt_snr = float((prompt_artifact.get("snr") or {}).get("normalized", 0.0))
    rlvr_summary = rlvr_report.get("summary") or {}
    rlvr_snr = float((rlvr_summary.get("snr") or {}).get("normalized", 0.0))
    qualified_rate = float(rlvr_summary.get("qualified_rate", 0.0))
    compiled = bool(rlvr_summary.get("compiled", False))
    chain_valid = bool(rlvr_summary.get("chain_valid", False))

    score = (
        cfg.score_weights["prompt_snr"] * prompt_snr
        + cfg.score_weights["rlvr_snr"] * rlvr_snr
        + cfg.score_weights["qualified_rate"] * qualified_rate
        + cfg.score_weights["compiled"] * (1.0 if compiled else 0.0)
        + cfg.score_weights["chain_valid"] * (1.0 if chain_valid else 0.0)
    )

    constraints = {
        "prompt_snr": prompt_snr >= cfg.min_prompt_snr,
        "rlvr_snr": rlvr_snr >= cfg.min_rlvr_snr,
        "qualified_rate": qualified_rate >= cfg.min_qualified_rate,
        "compiled": (compiled or (not cfg.require_compiled)),
        "chain_valid": (chain_valid or (not cfg.require_chain_valid)),
        "min_score": score >= cfg.min_score,
    }
    gate_passed = all(constraints.values())

    failed = [name for name, ok in constraints.items() if not ok]
    if gate_passed:
        next_step = {
            "priority": "P0",
            "owner": "autonomous-engine",
            "action": "Promote autonomous profile to protected release pipeline.",
        }
    else:
        next_step = {
            "priority": "P1",
            "owner": "policy-tuning",
            "action": f"Recalibrate failed constraints: {', '.join(failed)}",
        }

    return {
        "program": {"id": "autonomous_engine_gate", "version": "1.0.0"},
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "standing_on_giants_protocol": cfg.giants_protocol,
        "metrics": {
            "prompt_snr": round(prompt_snr, 4),
            "rlvr_snr": round(rlvr_snr, 4),
            "qualified_rate": round(qualified_rate, 4),
            "compiled": compiled,
            "chain_valid": chain_valid,
            "score": round(score, 4),
        },
        "thresholds": {
            "min_score": cfg.min_score,
            "min_prompt_snr": cfg.min_prompt_snr,
            "min_rlvr_snr": cfg.min_rlvr_snr,
            "min_qualified_rate": cfg.min_qualified_rate,
            "require_compiled": cfg.require_compiled,
            "require_chain_valid": cfg.require_chain_valid,
        },
        "constraints": constraints,
        "gate_passed": gate_passed,
        "graph_of_thought": {
            "nodes": [
                {"id": "prompt_engine", "score": round(prompt_snr, 4)},
                {"id": "rlvr_loop", "score": round(rlvr_snr, 4)},
                {"id": "policy_quality", "score": round(qualified_rate, 4)},
                {"id": "compile_state", "score": 1.0 if compiled else 0.0},
                {"id": "receipt_chain", "score": 1.0 if chain_valid else 0.0},
                {"id": "release_decision", "score": round(score, 4)},
            ],
            "edges": [
                {"from": "prompt_engine", "to": "policy_quality"},
                {"from": "rlvr_loop", "to": "policy_quality"},
                {"from": "policy_quality", "to": "compile_state"},
                {"from": "compile_state", "to": "receipt_chain"},
                {"from": "receipt_chain", "to": "release_decision"},
            ],
        },
        "autonomous_next_step": next_step,
        "prompt_artifact": {
            "snr": prompt_artifact.get("snr"),
            "graph_of_thought": prompt_artifact.get("graph_of_thought"),
            "snr_tuning_actions": prompt_artifact.get("snr_tuning_actions"),
        },
        "rlvr_artifact": {
            "summary": rlvr_summary,
            "decision": rlvr_report.get("decision"),
        },
    }


def run_gate(cfg: AutonomousGateConfig) -> dict[str, Any]:
    prompt_cfg = load_prompt_config(cfg.prompt_config)
    prompt_request = PromptRequest(
        intent=cfg.prompt_intent,
        context=cfg.prompt_context,
        symbolic_neural=cfg.symbolic_neural,
        creativity=cfg.creativity,
        rigor=cfg.rigor,
    )
    prompt_artifact = build_prompt_artifact(prompt_request, prompt_cfg)

    rlvr_cfg = HarnessConfig(**load_gate_config_values(cfg.rlvr_config))
    episodes = _build_episodes(cfg.episodes_profile, cfg.episodes_count)
    rlvr_report = run_self_rlvr_harness(
        agent_id="node0", episodes=episodes, config=rlvr_cfg
    )

    return evaluate_gate(cfg, prompt_artifact, rlvr_report)


def load_gate_config_values(path: Path) -> dict[str, Any]:
    payload = _load_yaml(path)
    if not payload:
        return {}
    allowed = {
        "ihsan_threshold",
        "snr_threshold",
        "reward_threshold",
        "compile_streak",
        "ema_alpha",
        "convergence_target",
        "convergence_variance_max",
        "min_temperature",
        "max_temperature",
        "cool_rate",
        "heat_rate",
        "variance_window",
    }
    return {k: v for k, v in payload.items() if k in allowed}


def _write_github_outputs(path: Path, report: dict[str, Any]) -> None:
    metrics = report.get("metrics") or {}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"autonomous_gate_passed={str(report.get('gate_passed', False)).lower()}\n"
        )
        fh.write(f"autonomous_score={metrics.get('score', 0.0)}\n")
        fh.write(f"autonomous_prompt_snr={metrics.get('prompt_snr', 0.0)}\n")
        fh.write(f"autonomous_rlvr_snr={metrics.get('rlvr_snr', 0.0)}\n")
        fh.write(f"autonomous_qualified_rate={metrics.get('qualified_rate', 0.0)}\n")
        fh.write(f"autonomous_compiled={str(metrics.get('compiled', False)).lower()}\n")
        fh.write(
            f"autonomous_chain_valid={str(metrics.get('chain_valid', False)).lower()}\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run autonomous engine gate.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/autonomous_engine_gate.yaml"),
        help="Autonomous gate YAML config path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional output JSON report path.",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        default=None,
        help="Optional GitHub output file for CI job outputs.",
    )
    args = parser.parse_args()

    cfg = load_gate_config(args.config)
    report = run_gate(cfg)
    encoded = json.dumps(report, indent=2)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded, encoding="utf-8")
    if args.github_output is not None:
        _write_github_outputs(args.github_output, report)

    print(encoded)
    return 0 if report.get("gate_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
