"""
Learning Loop Closure Gate

Produces a deterministic proof packet for the board-selected P0 workstream:
candidate -> training batch -> training receipt -> reflex compilation.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.autopoiesis.loop import IntegrationCandidate
from core.orchestration.learning_loop import LearningLoopOrchestrator
from core.sdpo.reflex_bridge import SDPOReflexBridge
from core.sdpo.training.bizra_sdpo_trainer import TrainingResult, TrainingState


@dataclass(frozen=True)
class LearningLoopClosureConfig:
    candidate_count: int
    min_observations: int
    candidate_fitness: float
    candidate_ihsan: float
    candidate_snr: float
    training_ihsan: float
    training_loss: float
    task_description: str
    task_output: str
    min_training_ihsan: float
    required_compiled_reflexes: int
    required_events: list[str]
    giants_protocol: list[str]
    program: dict[str, Any]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_config(path: Path) -> LearningLoopClosureConfig:
    payload = _load_json(path)
    scenario = payload.get("scenario") or {}
    thresholds = payload.get("thresholds") or {}
    return LearningLoopClosureConfig(
        candidate_count=max(1, int(scenario.get("candidate_count", 3))),
        min_observations=max(1, int(scenario.get("min_observations", 3))),
        candidate_fitness=float(scenario.get("candidate_fitness", 0.96)),
        candidate_ihsan=float(scenario.get("candidate_ihsan", 0.99)),
        candidate_snr=float(scenario.get("candidate_snr", 0.92)),
        training_ihsan=float(scenario.get("training_ihsan", 0.99)),
        training_loss=float(scenario.get("training_loss", 0.05)),
        task_description=str(
            scenario.get(
                "task_description", "Close learning loop with verified quality"
            )
        ),
        task_output=str(scenario.get("task_output", "Learning loop candidate output")),
        min_training_ihsan=float(thresholds.get("min_training_ihsan", 0.98)),
        required_compiled_reflexes=max(
            1, int(thresholds.get("required_compiled_reflexes", 1))
        ),
        required_events=[
            str(item)
            for item in (
                thresholds.get("required_events")
                or ["CANDIDATE_ACCEPTED", "TRAINING_COMPLETED", "REFLEX_COMPILED"]
            )
        ],
        giants_protocol=[str(item) for item in (payload.get("giants_protocol") or [])],
        program=payload.get("program") or {},
    )


class DeterministicTrainer:
    """Minimal trainer stub that emits a stable training result."""

    def __init__(self, *, final_ihsan: float, final_loss: float) -> None:
        self._final_ihsan = final_ihsan
        self._final_loss = final_loss

    async def train(
        self, batches: list[Any], resume_from_checkpoint: bool = True
    ) -> TrainingResult:
        total_samples = sum(len(batch) for batch in batches)
        state = TrainingState(
            epoch=1,
            global_step=total_samples,
            best_loss=self._final_loss,
            best_ihsan_score=self._final_ihsan,
            total_samples_processed=total_samples,
        )
        return TrainingResult(
            final_state=state,
            total_epochs_completed=1,
            total_steps=total_samples,
            final_loss=self._final_loss,
            final_ihsan_score=self._final_ihsan,
            training_time_seconds=0.0,
            checkpoints_saved=0,
        )


def _build_candidate(
    cfg: LearningLoopClosureConfig, index: int
) -> IntegrationCandidate:
    genome = SimpleNamespace(
        genome_id=f"loop-candidate-{index:02d}",
        snr_score=cfg.candidate_snr,
        task_description=cfg.task_description,
        task_output=f"{cfg.task_output} #{index + 1}",
        reasoning_steps=["observe", "distill", "compile"],
        improvement_suggestions=["preserve signal", "promote reflex"],
    )
    return IntegrationCandidate(
        genome=genome,
        fitness=cfg.candidate_fitness,
        novelty_score=0.6,
        ihsan_score=cfg.candidate_ihsan,
        recommendation="Integrate",
    )


async def _run_scenario(cfg: LearningLoopClosureConfig) -> dict[str, Any]:
    reflex_bridge = SDPOReflexBridge(min_observations=cfg.min_observations)
    reflex_cache: dict[bytes, Any] = {}
    trainer = DeterministicTrainer(
        final_ihsan=cfg.training_ihsan,
        final_loss=cfg.training_loss,
    )
    orchestrator = LearningLoopOrchestrator(
        enabled=True,
        sdpo_trainer=trainer,
        reflex_bridge=reflex_bridge,
        reflex_cache=reflex_cache,
    )

    for index in range(cfg.candidate_count):
        orchestrator.on_candidate(_build_candidate(cfg, index))

    training_result = await orchestrator.run_training_cycle()
    eligible_candidates = [c.to_dict() for c in reflex_bridge.get_eligible_candidates()]
    compiled_candidates = [c.to_dict() for c in orchestrator.run_compilation_cycle()]
    events = orchestrator.get_events(limit=100)
    event_types = [event["event_type"] for event in events]

    return {
        "training_executed": training_result is not None,
        "training_result": (
            {
                "final_ihsan_score": training_result.final_ihsan_score,
                "final_loss": training_result.final_loss,
                "total_steps": training_result.total_steps,
            }
            if training_result is not None
            else None
        ),
        "eligible_candidates": eligible_candidates,
        "compiled_candidates": compiled_candidates,
        "event_types": event_types,
        "metrics": {
            "candidates_accepted": orchestrator.metrics.candidates_accepted,
            "total_observations": orchestrator.metrics.total_observations,
            "training_runs": orchestrator.metrics.training_runs,
            "avg_training_ihsan": orchestrator.metrics.avg_training_ihsan,
            "reflexes_compiled": orchestrator.metrics.reflexes_compiled,
            "reflex_cache_size": len(reflex_cache),
            "avg_candidate_snr": (
                round(
                    sum(
                        float(candidate["avg_snr"])
                        for candidate in eligible_candidates or compiled_candidates
                    )
                    / max(len(eligible_candidates or compiled_candidates), 1),
                    4,
                )
                if (eligible_candidates or compiled_candidates)
                else 0.0
            ),
        },
    }


def build_report(
    cfg: LearningLoopClosureConfig, scenario: dict[str, Any]
) -> dict[str, Any]:
    metrics = scenario.get("metrics") or {}
    training_result = scenario.get("training_result") or {}
    constraints = {
        "candidates_accepted": int(metrics.get("candidates_accepted", 0))
        >= cfg.candidate_count,
        "training_executed": bool(scenario.get("training_executed", False)),
        "training_ihsan": float(training_result.get("final_ihsan_score", 0.0))
        >= cfg.min_training_ihsan,
        "observation_threshold": int(metrics.get("total_observations", 0))
        >= cfg.min_observations,
        "eligible_candidate": len(scenario.get("eligible_candidates") or [])
        >= cfg.required_compiled_reflexes,
        "compiled_reflexes": int(metrics.get("reflexes_compiled", 0))
        >= cfg.required_compiled_reflexes,
        "required_events": all(
            event_type in (scenario.get("event_types") or [])
            for event_type in cfg.required_events
        ),
    }
    gate_passed = all(constraints.values())
    closure_status = "CLOSED" if gate_passed else "DEGRADED"
    score = round(
        sum(1 for passed in constraints.values() if passed) / len(constraints),
        4,
    )

    receipt_body = {
        "program_id": cfg.program.get("id", "learning_loop_closure_gate"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "candidate_count": cfg.candidate_count,
        "min_observations": cfg.min_observations,
        "constraints": constraints,
        "metrics": metrics,
        "compiled_pattern_ids": [
            candidate.get("pattern_id")
            for candidate in (scenario.get("compiled_candidates") or [])
        ],
    }
    encoded = json.dumps(receipt_body, sort_keys=True, separators=(",", ":"))
    receipt_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    return {
        "program": cfg.program,
        "gate_passed": gate_passed,
        "status": closure_status,
        "closure_status": closure_status,
        "score": score,
        "constraints": constraints,
        "metrics": {
            "candidate_count": cfg.candidate_count,
            "min_observations": cfg.min_observations,
            "candidates_accepted": int(metrics.get("candidates_accepted", 0)),
            "total_observations": int(metrics.get("total_observations", 0)),
            "training_runs": int(metrics.get("training_runs", 0)),
            "training_ihsan": round(
                float(training_result.get("final_ihsan_score", 0.0)),
                4,
            ),
            "compiled_reflexes": int(metrics.get("reflexes_compiled", 0)),
            "reflex_cache_size": int(metrics.get("reflex_cache_size", 0)),
            "avg_candidate_snr": round(float(metrics.get("avg_candidate_snr", 0.0)), 4),
        },
        "standing_on_giants_protocol": cfg.giants_protocol,
        "event_types": scenario.get("event_types") or [],
        "eligible_candidates": scenario.get("eligible_candidates") or [],
        "compiled_candidates": scenario.get("compiled_candidates") or [],
        "receipt": {
            "receipt_id": f"llcg-{receipt_hash[:16]}",
            "receipt_hash": receipt_hash,
            "status": closure_status,
        },
    }


def render_markdown(report: dict[str, Any]) -> str:
    metrics = report.get("metrics") or {}
    lines = [
        "# BIZRA Learning Loop Closure Gate",
        "",
        f"- Gate passed: `{report.get('gate_passed', False)}`",
        f"- Score: `{report.get('score', 0.0):.4f}`",
        f"- Closure status: `{report.get('closure_status', 'DEGRADED')}`",
        f"- Candidates accepted: `{metrics.get('candidates_accepted', 0)}`",
        f"- Observations recorded: `{metrics.get('total_observations', 0)}`",
        f"- Training Ihsan: `{metrics.get('training_ihsan', 0.0):.4f}`",
        f"- Compiled reflexes: `{metrics.get('compiled_reflexes', 0)}`",
        f"- Avg candidate SNR: `{metrics.get('avg_candidate_snr', 0.0):.4f}`",
        "",
        "## Receipt",
        "",
        f"- Receipt ID: `{(report.get('receipt') or {}).get('receipt_id', '')}`",
        f"- Receipt Hash: `{(report.get('receipt') or {}).get('receipt_hash', '')}`",
        "",
        "## Events",
        "",
    ]
    for event_type in report.get("event_types") or []:
        lines.append(f"- {event_type}")
    return "\n".join(lines) + "\n"


def _write_github_outputs(path: Path, report: dict[str, Any]) -> None:
    metrics = report.get("metrics") or {}
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"learning_loop_closure_passed={str(report.get('gate_passed', False)).lower()}\n"
        )
        fh.write(
            f"learning_loop_closure_status={report.get('closure_status', 'DEGRADED')}\n"
        )
        fh.write(f"learning_loop_closure_score={report.get('score', 0.0)}\n")
        fh.write(
            f"learning_loop_compiled_reflexes={metrics.get('compiled_reflexes', 0)}\n"
        )
        fh.write(f"learning_loop_training_ihsan={metrics.get('training_ihsan', 0.0)}\n")


def run_learning_loop_closure_gate(
    *,
    config_path: Path,
    report_path: Path | None = None,
    markdown_report_path: Path | None = None,
    github_output: Path | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    scenario = asyncio.run(_run_scenario(cfg))
    report = build_report(cfg, scenario)
    encoded = json.dumps(report, indent=2)

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(encoded, encoding="utf-8")
    if markdown_report_path is not None:
        markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_report_path.write_text(render_markdown(report), encoding="utf-8")
    if github_output is not None:
        _write_github_outputs(github_output, report)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run learning loop closure gate.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/learning_loop_closure_gate.json"),
        help="Learning loop closure gate config JSON path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional output JSON report path.",
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=None,
        help="Optional output Markdown report path.",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        default=None,
        help="Optional GitHub outputs file path.",
    )
    args = parser.parse_args()

    report = run_learning_loop_closure_gate(
        config_path=args.config,
        report_path=args.report,
        markdown_report_path=args.markdown_report,
        github_output=args.github_output,
    )
    print(json.dumps(report, indent=2))
    return 0 if report.get("gate_passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
