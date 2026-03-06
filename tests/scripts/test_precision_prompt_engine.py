from __future__ import annotations

from pathlib import Path

import yaml

from scripts.ops.precision_prompt_engine import (
    PromptRequest,
    build_prompt_artifact,
    load_config,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_precision_prompt_engine_builds_all_tiers(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    _write(
        cfg_path,
        yaml.safe_dump(
            {
                "ihsan_floor": 0.95,
                "snr_target": 0.90,
                "constraints": ["bind claims", "show assumptions"],
                "giants_protocol": ["Shannon:SNR maximization"],
            },
            sort_keys=False,
        ),
    )

    config = load_config(cfg_path)
    req = PromptRequest(
        intent="Design an elite full-stack rollout plan",
        context={"evidence": ["report-a", "report-b"], "risk": "medium"},
        symbolic_neural=True,
        creativity=0.62,
        rigor=0.88,
    )
    artifact = build_prompt_artifact(req, config)
    assert len(artifact["tiers"]) == 4
    assert "Tier-1 | Rare-Circuit Probe" in artifact["tiers"][0]
    assert "Tier-2 | Symbolic-Neural Hybrid" in artifact["tiers"][1]
    assert "Tier-3 | Higher-Order Abstraction" in artifact["tiers"][2]
    assert "Tier-4 | Logic-Creative Tension Space" in artifact["tiers"][3]


def test_precision_prompt_engine_marks_symbolic_neural_disabled(tmp_path: Path) -> None:
    config = load_config(tmp_path / "missing.yaml")
    req = PromptRequest(
        intent="Plan architecture evolution",
        context={},
        symbolic_neural=False,
        creativity=0.55,
        rigor=0.9,
    )
    artifact = build_prompt_artifact(req, config)
    assert "disabled" in artifact["tiers"][1]
    assert artifact["graph_of_thought"]["edges"][-1] == {
        "from": "rare_probe",
        "to": "abstraction",
    }


def test_precision_prompt_engine_snr_range_and_fields(tmp_path: Path) -> None:
    config = load_config(tmp_path / "missing.yaml")
    req = PromptRequest(
        intent="Optimize CI/CD and quality gates",
        context={"sources": ["ci.yml", "gate_report.json"], "constraints": ["latency"]},
        symbolic_neural=True,
        creativity=0.6,
        rigor=0.92,
    )
    artifact = build_prompt_artifact(req, config)
    snr = artifact["snr"]
    assert 0.0 <= snr["signal"] <= 1.0
    assert 0.0 <= snr["noise"] <= 1.0
    assert 0.0 <= snr["normalized"] <= 1.0
    assert "signal_components" in snr
    assert "noise_components" in snr
    assert artifact["snr_tuning_actions"]


def test_precision_prompt_engine_normalizes_bad_weights(tmp_path: Path) -> None:
    cfg_path = tmp_path / "bad-weights.yaml"
    _write(
        cfg_path,
        yaml.safe_dump(
            {
                "snr_weights": {
                    "signal": {"specificity": "bad", "context_density": -1},
                    "noise": {"ambiguity": "bad"},
                }
            },
            sort_keys=False,
        ),
    )
    config = load_config(cfg_path)
    assert abs(sum(config.snr_signal_weights.values()) - 1.0) < 1e-9
    assert abs(sum(config.snr_noise_weights.values()) - 1.0) < 1e-9
    assert all(v >= 0.0 for v in config.snr_signal_weights.values())
    assert all(v >= 0.0 for v in config.snr_noise_weights.values())


def test_precision_prompt_engine_composite_prompt_contract(tmp_path: Path) -> None:
    config = load_config(tmp_path / "missing.yaml")
    req = PromptRequest(
        intent="Build a risk-aware deployment plan",
        context={"refs": ["runbook-v1"]},
        symbolic_neural=True,
        creativity=0.7,
        rigor=0.9,
    )
    artifact = build_prompt_artifact(req, config)
    prompt = artifact["composite_prompt"]
    assert "SYSTEM CONTRACT" in prompt
    assert "STANDING ON THE SHOULDER OF GIANTS PROTOCOL" in prompt
    assert "OUTPUT CONTRACT" in prompt
    assert "Section 5: SNR self-score + confidence" in prompt
