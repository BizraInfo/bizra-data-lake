from __future__ import annotations

from pathlib import Path

import yaml

from scripts.ops.self_rlvr_harness import (
    HarnessConfig,
    generate_synthetic_episodes,
    load_config,
    run_self_rlvr_harness,
    verify_receipt_chain,
)


def _episode(
    *,
    snr: float = 0.95,
    ihsan: float = 0.96,
    tokens_used: int = 640,
    quality: float = 0.93,
    user_feedback: float = 0.90,
    penalties: float = 0.0,
    verified: bool = True,
) -> dict:
    return {
        "snr": snr,
        "ihsan": ihsan,
        "tokens_used": tokens_used,
        "quality": quality,
        "user_feedback": user_feedback,
        "penalties": penalties,
        "verified": verified,
    }


def test_self_rlvr_harness_compiles_after_three_qualified() -> None:
    episodes = [_episode(), _episode(), _episode()]
    report = run_self_rlvr_harness(agent_id="node0", episodes=episodes)
    assert report["summary"]["compiled"] is True
    assert report["summary"]["qualified_count"] == 3
    assert report["summary"]["chain_valid"] is True
    assert report["decision"]["action"] in {
        "PROMOTE_TO_SYSTEM_1",
        "PROMOTE_WITH_SHADOW_MONITORING",
    }


def test_self_rlvr_harness_resets_streak_on_gate_failure() -> None:
    episodes = [
        _episode(),
        _episode(ihsan=0.70),  # fails Ihsan gate
        _episode(),
        _episode(),
    ]
    cfg = HarnessConfig(compile_streak=3)
    report = run_self_rlvr_harness(agent_id="node0", episodes=episodes, config=cfg)
    assert report["summary"]["compiled"] is False
    streaks = [r["hash_input"]["compile_streak"] for r in report["receipts"]]
    assert streaks == [1, 0, 1, 2]


def test_self_rlvr_harness_receipt_chain_detects_tamper() -> None:
    report = run_self_rlvr_harness(
        agent_id="node0",
        episodes=[_episode(), _episode(), _episode()],
    )
    receipts = report["receipts"]
    assert verify_receipt_chain(receipts) is True

    tampered = [dict(r) for r in receipts]
    tampered[1] = dict(tampered[1])
    tampered[1]["hash_input"] = dict(tampered[1]["hash_input"])
    tampered[1]["hash_input"]["reward"] = 0.0
    assert verify_receipt_chain(tampered) is False


def test_self_rlvr_harness_synthetic_generation_is_seeded() -> None:
    first = generate_synthetic_episodes(count=5, seed=11)
    second = generate_synthetic_episodes(count=5, seed=11)
    assert first == second


def test_self_rlvr_harness_config_loader(tmp_path: Path) -> None:
    cfg_file = tmp_path / "self_rlvr.yaml"
    cfg_file.write_text(
        yaml.safe_dump(
            {
                "ihsan_threshold": 0.92,
                "snr_threshold": 0.88,
                "compile_streak": 4,
                "ema_alpha": 0.25,
                "convergence_target": 0.80,
                "reward_threshold": 0.70,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.ihsan_threshold == 0.92
    assert cfg.snr_threshold == 0.88
    assert cfg.compile_streak == 4
    assert cfg.ema_alpha == 0.25
    assert cfg.convergence_target == 0.80
    assert cfg.reward_threshold == 0.70
