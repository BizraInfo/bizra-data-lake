"""
Self-RLVR Harness — Verifiable reward loop for reflex promotion.

This harness simulates/executes a self-reinforcement loop with:
1) Composite reward scoring (SNR/Ihsan/efficiency/feedback),
2) Constitutional gating (Ihsan + SNR + verification),
3) Hash-chained episode receipts for auditability,
4) Reflex promotion after consecutive qualified episodes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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

from core.integration.constants import (
    SNR_THRESHOLD_T2_STANDARD,
    UNIFIED_IHSAN_THRESHOLD,
)
from core.token.rl_rewards import composite_reward


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )


def _hash_payload(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class HarnessConfig:
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD
    snr_threshold: float = SNR_THRESHOLD_T2_STANDARD
    reward_threshold: float = 0.75
    compile_streak: int = 3
    ema_alpha: float = 0.30
    convergence_target: float = 0.85
    convergence_variance_max: float = 0.010
    min_temperature: float = 0.10
    max_temperature: float = 2.00
    cool_rate: float = 0.08
    heat_rate: float = 0.06
    variance_window: int = 5


def load_config(path: Path) -> HarnessConfig:
    if not path.exists():
        return HarnessConfig()

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = payload if isinstance(payload, dict) else {}

    def _num(name: str, default: float) -> float:
        value = cfg.get(name, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _int(name: str, default: int) -> int:
        value = cfg.get(name, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return int(default)

    return HarnessConfig(
        ihsan_threshold=_clamp01(_num("ihsan_threshold", UNIFIED_IHSAN_THRESHOLD)),
        snr_threshold=_clamp01(_num("snr_threshold", SNR_THRESHOLD_T2_STANDARD)),
        reward_threshold=_clamp01(_num("reward_threshold", 0.75)),
        compile_streak=max(1, _int("compile_streak", 3)),
        ema_alpha=_clamp01(_num("ema_alpha", 0.30)),
        convergence_target=_clamp01(_num("convergence_target", 0.85)),
        convergence_variance_max=max(0.0, _num("convergence_variance_max", 0.010)),
        min_temperature=max(0.01, _num("min_temperature", 0.10)),
        max_temperature=max(0.05, _num("max_temperature", 2.0)),
        cool_rate=max(0.0, _num("cool_rate", 0.08)),
        heat_rate=max(0.0, _num("heat_rate", 0.06)),
        variance_window=max(2, _int("variance_window", 5)),
    )


def generate_synthetic_episodes(count: int = 8, seed: int = 7) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    episodes: list[dict[str, Any]] = []
    for idx in range(max(count, 1)):
        baseline = 0.84 + min(idx, 6) * 0.02
        snr = _clamp01(baseline + rng.uniform(-0.015, 0.02))
        ihsan = _clamp01(0.86 + min(idx, 6) * 0.02 + rng.uniform(-0.015, 0.02))
        tokens_used = int(620 + rng.randint(-120, 180))
        quality = _clamp01((snr + ihsan) / 2.0)
        feedback = _clamp01(0.78 + rng.uniform(-0.06, 0.15))
        verified = rng.random() > 0.05
        episodes.append(
            {
                "snr": snr,
                "ihsan": ihsan,
                "tokens_used": tokens_used,
                "quality": quality,
                "user_feedback": feedback,
                "verified": verified,
                "penalties": 0.0 if verified else 0.15,
            }
        )
    return episodes


def verify_receipt_chain(receipts: list[dict[str, Any]]) -> bool:
    previous = "GENESIS"
    for receipt in receipts:
        hash_input = receipt.get("hash_input")
        receipt_hash = str(receipt.get("receipt_hash", ""))
        if not isinstance(hash_input, dict):
            return False
        if hash_input.get("previous_hash") != previous:
            return False
        if _hash_payload(hash_input) != receipt_hash:
            return False
        previous = receipt_hash
    return True


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def run_self_rlvr_harness(
    agent_id: str,
    episodes: list[dict[str, Any]],
    config: HarnessConfig | None = None,
) -> dict[str, Any]:
    cfg = config or HarnessConfig()

    receipts: list[dict[str, Any]] = []
    previous_hash = "GENESIS"
    reward_ema = 0.0
    reward_ema_initialized = False
    temperature = 1.0
    streak = 0
    compiled = False
    rewards_window: list[float] = []
    qualified_count = 0

    for index, raw in enumerate(episodes, start=1):
        snr = _clamp01(float(raw.get("snr", 0.0)))
        ihsan = _clamp01(float(raw.get("ihsan", 0.0)))
        tokens_used = max(int(raw.get("tokens_used", 0)), 0)
        quality = _clamp01(float(raw.get("quality", snr)))
        feedback = _clamp01(float(raw.get("user_feedback", 0.5)))
        penalties = _clamp01(float(raw.get("penalties", 0.0)))
        verified = bool(raw.get("verified", True))

        reward = composite_reward(
            {
                "snr": snr,
                "ihsan": ihsan,
                "tokens_used": tokens_used,
                "quality": quality,
                "user_feedback": feedback,
                "penalties": penalties,
            }
        )

        qualified = (
            verified
            and snr >= cfg.snr_threshold
            and ihsan >= cfg.ihsan_threshold
            and reward >= cfg.reward_threshold
        )
        if qualified:
            streak += 1
            qualified_count += 1
        else:
            streak = 0
        if streak >= cfg.compile_streak:
            compiled = True

        if not reward_ema_initialized:
            reward_ema = reward
            reward_ema_initialized = True
        else:
            reward_ema = cfg.ema_alpha * reward + (1.0 - cfg.ema_alpha) * reward_ema

        rewards_window.append(reward)
        if len(rewards_window) > cfg.variance_window:
            rewards_window.pop(0)
        current_variance = _variance(rewards_window)

        if reward >= reward_ema:
            temperature = max(cfg.min_temperature, temperature - cfg.cool_rate)
        else:
            temperature = min(cfg.max_temperature, temperature + cfg.heat_rate)

        hash_input = {
            "agent_id": agent_id,
            "episode_index": index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "snr": round(snr, 6),
                "ihsan": round(ihsan, 6),
                "tokens_used": tokens_used,
                "quality": round(quality, 6),
                "user_feedback": round(feedback, 6),
                "penalties": round(penalties, 6),
                "verified": verified,
            },
            "reward": round(reward, 6),
            "reward_ema": round(reward_ema, 6),
            "qualified": qualified,
            "compile_streak": streak,
            "compiled": compiled,
            "temperature": round(temperature, 6),
            "variance_window": round(current_variance, 6),
            "previous_hash": previous_hash,
        }
        receipt_hash = _hash_payload(hash_input)
        receipts.append(
            {
                "episode_index": index,
                "hash_input": hash_input,
                "receipt_hash": receipt_hash,
            }
        )
        previous_hash = receipt_hash

    reward_values = [r["hash_input"]["reward"] for r in receipts]
    avg_reward = sum(reward_values) / len(reward_values) if reward_values else 0.0
    final_variance = _variance(reward_values[-cfg.variance_window :])
    qualified_rate = qualified_count / max(len(receipts), 1)
    converged = (
        reward_ema >= cfg.convergence_target
        and final_variance <= cfg.convergence_variance_max
    )
    chain_valid = verify_receipt_chain(receipts)

    signal = qualified_count + int(compiled) + int(chain_valid) + int(converged)
    noise = (len(receipts) - qualified_count) + int(not chain_valid)
    snr_raw = signal / max(noise, 1)
    snr_normalized = snr_raw / (1.0 + snr_raw)

    if compiled and converged and chain_valid:
        decision = {
            "action": "PROMOTE_TO_SYSTEM_1",
            "owner": "reflex-compiler",
            "reason": "Compile streak satisfied with stable reward and valid receipts.",
        }
    elif compiled and chain_valid:
        decision = {
            "action": "PROMOTE_WITH_SHADOW_MONITORING",
            "owner": "runtime-governance",
            "reason": "Compile streak met; convergence incomplete. Run shadow audits.",
        }
    elif qualified_rate >= 0.60:
        decision = {
            "action": "CONTINUE_SELF_TRAINING",
            "owner": "rlvr-loop",
            "reason": "Signal improving but compile streak not yet stable.",
        }
    else:
        decision = {
            "action": "RECALIBRATE_POLICY",
            "owner": "policy-tuning",
            "reason": "Low qualified rate; adjust prompt/policy and re-run.",
        }

    return {
        "program": {
            "id": "self_rlvr_harness",
            "version": "1.0.0",
            "agent_id": agent_id,
        },
        "config": {
            "ihsan_threshold": cfg.ihsan_threshold,
            "snr_threshold": cfg.snr_threshold,
            "reward_threshold": cfg.reward_threshold,
            "compile_streak": cfg.compile_streak,
            "ema_alpha": cfg.ema_alpha,
            "convergence_target": cfg.convergence_target,
            "convergence_variance_max": cfg.convergence_variance_max,
            "temperature_bounds": [cfg.min_temperature, cfg.max_temperature],
        },
        "summary": {
            "episodes": len(receipts),
            "qualified_count": qualified_count,
            "qualified_rate": round(qualified_rate, 4),
            "avg_reward": round(avg_reward, 4),
            "reward_ema": round(reward_ema, 4),
            "final_variance": round(final_variance, 6),
            "compiled": compiled,
            "converged": converged,
            "chain_valid": chain_valid,
            "snr": {
                "signal": signal,
                "noise": noise,
                "raw": round(snr_raw, 4),
                "normalized": round(snr_normalized, 4),
            },
            "last_receipt_hash": (
                receipts[-1]["receipt_hash"] if receipts else "GENESIS"
            ),
        },
        "decision": decision,
        "receipts": receipts,
    }


def _load_episodes(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("episodes file must be a JSON list of episode objects")
    normalized = [item for item in payload if isinstance(item, dict)]
    if len(normalized) != len(payload):
        raise ValueError("episodes file must contain only JSON objects")
    return normalized


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run self-RLVR harness with verifiable receipts."
    )
    parser.add_argument("--agent-id", default="node0", help="Agent identifier.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/self_rlvr_harness.yaml"),
        help="YAML config path.",
    )
    parser.add_argument(
        "--episodes-file",
        type=Path,
        default=None,
        help="JSON file with episode list. If omitted, synthetic episodes are generated.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=8,
        help="Synthetic episode count when --episodes-file is not provided.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for synthetic generation.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output report path.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    episodes = (
        _load_episodes(args.episodes_file)
        if args.episodes_file is not None
        else generate_synthetic_episodes(count=args.episodes, seed=args.seed)
    )
    report = run_self_rlvr_harness(
        agent_id=args.agent_id, episodes=episodes, config=cfg
    )

    encoded = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(encoded, encoding="utf-8")
    print(encoded)
    return 0 if report["summary"]["chain_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
