#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Missing dependency: pyyaml ({exc})")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    constitution_path = repo_root / "constitution" / "ihsan_v1.yaml"
    data = yaml.safe_load(constitution_path.read_text(encoding="utf-8"))
    dims = data["dimensions"]
    weights = {k: float(v["weight"]) for k, v in dims.items()}
    threshold = float(data["units"]["threshold"])
    policy = data.get("threshold_policy") or {}
    if not isinstance(policy, dict):
        policy = {}

    combine = str(policy.get("combine", "max")).strip().lower()
    if combine not in ("max", "min"):
        combine = "max"

    default_env = str(policy.get("default_env", "development")).strip() or "development"
    thresholds_by_env_raw = policy.get("thresholds_by_env") or {}
    thresholds_by_artifact_raw = policy.get("thresholds_by_artifact_class") or {}

    normalization = policy.get("normalization") or {}
    if not isinstance(normalization, dict):
        normalization = {}

    env_aliases_raw = normalization.get("env_aliases") or {}
    artifact_aliases_raw = normalization.get("artifact_class_aliases") or {}

    def normalize_key(raw: str) -> str:
        return raw.strip().lower().replace("-", "_").replace(" ", "_")

    thresholds_by_env = {
        normalize_key(str(k)): float(v)
        for k, v in (thresholds_by_env_raw.items() if isinstance(thresholds_by_env_raw, dict) else [])
    }
    thresholds_by_artifact = {
        normalize_key(str(k)): float(v)
        for k, v in (thresholds_by_artifact_raw.items() if isinstance(thresholds_by_artifact_raw, dict) else [])
    }
    env_aliases = {
        normalize_key(str(k)): normalize_key(str(v))
        for k, v in (env_aliases_raw.items() if isinstance(env_aliases_raw, dict) else [])
    }
    artifact_aliases = {
        normalize_key(str(k)): normalize_key(str(v))
        for k, v in (artifact_aliases_raw.items() if isinstance(artifact_aliases_raw, dict) else [])
    }

    def expected_threshold(env_name: str, artifact_class: str) -> float:
        env_key = normalize_key(env_name) or normalize_key(default_env)
        artifact_key = normalize_key(artifact_class)

        env_key = env_aliases.get(env_key, env_key)
        artifact_key = artifact_aliases.get(artifact_key, artifact_key)

        candidates: list[float] = []
        if env_key in thresholds_by_env:
            candidates.append(thresholds_by_env[env_key])
        if artifact_key in thresholds_by_artifact:
            candidates.append(thresholds_by_artifact[artifact_key])

        if not candidates:
            return threshold
        return min(candidates) if combine == "min" else max(candidates)

    scores = {
        "correctness": 0.91,
        "safety": 0.97,
        "user_benefit": 0.88,
        "efficiency": 0.77,
        "auditability": 0.66,
        "anti_centralization": 0.55,
        "robustness": 0.93,
        "adl_fairness": 0.84,
    }

    expected = sum(weights[k] * scores[k] for k in weights)

    # Ensure repo root is importable for local Python parity (bizra_kernel package).
    sys.path.insert(0, str(repo_root))
    # Back-compat: if the legacy Node0 repo subtree exists, allow importing from there too.
    legacy_root = repo_root / "bizra-genesis-node"
    if legacy_root.exists():
        sys.path.insert(0, str(legacy_root))

    from bizra_kernel.ihsan_vector import constitution_snapshot, score_plain, threshold_for  # type: ignore

    py_score = float(score_plain(scores))
    snap = constitution_snapshot()

    eps = 1e-9
    if abs(sum(snap["weights"].values()) - 1.0) > eps:
        print(f"[FAIL] python weights do not sum to 1.0: {sum(snap['weights'].values())}")
        return 2

    if abs(float(snap["threshold"]) - threshold) > eps:
        print(
            "[FAIL] python threshold mismatch:",
            "python=",
            snap["threshold"],
            "constitution=",
            threshold,
        )
        return 2

    for k, w in weights.items():
        if abs(float(snap["weights"].get(k, -1.0)) - w) > eps:
            print(f"[FAIL] python weight mismatch for {k}: python={snap['weights'].get(k)} constitution={w}")
            return 2

    rust_out = subprocess.check_output(
        ["cargo", "run", "-q", "--bin", "ihsan_calc", "--", json.dumps(scores)],
        cwd=str(repo_root),
    ).decode("utf-8", errors="replace")
    rs_score = float(rust_out.strip())

    policy_cases = [
        ("development", "docs"),
        ("dev", "documentation"),
        ("ci", "docs"),
        ("production", "docs"),
        ("production", "code"),
        ("ci", "receipt"),
    ]

    for env_name, artifact_class in policy_cases:
        exp = expected_threshold(env_name, artifact_class)
        py_thr = float(threshold_for(env_name, artifact_class))
        if abs(py_thr - exp) > eps:
            print(
                f"[FAIL] python threshold_for mismatch env={env_name} artifact={artifact_class}: python={py_thr} expected={exp}"
            )
            return 2

        rs_thr_out = subprocess.check_output(
            ["cargo", "run", "-q", "--bin", "ihsan_threshold", "--", env_name, artifact_class],
            cwd=str(repo_root),
        ).decode("utf-8", errors="replace")
        rs_thr = float(rs_thr_out.strip())
        if abs(rs_thr - exp) > eps:
            print(
                f"[FAIL] rust threshold_for mismatch env={env_name} artifact={artifact_class}: rust={rs_thr} expected={exp}"
            )
            return 2

    if abs(py_score - expected) > eps:
        print(f"[FAIL] python score mismatch: python={py_score} expected={expected}")
        return 1

    if abs(rs_score - expected) > eps:
        print(f"[FAIL] rust score mismatch: rust={rs_score} expected={expected}")
        return 1

    if abs(rs_score - py_score) > eps:
        print(f"[FAIL] rust/python mismatch: rust={rs_score} python={py_score}")
        return 1

    print(
        "[OK] ihsan parity",
        f"score={expected:.9f}",
        f"threshold_baseline={threshold:.2f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
