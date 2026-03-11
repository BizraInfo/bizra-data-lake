"""
Constitution Parser — load and validate `constitution.toml`.

This module provides a safe bridge between the constitutional source-of-truth
file and current runtime code without forcing immediate replacement of existing
gate implementations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError as exc:  # pragma: no cover
    raise RuntimeError("tomllib is required (Python 3.11+)") from exc


class ConstitutionError(ValueError):
    """Raised when constitution.toml is missing or structurally invalid."""


def resolve_constitution_path(path: str | Path | None = None) -> Path:
    """Resolve constitution path from explicit arg, env, or repo root."""
    if path is not None:
        return Path(path).expanduser().resolve()

    env_path = os.environ.get("BIZRA_CONSTITUTION_PATH")
    if env_path:
        return Path(env_path).expanduser().resolve()

    # core/integration/constitution_parser.py -> repo_root
    repo_root = Path(__file__).resolve().parents[2]
    canonical = repo_root / "bizra-constitution" / "constitution.toml"
    if canonical.exists():
        return canonical
    return repo_root / "constitution.toml"


def load_constitution(
    path: str | Path | None = None, *, validate: bool = True
) -> dict[str, Any]:
    """Load constitution TOML as a dictionary."""
    resolved = resolve_constitution_path(path)
    if not resolved.exists():
        raise ConstitutionError(f"constitution.toml not found: {resolved}")

    with resolved.open("rb") as f:
        data = tomllib.load(f)

    if validate:
        validate_constitution(data)
    return data


def validate_constitution(data: dict[str, Any]) -> None:
    """Validate minimal structural and numerical invariants."""
    required_top = [
        "meta",
        "identity",
        "interaction_laws",
        "ihsan_tensor",
        "pat",
        "sat",
        "gates",
        "hhmm",
        "economics",
        "reflex",
        "conformance",
        "security",
        "daughter_test",
    ]
    for key in required_top:
        if key not in data:
            raise ConstitutionError(f"missing required section: [{key}]")

    # Identity rights invariant
    rights = data.get("identity", {}).get("rights", {}).get("rights", [])
    min_rights = (
        data.get("identity", {}).get("rights", {}).get("minimum_rights_count", 0)
    )
    if len(rights) < int(min_rights):
        raise ConstitutionError(
            f"identity rights count {len(rights)} < minimum_rights_count {min_rights}"
        )

    # Canonical ihsan weights must sum to ~1.0
    canonical = data["ihsan_tensor"]["canonical_weights"]
    weight_sum = float(sum(float(v) for v in canonical.values()))
    if abs(weight_sum - 1.0) > 1e-6:
        raise ConstitutionError(
            f"ihsan canonical weights sum={weight_sum}, expected 1.0"
        )

    # Operational dims must be subset of canonical keys
    op_dims = data["ihsan_tensor"]["operational_dimensions"]["dimensions"]
    missing = [d for d in op_dims if d not in canonical]
    if missing:
        raise ConstitutionError(
            f"operational dimensions missing from canonical: {missing}"
        )

    # Gate count and weights
    gates = data["gates"]
    expected_count = int(gates["count"])
    gate_keys = ["alpha_4", "alpha_7", "alpha_8", "alpha_9", "alpha_10"]
    if expected_count != len(gate_keys):
        raise ConstitutionError(
            f"gates.count={expected_count} must equal {len(gate_keys)}"
        )
    gate_weight_sum = sum(float(gates[k]["weight"]) for k in gate_keys)
    if abs(gate_weight_sum - 1.0) > 1e-6:
        raise ConstitutionError(f"gate weights sum={gate_weight_sum}, expected 1.0")

    # Bounded threshold checks
    thresholds = data["ihsan_tensor"]["thresholds"]
    for field in [
        "gate_minimum",
        "poi_consensus",
        "bloom_eligibility",
        "ihsan_excellence",
        "conformance_join",
    ]:
        value = float(thresholds[field])
        if value < 0.0 or value > 1.0:
            raise ConstitutionError(
                f"ihsan_tensor.thresholds.{field}={value} outside [0,1]"
            )

    zakat_rate = float(data["economics"]["zakat"]["rate"])
    if zakat_rate < 0.0 or zakat_rate > 1.0:
        raise ConstitutionError(f"economics.zakat.rate={zakat_rate} outside [0,1]")


def canonical_ihsan_weights(data: dict[str, Any]) -> dict[str, float]:
    """Return canonical ihsan weights as float map."""
    weights = data["ihsan_tensor"]["canonical_weights"]
    return {k: float(v) for k, v in weights.items()}


def operational_ihsan_weights(data: dict[str, Any]) -> dict[str, float]:
    """Return normalized projection onto operational dimensions."""
    canonical = canonical_ihsan_weights(data)
    dims = data["ihsan_tensor"]["operational_dimensions"]["dimensions"]
    projected = {k: canonical[k] for k in dims}
    total = sum(projected.values())
    if total <= 0:
        raise ConstitutionError("operational projection sum must be > 0")
    return {k: v / total for k, v in projected.items()}


__all__ = [
    "ConstitutionError",
    "resolve_constitution_path",
    "load_constitution",
    "validate_constitution",
    "canonical_ihsan_weights",
    "operational_ihsan_weights",
]
