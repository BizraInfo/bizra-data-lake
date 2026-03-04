from __future__ import annotations

from pathlib import Path

import pytest

from core.integration.constitution_parser import (
    ConstitutionError,
    canonical_ihsan_weights,
    load_constitution,
    operational_ihsan_weights,
    resolve_constitution_path,
)


def test_resolve_default_path_points_to_repo_root_constitution():
    path = resolve_constitution_path()
    assert path.name == "constitution.toml"
    assert path.exists()


def test_load_constitution_has_expected_sections():
    data = load_constitution()
    assert "meta" in data
    assert "ihsan_tensor" in data
    assert "gates" in data
    assert "economics" in data


def test_canonical_weights_sum_to_one():
    data = load_constitution()
    weights = canonical_ihsan_weights(data)
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    assert len(weights) == 8


def test_operational_projection_is_normalized_and_six_dim():
    data = load_constitution()
    weights = operational_ihsan_weights(data)
    assert len(weights) == 6
    assert abs(sum(weights.values()) - 1.0) < 1e-6
    for value in weights.values():
        assert 0.0 < value <= 1.0


def test_gate_weights_sum_to_one():
    data = load_constitution()
    gates = data["gates"]
    total = sum(
        float(gates[k]["weight"])
        for k in ["alpha_4", "alpha_7", "alpha_8", "alpha_9", "alpha_10"]
    )
    assert abs(total - 1.0) < 1e-6


def test_thresholds_are_bounded():
    data = load_constitution()
    thresholds = data["ihsan_tensor"]["thresholds"]
    for name in [
        "gate_minimum",
        "poi_consensus",
        "bloom_eligibility",
        "ihsan_excellence",
        "conformance_join",
    ]:
        v = float(thresholds[name])
        assert 0.0 <= v <= 1.0


def test_invalid_missing_section_raises(tmp_path: Path):
    bad = tmp_path / "bad.toml"
    bad.write_text("[meta]\nversion='x'\n", encoding="utf-8")
    with pytest.raises(ConstitutionError):
        load_constitution(bad)
