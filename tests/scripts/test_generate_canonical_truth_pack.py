"""Tests for ``scripts/generate_canonical_truth_pack.py``.

Covers:
    * Regression for the ``sys.modules`` registration bug that crashed
      ``@dataclass(frozen=True)`` resolution on Python 3.12 (see commit trail
      following dc732128).
    * Schema shape and value correctness for ``build_truth_pack``.
    * End-to-end ``write_truth_pack`` round-trip.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO / "scripts" / "generate_canonical_truth_pack.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "_canonical_truth_pack_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def generator():
    return _load_generator()


def test_load_api_policy_module_resolves_dataclass_classes(generator):
    """Regression test for sys.modules registration bug.

    Pre-fix: @dataclass(frozen=True) raised
      AttributeError: 'NoneType' object has no attribute '__dict__'
    because CPython 3.12 dataclasses.py calls sys.modules.get(cls.__module__).
    Post-fix: module is registered in sys.modules before exec_module().
    """
    module = generator._load_api_policy_module(REPO)
    assert hasattr(module, "API_ROUTE_POLICIES")
    assert hasattr(module, "APIRoutePolicy")
    policies = module.API_ROUTE_POLICIES
    assert len(policies) > 0
    first = policies[0]
    # Validate the dataclass decorator actually took effect (frozen=True).
    with pytest.raises((AttributeError, Exception)):
        first.path = "/mutated"  # type: ignore[misc]


def test_build_truth_pack_returns_expected_top_level_keys(generator):
    pack = generator.build_truth_pack(REPO)
    assert set(pack.keys()) == {"thresholds", "routes", "workspace"}


def test_build_truth_pack_thresholds_shape(generator):
    pack = generator.build_truth_pack(REPO)
    t = pack["thresholds"]
    for required in (
        "unified_ihsan_threshold",
        "unified_snr_threshold",
        "adl_gini_threshold",
        "source",
    ):
        assert required in t
    assert isinstance(t["unified_ihsan_threshold"], float)
    assert isinstance(t["unified_snr_threshold"], float)
    assert isinstance(t["adl_gini_threshold"], float)
    assert t["source"] == "core/integration/constants.py"


def test_build_truth_pack_threshold_values_match_constants(generator):
    """Values must match the canonical constants module."""
    from core.integration import constants as C

    pack = generator.build_truth_pack(REPO)
    t = pack["thresholds"]
    assert t["unified_ihsan_threshold"] == float(C.UNIFIED_IHSAN_THRESHOLD)
    assert t["unified_snr_threshold"] == float(C.UNIFIED_SNR_THRESHOLD)
    assert t["adl_gini_threshold"] == float(C.ADL_GINI_THRESHOLD)


def test_build_truth_pack_routes_shape(generator):
    pack = generator.build_truth_pack(REPO)
    r = pack["routes"]
    for required in ("total", "domains", "by_exposure", "source"):
        assert required in r
    by_exp = r["by_exposure"]
    for required in ("public", "bootstrap_public", "authenticated"):
        assert required in by_exp
        assert isinstance(by_exp[required], int)
    # Exposure counts must sum to total.
    assert (
        by_exp["public"] + by_exp["bootstrap_public"] + by_exp["authenticated"]
        == r["total"]
    )


def test_build_truth_pack_workspace_counts_are_positive(generator):
    pack = generator.build_truth_pack(REPO)
    w = pack["workspace"]
    assert w["rust_crates"] >= 20  # workspace has ~25+ crates; floor well below
    assert w["workflow_files"] >= 5
    assert w["rust_workspace_source"] == "bizra-omega/Cargo.toml"
    assert w["workflow_source"] == ".github/workflows/"


def test_write_truth_pack_roundtrip(generator, tmp_path: Path):
    out = tmp_path / "pack.json"
    written = generator.write_truth_pack(output_path=out, root=REPO)
    assert written == out
    data = json.loads(out.read_text(encoding="utf-8"))
    assert "thresholds" in data
    assert "routes" in data
    assert "workspace" in data


def test_write_truth_pack_is_deterministic(generator, tmp_path: Path):
    """Two back-to-back generations must produce byte-identical output.

    This is the property Gate 7 (freshness diff) relies on.
    """
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    generator.write_truth_pack(output_path=a, root=REPO)
    generator.write_truth_pack(output_path=b, root=REPO)
    assert a.read_bytes() == b.read_bytes()


def test_committed_truth_pack_matches_fresh_generation(generator, tmp_path: Path):
    """The committed JSON must equal a fresh regeneration — this mirrors Gate 7."""
    committed = REPO / "docs" / "knowledge" / "canonical_truth_pack.json"
    assert committed.exists(), "canonical_truth_pack.json must be committed"
    fresh = tmp_path / "fresh.json"
    generator.write_truth_pack(output_path=fresh, root=REPO)
    assert committed.read_bytes() == fresh.read_bytes(), (
        "Committed truth pack is stale. "
        "Run: python scripts/generate_canonical_truth_pack.py"
    )


def test_helpers_count_rust_workspace_members(generator):
    n = generator._count_rust_workspace_members(REPO)
    assert n > 0


def test_helpers_count_workflow_files(generator):
    n = generator._count_workflow_files(REPO)
    assert n > 0


def test_extract_float_constant_raises_on_unknown_name(generator, tmp_path: Path):
    # Fabricate a minimal fake constants.py so we exercise the error branch
    # without touching the real file.
    fake_root = tmp_path / "fakeroot"
    (fake_root / "core" / "integration").mkdir(parents=True)
    (fake_root / "core" / "integration" / "constants.py").write_text(
        "SOMETHING_ELSE: Final[float] = 0.1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="UNIFIED_IHSAN_THRESHOLD"):
        generator._extract_float_constant("UNIFIED_IHSAN_THRESHOLD", root=fake_root)
