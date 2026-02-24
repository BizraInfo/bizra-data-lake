"""Tests for scripts/atlas/atlas_gap_report.py runtime verification wiring."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# Ensure repo root is importable so we can import the script module directly.
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "atlas" / "atlas_gap_report.py"

sys.path.insert(0, str(REPO_ROOT))
from scripts.atlas.atlas_gap_report import (  # noqa: E402
    PRIORITY_TO_TIER,
    TIER_CAPABILITIES,
    TIER_ORDER,
    user_tier_report,
)


def _write_matrix(path: Path) -> None:
    matrix = {
        "schema_version": "1.0",
        "source": "test",
        "capabilities": [
            {
                "capability": "PAT-SAT negotiation protocol",
                "status": "implemented",
                "evidence": ["core/bridges/dual_agentic_bridge.py"],
                "owner": "bridges",
                "target_phase": "P1",
            }
        ],
    }
    path.write_text(json.dumps(matrix), encoding="utf-8")


def test_atlas_report_marks_pat_sat_verified_from_runtime_status(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "matrix.json"
    runtime_status_path = tmp_path / "runtime_status.json"
    out_path = tmp_path / "atlas_report.json"
    _write_matrix(matrix_path)

    runtime_status = {
        "pat_sat": {
            "negotiation_receipt_chain": {
                "verified_end_to_end": True,
                "chain_valid": True,
                "total_negotiation_receipts": 3,
                "latest_sequence": 12,
                "latest_entry_hash": "f" * 64,
                "latest_receipt_id": "a" * 32,
            }
        }
    }
    runtime_status_path.write_text(json.dumps(runtime_status), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--matrix",
            str(matrix_path),
            "--out",
            str(out_path),
            "--runtime-status",
            str(runtime_status_path),
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["pat_sat_receipt_chain_verified"] is True
    assert report["pat_sat_receipt_chain"]["latest_entry_hash"] == "f" * 64
    assert report["runtime_status_path"] == str(runtime_status_path)
    capability = report["capabilities"][0]
    assert capability["capability"] == "PAT-SAT negotiation protocol"
    assert capability["runtime_verification"]["status"] == "verified"
    assert capability["runtime_verification"]["receipt_chain"]["latest_sequence"] == 12


def test_atlas_report_defaults_pat_sat_unverified_without_runtime_status(
    tmp_path: Path,
) -> None:
    matrix_path = tmp_path / "matrix.json"
    out_path = tmp_path / "atlas_report.json"
    _write_matrix(matrix_path)

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--matrix",
            str(matrix_path),
            "--out",
            str(out_path),
        ],
        check=True,
        cwd=REPO_ROOT,
    )

    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["pat_sat_receipt_chain_verified"] is False
    assert report["runtime_status_path"] is None
    capability = report["capabilities"][0]
    assert capability["runtime_verification"]["status"] == "unverified"


# ── user_tier_report() tests ─────────────────────────────────────────


class TestUserTierReport:
    """Tests for the public user_tier_report() function."""

    def test_seed_tier_returns_base_capabilities(self) -> None:
        result = user_tier_report("seed")
        assert result["tier"] == "seed"
        assert result["capabilities_unlocked"] == TIER_CAPABILITIES["seed"]
        # Everything above seed should be locked
        assert len(result["capabilities_locked"]) > 0
        assert result["next_tier"] == "sprout"

    def test_flourishing_tier_unlocks_everything(self) -> None:
        result = user_tier_report("flourishing")
        assert result["tier"] == "flourishing"
        assert result["capabilities_locked"] == []
        assert result["next_tier"] is None
        assert result["unlock_criteria"] == "Max tier reached"
        # Should include all capabilities from every tier
        all_caps = []
        for tier_key in TIER_ORDER:
            all_caps.extend(TIER_CAPABILITIES[tier_key])
        assert result["capabilities_unlocked"] == all_caps

    def test_growing_tier_includes_seed_and_sprout_capabilities(self) -> None:
        result = user_tier_report("growing")
        unlocked = result["capabilities_unlocked"]
        # Must include seed, sprout, and growing capabilities
        for tier_key in ("seed", "sprout", "growing"):
            for cap in TIER_CAPABILITIES[tier_key]:
                assert cap in unlocked, f"{cap} should be unlocked at growing tier"
        # rooted + flourishing should be locked
        for tier_key in ("rooted", "flourishing"):
            for cap in TIER_CAPABILITIES[tier_key]:
                assert cap in result["capabilities_locked"]

    def test_invalid_tier_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown tier"):
            user_tier_report("nonexistent")

    @pytest.mark.parametrize("tier_name", TIER_ORDER)
    def test_all_tiers_have_required_keys(self, tier_name: str) -> None:
        result = user_tier_report(tier_name)
        required_keys = {
            "tier",
            "capabilities_unlocked",
            "capabilities_locked",
            "next_tier",
            "unlock_criteria",
            "available_priorities",
        }
        assert required_keys.issubset(result.keys())

    def test_available_priorities_for_seed(self) -> None:
        result = user_tier_report("seed")
        assert "P0" in result["available_priorities"]
        assert "P1" not in result["available_priorities"]

    def test_available_priorities_for_rooted(self) -> None:
        result = user_tier_report("rooted")
        # seed maps to P0, growing maps to P1, rooted maps to P2
        assert "P0" in result["available_priorities"]
        assert "P1" in result["available_priorities"]
        assert "P2" in result["available_priorities"]
        assert "P3" not in result["available_priorities"]

    def test_available_priorities_for_flourishing(self) -> None:
        result = user_tier_report("flourishing")
        for priority in PRIORITY_TO_TIER:
            assert priority in result["available_priorities"]


class TestUserTierCLI:
    """Tests for the --user-tier CLI flag."""

    def test_user_tier_cli_outputs_valid_json(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--user-tier", "seed"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert proc.returncode == 0
        result = json.loads(proc.stdout)
        assert result["tier"] == "seed"
        assert "capabilities_unlocked" in result
        assert "capabilities_locked" in result

    def test_user_tier_cli_flourishing(self) -> None:
        proc = subprocess.run(
            [sys.executable, str(SCRIPT_PATH), "--user-tier", "flourishing"],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        assert proc.returncode == 0
        result = json.loads(proc.stdout)
        assert result["tier"] == "flourishing"
        assert result["capabilities_locked"] == []
        assert result["next_tier"] is None
