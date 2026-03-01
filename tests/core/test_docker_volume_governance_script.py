from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ops" / "docker_volume_governance.py"


def test_script_declares_expected_commands():
    content = SCRIPT.read_text(encoding="utf-8")
    assert "choices=[\"inventory\", \"orphans\", \"reclaim-k3d\", \"reclaim-all\"]" in content
    assert "cmd_reclaim_k3d" in content
    assert "cmd_reclaim_all" in content


def test_script_has_dry_run_and_confirmation_safety():
    content = SCRIPT.read_text(encoding="utf-8")
    assert "--dry-run" in content
    assert "--yes" in content
    assert "type YES to continue" in content
    assert "_confirm_or_die" in content


def test_script_collects_evidence_report():
    content = SCRIPT.read_text(encoding="utf-8")
    assert "docker_volume_governance_" in content
    assert "_write_report" in content
    assert "timestamp_utc" in content


def test_script_contains_k3d_reclaim_flow():
    content = SCRIPT.read_text(encoding="utf-8")
    assert "k3d\", \"cluster\", \"list\", \"-o\", \"json\"" in content
    assert "crictl rmi --prune" in content
    assert "k3s crictl rmi --prune" in content
    assert "FAILED prune on" in content
    assert "--restart-cluster" in content
