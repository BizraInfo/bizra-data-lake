from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTROL_CENTER = ROOT / "scripts" / "ops" / "CONTROL-CENTER.bat"
SHORTCUT = ROOT / "scripts" / "ops" / "Create-Shortcut.ps1"


def test_control_center_exposes_performance_recovery_menu_entries():
    content = CONTROL_CENTER.read_text(encoding="utf-8")
    assert "Node0 Performance Snapshot (Analyze)" in content
    assert "Node0 Performance Recovery (Dry Run)" in content
    assert "Node0 Performance Recovery (Execute)" in content
    assert "Docker Volume Governance (Inventory)" in content
    assert "Docker Volume Governance (Dry Run Reclaim)" in content
    assert "Docker Volume Governance (Execute k3d Reclaim)" in content
    assert "VHDX Compaction Snapshot (Analyze)" in content
    assert "VHDX Compaction (Dry Run)" in content
    assert "VHDX Compaction (Execute)" in content
    assert "Pagefile Governance Snapshot (Analyze)" in content
    assert "Pagefile Governance (Dry Run Apply)" in content
    assert "Pagefile Governance (Execute Apply)" in content
    assert "Schedule Post-Reboot VHDX Compact (One-Time)" in content
    assert "Select option (1-21): " in content


def test_control_center_routes_performance_commands():
    content = CONTROL_CENTER.read_text(encoding="utf-8")
    assert 'if "%choice%"=="8" goto PERF_ANALYZE' in content
    assert 'if "%choice%"=="9" goto PERF_REMEDIATE_DRYRUN' in content
    assert 'if "%choice%"=="10" goto PERF_REMEDIATE_EXEC' in content
    assert 'if "%choice%"=="11" goto VOL_GOV_INVENTORY' in content
    assert 'if "%choice%"=="12" goto VOL_GOV_DRYRUN' in content
    assert 'if "%choice%"=="13" goto VOL_GOV_EXEC' in content
    assert 'if "%choice%"=="14" goto VHDX_ANALYZE' in content
    assert 'if "%choice%"=="15" goto VHDX_DRYRUN' in content
    assert 'if "%choice%"=="16" goto VHDX_EXEC' in content
    assert 'if "%choice%"=="17" goto PAGEFILE_ANALYZE' in content
    assert 'if "%choice%"=="18" goto PAGEFILE_DRYRUN' in content
    assert 'if "%choice%"=="19" goto PAGEFILE_EXEC' in content
    assert 'if "%choice%"=="20" goto SCHEDULE_REBOOT_COMPACT' in content
    assert 'if "%choice%"=="21" goto EXIT' in content
    assert "-Mode Analyze" in content
    assert "-Mode Remediate -DryRun:$true" in content
    assert "-Mode Remediate -DryRun:$false" in content
    assert 'docker_volume_governance.py" inventory' in content
    assert 'docker_volume_governance.py" --dry-run reclaim-k3d' in content
    assert 'docker_volume_governance.py" reclaim-k3d --restart-cluster' in content
    assert 'VHDX-COMPACTION-LAUNCHER.bat" -Mode Analyze' in content
    assert 'VHDX-COMPACTION-LAUNCHER.bat" -Mode Compact -DryRun:$true' in content
    assert 'VHDX-COMPACTION-LAUNCHER.bat" -Mode Compact -DryRun:$false' in content
    assert 'PAGEFILE-GOVERNANCE-LAUNCHER.bat" -Mode Analyze' in content
    assert 'PAGEFILE-GOVERNANCE-LAUNCHER.bat" -Mode Apply -DryRun:$true' in content
    assert 'PAGEFILE-GOVERNANCE-LAUNCHER.bat" -Mode Apply -DryRun:$false' in content
    assert 'SCHEDULE-POST-REBOOT-COMPACTION.bat" -Target docker' in content


def test_control_center_paths_reference_real_workspace_files():
    content = CONTROL_CENTER.read_text(encoding="utf-8")
    assert "C:\\BIZRA-DATA-LAKE\\scripts\\ops\\CloudIngestion.ps1" in content
    assert "C:\\BIZRA-DATA-LAKE\\docs\\QUICK-START.md" in content
    assert "C:\\BIZRA-DATA-LAKE\\scripts\\ops\\VHDX-COMPACTION-LAUNCHER.bat" in content
    assert "C:\\BIZRA-DATA-LAKE\\scripts\\ops\\PAGEFILE-GOVERNANCE-LAUNCHER.bat" in content
    assert "C:\\BIZRA-DATA-LAKE\\scripts\\ops\\SCHEDULE-POST-REBOOT-COMPACTION.bat" in content


def test_shortcut_targets_actual_control_center_path():
    content = SHORTCUT.read_text(encoding="utf-8")
    assert "C:\\BIZRA-DATA-LAKE\\scripts\\ops\\CONTROL-CENTER.bat" in content
    assert "C:\\BIZRA-DATA-LAKE\\scripts\\ops" in content
