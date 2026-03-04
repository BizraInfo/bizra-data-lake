from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ops" / "vhdx_compaction_governance.ps1"


def test_script_exposes_modes_and_safety_flags():
    content = SCRIPT.read_text(encoding="utf-8")
    assert '[ValidateSet("Analyze", "Compact")]' in content
    assert "[object]$DryRun = $true" in content
    assert "Resolve-Bool" in content
    assert "[switch]$SkipElevation" in content
    assert "[switch]$RequireAdminForCompact = $true" in content
    assert 'Start-Process -FilePath "powershell.exe"' in content
    assert "-Verb RunAs" in content
    assert "[double]$MinFreeVirtualMemoryGB = 1.0" in content
    assert "[double]$MinPagefileAllocatedGB = 8.0" in content
    assert "type YES to continue" in content
    assert "Confirm-OrDie" in content


def test_script_targets_expected_vhdx_paths():
    content = SCRIPT.read_text(encoding="utf-8")
    assert "Docker\\wsl\\disk\\docker_data.vhdx" in content
    assert (
        "CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc\\LocalState\\ext4.vhdx" in content
    )
    assert "Test-WslDistroExists" in content
    assert "[SKIP] docker-desktop-data distro not present" in content
    assert "wsl --shutdown" in content
    assert "Win32_PageFileUsage" in content
    assert "Insufficient free virtual memory for safe compaction." in content
    assert "Pagefile allocation is below safe compaction threshold." in content


def test_script_uses_diskpart_compaction_and_writes_report():
    content = SCRIPT.read_text(encoding="utf-8")
    assert "compact vdisk" in content
    assert "vhdx_compaction_governance_" in content
    assert "ConvertTo-Json" in content
    assert "$report.summary" in content
    assert "failed_compacts" in content
    assert "failed_steps" in content
    assert "Test-IsAdministrator" in content
    assert "preflight_failure" in content
