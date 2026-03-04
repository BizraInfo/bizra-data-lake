from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ops" / "pagefile_governance.ps1"


def test_script_exposes_modes_and_safety_flags():
    content = SCRIPT.read_text(encoding="utf-8")
    assert '[ValidateSet("Analyze", "Apply")]' in content
    assert "[object]$DryRun = $true" in content
    assert "[switch]$SkipElevation" in content
    assert "[switch]$RequireAdminForApply = $true" in content
    assert "type YES to continue" in content
    assert "Resolve-Bool" in content


def test_script_reads_virtual_memory_and_pagefile_state():
    content = SCRIPT.read_text(encoding="utf-8")
    assert "Win32_OperatingSystem" in content
    assert "Win32_PageFileSetting" in content
    assert "Win32_PageFileUsage" in content
    assert "automatic_managed_pagefile" in content


def test_script_writes_evidence_report():
    content = SCRIPT.read_text(encoding="utf-8")
    assert "pagefile_governance_" in content
    assert "ConvertTo-Json" in content
    assert "reboot_required" in content
    assert "preflight_failure" in content
