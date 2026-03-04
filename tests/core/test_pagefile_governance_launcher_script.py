from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ops" / "PAGEFILE-GOVERNANCE-LAUNCHER.bat"


def test_launcher_references_governance_script():
    content = SCRIPT.read_text(encoding="utf-8")
    assert "pagefile_governance.ps1" in content
    assert "ExecutionPolicy Bypass" in content


def test_launcher_elevates_only_for_non_dryrun_apply():
    content = SCRIPT.read_text(encoding="utf-8")
    assert 'if /I "%~1"=="-Mode"' in content
    assert 'if /I "%~2"=="Apply" (' in content
    assert "DRYRUN_HINT" in content
    assert "SKIP_ELEVATION" in content
    assert "Start-Process -FilePath 'powershell.exe' -Verb RunAs" in content
