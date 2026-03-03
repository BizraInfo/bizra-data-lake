from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ops" / "VHDX-COMPACTION-LAUNCHER.bat"


def test_launcher_references_governance_script():
    content = SCRIPT.read_text(encoding="utf-8")
    assert "vhdx_compaction_governance.ps1" in content
    assert "ExecutionPolicy Bypass" in content


def test_launcher_requires_mode_and_supports_compact_elevation():
    content = SCRIPT.read_text(encoding="utf-8")
    assert 'if "%~1"==""' in content
    assert 'if /I "%~1"=="-Mode"' in content
    assert 'if /I "%~2"=="Compact" (' in content
    assert 'set "DRYRUN_HINT=0"' in content
    assert 'set "SKIP_ELEVATION=0"' in content
    assert (
        'findstr /I /C:"-DryRun:$true" /C:"-DryRun:true" /C:"-DryRun true" /C:"-DryRun 1"'
        in content
    )
    assert 'if "%DRYRUN_HINT%"=="0" set "NEED_ELEVATION=1"' in content
    assert "Start-Process -FilePath 'powershell.exe' -Verb RunAs" in content
