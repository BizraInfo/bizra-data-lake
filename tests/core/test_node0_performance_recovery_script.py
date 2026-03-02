from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ops" / "node0_performance_recovery.ps1"


def test_script_exists_and_has_modes():
    content = SCRIPT.read_text(encoding="utf-8")
    assert "[ValidateSet(\"Analyze\", \"Remediate\")]" in content
    assert "[bool]$DryRun = $true" in content
    assert "if ($Mode -eq \"Analyze\")" in content


def test_script_detects_known_pressure_paths():
    content = SCRIPT.read_text(encoding="utf-8")
    assert "docker_data.vhdx" in content
    assert "CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc" in content
    assert ".cache\\huggingface\\hub" in content
    assert ".wslconfig" in content
    assert "Get-Counter -Counter" in content
    assert "process_hotspots" in content
    assert "Win32_PageFileUsage" in content
    assert "FreeVirtualMemory" in content
    assert "LOW_VIRTUAL_MEMORY" in content
    assert "PAGEFILE_UNDERSIZED" in content
    assert "CPU_PRESSURE" in content
    assert "DISK_QUEUE_PRESSURE" in content


def test_script_remediation_has_explicit_steps():
    content = SCRIPT.read_text(encoding="utf-8")
    assert "docker system prune -a --volumes -f" in content
    assert "docker builder prune -a -f" in content
    assert "wsl --shutdown" in content
    assert "Optimize-VHD" in content


def test_script_requires_confirmation_before_mutation():
    content = SCRIPT.read_text(encoding="utf-8")
    assert "Confirm-OrDie" in content
    assert "type YES to continue" in content
    assert "No changes applied." in content


def test_script_emits_ranked_summary_and_recommendations():
    content = SCRIPT.read_text(encoding="utf-8")
    assert "$dominantActionByFinding" in content
    assert "recommended_next_step" in content
    assert "dominant_bottleneck" in content
    assert "$snapshot.recommendations" in content
    assert "Top recommended actions:" in content
