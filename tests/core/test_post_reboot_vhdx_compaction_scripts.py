from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN_ONCE = ROOT / "scripts" / "ops" / "post_reboot_vhdx_compact_once.ps1"
SCHEDULER = ROOT / "scripts" / "ops" / "schedule_post_reboot_vhdx_compaction.ps1"
SCHEDULER_BAT = ROOT / "scripts" / "ops" / "SCHEDULE-POST-REBOOT-COMPACTION.bat"


def test_run_once_script_executes_compaction_and_unregisters_task():
    content = RUN_ONCE.read_text(encoding="utf-8")
    assert "BIZRA-PostReboot-VHDX-Compact" in content
    assert "vhdx_compaction_governance.ps1" in content
    assert "elseif ($?) { 0 } else { 1 }" in content
    assert "Unregister-ScheduledTask" in content
    assert "post_reboot_vhdx_compact_once_" in content


def test_scheduler_script_registers_startup_task():
    content = SCHEDULER.read_text(encoding="utf-8")
    assert "Register-ScheduledTask" in content
    assert "New-ScheduledTaskTrigger -AtLogOn -User $RunAsUser" in content
    assert "LogonType Interactive" in content
    assert "Unregister-ScheduledTask" in content
    assert "schedule_post_reboot_vhdx_compaction_" in content


def test_scheduler_bat_invokes_powershell_with_bypass():
    content = SCHEDULER_BAT.read_text(encoding="utf-8")
    assert "schedule_post_reboot_vhdx_compaction.ps1" in content
    assert "ExecutionPolicy Bypass" in content
