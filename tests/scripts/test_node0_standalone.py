from __future__ import annotations

from pathlib import Path

import pytest

from scripts.node0_standalone import Node0StandaloneManager


def test_resolve_workspace_path_blocks_outside_workspace(tmp_path: Path) -> None:
    manager = Node0StandaloneManager(project_root=tmp_path)

    outside = Path("/tmp") / "outside.txt"
    with pytest.raises(ValueError, match="outside workspace"):
        manager._resolve_workspace_path(str(outside))


def test_filesystem_action_write_read_list(tmp_path: Path) -> None:
    manager = Node0StandaloneManager(project_root=tmp_path)

    write = manager._maybe_execute_filesystem_action(
        "write file missions/demo.txt :: hello from node0"
    )
    assert write is not None
    assert write["action"] == "write"

    target = tmp_path / "missions" / "demo.txt"
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "hello from node0"

    read = manager._maybe_execute_filesystem_action("read file missions/demo.txt")
    assert read is not None
    assert read["action"] == "read"
    assert "hello from node0" in read["preview"]

    listed = manager._maybe_execute_filesystem_action("list dir missions")
    assert listed is not None
    assert listed["action"] == "list"
    assert "demo.txt" in listed["entries"]


def test_health_is_degraded_without_activation(tmp_path: Path) -> None:
    manager = Node0StandaloneManager(project_root=tmp_path)

    report = manager.health()
    assert report["status"] == "degraded"
    assert report["gates"]["identity_credentials"] is False
    assert report["gates"]["assets_file"] is False


@pytest.mark.asyncio
async def test_run_task_reports_filesystem_action(tmp_path: Path) -> None:
    manager = Node0StandaloneManager(project_root=tmp_path)

    result = await manager.run_task(
        "write file missions/from_mission.txt :: hello from mission",
        browser_mode="mock",
    )

    fs = result.get("filesystem_action")
    assert fs is not None
    assert fs["action"] == "write"
    target = tmp_path / "missions" / "from_mission.txt"
    assert target.exists()
    assert target.read_text(encoding="utf-8") == "hello from mission"
