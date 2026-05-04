from __future__ import annotations

from pathlib import Path

from scripts.ci_workflow_action_pinning import (
    WorkflowActionRef,
    scan_workflow_file,
    validate_workflow_action_pinning,
)

PINNED_CHECKOUT = "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5"


def _write_workflow(workflows_dir: Path, name: str, body: str) -> Path:
    workflows_dir.mkdir(parents=True, exist_ok=True)
    path = workflows_dir / name
    path.write_text(body, encoding="utf-8")
    return path


def test_scan_workflow_file_flags_moving_action_refs(tmp_path: Path) -> None:
    path = _write_workflow(
        tmp_path,
        "ci.yml",
        """
jobs:
  test:
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
      - uses: ./local-action
      - uses: docker://alpine:3.20
      - uses: actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065
""".lstrip(),
    )

    findings = scan_workflow_file(path)

    assert [
        WorkflowActionRef(
            path=path,
            line_number=4,
            target="actions/checkout@v4",
            ref="v4",
        ),
        WorkflowActionRef(
            path=path,
            line_number=5,
            target="dtolnay/rust-toolchain@stable",
            ref="stable",
        ),
    ] == findings


def test_validate_workflow_action_pinning_passes_when_all_remote_refs_are_shas(
    tmp_path: Path,
) -> None:
    workflows_dir = tmp_path / ".github" / "workflows"
    _write_workflow(
        workflows_dir,
        "ci.yml",
        f"""
jobs:
  test:
    steps:
      - uses: {PINNED_CHECKOUT}  # v4
      - uses: ./local-action
""".lstrip(),
    )

    report = validate_workflow_action_pinning(workflows_dir)

    assert report.ok
    assert report.unpinned == ()


def test_validate_workflow_action_pinning_reports_relative_locations(
    tmp_path: Path,
) -> None:
    workflows_dir = tmp_path / ".github" / "workflows"
    _write_workflow(
        workflows_dir,
        "release.yaml",
        """
jobs:
  release:
    steps:
      - uses: actions/upload-artifact@v4
""".lstrip(),
    )

    report = validate_workflow_action_pinning(workflows_dir)

    assert not report.ok
    assert ".github/workflows/release.yaml:4" in report.format()
    assert "actions/upload-artifact@v4 uses moving ref @v4" in report.format()
