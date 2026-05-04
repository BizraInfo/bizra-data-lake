#!/usr/bin/env python3
"""Gate GitHub workflow actions on immutable commit SHAs.

GitHub Action tags such as ``@v4`` and branches such as ``@main`` are moving
references. This gate keeps the CI supply chain auditable by requiring every
remote ``uses: owner/repo[/path]@ref`` reference under ``.github/workflows`` to
use a full 40-character commit SHA.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKFLOWS_DIR = ROOT / ".github" / "workflows"

USES_RE = re.compile(r"^\s*(?:-\s*)?uses:\s*(?P<target>[^\s#]+)")
PINNED_REF_RE = re.compile(r"^[0-9a-fA-F]{40}$")
LOCAL_OR_DOCKER_PREFIXES = ("./", "../", "docker://")


@dataclass(frozen=True)
class WorkflowActionRef:
    path: Path
    line_number: int
    target: str
    ref: str

    def format(self, root: Path) -> str:
        rel = self.path.relative_to(root)
        return f"{rel}:{self.line_number}: {self.target} uses moving ref @{self.ref}"


@dataclass(frozen=True)
class WorkflowPinningReport:
    unpinned: tuple[WorkflowActionRef, ...]
    root: Path

    @property
    def ok(self) -> bool:
        return not self.unpinned

    def format(self) -> str:
        if self.ok:
            return "All remote GitHub Actions are pinned to full commit SHAs."
        header = "Unpinned GitHub Action references:"
        lines = [item.format(self.root) for item in self.unpinned]
        return "\n".join([header, *[f"- {line}" for line in lines]])


def _workflow_files(workflows_dir: Path) -> list[Path]:
    if not workflows_dir.exists():
        return []
    return sorted(
        [
            *workflows_dir.glob("*.yml"),
            *workflows_dir.glob("*.yaml"),
        ]
    )


def _extract_remote_action(
    line: str, path: Path, line_number: int
) -> WorkflowActionRef | None:
    match = USES_RE.match(line)
    if not match:
        return None

    target = match.group("target").strip().strip("\"'")
    if target.startswith(LOCAL_OR_DOCKER_PREFIXES):
        return None
    if "@" not in target:
        return WorkflowActionRef(
            path=path, line_number=line_number, target=target, ref=""
        )

    _, ref = target.rsplit("@", 1)
    if PINNED_REF_RE.fullmatch(ref):
        return None
    return WorkflowActionRef(path=path, line_number=line_number, target=target, ref=ref)


def scan_workflow_file(path: Path) -> list[WorkflowActionRef]:
    unpinned: list[WorkflowActionRef] = []
    for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        finding = _extract_remote_action(line, path, idx)
        if finding is not None:
            unpinned.append(finding)
    return unpinned


def validate_workflow_action_pinning(
    workflows_dir: Path = DEFAULT_WORKFLOWS_DIR,
) -> WorkflowPinningReport:
    workflows_dir = workflows_dir.resolve()
    unpinned: list[WorkflowActionRef] = []
    for path in _workflow_files(workflows_dir):
        unpinned.extend(scan_workflow_file(path))
    return WorkflowPinningReport(
        unpinned=tuple(unpinned), root=workflows_dir.parent.parent
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflows-dir",
        type=Path,
        default=DEFAULT_WORKFLOWS_DIR,
        help="Directory containing GitHub Actions workflow YAML files.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    report = validate_workflow_action_pinning(args.workflows_dir)
    print(report.format())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
