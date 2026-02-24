#!/usr/bin/env python3
"""SNR-based worktree hygiene guard.

Classifies current git worktree changes into:
- KEEP_TRACK
- KEEP_UNTRACKED
- ARCHIVE
- REVIEW

Then emits deterministic JSON/Markdown reports and can optionally:
- archive selected untracked paths (non-destructive move)
- sync KEEP_UNTRACKED patterns to .git/info/exclude
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class GitEntry:
    status: str
    path: str

    @property
    def tracked(self) -> bool:
        return self.status != "??"


@dataclass(frozen=True)
class Decision:
    path: str
    status: str
    tracked: bool
    recommendation: str
    snr_tier: str
    hhmm_layer: str
    reason: str


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_status_line(line: str) -> GitEntry | None:
    if not line:
        return None
    status = line[:2]
    raw = line[3:].strip()
    if not raw:
        return None

    # Rename/copy format in porcelain v1: "from -> to"
    if " -> " in raw:
        raw = raw.split(" -> ", 1)[1]

    # Quoted path with spaces
    if raw.startswith('"') and raw.endswith('"') and len(raw) >= 2:
        raw = raw[1:-1].replace('\\"', '"')

    raw = raw.replace("\\", "/")
    return GitEntry(status=status, path=raw)


def list_git_entries(repo_root: Path) -> list[GitEntry]:
    proc = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    out: list[GitEntry] = []
    for line in proc.stdout.splitlines():
        entry = _parse_status_line(line)
        if entry is not None:
            out.append(entry)
    return out


def _load_policy(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid policy yaml: {path}")
    groups = raw.get("groups")
    if not isinstance(groups, dict):
        raise ValueError(f"Policy missing groups: {path}")
    return raw


def _match(path: str, pattern: str) -> bool:
    if fnmatch.fnmatch(path, pattern):
        return True
    if not path.endswith("/") and fnmatch.fnmatch(path + "/", pattern):
        return True
    return False


def _find_rule(path: str, rules: list[dict[str, Any]]) -> dict[str, Any] | None:
    for rule in rules:
        pattern = str(rule.get("pattern", ""))
        if pattern and _match(path, pattern):
            return rule
    return None


def classify(entries: list[GitEntry], policy: dict[str, Any]) -> list[Decision]:
    groups = policy["groups"]
    keep_track_rules = list(groups.get("keep_track", []))
    keep_untracked_rules = list(groups.get("keep_untracked", []))
    archive_rules = list(groups.get("archive", []))

    out: list[Decision] = []
    for entry in entries:
        if entry.tracked:
            rule = _find_rule(entry.path, keep_track_rules)
            if rule is None:
                out.append(
                    Decision(
                        path=entry.path,
                        status=entry.status,
                        tracked=True,
                        recommendation="KEEP_TRACK",
                        snr_tier="medium",
                        hhmm_layer="slow",
                        reason="Tracked change; requires explicit review, not auto-cleanup.",
                    )
                )
                continue
            out.append(
                Decision(
                    path=entry.path,
                    status=entry.status,
                    tracked=True,
                    recommendation="KEEP_TRACK",
                    snr_tier=str(rule.get("snr_tier", "medium")),
                    hhmm_layer=str(rule.get("hhmm_layer", "slow")),
                    reason=str(rule.get("reason", "Tracked signal path.")),
                )
            )
            continue

        # Untracked path handling
        keep_track_rule = _find_rule(entry.path, keep_track_rules)
        if keep_track_rule is not None:
            out.append(
                Decision(
                    path=entry.path,
                    status=entry.status,
                    tracked=False,
                    recommendation="KEEP_TRACK",
                    snr_tier=str(keep_track_rule.get("snr_tier", "medium")),
                    hhmm_layer=str(keep_track_rule.get("hhmm_layer", "slow")),
                    reason=str(
                        keep_track_rule.get(
                            "reason",
                            "High-signal path; untracked but should be committed.",
                        )
                    ),
                )
            )
            continue

        keep_untracked_rule = _find_rule(entry.path, keep_untracked_rules)
        if keep_untracked_rule is not None:
            out.append(
                Decision(
                    path=entry.path,
                    status=entry.status,
                    tracked=False,
                    recommendation="KEEP_UNTRACKED",
                    snr_tier=str(keep_untracked_rule.get("snr_tier", "medium")),
                    hhmm_layer=str(keep_untracked_rule.get("hhmm_layer", "fast")),
                    reason=str(
                        keep_untracked_rule.get(
                            "reason", "Local generated state to retain outside commits."
                        )
                    ),
                )
            )
            continue

        archive_rule = _find_rule(entry.path, archive_rules)
        if archive_rule is not None:
            out.append(
                Decision(
                    path=entry.path,
                    status=entry.status,
                    tracked=False,
                    recommendation="ARCHIVE",
                    snr_tier=str(archive_rule.get("snr_tier", "low")),
                    hhmm_layer=str(archive_rule.get("hhmm_layer", "fast")),
                    reason=str(
                        archive_rule.get("reason", "Archive outside repo root.")
                    ),
                )
            )
            continue

        out.append(
            Decision(
                path=entry.path,
                status=entry.status,
                tracked=False,
                recommendation="REVIEW",
                snr_tier="unknown",
                hhmm_layer="unknown",
                reason="No policy rule matched; manual SNR triage required.",
            )
        )

    out.sort(key=lambda d: d.path)
    return out


def _summary(decisions: list[Decision]) -> dict[str, int]:
    keys = ["KEEP_TRACK", "KEEP_UNTRACKED", "ARCHIVE", "REVIEW"]
    out = {k: 0 for k in keys}
    for d in decisions:
        out[d.recommendation] = out.get(d.recommendation, 0) + 1
    return out


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def _write_markdown(
    path: Path, decisions: list[Decision], summary: dict[str, int]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Worktree SNR Triage Report",
        "",
        f"- Generated: `{utc_now()}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Decisions",
            "",
            "| Path | Status | Recommendation | SNR | HHMM | Reason |",
            "|---|---|---|---|---|---|",
        ]
    )
    for d in decisions:
        lines.append(
            f"| `{d.path}` | `{d.status}` | `{d.recommendation}` | `{d.snr_tier}` | `{d.hhmm_layer}` | {d.reason} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _sync_exclude(repo_root: Path, patterns: list[str]) -> None:
    exclude_path = repo_root / ".git" / "info" / "exclude"
    exclude_path.parent.mkdir(parents=True, exist_ok=True)
    existing = (
        exclude_path.read_text(encoding="utf-8").splitlines()
        if exclude_path.exists()
        else []
    )
    existing_set = set(existing)
    new_lines: list[str] = []
    if "# snr-worktree-guard" not in existing_set:
        new_lines.append("# snr-worktree-guard")
    for pattern in patterns:
        if pattern not in existing_set:
            new_lines.append(pattern)
    if new_lines:
        with exclude_path.open("a", encoding="utf-8") as f:
            for line in new_lines:
                f.write(line + "\n")


def _archive_paths(
    repo_root: Path, decisions: list[Decision], archive_root: Path
) -> list[str]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    moved: list[str] = []
    for d in decisions:
        if d.recommendation != "ARCHIVE" or d.tracked:
            continue
        src = repo_root / d.path
        if not src.exists():
            continue
        dst = archive_root / stamp / d.path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        moved.append(d.path)
    return moved


def run(
    *,
    repo_root: Path,
    policy_path: Path,
    out_json: Path,
    out_md: Path,
    sync_exclude: bool,
    apply_archive: bool,
    archive_root: Path,
) -> int:
    policy = _load_policy(policy_path)
    entries = list_git_entries(repo_root)
    decisions = classify(entries, policy)
    summary = _summary(decisions)

    report = {
        "generated_at": utc_now(),
        "repo_root": repo_root.as_posix(),
        "policy": policy_path.as_posix(),
        "summary": summary,
        "decisions": [asdict(d) for d in decisions],
    }
    _write_json(out_json, report)
    _write_markdown(out_md, decisions, summary)

    if sync_exclude:
        patterns = [
            str(r.get("pattern", ""))
            for r in policy["groups"].get("keep_untracked", [])
            if r.get("pattern")
        ]
        _sync_exclude(repo_root, patterns)

    moved = _archive_paths(repo_root, decisions, archive_root) if apply_archive else []

    output = {
        "report_json": out_json.as_posix(),
        "report_md": out_md.as_posix(),
        "summary": summary,
        "synced_exclude": bool(sync_exclude),
        "archived_count": len(moved),
        "archived_paths": moved,
    }
    print(json.dumps(output, indent=2, ensure_ascii=False))

    # Non-zero if policy left unresolved items.
    return 1 if summary.get("REVIEW", 0) > 0 else 0


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    policy_default = Path(__file__).resolve().with_name("snr_worktree_policy.yaml")
    out_dir = repo_root / "artifacts" / "ops"
    parser = argparse.ArgumentParser(description="SNR worktree guard")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--policy", type=Path, default=policy_default)
    parser.add_argument(
        "--out-json", type=Path, default=out_dir / "worktree_snr_report.json"
    )
    parser.add_argument(
        "--out-md", type=Path, default=out_dir / "worktree_snr_report.md"
    )
    parser.add_argument(
        "--sync-exclude",
        action="store_true",
        help="Append KEEP_UNTRACKED patterns to .git/info/exclude",
    )
    parser.add_argument(
        "--apply-archive",
        action="store_true",
        help="Move ARCHIVE paths out of repo root",
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path.home() / ".local" / "share" / "bizra-archive",
    )
    args = parser.parse_args()

    raise SystemExit(
        run(
            repo_root=args.repo_root,
            policy_path=args.policy,
            out_json=args.out_json,
            out_md=args.out_md,
            sync_exclude=args.sync_exclude,
            apply_archive=args.apply_archive,
            archive_root=args.archive_root,
        )
    )


if __name__ == "__main__":
    main()
