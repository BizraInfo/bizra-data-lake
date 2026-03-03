#!/usr/bin/env python3
"""Branch protection policy guard.

Modes:
- audit: verify GitHub branch protection matches local policy
- apply: apply local policy to target branches via GitHub API
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BranchProtectionPolicy:
    branches: list[str]
    ignore_missing_branches: bool
    strict_status_checks: bool
    required_status_checks: list[str]
    required_approving_review_count: int
    dismiss_stale_reviews: bool
    require_code_owner_reviews: bool
    require_conversation_resolution: bool
    enforce_admins: bool
    require_linear_history: bool
    allow_force_pushes: bool
    allow_deletions: bool
    block_creations: bool
    lock_branch: bool
    allow_fork_syncing: bool

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BranchProtectionPolicy":
        return cls(
            branches=list(raw.get("branches", ["main"])),
            ignore_missing_branches=bool(raw.get("ignore_missing_branches", True)),
            strict_status_checks=bool(raw.get("strict_status_checks", True)),
            required_status_checks=sorted(set(raw.get("required_status_checks", []))),
            required_approving_review_count=int(
                raw.get("required_approving_review_count", 1)
            ),
            dismiss_stale_reviews=bool(raw.get("dismiss_stale_reviews", True)),
            require_code_owner_reviews=bool(raw.get("require_code_owner_reviews", False)),
            require_conversation_resolution=bool(
                raw.get("require_conversation_resolution", True)
            ),
            enforce_admins=bool(raw.get("enforce_admins", True)),
            require_linear_history=bool(raw.get("require_linear_history", True)),
            allow_force_pushes=bool(raw.get("allow_force_pushes", False)),
            allow_deletions=bool(raw.get("allow_deletions", False)),
            block_creations=bool(raw.get("block_creations", False)),
            lock_branch=bool(raw.get("lock_branch", False)),
            allow_fork_syncing=bool(raw.get("allow_fork_syncing", True)),
        )


def load_policy(path: Path) -> BranchProtectionPolicy:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid policy format at {path}")
    return BranchProtectionPolicy.from_dict(raw)


def build_protection_payload(policy: BranchProtectionPolicy) -> dict[str, Any]:
    return {
        "required_status_checks": {
            "strict": policy.strict_status_checks,
            "contexts": policy.required_status_checks,
        },
        "enforce_admins": policy.enforce_admins,
        "required_pull_request_reviews": {
            "dismiss_stale_reviews": policy.dismiss_stale_reviews,
            "require_code_owner_reviews": policy.require_code_owner_reviews,
            "required_approving_review_count": policy.required_approving_review_count,
            "require_last_push_approval": False,
        },
        "restrictions": None,
        "required_linear_history": policy.require_linear_history,
        "allow_force_pushes": policy.allow_force_pushes,
        "allow_deletions": policy.allow_deletions,
        "block_creations": policy.block_creations,
        "required_conversation_resolution": policy.require_conversation_resolution,
        "lock_branch": policy.lock_branch,
        "allow_fork_syncing": policy.allow_fork_syncing,
    }


def run_gh_api(
    args: list[str], payload: dict[str, Any] | None = None
) -> tuple[int, str, str]:
    command = ["gh", "api", *args]
    input_text = None
    if payload is not None:
        input_text = json.dumps(payload)
    proc = subprocess.run(
        command,
        text=True,
        capture_output=True,
        input=input_text,
    )
    return proc.returncode, proc.stdout, proc.stderr


def parse_api_json(stdout: str) -> dict[str, Any]:
    data = json.loads(stdout or "{}")
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object from GitHub API")
    return data


def branch_exists(repo: str, branch: str) -> bool:
    code, _, _ = run_gh_api([f"/repos/{repo}/branches/{branch}"])
    return code == 0


def extract_actual_contexts(required_status_checks: dict[str, Any]) -> set[str]:
    contexts: set[str] = set()
    for value in required_status_checks.get("contexts", []) or []:
        if isinstance(value, str):
            contexts.add(value)
    for value in required_status_checks.get("checks", []) or []:
        if isinstance(value, dict):
            context = value.get("context")
            if isinstance(context, str) and context:
                contexts.add(context)
    return contexts


def evaluate_drift(
    actual: dict[str, Any], policy: BranchProtectionPolicy
) -> list[str]:
    drifts: list[str] = []
    actual_rss = actual.get("required_status_checks") or {}
    actual_contexts = extract_actual_contexts(actual_rss)
    missing_contexts = sorted(set(policy.required_status_checks) - actual_contexts)
    if missing_contexts:
        drifts.append(f"Missing required status checks: {', '.join(missing_contexts)}")

    if bool(actual_rss.get("strict")) != policy.strict_status_checks:
        drifts.append("strict_status_checks mismatch")

    reviews = actual.get("required_pull_request_reviews") or {}
    review_count = int(reviews.get("required_approving_review_count") or 0)
    if review_count < policy.required_approving_review_count:
        drifts.append(
            f"required_approving_review_count too low ({review_count} < {policy.required_approving_review_count})"
        )

    if bool(reviews.get("dismiss_stale_reviews")) != policy.dismiss_stale_reviews:
        drifts.append("dismiss_stale_reviews mismatch")
    if bool(reviews.get("require_code_owner_reviews")) != policy.require_code_owner_reviews:
        drifts.append("require_code_owner_reviews mismatch")
    if bool_field(actual.get("required_conversation_resolution")) != policy.require_conversation_resolution:
        drifts.append("required_conversation_resolution mismatch")
    if bool_field(actual.get("required_linear_history")) != policy.require_linear_history:
        drifts.append("required_linear_history mismatch")
    if bool_field(actual.get("allow_force_pushes")) != policy.allow_force_pushes:
        drifts.append("allow_force_pushes mismatch")
    if bool_field(actual.get("allow_deletions")) != policy.allow_deletions:
        drifts.append("allow_deletions mismatch")
    if bool_field(actual.get("block_creations")) != policy.block_creations:
        drifts.append("block_creations mismatch")
    if bool_field(actual.get("lock_branch")) != policy.lock_branch:
        drifts.append("lock_branch mismatch")
    if bool_field(actual.get("allow_fork_syncing")) != policy.allow_fork_syncing:
        drifts.append("allow_fork_syncing mismatch")

    enforce_admins = actual.get("enforce_admins") or {}
    enabled = bool_field(enforce_admins.get("enabled"))
    if enabled != policy.enforce_admins:
        drifts.append("enforce_admins mismatch")

    return drifts


def audit_branch(repo: str, branch: str, policy: BranchProtectionPolicy) -> list[str]:
    code, stdout, stderr = run_gh_api([f"/repos/{repo}/branches/{branch}/protection"])
    if code != 0:
        raise RuntimeError(
            f"Failed to read branch protection for '{branch}': {stderr.strip()}"
        )
    actual = parse_api_json(stdout)
    return evaluate_drift(actual, policy)


def apply_branch(repo: str, branch: str, policy: BranchProtectionPolicy) -> None:
    payload = build_protection_payload(policy)
    code, _, stderr = run_gh_api(
        [
            "-X",
            "PUT",
            f"/repos/{repo}/branches/{branch}/protection",
            "--input",
            "-",
        ],
        payload=payload,
    )
    if code != 0:
        raise RuntimeError(
            f"Failed to apply branch protection for '{branch}': {stderr.strip()}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Branch protection policy guard.")
    parser.add_argument(
        "mode",
        choices=["audit", "apply", "print"],
        help="Run mode: audit, apply, or print payload.",
    )
    parser.add_argument(
        "--policy",
        default=".github/branch_protection_policy.json",
        help="Path to policy JSON file.",
    )
    parser.add_argument(
        "--repo",
        default=None,
        help="GitHub repo in owner/name form (default: $GITHUB_REPOSITORY).",
    )
    parser.add_argument(
        "--branches",
        default=None,
        help="Comma-separated override branch list.",
    )
    return parser.parse_args()


def resolve_repo(cli_repo: str | None) -> str:
    if cli_repo:
        return cli_repo
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        raise ValueError("Missing --repo (or set GITHUB_REPOSITORY).")
    return repo


def bool_field(value: Any) -> bool:
    """Normalize GitHub API boolean-like fields.

    Some fields are booleans, others are objects with an `enabled` flag.
    """
    if isinstance(value, dict):
        return bool(value.get("enabled"))
    return bool(value)


def resolve_branches(policy: BranchProtectionPolicy, override: str | None) -> list[str]:
    if override:
        return [b.strip() for b in override.split(",") if b.strip()]
    return policy.branches


def main() -> int:
    args = parse_args()
    policy = load_policy(Path(args.policy))

    if args.mode == "print":
        print(json.dumps(build_protection_payload(policy), indent=2))
        return 0

    repo = resolve_repo(args.repo)
    branches = resolve_branches(policy, args.branches)

    has_failures = False
    for branch in branches:
        exists = branch_exists(repo, branch)
        if not exists:
            if policy.ignore_missing_branches:
                print(f"[branch-protection] {branch}: SKIP (missing branch)")
                continue
            print(f"[branch-protection] {branch}: FAIL (missing branch)")
            has_failures = True
            continue

        if args.mode == "apply":
            print(f"[branch-protection] {branch}: APPLY")
            apply_branch(repo, branch, policy)

        drifts = audit_branch(repo, branch, policy)
        if drifts:
            has_failures = True
            print(f"[branch-protection] {branch}: FAIL")
            for drift in drifts:
                print(f"  - {drift}")
        else:
            print(f"[branch-protection] {branch}: PASS")

    return 1 if has_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
