#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

try:
    import yaml  # type: ignore
except Exception as exc:  # pragma: no cover
    print(f"genesis manifest tool failed: PyYAML is required ({exc})")
    raise SystemExit(2)


@dataclass(frozen=True)
class RepoProfile:
    name: str
    root: Path
    include: list[str]
    exclude: list[str]


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def is_git_repo(repo_root: Path) -> bool:
    res = git(repo_root, "rev-parse", "--is-inside-work-tree")
    return res.returncode == 0 and res.stdout.strip().lower() == "true"


def git_commit(repo_root: Path, commit: str | None) -> str:
    if commit:
        commit = commit.strip()
        if not commit:
            raise ValueError("commit must be non-empty")
        return commit

    res = git(repo_root, "rev-parse", "HEAD")
    if res.returncode != 0:
        raise ValueError(f"could not determine git commit in {repo_root}: {res.stderr.strip()}")
    return res.stdout.strip()


def git_ls_files(repo_root: Path) -> list[str]:
    res = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if res.returncode != 0:
        raise ValueError(f"git ls-files failed in {repo_root}")

    raw = res.stdout.decode("utf-8", errors="replace")
    items = [p for p in raw.split("\0") if p]
    return items


def git_show_bytes(repo_root: Path, commit: str, relpath: str) -> bytes:
    res = subprocess.run(
        ["git", "show", f"{commit}:{relpath}"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if res.returncode != 0:
        raise ValueError(f"git show failed for {commit}:{relpath} in {repo_root}")
    return res.stdout


def worktree_read_bytes(repo_root: Path, relpath: str) -> bytes:
    full = (repo_root / relpath).resolve()
    if not full.exists():
        raise ValueError(f"missing file in worktree: {relpath}")
    return full.read_bytes()


def repo_is_dirty(repo_root: Path) -> bool:
    res = git(repo_root, "status", "--porcelain=v1")
    return res.returncode == 0 and bool(res.stdout.strip())


def path_matches_any(path: str, patterns: list[str]) -> bool:
    p = PurePosixPath(path)
    return any(p.match(pattern) for pattern in patterns)


def filter_allowlisted(paths: list[str], include: list[str], exclude: list[str]) -> list[str]:
    allow: list[str] = []
    for p in paths:
        if include and not path_matches_any(p, include):
            continue
        if exclude and path_matches_any(p, exclude):
            continue
        allow.append(p)
    return sorted(set(allow), key=str.casefold)


def load_profile(profile_path: Path) -> tuple[str, list[RepoProfile]]:
    data = yaml.safe_load(profile_path.read_text(encoding="utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError("profile must be a YAML mapping")

    profile_id = data.get("id")
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise ValueError("profile missing non-empty 'id'")

    repos_raw = data.get("repositories")
    if not isinstance(repos_raw, list) or not repos_raw:
        raise ValueError("profile missing non-empty 'repositories' list")

    repos: list[RepoProfile] = []
    seen: set[str] = set()
    for item in repos_raw:
        if not isinstance(item, dict):
            raise ValueError("each repository entry must be a mapping")
        name = item.get("name")
        root = item.get("root")
        include = item.get("include", [])
        exclude = item.get("exclude", [])

        if not isinstance(name, str) or not name.strip():
            raise ValueError("repository 'name' must be a non-empty string")
        if name in seen:
            raise ValueError(f"duplicate repository name: {name}")
        seen.add(name)

        if not isinstance(root, str) or not root.strip():
            raise ValueError(f"repository '{name}' root must be a non-empty string")
        if not isinstance(include, list) or not all(isinstance(x, str) for x in include):
            raise ValueError(f"repository '{name}' include must be a list of strings")
        if not isinstance(exclude, list) or not all(isinstance(x, str) for x in exclude):
            raise ValueError(f"repository '{name}' exclude must be a list of strings")
        if not include:
            raise ValueError(f"repository '{name}' include must be non-empty")

        repos.append(
            RepoProfile(
                name=name.strip(),
                root=Path(root.strip()),
                include=[x.strip() for x in include if x.strip()],
                exclude=[x.strip() for x in exclude if x.strip()],
            )
        )

    return profile_id.strip(), repos


def load_core_policy_hashes(repo_root: Path, commit: str) -> dict[str, str]:
    lexicon_rel = "constitution/lexicon_v1.yaml"
    ihsan_rel = "constitution/ihsan_v1.yaml"

    lexicon_bytes = git_show_bytes(repo_root, commit, lexicon_rel)
    ihsan_bytes = git_show_bytes(repo_root, commit, ihsan_rel)

    lexicon = yaml.safe_load(lexicon_bytes.decode("utf-8", errors="replace"))
    if not isinstance(lexicon, dict):
        raise ValueError("constitution/lexicon_v1.yaml must be a mapping")

    contract_rel = lexicon.get("contract")
    schema_rel = lexicon.get("schema")
    receipt_schema_rel = lexicon.get("receipt_schema")
    if not all(isinstance(v, str) and v.strip() for v in [contract_rel, schema_rel, receipt_schema_rel]):
        raise ValueError("lexicon must define non-empty contract/schema/receipt_schema paths")

    contract_bytes = git_show_bytes(repo_root, commit, str(contract_rel))
    schema_bytes = git_show_bytes(repo_root, commit, str(schema_rel))
    receipt_schema_bytes = git_show_bytes(repo_root, commit, str(receipt_schema_rel))

    return {
        "ihsan_constitution_sha256": sha256_bytes(ihsan_bytes),
        "lexicon_sha256": sha256_bytes(lexicon_bytes),
        "lexicon_contract_sha256": sha256_bytes(contract_bytes),
        "lexicon_schema_sha256": sha256_bytes(schema_bytes),
        "lexicon_receipt_schema_sha256": sha256_bytes(receipt_schema_bytes),
    }


def load_core_policy_hashes_worktree(repo_root: Path) -> dict[str, str]:
    lexicon_path = repo_root / "constitution" / "lexicon_v1.yaml"
    ihsan_path = repo_root / "constitution" / "ihsan_v1.yaml"

    lexicon_bytes = lexicon_path.read_bytes()
    ihsan_bytes = ihsan_path.read_bytes()

    lexicon = yaml.safe_load(lexicon_bytes.decode("utf-8", errors="replace"))
    if not isinstance(lexicon, dict):
        raise ValueError("constitution/lexicon_v1.yaml must be a mapping")

    contract_rel = lexicon.get("contract")
    schema_rel = lexicon.get("schema")
    receipt_schema_rel = lexicon.get("receipt_schema")
    if not all(isinstance(v, str) and v.strip() for v in [contract_rel, schema_rel, receipt_schema_rel]):
        raise ValueError("lexicon must define non-empty contract/schema/receipt_schema paths")

    contract_bytes = (repo_root / str(contract_rel)).read_bytes()
    schema_bytes = (repo_root / str(schema_rel)).read_bytes()
    receipt_schema_bytes = (repo_root / str(receipt_schema_rel)).read_bytes()

    return {
        "ihsan_constitution_sha256": sha256_bytes(ihsan_bytes),
        "lexicon_sha256": sha256_bytes(lexicon_bytes),
        "lexicon_contract_sha256": sha256_bytes(contract_bytes),
        "lexicon_schema_sha256": sha256_bytes(schema_bytes),
        "lexicon_receipt_schema_sha256": sha256_bytes(receipt_schema_bytes),
    }


def generate_manifest(
    *,
    repo_root: Path,
    profile_path: Path,
    out_path: Path,
    commit_override: str | None,
    source_mode: str,
    allow_missing_repos: bool,
) -> int:
    profile_path = profile_path if profile_path.is_absolute() else (repo_root / profile_path)
    profile_path = profile_path.resolve()

    profile_id, repos = load_profile(profile_path)

    if not is_git_repo(repo_root):
        print(f"genesis manifest generation failed: not a git repo: {repo_root}")
        return 2

    core_commit = git_commit(repo_root, commit_override)

    profile_rel = str(profile_path.relative_to(repo_root)).replace("\\", "/")
    if source_mode == "git":
        profile_bytes = git_show_bytes(repo_root, core_commit, profile_rel)
    elif source_mode == "worktree":
        profile_bytes = profile_path.read_bytes()
    else:
        raise ValueError(f"unsupported source mode: {source_mode}")
    profile_sha = sha256_bytes(profile_bytes)

    policy_hashes = (
        load_core_policy_hashes(repo_root, core_commit)
        if source_mode == "git"
        else load_core_policy_hashes_worktree(repo_root)
    )
    policy_sha = sha256_bytes(json.dumps(policy_hashes, sort_keys=True).encode("utf-8"))

    manifest_repos: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []

    for repo in repos:
        abs_root = (repo_root / repo.root).resolve()
        if not abs_root.exists():
            if allow_missing_repos:
                continue
            print(f"genesis manifest generation failed: repo root missing: {repo.root}")
            return 2
        if not is_git_repo(abs_root):
            if allow_missing_repos:
                continue
            print(f"genesis manifest generation failed: repo root is not a git repo: {repo.root}")
            return 2

        repo_commit = git_commit(abs_root, None)
        manifest_repos.append(
            {
                "name": repo.name,
                "root": str(repo.root).replace("\\", "/"),
                "commit": repo_commit,
                "dirty": repo_is_dirty(abs_root),
            }
        )

        tracked = git_ls_files(abs_root)
        allowlisted = filter_allowlisted(tracked, repo.include, repo.exclude)
        for relpath in allowlisted:
            blob = (
                git_show_bytes(abs_root, repo_commit, relpath)
                if source_mode == "git"
                else worktree_read_bytes(abs_root, relpath)
            )
            entries.append(
                {
                    "repo": repo.name,
                    "path": relpath,
                    "sha256": sha256_bytes(blob),
                    "bytes": len(blob),
                }
            )

    entries = sorted(entries, key=lambda e: (str(e["repo"]).casefold(), str(e["path"]).casefold()))

    manifest = {
        "type": "GenesisManifest",
        "version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": {"id": profile_id, "path": profile_rel, "sha256": profile_sha},
        "source": {"mode": source_mode, "core_commit": core_commit, "core_dirty": repo_is_dirty(repo_root)},
        "policy_sha256": policy_sha,
        "policy_hashes": policy_hashes,
        "repositories": manifest_repos,
        "entries": entries,
        "generator": {"name": "tools/genesis_manifest.py", "version": "1.0.0"},
    }

    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload, encoding="utf-8")
    return 0


def verify_manifest(
    *,
    repo_root: Path,
    manifest_path: Path,
    source_mode: str,
    optional: bool,
    allow_missing_repos: bool,
) -> int:
    if not manifest_path.exists():
        return 0 if optional else 2

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"genesis manifest verify failed: invalid JSON: {exc}")
        return 2

    if not isinstance(manifest, dict):
        print("genesis manifest verify failed: manifest must be a JSON object")
        return 2

    if manifest.get("type") != "GenesisManifest" or manifest.get("version") != 1:
        print("genesis manifest verify failed: expected type=GenesisManifest version=1")
        return 2

    profile = manifest.get("profile")
    if not isinstance(profile, dict):
        print("genesis manifest verify failed: missing profile object")
        return 2

    repos = manifest.get("repositories")
    entries = manifest.get("entries")
    if not isinstance(repos, list) or not isinstance(entries, list):
        print("genesis manifest verify failed: repositories and entries must be lists")
        return 2

    failures: list[str] = []

    entries_sorted = sorted(
        [e for e in entries if isinstance(e, dict)],
        key=lambda e: (str(e.get("repo", "")).casefold(), str(e.get("path", "")).casefold()),
    )
    if entries != entries_sorted:
        failures.append("entries must be sorted by (repo, path) case-insensitive")

    repo_index: dict[str, dict[str, str]] = {}
    for r in repos:
        if not isinstance(r, dict):
            failures.append("repository entry is not an object")
            continue
        name = r.get("name")
        root = r.get("root")
        commit = r.get("commit")
        if not all(isinstance(v, str) and v.strip() for v in [name, root, commit]):
            failures.append(f"invalid repository entry: {r}")
            continue
        if str(name) in repo_index:
            failures.append(f"duplicate repository name in manifest: {name}")
            continue
        repo_index[str(name)] = {"root": str(root), "commit": str(commit)}

        abs_root = (repo_root / root).resolve()
        if not abs_root.exists():
            if allow_missing_repos:
                continue
            failures.append(f"repo root missing: {root}")
            continue
        if not is_git_repo(abs_root):
            if allow_missing_repos:
                continue
            failures.append(f"repo root is not a git repo: {root}")
            continue

        for e in entries:
            if not isinstance(e, dict):
                failures.append("entry is not an object")
                continue
            if e.get("repo") != name:
                continue
            relpath = e.get("path")
            expected = e.get("sha256")
            if not isinstance(relpath, str) or not isinstance(expected, str):
                failures.append(f"invalid entry for repo {name}: {e}")
                continue

            try:
                blob = (
                    git_show_bytes(abs_root, commit, relpath)
                    if source_mode == "git"
                    else worktree_read_bytes(abs_root, relpath)
                )
            except Exception as exc:
                failures.append(f"missing at {name}@{commit}:{relpath} ({exc})")
                continue

            actual = sha256_bytes(blob)
            if actual != expected:
                failures.append(f"hash mismatch {name}:{relpath} expected={expected} actual={actual}")

    profile = manifest.get("profile")
    if isinstance(profile, dict):
        profile_path = profile.get("path")
        profile_sha = profile.get("sha256")
        if isinstance(profile_path, str) and isinstance(profile_sha, str):
            core = repo_index.get("core")
            if core:
                try:
                    profile_bytes = (
                        git_show_bytes(repo_root, core["commit"], profile_path)
                        if source_mode == "git"
                        else (repo_root / profile_path).read_bytes()
                    )
                    actual_profile_sha = sha256_bytes(profile_bytes)
                    if actual_profile_sha != profile_sha:
                        failures.append(
                            f"profile sha mismatch expected={profile_sha} actual={actual_profile_sha} path={profile_path}"
                        )
                except Exception as exc:
                    failures.append(f"could not load profile from git for verification ({exc})")

    policy_hashes = manifest.get("policy_hashes")
    policy_sha = manifest.get("policy_sha256")
    if isinstance(policy_hashes, dict) and isinstance(policy_sha, str):
        core = repo_index.get("core")
        if core:
            try:
                expected_hashes = (
                    load_core_policy_hashes(repo_root, core["commit"])
                    if source_mode == "git"
                    else load_core_policy_hashes_worktree(repo_root)
                )
                if policy_hashes != expected_hashes:
                    failures.append("policy_hashes mismatch vs core commit")
                expected_sha = sha256_bytes(json.dumps(expected_hashes, sort_keys=True).encode("utf-8"))
                if policy_sha != expected_sha:
                    failures.append(f"policy_sha256 mismatch expected={expected_sha} actual={policy_sha}")
            except Exception as exc:
                failures.append(f"could not recompute policy hashes ({exc})")

    if failures:
        print("genesis manifest verify failed:")
        for item in failures:
            print(f"- {item}")
        return 1

    return 0


def profile_lint(*, profile_path: Path) -> int:
    try:
        _profile_id, repos = load_profile(profile_path)
    except Exception as exc:
        print(f"genesis profile lint failed: {exc}")
        return 1

    for repo in repos:
        if any(Path(p).is_absolute() for p in [repo.root]):
            print(f"genesis profile lint failed: repo root must be relative: {repo.root}")
            return 1
        for pattern in repo.include + repo.exclude:
            if pattern.startswith("../") or pattern.startswith("..\\"):
                print(f"genesis profile lint failed: pattern must not escape repo: {pattern}")
                return 1

    return 0


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(description="Generate/verify a GenesisManifest (multi-repo)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_lint = sub.add_parser("profile-lint", help="Validate the manifest profile YAML")
    p_lint.add_argument(
        "--profile",
        default=str(repo_root / "constitution" / "genesis_manifest_profile_v1.yaml"),
        help="Path to manifest profile YAML",
    )

    p_gen = sub.add_parser("generate", help="Generate a GenesisManifest JSON from the profile")
    p_gen.add_argument(
        "--profile",
        default=str(repo_root / "constitution" / "genesis_manifest_profile_v1.yaml"),
        help="Path to manifest profile YAML",
    )
    p_gen.add_argument(
        "--output",
        default=str(repo_root / "evidence" / "genesis" / "GENESIS_MANIFEST.json"),
        help="Where to write the manifest JSON",
    )
    p_gen.add_argument(
        "--core-commit",
        help="Optional commit/ref to bind the core policy hashes/profile hash to (default: HEAD)",
    )
    p_gen.add_argument(
        "--source",
        default="git",
        choices=["git", "worktree"],
        help="Hash source: git (tracked content at commit) or worktree (current filesystem bytes)",
    )
    p_gen.add_argument(
        "--allow-missing-repos",
        action="store_true",
        help="Skip profile repos that are missing or not git repos",
    )

    p_verify = sub.add_parser("verify", help="Verify a GenesisManifest JSON against git content")
    p_verify.add_argument(
        "--manifest",
        default=str(repo_root / "evidence" / "genesis" / "GENESIS_MANIFEST.json"),
        help="Path to the manifest JSON",
    )
    p_verify.add_argument(
        "--optional",
        action="store_true",
        help="Exit 0 if manifest file is missing",
    )
    p_verify.add_argument(
        "--source",
        default="git",
        choices=["git", "worktree"],
        help="Verify against git (tracked content at commit) or worktree (current filesystem bytes)",
    )
    p_verify.add_argument(
        "--allow-missing-repos",
        action="store_true",
        help="Skip verification for repos that are missing or not git repos",
    )

    args = parser.parse_args()

    if args.cmd == "profile-lint":
        return profile_lint(profile_path=Path(args.profile))

    if args.cmd == "generate":
        return generate_manifest(
            repo_root=repo_root,
            profile_path=Path(args.profile),
            out_path=Path(args.output),
            commit_override=args.core_commit,
            source_mode=str(args.source),
            allow_missing_repos=bool(args.allow_missing_repos),
        )

    if args.cmd == "verify":
        return verify_manifest(
            repo_root=repo_root,
            manifest_path=Path(args.manifest),
            source_mode=str(args.source),
            optional=bool(args.optional),
            allow_missing_repos=bool(args.allow_missing_repos),
        )

    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
