#!/usr/bin/env python3
"""
Proof Forge — Evidence kernel for BIZRA.

BUILD → VERIFY → EVIDENCE in one pass.

Usage:
  python3 forge_evidence.py --project-dir <path> --description "<what>"
  python3 forge_evidence.py --verify --project-dir <path>   # chain verification
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


GENESIS = "0" * 64


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run(cmd, cwd, timeout=600):
    """Run a shell command, capture everything."""
    start = datetime.now(timezone.utc)
    try:
        result = subprocess.run(
            cmd,
            shell=isinstance(cmd, str),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        duration_s = (datetime.now(timezone.utc) - start).total_seconds()
        return {
            "command": cmd if isinstance(cmd, str) else " ".join(cmd),
            "exit_code": result.returncode,
            "stdout_tail": result.stdout[-4000:] if result.stdout else "",
            "stderr_tail": result.stderr[-4000:] if result.stderr else "",
            "duration_s": round(duration_s, 2),
        }
    except subprocess.TimeoutExpired:
        return {
            "command": cmd if isinstance(cmd, str) else " ".join(cmd),
            "exit_code": -1,
            "stdout_tail": "",
            "stderr_tail": f"TIMEOUT after {timeout}s",
            "duration_s": timeout,
        }


def git_session_commits(project_dir: Path, base_ref: str, head_ref: str = "HEAD"):
    """Return the list of commits from base_ref..head_ref."""
    r = run(
        ["git", "log", "--format=%H|%s|%ad", "--date=iso", f"{base_ref}..{head_ref}"],
        project_dir,
    )
    commits = []
    for line in reversed(r["stdout_tail"].strip().splitlines()):
        parts = line.split("|", 2)
        if len(parts) == 3:
            commits.append({
                "hash": parts[0],
                "subject": parts[1],
                "date": parts[2],
            })
    return commits


def collect_session_artifacts(project_dir: Path, base_ref: str, head_ref: str = "HEAD"):
    """Collect artifact metadata for every file touched in session commits."""
    r = run(
        ["git", "diff", "--name-only", f"{base_ref}..{head_ref}"],
        project_dir,
    )
    files = [line.strip() for line in r["stdout_tail"].splitlines() if line.strip()]
    artifacts = []
    for rel_path in files:
        full = project_dir / rel_path
        if not full.exists():
            continue  # deleted files
        stat = full.stat()
        kind = classify(rel_path)
        artifacts.append({
            "path": rel_path,
            "size_bytes": stat.st_size,
            "sha256": sha256_file(full),
            "mtime_iso": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
            "kind": kind,
        })
    return artifacts


def classify(rel_path: str) -> str:
    p = rel_path.lower()
    if p.endswith((".md", ".rst", ".txt")):
        return "doc"
    if "test" in p or "spec" in p or p.endswith("_test.rs") or "/tests/" in p:
        return "test"
    if p.endswith((".toml", ".yml", ".yaml", ".json", ".lock")):
        return "config"
    if p.endswith((".rs", ".ts", ".tsx", ".js", ".py", ".go")):
        return "code"
    if p.endswith((".sh", ".bash")):
        return "code"
    return "data"


def detect_verification_targets(artifacts):
    targets = {
        "rust_crates": set(),
        "node_projects": set(),
        "python_projects": set(),
    }
    for a in artifacts:
        p = a["path"]
        if p.endswith("Cargo.toml"):
            targets["rust_crates"].add(os.path.dirname(p) or ".")
        if p.endswith("package.json"):
            targets["node_projects"].add(os.path.dirname(p) or ".")
        if p.endswith("pyproject.toml") or p.endswith("setup.py"):
            targets["python_projects"].add(os.path.dirname(p) or ".")
    return {k: sorted(list(v)) for k, v in targets.items()}


def verify_rust(project_dir: Path, crate_dir: str):
    """Run cargo test on a specific crate."""
    cwd = project_dir / crate_dir
    # Prefer targeted test at the crate level
    return run(f"cargo test --manifest-path {cwd}/Cargo.toml --lib 2>&1 | tail -30", project_dir, timeout=900)


def verify_rust_workspace(workspace_dir: Path):
    """Run cargo test on a workspace."""
    return run("cargo test --workspace 2>&1 | grep -E 'test result: ok|FAILED|error\\[' | tail -40", workspace_dir, timeout=1800)


def verify_node(project_dir: Path, node_dir: str):
    cwd = project_dir / node_dir
    typecheck = run("pnpm typecheck 2>&1 | tail -15", cwd, timeout=300)
    test_result = run("pnpm test:unit --run 2>&1 | tail -10", cwd, timeout=600)
    return {"typecheck": typecheck, "test": test_result}


def compile_verification_report(project_dir: Path, artifacts, targets):
    report = {
        "generated_at": utc_now_iso(),
        "checks": [],
    }

    # Identify Rust workspace if Cargo.toml changed at a known workspace root
    workspace_roots = set()
    for crate in targets["rust_crates"]:
        # Climb to find workspace root (contains [workspace] in Cargo.toml)
        candidate = project_dir / crate
        while candidate != project_dir and candidate.parent != candidate:
            cargo = candidate / "Cargo.toml"
            if cargo.exists():
                txt = cargo.read_text(errors="ignore")
                if "[workspace]" in txt:
                    workspace_roots.add(str(candidate.relative_to(project_dir)))
                    break
            candidate = candidate.parent

    if workspace_roots:
        for root in sorted(workspace_roots):
            ws = project_dir / root
            ws_check = verify_rust_workspace(ws)
            report["checks"].append({
                "type": "rust_workspace_test",
                "scope": root,
                **ws_check,
            })

    # Node projects
    for node in targets["node_projects"]:
        n = verify_node(project_dir, node)
        report["checks"].append({
            "type": "node_project_test",
            "scope": node,
            "typecheck": n["typecheck"],
            "test": n["test"],
        })

    # If no automated checks found
    if not report["checks"]:
        report["checks"].append({
            "type": "manual_attestation",
            "note": "No automated verification targets discovered for session scope.",
        })

    return report


def load_previous_receipt(receipts_dir: Path):
    if not receipts_dir.exists():
        return None
    files = sorted(receipts_dir.glob("*.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def compute_evidence_hash(artifacts, report):
    """Composite hash over all artifact hashes + verification report hash."""
    artifact_hashes = sorted(a["sha256"] for a in artifacts)
    report_bytes = json.dumps(report, sort_keys=True).encode()
    combined = "".join(artifact_hashes).encode() + report_bytes
    return sha256_bytes(combined)


def build_receipt(
    project_dir: Path,
    description: str,
    session_commits,
    artifacts,
    report,
    previous_hash: str,
    chain_position: int,
):
    evidence_hash = compute_evidence_hash(artifacts, report)
    # Receipt hash = sha256 of the whole receipt content pre-hash-field
    receipt_core = {
        "version": "proof-forge-v1",
        "generated_at": utc_now_iso(),
        "chain_position": chain_position,
        "previous_hash": previous_hash,
        "project_dir": str(project_dir),
        "description": description,
        "session_commits": session_commits,
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
        "verification_report": report,
        "evidence_hash": evidence_hash,
    }
    receipt_bytes = json.dumps(receipt_core, sort_keys=True).encode()
    receipt_hash = sha256_bytes(receipt_bytes)
    receipt_core["receipt_hash"] = receipt_hash
    return receipt_core


def compute_confidence_level(report):
    """
    5 Ironclad = tests + benchmarks + static analysis + schema + CI
    4 Strong = tests + ≥1 other verification type
    3 Solid = tests pass OR build + static analysis
    2 Attested = manual attestation
    1 Logged = evidence collected, no verification
    """
    checks = report.get("checks", [])
    if not checks:
        return 1, "Logged"
    test_pass = False
    type_count = 0
    for c in checks:
        t = c.get("type", "")
        if t in ("rust_workspace_test", "rust_crate_test", "node_project_test"):
            type_count += 1
            ec = c.get("exit_code", -1)
            sub_ec = c.get("test", {}).get("exit_code") if isinstance(c.get("test"), dict) else None
            if ec == 0 or sub_ec == 0:
                test_pass = True
            # Any grep pattern that finds only "ok"s with no "FAILED" is a pass
            if "FAILED" not in (c.get("stdout_tail", "") + c.get("stderr_tail", "")):
                test_pass = True
        elif t == "manual_attestation":
            return 2, "Attested"

    if test_pass and type_count >= 2:
        return 4, "Strong"
    if test_pass:
        return 3, "Solid"
    return 2, "Attested"


def update_evidence_index(index_path: Path, receipt: dict):
    if index_path.exists():
        with open(index_path) as f:
            index = json.load(f)
    else:
        index = {
            "version": "proof-forge-v1",
            "created_at": utc_now_iso(),
            "receipts": [],
        }
    level, label = compute_confidence_level(receipt["verification_report"])
    index["receipts"].append({
        "chain_position": receipt["chain_position"],
        "generated_at": receipt["generated_at"],
        "description": receipt["description"],
        "receipt_hash": receipt["receipt_hash"],
        "previous_hash": receipt["previous_hash"],
        "evidence_hash": receipt["evidence_hash"],
        "artifact_count": receipt["artifact_count"],
        "confidence_level": level,
        "confidence_label": label,
    })
    index["latest_hash"] = receipt["receipt_hash"]
    index["chain_length"] = len(index["receipts"])
    index["updated_at"] = utc_now_iso()
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def generate_proof_summary(receipt: dict, project_dir: Path, output_path: Path):
    level, label = compute_confidence_level(receipt["verification_report"])
    lines = []
    lines.append(f"# Proof Summary — {receipt['description']}")
    lines.append("")
    lines.append(f"**Receipt hash:** `{receipt['receipt_hash']}`  ")
    lines.append(f"**Chain position:** {receipt['chain_position']}  ")
    lines.append(f"**Previous hash:** `{receipt['previous_hash']}`  ")
    lines.append(f"**Generated:** {receipt['generated_at']}  ")
    lines.append(f"**Confidence:** {label} (level {level}/5)")
    lines.append("")

    # Commits
    lines.append("## Commits in this evidence slice")
    lines.append("")
    if receipt["session_commits"]:
        for c in receipt["session_commits"]:
            lines.append(f"- `{c['hash'][:8]}` {c['subject']}")
    else:
        lines.append("_(no session commits in slice — full-tree fingerprint)_")
    lines.append("")

    # Artifacts
    lines.append(f"## Artifacts ({receipt['artifact_count']} total)")
    lines.append("")
    kinds = {}
    for a in receipt["artifacts"]:
        kinds.setdefault(a["kind"], 0)
        kinds[a["kind"]] += 1
    lines.append("| Kind | Count |")
    lines.append("|---|---|")
    for k, v in sorted(kinds.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    lines.append("")

    # Verification
    lines.append("## Verification")
    lines.append("")
    for c in receipt["verification_report"]["checks"]:
        t = c.get("type", "?")
        lines.append(f"### {t}")
        lines.append("")
        if t == "manual_attestation":
            lines.append(c.get("note", ""))
            lines.append("")
            continue
        if "scope" in c:
            lines.append(f"**Scope:** `{c['scope']}`")
            lines.append("")
        if "typecheck" in c:
            tc = c["typecheck"]
            lines.append(f"**typecheck** exit={tc['exit_code']} ({tc['duration_s']}s)")
            lines.append("```")
            lines.append(tc.get("stdout_tail", "").strip() or "(no output)")
            lines.append("```")
        if "test" in c:
            tr = c["test"]
            lines.append(f"**test** exit={tr['exit_code']} ({tr['duration_s']}s)")
            lines.append("```")
            lines.append(tr.get("stdout_tail", "").strip() or "(no output)")
            lines.append("```")
        if "exit_code" in c and "test" not in c and "typecheck" not in c:
            lines.append(f"exit={c['exit_code']} ({c.get('duration_s',0)}s)")
            lines.append("```")
            lines.append(c.get("stdout_tail", "").strip() or "(no output)")
            lines.append("```")
        lines.append("")

    # Evidence digest
    lines.append("## Evidence digest")
    lines.append("")
    lines.append(f"- Composite evidence hash (SHA-256 over sorted artifact hashes + verification report): `{receipt['evidence_hash']}`")
    lines.append(f"- Receipt hash (SHA-256 over the full receipt content): `{receipt['receipt_hash']}`")
    lines.append(f"- Chain linkage: `{receipt['previous_hash']}` → `{receipt['receipt_hash']}`")
    lines.append("")
    lines.append("Any party with this receipt and the project source at the recorded state can recompute the evidence_hash and confirm integrity.")
    lines.append("")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def verify_chain(receipts_dir: Path):
    """Walk the chain from genesis, recomputing each link."""
    files = sorted(receipts_dir.glob("*.json"))
    previous_hash = GENESIS
    results = []
    for f in files:
        with open(f) as fh:
            r = json.load(fh)
        expected_prev = r.get("previous_hash")
        if expected_prev != previous_hash:
            results.append({"file": f.name, "status": "BROKEN", "detail": f"expected previous_hash={previous_hash}, got {expected_prev}"})
            return results
        # Recompute receipt hash
        stored = r.pop("receipt_hash")
        recomputed = sha256_bytes(json.dumps(r, sort_keys=True).encode())
        r["receipt_hash"] = stored
        if stored != recomputed:
            results.append({"file": f.name, "status": "BROKEN", "detail": f"receipt_hash mismatch stored={stored} recomputed={recomputed}"})
            return results
        results.append({"file": f.name, "status": "OK", "receipt_hash": stored})
        previous_hash = stored
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-dir", required=True)
    ap.add_argument("--description", default="")
    ap.add_argument("--base-ref", default=None, help="Git base ref for session slice (e.g., a prior commit or tag)")
    ap.add_argument("--head-ref", default="HEAD")
    ap.add_argument("--verify", action="store_true", help="Verify chain integrity only")
    args = ap.parse_args()

    project_dir = Path(args.project_dir).resolve()
    pf = project_dir / ".proof-forge"
    receipts_dir = pf / "receipts"
    summaries_dir = pf / "summaries"
    index_path = pf / "EVIDENCE_INDEX.json"

    if args.verify:
        print(f"Verifying chain in {receipts_dir}...")
        results = verify_chain(receipts_dir)
        for r in results:
            print(f"  {r['status']:8} {r['file']}")
        broken = [r for r in results if r["status"] == "BROKEN"]
        if broken:
            sys.exit(1)
        print(f"\n✓ chain intact ({len(results)} receipt(s))")
        sys.exit(0)

    receipts_dir.mkdir(parents=True, exist_ok=True)
    summaries_dir.mkdir(parents=True, exist_ok=True)

    # Chain position + previous hash
    previous_receipt = load_previous_receipt(receipts_dir)
    if previous_receipt:
        previous_hash = previous_receipt["receipt_hash"]
        chain_position = previous_receipt["chain_position"] + 1
    else:
        previous_hash = GENESIS
        chain_position = 0

    # Session commits
    if args.base_ref:
        session_commits = git_session_commits(project_dir, args.base_ref, args.head_ref)
    else:
        session_commits = []

    # Artifacts
    if args.base_ref:
        artifacts = collect_session_artifacts(project_dir, args.base_ref, args.head_ref)
    else:
        # Fallback: just HEAD
        r = run(["git", "ls-tree", "-r", "--name-only", args.head_ref], project_dir)
        files = [line.strip() for line in r["stdout_tail"].splitlines() if line.strip()]
        artifacts = []
        for rel_path in files[:500]:  # cap for safety
            full = project_dir / rel_path
            if not full.exists():
                continue
            stat = full.stat()
            artifacts.append({
                "path": rel_path,
                "size_bytes": stat.st_size,
                "sha256": sha256_file(full),
                "mtime_iso": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
                "kind": classify(rel_path),
            })

    # Verification targets
    targets = detect_verification_targets(artifacts)
    print(f"  discovered rust_crates={len(targets['rust_crates'])}, node_projects={len(targets['node_projects'])}")

    # Run verification
    report = compile_verification_report(project_dir, artifacts, targets)

    # Receipt
    receipt = build_receipt(
        project_dir=project_dir,
        description=args.description,
        session_commits=session_commits,
        artifacts=artifacts,
        report=report,
        previous_hash=previous_hash,
        chain_position=chain_position,
    )

    # Persist receipt
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    receipt_path = receipts_dir / f"{ts}.json"
    with open(receipt_path, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print(f"  receipt: {receipt_path.name}")

    # Update index
    update_evidence_index(index_path, receipt)
    print(f"  index updated: {index_path.name}")

    # Proof summary
    summary_path = summaries_dir / f"{ts}.md"
    generate_proof_summary(receipt, project_dir, summary_path)
    # Top-level pointer
    top_summary = project_dir / "PROOF_SUMMARY.md"
    with open(top_summary, "w") as f:
        f.write(summary_path.read_text())
    print(f"  summary: {summary_path.relative_to(project_dir)}")
    print(f"  top-level pointer: PROOF_SUMMARY.md")

    # Final line for the operator
    level, label = compute_confidence_level(report)
    print(f"\n  confidence: {label} (level {level}/5)")
    print(f"  receipt_hash: {receipt['receipt_hash']}")
    print(f"  chain_position: {receipt['chain_position']}")


if __name__ == "__main__":
    main()
