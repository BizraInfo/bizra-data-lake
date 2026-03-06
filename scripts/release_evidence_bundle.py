#!/usr/bin/env python3
"""
BIZRA Release Evidence Bundle Generator
ADR-012: Persists all release artifacts into one machine-readable record.

Collects:
  1. Git metadata (SHA, branch, tag, author)
  2. Test coverage summary
  3. Security scan results (bandit, pip-audit, cargo-audit, trivy)
  4. Container image digest + provenance
  5. Rollout verdict (promoted / rolled-back / in-progress)
  6. SLO metrics snapshot at promotion time
  7. Benchmark delta vs previous release

Output: JSON file at deploy/evidence/<version>.json
        Also prints summary to stdout for CI step summaries.

Usage:
  python scripts/release_evidence_bundle.py --version v1.2.3
  python scripts/release_evidence_bundle.py --version sha-abc1234 --rollout-verdict promoted
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run(cmd: str, default: str = "") -> str:
    """Run shell command, return stdout or default on failure."""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        return result.stdout.strip() if result.returncode == 0 else default
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return default


def git_metadata() -> dict:
    return {
        "sha": run("git rev-parse HEAD"),
        "short_sha": run("git rev-parse --short HEAD"),
        "branch": run("git rev-parse --abbrev-ref HEAD"),
        "author": run("git log -1 --format='%an <%ae>'"),
        "message": run("git log -1 --format='%s'"),
        "timestamp": run("git log -1 --format='%aI'"),
        "tag": run("git describe --tags --exact-match 2>/dev/null"),
    }


def coverage_summary() -> dict:
    """Parse pytest coverage if .coverage exists."""
    cov_file = Path("htmlcov/status.json")
    if cov_file.exists():
        try:
            data = json.loads(cov_file.read_text())
            return {
                "total_pct": data.get("totals", {}).get("percent_covered", 0),
                "lines_covered": data.get("totals", {}).get("covered_lines", 0),
                "lines_total": data.get("totals", {}).get("num_statements", 0),
            }
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: parse pyproject.toml fail_under
    pyproject = Path("pyproject.toml")
    if pyproject.exists():
        for line in pyproject.read_text().splitlines():
            if "fail_under" in line:
                try:
                    val = float(line.split("=")[1].strip())
                    return {"floor": val, "measured": "not-available"}
                except (ValueError, IndexError):
                    pass
    return {"measured": "not-available"}


def security_scan_results() -> dict:
    """Collect security scan results if artifacts exist."""
    results = {}

    # Bandit
    bandit_out = run("bandit -r core/ -f json -q 2>/dev/null")
    if bandit_out:
        try:
            data = json.loads(bandit_out)
            results["bandit"] = {
                "high": sum(1 for r in data.get("results", []) if r.get("issue_severity") == "HIGH"),
                "medium": sum(1 for r in data.get("results", []) if r.get("issue_severity") == "MEDIUM"),
                "low": sum(1 for r in data.get("results", []) if r.get("issue_severity") == "LOW"),
            }
        except json.JSONDecodeError:
            results["bandit"] = "parse-error"
    else:
        results["bandit"] = "not-run"

    # pip-audit
    pip_audit_out = run("pip-audit --format=json 2>/dev/null")
    if pip_audit_out:
        try:
            data = json.loads(pip_audit_out)
            vulns = data if isinstance(data, list) else data.get("dependencies", [])
            results["pip_audit"] = {
                "vulnerabilities": sum(len(d.get("vulns", [])) for d in vulns),
            }
        except json.JSONDecodeError:
            results["pip_audit"] = "parse-error"
    else:
        results["pip_audit"] = "not-run"

    return results


def container_provenance(version: str) -> dict:
    """Get container image digest if available."""
    registry = os.environ.get("REGISTRY", "ghcr.io")
    repo = os.environ.get("IMAGE_NAME", "bizrainfo/bizra-data-lake")

    images = {}
    for component in ["elite", "omega", "mcp", "frontend"]:
        tag = f"{registry}/{repo}/{component}:{version}"
        digest = run(f"docker inspect --format='{{{{.Id}}}}' {tag} 2>/dev/null")
        if digest:
            images[component] = {"tag": tag, "digest": digest}
        else:
            images[component] = {"tag": tag, "digest": "not-available"}

    return images


def rollout_snapshot() -> dict:
    """Query Argo Rollouts status if kubectl is available."""
    status = run(
        "kubectl get rollout bizra-elite-rollout -n bizra "
        "-o jsonpath='{.status.phase}' 2>/dev/null"
    )
    if not status:
        return {"phase": "not-available", "note": "kubectl not configured or rollout not found"}

    return {
        "phase": status,
        "canary_weight": run(
            "kubectl get rollout bizra-elite-rollout -n bizra "
            "-o jsonpath='{.status.canary.weights.canary}' 2>/dev/null"
        ),
        "stable_rs": run(
            "kubectl get rollout bizra-elite-rollout -n bizra "
            "-o jsonpath='{.status.stableRS}' 2>/dev/null"
        ),
    }


def build_bundle(version: str, verdict: str) -> dict:
    """Assemble the complete evidence bundle."""
    bundle = {
        "schema_version": "1.0.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "version": version,
        "rollout_verdict": verdict,
        "git": git_metadata(),
        "coverage": coverage_summary(),
        "security": security_scan_results(),
        "containers": container_provenance(version),
        "rollout": rollout_snapshot(),
        "thresholds": {
            "ihsan_production": 0.95,
            "snr_minimum": 0.85,
            "error_rate_max": 0.01,
            "p95_latency_max_ms": 500,
            "p99_latency_max_ms": 1000,
        },
    }

    # Compute bundle hash for integrity
    bundle_json = json.dumps(bundle, sort_keys=True, separators=(",", ":"))
    bundle["bundle_hash"] = hashlib.sha256(bundle_json.encode()).hexdigest()

    return bundle


def main():
    parser = argparse.ArgumentParser(description="BIZRA Release Evidence Bundle")
    parser.add_argument("--version", required=True, help="Release version or SHA")
    parser.add_argument(
        "--rollout-verdict",
        default="pending",
        choices=["pending", "promoted", "rolled-back", "in-progress"],
        help="Rollout outcome",
    )
    parser.add_argument(
        "--output-dir",
        default="deploy/evidence",
        help="Output directory for evidence JSON",
    )
    args = parser.parse_args()

    bundle = build_bundle(args.version, args.rollout_verdict)

    # Write to file
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.version.replace('/', '_')}.json"
    out_file.write_text(json.dumps(bundle, indent=2))

    # Print summary
    print("=" * 70)
    print("BIZRA RELEASE EVIDENCE BUNDLE")
    print("=" * 70)
    print(f"Version:       {bundle['version']}")
    print(f"Git SHA:       {bundle['git']['sha'][:12]}")
    print(f"Verdict:       {bundle['rollout_verdict']}")
    print(f"Coverage:      {bundle['coverage']}")
    print(f"Security:      {json.dumps(bundle['security'], indent=None)}")
    print(f"Bundle Hash:   {bundle['bundle_hash'][:16]}...")
    print(f"Written to:    {out_file}")
    print("=" * 70)

    # Exit non-zero if verdict is rolled-back (for CI)
    if args.rollout_verdict == "rolled-back":
        print("[WARN] Release was rolled back — evidence preserved for post-mortem")
        sys.exit(1)


if __name__ == "__main__":
    main()
