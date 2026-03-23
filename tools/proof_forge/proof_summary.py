#!/usr/bin/env python3
"""
Proof Summary Generator — Investor-Readable Evidence Report

Takes a receipt JSON and produces a concise, professional Markdown summary
suitable for pitch decks, due diligence, and stakeholder communication.

Usage:
    proof_summary.py --receipt .proof-forge/receipts/2026-02-07_001823.json \
                     --project-dir ./my-project

    proof_summary.py --latest --project-dir ./my-project
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

CONFIDENCE_ICONS = {
    5: "🛡️",  # Ironclad
    4: "💪",  # Strong
    3: "✅",  # Solid
    2: "📝",  # Attested
    1: "📋",  # Logged
}

CONFIDENCE_BARS = {
    5: "█████",
    4: "████░",
    3: "███░░",
    2: "██░░░",
    1: "█░░░░",
}


def load_receipt(receipt_path: Path) -> dict:
    """Load a receipt JSON file."""
    with open(receipt_path) as f:
        return json.load(f)


def find_latest_receipt(project_dir: Path) -> Path:
    """Find the most recent receipt file."""
    proof_dir = project_dir / ".proof-forge"
    index_path = proof_dir / "EVIDENCE_INDEX.json"

    if not index_path.exists():
        print("❌ No evidence index found.")
        sys.exit(1)

    with open(index_path) as f:
        index = json.load(f)

    if not index.get("latest_receipt"):
        print("❌ No receipts in index.")
        sys.exit(1)

    return proof_dir / "receipts" / index["latest_receipt"]


def format_timestamp(iso_str: str) -> str:
    """Format ISO timestamp for display."""
    try:
        dt = datetime.fromisoformat(iso_str)
        return dt.strftime("%B %d, %Y at %H:%M UTC")
    except (ValueError, TypeError):
        return iso_str


def format_bytes(size: int) -> str:
    """Format byte count for display."""
    if size < 1024:
        return f"{size} B"
    elif size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    elif size < 1024 * 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    else:
        return f"{size / (1024 * 1024 * 1024):.2f} GB"


def generate_summary(receipt: dict, project_dir: Path) -> str:
    """Generate investor-readable Proof Summary markdown."""
    conf = receipt.get("confidence", {})
    level = conf.get("level", 1)
    label = conf.get("label", "Unknown")
    criteria = conf.get("criteria", "")
    icon = CONFIDENCE_ICONS.get(level, "❓")
    bar = CONFIDENCE_BARS.get(level, "░░░░░")

    hashes = receipt.get("hashes", {})
    artifacts = receipt.get("artifacts", {})
    verification = receipt.get("verification", {})
    checks = verification.get("checks", [])

    chain_pos = receipt.get("chain_position", 0)
    timestamp = format_timestamp(receipt.get("timestamp", ""))

    # Artifact stats
    total_files = artifacts.get("count", 0)
    summary = artifacts.get("summary", {})
    total_size = sum(a.get("size_bytes", 0) for a in artifacts.get("details", []))

    # Verification stats
    checks_run = verification.get("checks_run", len(checks))
    checks_passed = verification.get(
        "checks_passed", sum(1 for c in checks if c.get("passed"))
    )

    lines = []

    # ── Header ──
    lines.append(f"# Proof Summary — Receipt #{chain_pos}")
    lines.append("")
    lines.append(f"> **{receipt.get('description', 'No description')}**")
    lines.append(">")
    lines.append(f"> {timestamp}")
    lines.append("")

    # ── Confidence ──
    lines.append(f"## {icon} Confidence: {label} (Level {level}/5)")
    lines.append("")
    lines.append("```")
    lines.append(f"  {bar}  {label}")
    lines.append("```")
    lines.append("")
    lines.append(f"*{criteria}*")
    lines.append("")

    # ── What Was Verified ──
    lines.append("## Verification Results")
    lines.append("")

    if checks:
        lines.append(f"**{checks_passed}/{checks_run} checks passed**")
        lines.append("")
        lines.append("| Check | Type | Result | Duration |")
        lines.append("|-------|------|--------|----------|")
        for check in checks:
            status = "✅ Pass" if check.get("passed") else "❌ Fail"
            ctype = check.get("type", "unknown").replace("_", " ").title()
            duration = (
                f"{check.get('duration_ms', 0)}ms" if check.get("duration_ms") else "—"
            )
            cmd = check.get("command", "—")
            lines.append(f"| `{cmd[:40]}` | {ctype} | {status} | {duration} |")
        lines.append("")
    elif verification.get("manual_attestation"):
        lines.append(f"**Manual Attestation**: {verification['manual_attestation']}")
        lines.append("")
    else:
        lines.append("*No automated verification was available for this evidence.*")
        lines.append("")

    # ── Artifacts ──
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"**{total_files} files** ({format_bytes(total_size)} total)")
    lines.append("")
    if summary:
        for ftype, count in sorted(summary.items()):
            lines.append(f"- {ftype}: {count} files")
        lines.append("")

    # ── Evidence Chain ──
    lines.append("## Evidence Chain")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| **Chain Position** | #{chain_pos} |")
    lines.append(
        f"| **Evidence Hash** | `{hashes.get('evidence_hash', 'N/A')[:32]}...` |"
    )
    lines.append(f"| **Chain Hash** | `{hashes.get('chain_hash', 'N/A')[:32]}...` |")
    lines.append(
        f"| **Previous Hash** | `{hashes.get('previous_hash', 'N/A')[:32]}...` |"
    )
    lines.append(f"| **Receipt ID** | `{receipt.get('receipt_id', 'N/A')}` |")
    lines.append("")

    if chain_pos > 1:
        lines.append(
            f"*This receipt is cryptographically linked to {chain_pos - 1} prior receipt(s).*"
        )
        lines.append(
            "*The complete chain can be independently verified by recomputing all hashes.*"
        )
    else:
        lines.append(
            "*This is the genesis receipt — the first link in this evidence chain.*"
        )
    lines.append("")

    # ── Full Hashes (for verification) ──
    lines.append("---")
    lines.append("")
    lines.append("<details>")
    lines.append("<summary>Full Hash Values (for independent verification)</summary>")
    lines.append("")
    lines.append("```")
    lines.append(f"Evidence Hash:  {hashes.get('evidence_hash', 'N/A')}")
    lines.append(f"Chain Hash:     {hashes.get('chain_hash', 'N/A')}")
    lines.append(f"Previous Hash:  {hashes.get('previous_hash', 'N/A')}")
    lines.append("```")
    lines.append("")
    lines.append("</details>")
    lines.append("")

    # ── Footer ──
    lines.append("---")
    lines.append("*Generated by Proof Forge v1.0 — BUILD → VERIFY → EVIDENCE*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Proof Summary Generator")
    parser.add_argument("--receipt", help="Path to receipt JSON file")
    parser.add_argument("--latest", action="store_true", help="Use the latest receipt")
    parser.add_argument("--project-dir", required=True, help="Project directory path")

    args = parser.parse_args()
    project_dir = Path(args.project_dir).resolve()

    if args.latest:
        receipt_path = find_latest_receipt(project_dir)
    elif args.receipt:
        receipt_path = Path(args.receipt)
    else:
        print("❌ Provide either --receipt or --latest")
        sys.exit(1)

    if not receipt_path.exists():
        print(f"❌ Receipt not found: {receipt_path}")
        sys.exit(1)

    receipt = load_receipt(receipt_path)
    summary_md = generate_summary(receipt, project_dir)

    # Write to summaries dir
    proof_dir = project_dir / ".proof-forge"
    summaries_dir = proof_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    # Use same filename as receipt but .md
    summary_filename = receipt_path.stem + ".md"
    summary_path = summaries_dir / summary_filename
    summary_path.write_text(summary_md, encoding="utf-8")
    print(f"✅ Summary written: {summary_path}")

    # Also write to project root as PROOF_SUMMARY.md
    root_summary = project_dir / "PROOF_SUMMARY.md"
    root_summary.write_text(summary_md, encoding="utf-8")
    print(f"✅ Latest summary: {root_summary}")

    print(f"\nSUMMARY_PATH={summary_path}")


if __name__ == "__main__":
    main()
