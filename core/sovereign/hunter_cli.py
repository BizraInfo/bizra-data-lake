"""
BIZRA Hunter CLI — Bounty Revenue Pipeline
═══════════════════════════════════════════════════════════════════

Wire: HunterAgent → UERS Analysis → ImpactProof → Immunefi Report

Standing on Giants:
- Shannon (1948): SNR filtering separates real vulns from noise
- Saltzer & Schroeder (1975): Fail-closed — no unsigned proofs
- Bugcrowd/Immunefi: Platform report format standards
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.bounty.hunter import HunterAgent, ScanTarget
from core.bounty.impact_proof import ImpactProof
from core.proof_engine.receipt import SimpleSigner


def _load_signer() -> SimpleSigner:
    """Load signing key from environment."""
    key_hex = os.getenv("BIZRA_RECEIPT_PRIVATE_KEY_HEX", "")
    if key_hex:
        return SimpleSigner(bytes.fromhex(key_hex))
    # Fallback: derive from node identity
    return SimpleSigner(b"bizra-hunter-node0-key")


def _fetch_bytecode_rpc(address: str, rpc_url: str) -> Optional[bytes]:
    """Fetch contract bytecode via JSON-RPC eth_getCode."""
    import urllib.request

    payload = json.dumps(
        {
            "jsonrpc": "2.0",
            "method": "eth_getCode",
            "params": [address, "latest"],
            "id": 1,
        }
    ).encode()

    req = urllib.request.Request(
        rpc_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310
            data = json.loads(resp.read().decode())
            code_hex = data.get("result", "0x")
            if code_hex and code_hex != "0x":
                return bytes.fromhex(code_hex[2:])
    except Exception as e:
        print(f"  RPC fetch failed: {e}")

    return None


async def run_hunter_scan(
    address: str,
    chain: str = "ethereum",
    name: Optional[str] = None,
    bytecode_file: Optional[str] = None,
    source_file: Optional[str] = None,
    abi_file: Optional[str] = None,
    rpc_url: Optional[str] = None,
    json_output: bool = False,
    output_dir: Optional[str] = None,
) -> int:
    """
    Run vulnerability scan on a target.

    Returns exit code (0 = findings, 1 = error, 2 = no findings).
    """
    print()
    print("=" * 60)
    print("  BIZRA HUNTER — Vulnerability Scan")
    print("=" * 60)
    print(f"  Target:  {address}")
    print(f"  Chain:   {chain}")
    if name:
        print(f"  Name:    {name}")
    print()

    # Build scan target
    bytecode = None
    source_code = None
    abi = None

    # Load bytecode
    if bytecode_file:
        path = Path(bytecode_file)
        if path.exists():
            raw = path.read_bytes()
            # Handle hex-encoded bytecode files
            try:
                text = raw.decode("utf-8").strip()
                if text.startswith("0x"):
                    text = text[2:]
                bytecode = bytes.fromhex(text)
            except (UnicodeDecodeError, ValueError):
                bytecode = raw
            print(f"  Bytecode: {len(bytecode):,} bytes from {path.name}")
        else:
            print(f"  ERROR: Bytecode file not found: {bytecode_file}")
            return 1
    elif rpc_url:
        print("  Fetching bytecode from RPC...")
        bytecode = _fetch_bytecode_rpc(address, rpc_url)
        if bytecode:
            print(f"  Bytecode: {len(bytecode):,} bytes fetched")
        else:
            print("  WARNING: No bytecode returned (EOA or empty contract)")

    # Load source code
    if source_file:
        path = Path(source_file)
        if path.exists():
            source_code = path.read_text(encoding="utf-8", errors="replace")
            print(f"  Source:   {len(source_code):,} chars from {path.name}")
        else:
            print(f"  ERROR: Source file not found: {source_file}")
            return 1

    # Load ABI
    if abi_file:
        path = Path(abi_file)
        if path.exists():
            abi = json.loads(path.read_text())
            print(f"  ABI:      {len(abi)} entries from {path.name}")

    if not bytecode and not source_code and not abi:
        print()
        print("  WARNING: No bytecode, source, or ABI provided.")
        print("  Analysis will be limited. Use --bytecode, --source, --abi, or --rpc.")
        print()

    target = ScanTarget(
        address=address,
        chain=chain,
        name=name,
        bytecode=bytecode,
        source_code=source_code,
        abi=abi,
    )

    # Initialize hunter
    signer = _load_signer()
    hunter = HunterAgent(signer)

    print()
    print("  Running UERS 5D analysis...")
    print("  " + "-" * 40)

    start = time.perf_counter()
    result, proofs = await hunter.hunt(target, generate_proofs=True)
    elapsed = time.perf_counter() - start

    # Display results
    print(f"  Duration:   {elapsed * 1000:.1f}ms")
    print(f"  Status:     {result.status.value}")
    print(f"  Findings:   {len(result.vulnerabilities)}")
    print(f"  Proofs:     {len(proofs)}")
    print()

    # Entropy measurements
    e = result.entropy_measurements
    print("  UERS Entropy Vectors:")
    print(f"    Surface:      {e.surface_entropy:.3f}")
    print(f"    Structural:   {e.structural_entropy:.3f}")
    print(f"    Behavioral:   {e.behavioral_entropy:.3f}")
    print(f"    Hypothetical: {e.hypothetical_entropy:.3f}")
    print(f"    Contextual:   {e.contextual_entropy:.3f}")
    print(f"    Average:      {e.average_entropy:.3f}")
    print()

    if result.vulnerabilities:
        print("  FINDINGS:")
        print("  " + "-" * 40)
        for i, vuln in enumerate(result.vulnerabilities, 1):
            sev = vuln.get("severity", "unknown").upper()
            cat = vuln.get("category", "unknown")
            title = vuln.get("title", "Untitled")
            conf = vuln.get("confidence", 0.0)
            print(f"  [{i}] [{sev}] {title}")
            print(f"      Category: {cat} | Confidence: {conf:.1%}")
            if vuln.get("description"):
                print(f"      {vuln['description']}")
            print()

    # Output
    out_dir = Path(output_dir) if output_dir else Path("sovereign_state/hunter")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save scan result
    scan_data = {
        "scan": result.to_dict(),
        "proofs": [_proof_to_dict(p) for p in proofs],
        "stats": hunter.get_stats(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    scan_file = out_dir / f"scan_{address[:10]}_{int(time.time())}.json"
    scan_file.write_text(json.dumps(scan_data, indent=2, default=str))
    print(f"  Scan saved: {scan_file}")

    if json_output:
        print()
        print(json.dumps(scan_data, indent=2, default=str))

    print("=" * 60)

    if not result.vulnerabilities:
        return 2  # No findings
    return 0


async def run_hunter_report(
    scan_file: str,
    platform: str = "immunefi",
    output_file: Optional[str] = None,
) -> int:
    """
    Generate platform-formatted report from a scan result.

    Returns exit code (0 = success).
    """
    path = Path(scan_file)
    if not path.exists():
        print(f"ERROR: Scan file not found: {scan_file}")
        return 1

    scan_data = json.loads(path.read_text())
    proofs_data = scan_data.get("proofs", [])

    if not proofs_data:
        print("No proofs found in scan file. Nothing to report.")
        return 2

    print()
    print("=" * 60)
    print(f"  BIZRA HUNTER — {platform.upper()} Report Generator")
    print("=" * 60)
    print(f"  Source: {path.name}")
    print(f"  Proofs: {len(proofs_data)}")
    print()

    # Generate reports
    reports = []
    for i, proof_data in enumerate(proofs_data, 1):
        report = _format_immunefi_report(proof_data)
        reports.append(report)
        print(f"  [{i}] {report.get('title', 'Finding')}")
        print(f"      Severity: {report.get('severity', 'Unknown')}")
        print(f"      Target:   {report.get('target', {}).get('address', 'N/A')}")
        print()

    # Write output
    out_path = (
        Path(output_file) if output_file else path.with_suffix(f".{platform}.json")
    )
    out_path.write_text(json.dumps(reports, indent=2, default=str))
    print(f"  Report saved: {out_path}")

    # Also generate markdown for manual review
    md_path = out_path.with_suffix(".md")
    md_content = _generate_markdown_report(reports, platform)
    md_path.write_text(md_content)
    print(f"  Markdown:   {md_path}")

    print()
    print("  Next steps:")
    print("  1. Review the markdown report carefully")
    print("  2. Verify the finding is genuine and reproducible")
    print("  3. Submit through the Immunefi web portal")
    print("     https://immunefi.com/bounty/")
    print("=" * 60)

    return 0


async def run_hunter_list(output_dir: Optional[str] = None) -> int:
    """List previous scan results."""
    out_dir = Path(output_dir) if output_dir else Path("sovereign_state/hunter")

    if not out_dir.exists():
        print("No scans found. Run 'bizra hunter scan' first.")
        return 0

    # Match scan files but not report files (*.immunefi.json etc.)
    scans = sorted(
        [
            f
            for f in out_dir.glob("scan_*.json")
            if not any(
                f.name.endswith(f".{p}.json")
                for p in ["immunefi", "hackerone", "bugcrowd"]
            )
        ],
        reverse=True,
    )

    if not scans:
        print("No scans found.")
        return 0

    print()
    print("=" * 60)
    print("  BIZRA HUNTER — Scan History")
    print("=" * 60)
    print()

    for scan_file in scans[:20]:
        try:
            data = json.loads(scan_file.read_text())
            scan = data.get("scan", {})
            target = scan.get("target", {})
            findings = scan.get("finding_count", 0)
            ts = scan.get("timestamp", "")[:19]
            addr = target.get("address", "?")[:20]
            status = scan.get("status", "?")
            print(f"  {scan_file.name}")
            print(f"    Target: {addr}  Findings: {findings}  Status: {status}  {ts}")
        except Exception:
            print(f"  {scan_file.name}  (corrupt)")
    print()
    print("=" * 60)
    return 0


def _proof_to_dict(proof: ImpactProof) -> Dict[str, Any]:
    """Convert ImpactProof to JSON-serializable dict."""
    return {
        "proof_id": proof.proof_id,
        "title": proof.title,
        "description": (
            proof.description_hash.hex()
            if isinstance(proof.description_hash, bytes)
            else str(proof.description_hash)
        ),
        "target_address": proof.target_address,
        "target_chain": proof.target_chain,
        "target_name": proof.target_name,
        "vuln_category": proof.vuln_category.value,
        "severity": proof.severity.value,
        "delta_e": proof.delta_e,
        "funds_at_risk": proof.funds_at_risk,
        "snr_score": proof.snr_score,
        "ihsan_score": proof.ihsan_score,
        "exploit_hash": (
            proof.exploit_hash.hex()
            if isinstance(proof.exploit_hash, bytes)
            else str(proof.exploit_hash)
        ),
        "entropy_before": (
            proof.entropy_before.to_dict() if proof.entropy_before else None
        ),
        "entropy_after": proof.entropy_after.to_dict() if proof.entropy_after else None,
        "timestamp": (
            proof.timestamp.isoformat()
            if hasattr(proof, "timestamp")
            else datetime.now(timezone.utc).isoformat()
        ),
    }


def _format_immunefi_report(proof_data: Dict[str, Any]) -> Dict[str, Any]:
    """Format proof data into Immunefi submission format."""
    severity_display = (
        (proof_data.get("severity", "medium") or "medium").replace("_", " ").title()
    )

    return {
        "title": proof_data.get("title", "Vulnerability Finding"),
        "severity": severity_display,
        "target": {
            "address": proof_data.get("target_address", ""),
            "chain": proof_data.get("target_chain", "ethereum"),
            "protocol": proof_data.get("target_name", ""),
        },
        "vulnerability_type": (
            proof_data.get("vuln_category", "logic_error") or "logic_error"
        )
        .replace("_", " ")
        .title(),
        "impact": {
            "funds_at_risk_usd": proof_data.get("funds_at_risk", 0),
            "entropy_delta": proof_data.get("delta_e", 0),
        },
        "proof_of_concept": {
            "exploit_hash": proof_data.get("exploit_hash", ""),
        },
        "quality_metrics": {
            "snr_score": proof_data.get("snr_score", 0),
            "ihsan_score": proof_data.get("ihsan_score", 0),
        },
        "metadata": {
            "proof_id": proof_data.get("proof_id", ""),
            "submitted_via": "BIZRA PoI Hunter v1.0",
            "scanner_version": "UERS-5D",
        },
    }


def _generate_markdown_report(reports: List[Dict], platform: str) -> str:
    """Generate human-readable markdown report."""
    lines = [
        f"# BIZRA Hunter — {platform.title()} Submission Report",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "---",
        "",
    ]

    for i, report in enumerate(reports, 1):
        target = report.get("target", {})
        impact = report.get("impact", {})
        quality = report.get("quality_metrics", {})
        poc = report.get("proof_of_concept", {})

        lines.extend(
            [
                f"## Finding {i}: {report.get('title', 'Untitled')}",
                "",
                f"**Severity:** {report.get('severity', 'Unknown')}",
                f"**Vulnerability Type:** {report.get('vulnerability_type', 'Unknown')}",
                "",
                "### Target",
                f"- Address: `{target.get('address', 'N/A')}`",
                f"- Chain: {target.get('chain', 'N/A')}",
                f"- Protocol: {target.get('protocol', 'N/A')}",
                "",
                "### Impact",
                f"- Funds at Risk: ${impact.get('funds_at_risk_usd', 0):,.2f}",
                f"- Entropy Delta: {impact.get('entropy_delta', 0):.4f}",
                "",
                "### Proof of Concept",
                f"- Exploit Hash: `{poc.get('exploit_hash', 'N/A')[:32]}...`",
                "",
                "### Quality",
                f"- SNR Score: {quality.get('snr_score', 0):.3f}",
                f"- Ihsan Score: {quality.get('ihsan_score', 0):.3f}",
                "",
                "---",
                "",
            ]
        )

    lines.extend(
        [
            "## Submission Checklist",
            "",
            "- [ ] Finding verified manually",
            "- [ ] PoC reproducible on local fork",
            "- [ ] No false positive indicators",
            "- [ ] Report reviewed for accuracy",
            "- [ ] Submitted through platform portal",
            "",
        ]
    )

    return "\n".join(lines)
