"""
Phase65 alpha launch readiness packet generator.

Creates a signed, auditable launch packet from:
- lifecycle summary
- blueprint gate report
- KPI snapshot

Decision states:
- GO
- CONDITIONAL_GO
- NO_GO
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import blake3
import yaml

try:
    from core.pci.crypto import (
        PrivateKeyWrapper,
        canonicalize_json,
        domain_separated_digest,
        sign_message,
    )
except ModuleNotFoundError:  # pragma: no cover - CLI fallback
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    for mod_name in list(sys.modules):
        if mod_name == "core" or mod_name.startswith("core."):
            sys.modules.pop(mod_name, None)
    from core.pci.crypto import (
        PrivateKeyWrapper,
        canonicalize_json,
        domain_separated_digest,
        sign_message,
    )


MANUAL_CHECK_SPECS: list[tuple[str, str]] = [
    ("website_updated", "Website reflects latest architecture and lifecycle"),
    ("unified_installer_ready", "Unified installer is validated end-to-end"),
    ("onboarding_lifecycle_ready", "Onboarding lifecycle is ready for human users"),
    ("urp_active", "Universal Resource Pool is active"),
    ("identity_activated", "Sovereign identity is activated"),
    ("pat_minted", "PAT is minted and operating"),
    ("sat_minted", "SAT is minted and operating"),
    ("local_filesystem_automation_ready", "Local filesystem automation is verified"),
    ("web_autonomy_ready", "Autonomous browser operation is verified"),
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _file_blake3(path: Path) -> str:
    hasher = blake3.blake3()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def _check(
    name: str, passed: bool, expected: Any, actual: Any, source: str
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "expected": expected,
        "actual": actual,
        "source": source,
    }


def _build_automated_checks(
    *,
    summary: dict[str, Any],
    gate: dict[str, Any],
    kpi: dict[str, Any],
    cfg: dict[str, Any],
    require_tier: str,
) -> list[dict[str, Any]]:
    required = cfg.get("quality_gates", {}).get("required", {})
    scoring = cfg.get("quality_gates", {}).get("scoring", {})
    return [
        _check(
            "final_state",
            summary.get("final_state") == required.get("final_state"),
            required.get("final_state"),
            summary.get("final_state"),
            "summary.final_state",
        ),
        _check(
            "gate_passed",
            bool(gate.get("gate_passed")),
            True,
            gate.get("gate_passed"),
            "gate_report.gate_passed",
        ),
        _check(
            "snr_score",
            float(gate.get("snr_score", 0.0))
            >= float(scoring.get("min_snr_score", 0.0)),
            f">={scoring.get('min_snr_score', 0.0)}",
            gate.get("snr_score"),
            "gate_report.snr_score",
        ),
        _check(
            "ledger_chain_valid",
            bool(summary.get("ledger_chain_valid"))
            is bool(required.get("ledger_chain_valid", True)),
            required.get("ledger_chain_valid", True),
            summary.get("ledger_chain_valid"),
            "summary.ledger_chain_valid",
        ),
        _check(
            "signed_receipts",
            bool(summary.get("signed_receipts")),
            True,
            summary.get("signed_receipts"),
            "summary.signed_receipts",
        ),
        _check(
            "avg_ihsan",
            float(summary.get("avg_ihsan", 0.0))
            >= float(required.get("min_avg_ihsan", 0.0)),
            f">={required.get('min_avg_ihsan', 0.0)}",
            summary.get("avg_ihsan"),
            "summary.avg_ihsan",
        ),
        _check(
            "speedup_system1_vs_system2",
            float(summary.get("speedup_system1_vs_system2", 0.0))
            >= float(required.get("min_speedup_system1_vs_system2", 0.0)),
            f">={required.get('min_speedup_system1_vs_system2', 0.0)}",
            summary.get("speedup_system1_vs_system2"),
            "summary.speedup_system1_vs_system2",
        ),
        _check(
            "avg_latency_ms",
            float(summary.get("avg_latency_ms", 1e9))
            <= float(required.get("max_avg_latency_ms", 1e9)),
            f"<={required.get('max_avg_latency_ms', 1e9)}",
            summary.get("avg_latency_ms"),
            "summary.avg_latency_ms",
        ),
        _check(
            "tier",
            str(kpi.get("tier", "")) == require_tier,
            require_tier,
            kpi.get("tier"),
            "kpi_snapshot.tier",
        ),
    ]


def _build_manual_checks(
    manual_values: dict[str, Any] | None,
) -> tuple[list[dict[str, Any]], int, int, int]:
    values = manual_values or {}
    checks: list[dict[str, Any]] = []
    passed = 0
    failed = 0
    pending = 0
    for key, description in MANUAL_CHECK_SPECS:
        raw = values.get(key, None)
        if raw is True:
            status = "passed"
            passed += 1
        elif raw is False:
            status = "failed"
            failed += 1
        else:
            status = "pending"
            pending += 1
        checks.append(
            {
                "name": key,
                "description": description,
                "status": status,
                "value": raw,
            }
        )
    return checks, passed, failed, pending


def build_alpha_launch_packet(
    *,
    summary_payload: dict[str, Any],
    gate_payload: dict[str, Any],
    kpi_payload: dict[str, Any],
    cfg: dict[str, Any],
    summary_path: Path,
    gate_path: Path,
    kpi_path: Path,
    alpha_users_target: int,
    require_tier: str,
    strict_manual: bool,
    manual_values: dict[str, Any] | None = None,
    signer_private_key_hex: str = "",
) -> dict[str, Any]:
    summary = summary_payload.get("summary", summary_payload)
    automated_checks = _build_automated_checks(
        summary=summary,
        gate=gate_payload,
        kpi=kpi_payload,
        cfg=cfg,
        require_tier=require_tier,
    )
    automated_pass = all(c["passed"] for c in automated_checks)

    manual_checks, manual_passed, manual_failed, manual_pending = _build_manual_checks(
        manual_values
    )
    if strict_manual:
        manual_blocking = manual_failed > 0 or manual_pending > 0
    else:
        manual_blocking = manual_failed > 0

    blockers: list[str] = []
    for check in automated_checks:
        if not check["passed"]:
            blockers.append(f"auto:{check['name']}")
    for check in manual_checks:
        if check["status"] == "failed":
            blockers.append(f"manual:{check['name']}")
        elif check["status"] == "pending" and strict_manual:
            blockers.append(f"manual_pending:{check['name']}")

    if (not automated_pass) or manual_blocking:
        decision = "NO_GO"
    elif manual_pending > 0:
        decision = "CONDITIONAL_GO"
    else:
        decision = "GO"

    packet = {
        "packet_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "alpha_users_target": int(alpha_users_target),
        "decision": decision,
        "launch_ready": decision == "GO",
        "strict_manual": bool(strict_manual),
        "automated": {
            "pass": automated_pass,
            "checks": automated_checks,
        },
        "manual": {
            "checks": manual_checks,
            "counts": {
                "passed": manual_passed,
                "failed": manual_failed,
                "pending": manual_pending,
            },
        },
        "blockers": blockers,
        "metrics": {
            "final_state": summary.get("final_state"),
            "snr_score": gate_payload.get("snr_score"),
            "avg_ihsan": summary.get("avg_ihsan"),
            "avg_latency_ms": summary.get("avg_latency_ms"),
            "speedup_system1_vs_system2": summary.get("speedup_system1_vs_system2"),
            "tier": kpi_payload.get("tier"),
            "signed_receipts": summary.get("signed_receipts"),
            "ledger_chain_valid": summary.get("ledger_chain_valid"),
        },
        "artifacts": {
            "summary_path": str(summary_path),
            "gate_report_path": str(gate_path),
            "kpi_snapshot_path": str(kpi_path),
            "summary_blake3": _file_blake3(summary_path),
            "gate_report_blake3": _file_blake3(gate_path),
            "kpi_snapshot_blake3": _file_blake3(kpi_path),
        },
    }

    canonical = canonicalize_json(packet, ensure_ascii=True)
    digest_hex = domain_separated_digest(canonical)
    signature: dict[str, Any] = {"signed": False}
    if signer_private_key_hex.strip():
        signer = PrivateKeyWrapper(signer_private_key_hex.strip())
        signature = {
            "signed": True,
            "public_key": signer.public_key_hex,
            "digest": digest_hex,
            "value": sign_message(digest_hex, signer_private_key_hex.strip()),
        }
    else:
        signature = {
            "signed": False,
            "digest": digest_hex,
            "reason": "BIZRA_RECEIPT_PRIVATE_KEY_HEX not provided",
        }

    packet["signature"] = signature
    return packet


def render_markdown(packet: dict[str, Any]) -> str:
    counts = packet["manual"]["counts"]
    return "\n".join(
        [
            "# Phase65 Alpha Launch Packet",
            "",
            f"- Decision: **{packet['decision']}**",
            f"- Launch Ready: **{packet['launch_ready']}**",
            f"- Alpha Users Target: **{packet['alpha_users_target']}**",
            f"- Strict Manual Mode: **{packet['strict_manual']}**",
            "",
            "## Automated Checks",
            "",
            "| Check | Passed | Expected | Actual |",
            "|-------|--------|----------|--------|",
            *[
                f"| {c['name']} | {c['passed']} | {c['expected']} | {c['actual']} |"
                for c in packet["automated"]["checks"]
            ],
            "",
            "## Manual Checks",
            "",
            "| Check | Status |",
            "|-------|--------|",
            *[f"| {c['name']} | {c['status']} |" for c in packet["manual"]["checks"]],
            "",
            (
                f"Manual Counts: passed={counts['passed']} "
                f"failed={counts['failed']} pending={counts['pending']}"
            ),
            "",
            "## Signature",
            "",
            f"- Signed: {packet['signature'].get('signed')}",
            f"- Digest: `{packet['signature'].get('digest')}`",
            f"- Public Key: `{packet['signature'].get('public_key', '')}`",
        ]
    )


def _emit_github_outputs(packet: dict[str, Any], output_path: Path) -> None:
    counts = packet["manual"]["counts"]
    lines = [
        f"launch_decision={packet['decision']}",
        f"launch_ready={'true' if packet['launch_ready'] else 'false'}",
        f"packet_signed={'true' if packet['signature'].get('signed') else 'false'}",
        f"manual_pending_count={counts['pending']}",
        f"manual_failed_count={counts['failed']}",
        f"alpha_users_target={packet['alpha_users_target']}",
        f"packet_digest={packet['signature'].get('digest', '')}",
    ]
    with output_path.open("a", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Phase65 alpha launch readiness packet."
    )
    parser.add_argument(
        "--summary", type=Path, required=True, help="Lifecycle summary JSON path."
    )
    parser.add_argument(
        "--gate-report", type=Path, required=True, help="Gate report JSON path."
    )
    parser.add_argument(
        "--kpi", type=Path, required=True, help="KPI snapshot JSON path."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/phase65_masterpiece_roadmap.yaml"),
        help="Blueprint config YAML path.",
    )
    parser.add_argument(
        "--manual-checks",
        type=Path,
        default=None,
        help="Optional manual checks JSON path.",
    )
    parser.add_argument(
        "--alpha-users-target", type=int, default=100, help="Alpha user cohort target."
    )
    parser.add_argument(
        "--require-tier",
        type=str,
        default="elite-operational",
        help="Required KPI tier to classify automated readiness as pass.",
    )
    parser.add_argument(
        "--strict-manual",
        action="store_true",
        help="Fail launch decision if any manual check is pending.",
    )
    parser.add_argument(
        "--out-json", type=Path, required=True, help="Output packet JSON path."
    )
    parser.add_argument(
        "--out-md", type=Path, default=None, help="Optional output markdown path."
    )
    parser.add_argument(
        "--github-output", type=Path, default=None, help="Optional GITHUB_OUTPUT path."
    )
    args = parser.parse_args()

    summary_payload = _load_json(args.summary)
    gate_payload = _load_json(args.gate_report)
    kpi_payload = _load_json(args.kpi)
    cfg = _load_yaml(args.config)
    manual_values = (
        _load_json(args.manual_checks) if args.manual_checks is not None else None
    )

    # Explicit signer key for packet signing is optional but recommended.
    # Uses same key as receipt signing when present.
    signer_private_key_hex = os.getenv("BIZRA_RECEIPT_PRIVATE_KEY_HEX", "").strip()

    packet = build_alpha_launch_packet(
        summary_payload=summary_payload,
        gate_payload=gate_payload,
        kpi_payload=kpi_payload,
        cfg=cfg,
        summary_path=args.summary,
        gate_path=args.gate_report,
        kpi_path=args.kpi,
        alpha_users_target=args.alpha_users_target,
        require_tier=args.require_tier,
        strict_manual=args.strict_manual,
        manual_values=manual_values,
        signer_private_key_hex=signer_private_key_hex,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(packet, indent=2), encoding="utf-8")
    if args.out_md is not None:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(render_markdown(packet), encoding="utf-8")
    if args.github_output is not None:
        _emit_github_outputs(packet, args.github_output)

    print(json.dumps(packet, indent=2))
    return 0 if packet["decision"] != "NO_GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
