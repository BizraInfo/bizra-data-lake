#!/usr/bin/env python3
"""Compile normalized conversations into a stereoscopic graph-of-thoughts report."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine import AutonomousSNRGoTEngine  # noqa: E402
from genesis_gate import (  # noqa: E402
    GenesisGateConfig,
    NodeMaturityStage,
    evaluate_genesis_gate,
)
from memory_bridge import ingest_report_nodes  # noqa: E402
from normalizers import (  # noqa: E402
    COLLECTION_GAP,
    CONVERSATION_GAP,
    CONVERSATION_PLATFORMS,
    CORE8,
    EXPORTABLE_NOW,
)

_GATE_PROVIDER_SET_CHOICES = ("exportable_now", "conversation_platforms", "core8")
_GATE_PROFILE_CHOICES = ("seed", "sprout", "growing", "rooted")


def _format_summary(report: dict) -> str:
    cv_top = float(report.get("cv", 0.0))
    cv_core8 = float(report.get("cv_core8", cv_top))
    cv_gate = report.get("cv_gate")
    lines = [
        "Stereoscopic Graph Report",
        f"Turns: {report['total_turns']}",
        f"Hints: {report['total_hints']}",
        f"Providers: {len(report['provider_coverage'])}/{len(CONVERSATION_PLATFORMS)}",
        f"CV (Conversation): {cv_core8:.4f}",
        f"Nodes (>= SNR): {report['node_count']}",
        f"Edges: {report['edge_count']}",
        f"Elite Nodes: {report['elite_count']}",
    ]
    gate = report.get("genesis_gate")
    if isinstance(gate, dict):
        if cv_gate is not None:
            lines.append(f"CV (Gate Set): {float(cv_gate):.4f}")
        lines.append(f"GENESIS Gate: {'PASS' if gate.get('passed') else 'FAIL'}")
        reasons = gate.get("reasons") or []
        if reasons:
            lines.append(f"GENESIS Reasons: {json.dumps(reasons, ensure_ascii=False)}")
    return "\n".join(lines)


def _iso_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


def _resolve_gate_provider_set(mode: str) -> set[str]:
    if mode == "exportable_now":
        return set(EXPORTABLE_NOW)
    if mode == "conversation_platforms":
        return set(CONVERSATION_PLATFORMS)
    if mode == "core8":
        return set(CORE8)
    raise ValueError(f"Unsupported gate provider set: {mode}")


def _write_checkpoint(
    checkpoint_dir: Path,
    report: dict[str, Any],
    label: str,
) -> dict[str, str]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stamp = _iso_now()
    slug = "".join(
        ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label
    ).strip("_")
    if not slug:
        slug = "run"

    report_path = checkpoint_dir / f"{stamp}_report.json"
    canonical = json.dumps(
        report, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    report_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    latest_path = checkpoint_dir / "latest_report.json"
    latest_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )

    checkpoint_row = {
        "timestamp": stamp,
        "label": slug,
        "report_path": str(report_path),
        "report_hash": report_hash,
        "cv": report.get("cv", 0.0),
        "provider_count": len(report.get("provider_coverage") or []),
        "total_turns": report.get("total_turns", 0),
        "total_hints": report.get("total_hints", 0),
        "node_count": report.get("node_count", 0),
        "edge_count": report.get("edge_count", 0),
        "elite_count": report.get("elite_count", 0),
        "genesis_gate_passed": bool(
            (report.get("genesis_gate") or {}).get("passed", False)
        ),
    }
    checkpoints_path = checkpoint_dir / "drift_checkpoints.jsonl"
    with checkpoints_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(checkpoint_row, ensure_ascii=False) + "\n")

    return {
        "report_path": str(report_path),
        "latest_path": str(latest_path),
        "checkpoints_path": str(checkpoints_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compile graph-of-thoughts report from platform conversation exports."
    )
    parser.add_argument("paths", nargs="*", help="Corpus directories/files to scan")
    parser.add_argument(
        "--fixtures",
        action="store_true",
        help="Scan built-in fixtures and include legacy Core-4 provider coverage",
    )
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=0.85,
        help="Minimum SNR required for node inclusion",
    )
    parser.add_argument(
        "--elite-threshold",
        type=float,
        default=0.95,
        help="Minimum SNR required for elite node inclusion",
    )
    parser.add_argument(
        "--genesis-gate",
        action="store_true",
        help="Enable fail-closed GENESIS gate checks on CV and elite node minimum",
    )
    parser.add_argument(
        "--min-cv",
        type=float,
        default=1.0,
        help="GENESIS gate minimum CV threshold",
    )
    parser.add_argument(
        "--min-nodes",
        type=int,
        default=1,
        help="GENESIS gate minimum node count",
    )
    parser.add_argument(
        "--min-elite-nodes",
        type=int,
        default=1,
        help="GENESIS gate minimum elite node count",
    )
    parser.add_argument(
        "--gate-provider-set",
        choices=_GATE_PROVIDER_SET_CHOICES,
        default="exportable_now",
        help=(
            "Provider set used for fail-closed GENESIS gating "
            "(default: exportable_now)"
        ),
    )
    parser.add_argument(
        "--fail-open",
        action="store_true",
        help="Override fail-closed behavior (for diagnostics only)",
    )
    parser.add_argument(
        "--gate-profile",
        choices=_GATE_PROFILE_CHOICES,
        default=None,
        help=(
            "Node maturity profile for tiered GENESIS gate thresholds. "
            "Overrides --min-cv, --min-nodes, --min-elite-nodes when set. "
            "Choices: seed (zero-data), sprout (10+ atoms), "
            "growing (100+ msgs), rooted (full enforcement)."
        ),
    )
    parser.add_argument(
        "--ingest-bizra-memory",
        action="store_true",
        help="Bridge report nodes into bizra-memory via typed adapter",
    )
    parser.add_argument(
        "--ingest-min-snr",
        type=float,
        default=0.85,
        help="Minimum node SNR to include in memory ingest bridge",
    )
    parser.add_argument(
        "--ingest-session-id",
        type=int,
        default=9000,
        help="Session ID used for memory bridge ingest",
    )
    parser.add_argument(
        "--export-ingest-jsonl",
        type=str,
        default="",
        help="Optional path to export typed ingest payloads as JSONL",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="",
        help="Directory for timestamped report checkpoints and drift JSONL",
    )
    parser.add_argument(
        "--checkpoint-label",
        type=str,
        default="manual",
        help="Label stamped into drift checkpoint rows",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON report")
    parser.add_argument(
        "--out", type=str, default="", help="Write JSON report to file path"
    )
    args = parser.parse_args()

    paths: list[Path] = []
    if args.fixtures:
        paths.append(ROOT / "fixtures")
    if args.paths:
        paths.extend(Path(raw).expanduser().resolve() for raw in args.paths)
    if not paths:
        parser.error("Provide --fixtures or at least one path")

    engine = AutonomousSNRGoTEngine(
        snr_threshold=args.snr_threshold,
        elite_threshold=args.elite_threshold,
    )
    report = engine.compile_paths(paths).to_dict()

    if args.fixtures:
        merged_coverage = sorted(
            set(report["provider_coverage"]) | set(CONVERSATION_PLATFORMS)
        )
        report["provider_coverage"] = merged_coverage
        report["cv"] = round(len(merged_coverage) / len(CONVERSATION_PLATFORMS), 4)

    if args.genesis_gate:
        required_providers = sorted(_resolve_gate_provider_set(args.gate_provider_set))
        target_providers = sorted(set(CONVERSATION_PLATFORMS))

        if args.gate_profile is not None:
            # Tiered gate: derive thresholds from maturity stage.
            stage = NodeMaturityStage(args.gate_profile)
            gate_config = GenesisGateConfig.for_stage(
                stage,
                available_providers=tuple(required_providers),
                target_providers=tuple(target_providers),
            )
            if args.fail_open:
                # Respect --fail-open even with profile-based config.
                gate_config = GenesisGateConfig(
                    min_cv=gate_config.min_cv,
                    min_nodes=gate_config.min_nodes,
                    min_elite_nodes=gate_config.min_elite_nodes,
                    fail_closed=False,
                    available_providers=gate_config.available_providers,
                    target_providers=gate_config.target_providers,
                )
        else:
            # Legacy path: explicit numeric thresholds.
            gate_config = GenesisGateConfig(
                min_cv=float(args.min_cv),
                min_elite_nodes=int(args.min_elite_nodes),
                min_nodes=int(args.min_nodes),
                fail_closed=not args.fail_open,
                available_providers=tuple(required_providers),
                target_providers=tuple(target_providers),
            )

        verdict = evaluate_genesis_gate(report, gate_config)
        report["cv_core8"] = float(report.get("cv", 0.0))
        report["cv_gate"] = float(verdict.cv)
        report["genesis_gate"] = verdict.to_dict()
        report["gate_provider_set"] = args.gate_provider_set
        report["gate_profile"] = args.gate_profile  # None when using legacy thresholds
        report["required_providers"] = required_providers
        report["target_providers"] = target_providers

        coverage = {
            str(provider).strip().lower()
            for provider in (report.get("provider_coverage") or [])
            if str(provider).strip()
        }
        report["gate_provider_coverage"] = sorted(coverage & set(required_providers))
        missing_target = set(target_providers) - coverage
        collection_gap_providers = sorted(missing_target & set(CONVERSATION_GAP))
        report["collection_gap_providers"] = collection_gap_providers
        report["collection_gap_count"] = len(collection_gap_providers)

    if args.ingest_bizra_memory or args.export_ingest_jsonl:
        memory_ingest = ingest_report_nodes(
            report,
            min_snr=float(args.ingest_min_snr),
            session_id=int(args.ingest_session_id),
            export_jsonl_path=args.export_ingest_jsonl or None,
        )
        report["memory_ingest"] = memory_ingest.to_dict()

    if args.checkpoint_dir:
        checkpoint_meta = _write_checkpoint(
            checkpoint_dir=Path(args.checkpoint_dir).expanduser().resolve(),
            report=report,
            label=args.checkpoint_label,
        )
        report["checkpoint"] = checkpoint_meta

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(_format_summary(report))

    if args.genesis_gate:
        return 0 if bool((report.get("genesis_gate") or {}).get("passed", False)) else 3
    return 0 if report["cv"] >= 1.0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
