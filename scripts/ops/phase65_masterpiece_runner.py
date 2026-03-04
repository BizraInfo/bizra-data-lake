"""
Phase65 masterpiece runner.

Single-command orchestrator for:
1) lifecycle emulation
2) blueprint gate evaluation
3) KPI snapshot generation
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.node0_lifecycle_emulation import (
        EmulationConfig,
        run_lifecycle_emulation,
    )
    from scripts.ops.phase65_alpha_launch_packet import (
        _emit_github_outputs as _emit_alpha_outputs,
    )
    from scripts.ops.phase65_alpha_launch_packet import (
        _load_json as _load_manual_checks_json,
    )
    from scripts.ops.phase65_alpha_launch_packet import (
        _load_yaml as _load_blueprint_yaml,
    )
    from scripts.ops.phase65_alpha_launch_packet import (
        build_alpha_launch_packet,
    )
    from scripts.ops.phase65_alpha_launch_packet import (
        render_markdown as render_alpha_markdown,
    )
    from scripts.ops.phase65_blueprint_gate import _load_yaml, evaluate
    from scripts.ops.phase65_kpi_pack import (
        _emit_github_outputs,
        build_kpi_snapshot,
        render_markdown,
    )
except ModuleNotFoundError:  # pragma: no cover - CLI fallback
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    for mod_name in list(sys.modules):
        if mod_name == "scripts" or mod_name.startswith("scripts."):
            sys.modules.pop(mod_name, None)
    from scripts.node0_lifecycle_emulation import (
        EmulationConfig,
        run_lifecycle_emulation,
    )
    from scripts.ops.phase65_alpha_launch_packet import (
        _emit_github_outputs as _emit_alpha_outputs,
    )
    from scripts.ops.phase65_alpha_launch_packet import (
        _load_json as _load_manual_checks_json,
    )
    from scripts.ops.phase65_alpha_launch_packet import (
        _load_yaml as _load_blueprint_yaml,
    )
    from scripts.ops.phase65_alpha_launch_packet import (
        build_alpha_launch_packet,
    )
    from scripts.ops.phase65_alpha_launch_packet import (
        render_markdown as render_alpha_markdown,
    )
    from scripts.ops.phase65_blueprint_gate import _load_yaml, evaluate
    from scripts.ops.phase65_kpi_pack import (
        _emit_github_outputs,
        build_kpi_snapshot,
        render_markdown,
    )


def run_phase65_masterpiece(
    *,
    state_dir: Path,
    out_dir: Path,
    config_path: Path,
    strict_signing: bool,
    alpha_users_target: int = 100,
    require_tier: str = "elite-operational",
    strict_manual: bool = False,
    manual_checks_path: Path | None = None,
    github_output: Path | None = None,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    lifecycle_payload = run_lifecycle_emulation(
        state_dir=state_dir,
        config=EmulationConfig(strict_signing=strict_signing),
    )
    summary_path = out_dir / "lifecycle_summary.json"
    summary_path.write_text(json.dumps(lifecycle_payload, indent=2), encoding="utf-8")

    summary = lifecycle_payload.get("summary", lifecycle_payload)
    config = _load_yaml(config_path)
    gate_report = evaluate(summary, config, payload=lifecycle_payload)
    gate_path = out_dir / "phase65_gate_report.json"
    gate_path.write_text(json.dumps(gate_report, indent=2), encoding="utf-8")

    kpi_snapshot = build_kpi_snapshot(lifecycle_payload, gate_report)
    kpi_json_path = out_dir / "phase65_kpi_snapshot.json"
    kpi_md_path = out_dir / "phase65_kpi_snapshot.md"
    kpi_json_path.write_text(json.dumps(kpi_snapshot, indent=2), encoding="utf-8")
    kpi_md_path.write_text(render_markdown(kpi_snapshot), encoding="utf-8")
    if github_output is not None:
        _emit_github_outputs(kpi_snapshot, github_output)

    manual_values: dict[str, Any] | None = None
    if manual_checks_path is not None:
        manual_values = _load_manual_checks_json(manual_checks_path)

    alpha_packet = build_alpha_launch_packet(
        summary_payload=lifecycle_payload,
        gate_payload=gate_report,
        kpi_payload=kpi_snapshot,
        cfg=_load_blueprint_yaml(config_path),
        summary_path=summary_path,
        gate_path=gate_path,
        kpi_path=kpi_json_path,
        alpha_users_target=alpha_users_target,
        require_tier=require_tier,
        strict_manual=strict_manual,
        manual_values=manual_values,
        signer_private_key_hex=os.getenv("BIZRA_RECEIPT_PRIVATE_KEY_HEX", "").strip(),
    )
    alpha_json_path = out_dir / "phase65_alpha_launch_packet.json"
    alpha_md_path = out_dir / "phase65_alpha_launch_packet.md"
    alpha_json_path.write_text(json.dumps(alpha_packet, indent=2), encoding="utf-8")
    alpha_md_path.write_text(render_alpha_markdown(alpha_packet), encoding="utf-8")
    if github_output is not None:
        _emit_alpha_outputs(alpha_packet, github_output)

    return {
        "summary_path": str(summary_path),
        "gate_report_path": str(gate_path),
        "kpi_json_path": str(kpi_json_path),
        "kpi_md_path": str(kpi_md_path),
        "alpha_packet_json_path": str(alpha_json_path),
        "alpha_packet_md_path": str(alpha_md_path),
        "gate_passed": bool(gate_report.get("gate_passed", False)),
        "snr_score": float(gate_report.get("snr_score", 0.0)),
        "launch_decision": str(alpha_packet.get("decision", "NO_GO")),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Phase65 lifecycle + gate + KPI pipeline."
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("/tmp/phase65/state"),
        help="Lifecycle state directory.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/tmp/phase65"),
        help="Output directory for summary, gate report, KPI artifacts.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/phase65_masterpiece_roadmap.yaml"),
        help="Blueprint config YAML path.",
    )
    parser.add_argument(
        "--strict-signing",
        action="store_true",
        help="Require explicit signer key via environment variables.",
    )
    parser.add_argument(
        "--alpha-users-target",
        type=int,
        default=100,
        help="Target number of alpha users for launch packet.",
    )
    parser.add_argument(
        "--require-tier",
        type=str,
        default="elite-operational",
        help="Required KPI tier for automated launch readiness.",
    )
    parser.add_argument(
        "--strict-manual",
        action="store_true",
        help="Treat pending manual checks as launch blockers.",
    )
    parser.add_argument(
        "--manual-checks",
        type=Path,
        default=None,
        help="Optional JSON path for manual readiness checks.",
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        default=None,
        help="Optional GITHUB_OUTPUT path for CI job outputs.",
    )
    args = parser.parse_args()

    result = run_phase65_masterpiece(
        state_dir=args.state_dir,
        out_dir=args.out_dir,
        config_path=args.config,
        strict_signing=args.strict_signing,
        alpha_users_target=args.alpha_users_target,
        require_tier=args.require_tier,
        strict_manual=args.strict_manual,
        manual_checks_path=args.manual_checks,
        github_output=args.github_output,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
