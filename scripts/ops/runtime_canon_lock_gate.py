"""
Runtime Canon Lock Gate

Machine-enforced verification that public canonical mission surfaces converge on
one authority path:

surface -> runtime.mission() -> organism receipt -> Node0 ingest -> breathe
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimeCanonLockConfig:
    program: dict[str, Any]
    repo_root: Path
    files: dict[str, Path]
    checks: dict[str, list[str]]
    thresholds: dict[str, Any]
    giants_protocol: list[str]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def load_config(path: Path) -> RuntimeCanonLockConfig:
    payload = _load_json(path)
    repo_root = Path(str(payload.get("repo_root", "."))).resolve()
    files = {
        key: repo_root / str(value)
        for key, value in (payload.get("files") or {}).items()
    }
    return RuntimeCanonLockConfig(
        program=payload.get("program") or {},
        repo_root=repo_root,
        files=files,
        checks={
            key: [str(item) for item in value]
            for key, value in (payload.get("checks") or {}).items()
            if isinstance(value, list)
        },
        thresholds=payload.get("thresholds") or {},
        giants_protocol=[str(item) for item in (payload.get("giants_protocol") or [])],
    )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _contains_all(text: str, patterns: list[str]) -> tuple[bool, list[str]]:
    missing = [pattern for pattern in patterns if pattern not in text]
    return not missing, missing


def _check_sources(cfg: RuntimeCanonLockConfig) -> tuple[list[dict[str, Any]], bool]:
    api_text = _read(cfg.files["api"])
    cli_text = _read(cfg.files["cli"])
    plan_tests_text = _read(cfg.files["plan_tests"])
    cli_tests_text = _read(cfg.files["cli_tests"])

    checks: list[dict[str, Any]] = []

    def _append(name: str, text: str, patterns: list[str], detail: str) -> None:
        passed, missing = _contains_all(text, patterns)
        checks.append(
            {
                "name": name,
                "passed": passed,
                "detail": detail,
                "missing": missing,
            }
        )

    _append(
        "api_canonical_runtime_authority",
        api_text,
        cfg.checks.get("api_canonical_runtime_authority", []),
        "Canonical /v1/plan must route through runtime mission authority.",
    )
    _append(
        "api_noncanonical_shim_explicit",
        api_text,
        cfg.checks.get("api_noncanonical_shim_explicit", []),
        "API-local reflex shortcut must be scoped to noncanonical runtime paths only.",
    )
    _append(
        "api_runtime_reflex_lineage",
        api_text,
        cfg.checks.get("api_runtime_reflex_lineage", []),
        "Canonical reflex lineage must come from runtime-owned receipt metadata.",
    )
    _append(
        "cli_runtime_authority",
        cli_text,
        cfg.checks.get("cli_runtime_authority", []),
        "Canonical CLI mission command must call runtime.mission().",
    )
    _append(
        "plan_tests_cover_canonical_authority",
        plan_tests_text,
        cfg.checks.get("plan_tests_cover_canonical_authority", []),
        "Integration tests must cover canonical API authority and runtime-owned S1.",
    )
    _append(
        "cli_tests_cover_runtime_authority",
        cli_tests_text,
        cfg.checks.get("cli_tests_cover_runtime_authority", []),
        "CLI tests must prove runtime.mission() is the canonical mission authority.",
    )

    passed = all(item["passed"] for item in checks)
    return checks, passed


def build_report(cfg: RuntimeCanonLockConfig) -> dict[str, Any]:
    checks, gate_passed = _check_sources(cfg)
    score = round(
        sum(1 for item in checks if item["passed"]) / max(len(checks), 1),
        4,
    )
    status = "LOCKED" if gate_passed else "DEGRADED"
    receipt_body = {
        "program_id": cfg.program.get("id", "runtime_canon_lock_gate"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checks": checks,
        "score": score,
        "status": status,
    }
    encoded = json.dumps(receipt_body, sort_keys=True, separators=(",", ":"))
    receipt_hash = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    return {
        "program": cfg.program,
        "gate_passed": gate_passed,
        "status": status,
        "score": score,
        "checks": checks,
        "standing_on_giants_protocol": cfg.giants_protocol,
        "metrics": {
            "checks_total": len(checks),
            "checks_passed": sum(1 for item in checks if item["passed"]),
            "required_score": float(cfg.thresholds.get("min_score", 1.0)),
        },
        "receipt": {
            "receipt_id": f"rclg-{receipt_hash[:16]}",
            "payload_hash": receipt_hash,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def _write_github_output(path: Path, report: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            f"runtime_canon_lock_passed={str(report['gate_passed']).lower()}\n"
        )
        handle.write(f"runtime_canon_lock_status={report['status']}\n")
        handle.write(f"runtime_canon_lock_score={report['score']}\n")


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Runtime Canon Lock Gate",
        "",
        f"- Status: `{report['status']}`",
        f"- Gate passed: `{str(report['gate_passed']).lower()}`",
        f"- Score: `{report['score']:.4f}`",
        "",
        "## Checks",
        "",
    ]
    for check in report["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        lines.append(f"- `{marker}` {check['name']}: {check['detail']}")
        if check["missing"]:
            lines.append(f"  Missing: {', '.join(check['missing'])}")
    lines.extend(
        [
            "",
            "## Standing On Giants",
            "",
        ]
    )
    for item in report["standing_on_giants_protocol"]:
        lines.append(f"- {item}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_runtime_canon_lock_gate(
    *,
    config_path: Path,
    report_path: Path | None = None,
    markdown_report_path: Path | None = None,
    github_output: Path | None = None,
) -> dict[str, Any]:
    cfg = load_config(config_path)
    report = build_report(cfg)
    min_score = float(cfg.thresholds.get("min_score", 1.0))
    if report["score"] < min_score:
        report["gate_passed"] = False
        report["status"] = "DEGRADED"
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if markdown_report_path is not None:
        markdown_report_path.parent.mkdir(parents=True, exist_ok=True)
        _write_markdown(markdown_report_path, report)
    if github_output is not None:
        github_output.parent.mkdir(parents=True, exist_ok=True)
        _write_github_output(github_output, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the runtime canon lock gate.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/runtime_canon_lock_gate.json"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/tmp/phase65/runtime_canon_lock_gate.json"),
    )
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=Path("/tmp/phase65/runtime_canon_lock_gate.md"),
    )
    parser.add_argument(
        "--github-output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    report = run_runtime_canon_lock_gate(
        config_path=args.config,
        report_path=args.report,
        markdown_report_path=args.markdown_report,
        github_output=args.github_output,
    )
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
