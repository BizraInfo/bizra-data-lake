"""
Phase 65 blueprint quality gate.

Evaluates lifecycle emulation summary against machine-readable roadmap gates and
computes a weighted SNR score for release readiness.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class CheckResult:
    name: str
    stream: str
    passed: bool
    actual: Any
    expected: Any
    weight: float

    def score(self) -> float:
        return self.weight if self.passed else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stream": self.stream,
            "passed": self.passed,
            "actual": self.actual,
            "expected": self.expected,
            "weight": self.weight,
            "score": self.score(),
        }


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check_signed_receipts(
    payload: dict[str, Any], summary: dict[str, Any]
) -> tuple[bool, str]:
    """Verify receipts are signed based on ledger artifacts or summary fallback."""
    artifacts = payload.get("artifacts", {}) if isinstance(payload, dict) else {}
    ledger_path = artifacts.get("ledger_path")
    if ledger_path:
        path = Path(str(ledger_path))
        if not path.exists():
            return False, f"ledger file missing: {path}"
        try:
            lines = [
                ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()
            ]
            if not lines:
                return False, "ledger has no entries"
            for ln in lines:
                entry = json.loads(ln)
                receipt = entry.get("receipt", {})
                signature = receipt.get("signature")
                if isinstance(signature, str):
                    if not signature:
                        return False, "empty receipt signature"
                    if not isinstance(
                        receipt.get("signer_pubkey"), str
                    ) or not receipt.get("signer_pubkey"):
                        return False, "missing signer_pubkey"
                elif isinstance(signature, dict):
                    if not isinstance(signature.get("value"), str) or not signature.get(
                        "value"
                    ):
                        return False, "missing signature.value"
                    if not isinstance(
                        signature.get("public_key"), str
                    ) or not signature.get("public_key"):
                        return False, "missing signature.public_key"
                else:
                    return False, "missing receipt signature"
            return True, f"signed_receipts={len(lines)}"
        except Exception as exc:  # pragma: no cover - defensive parsing
            return False, f"ledger parse error: {exc}"

    # Fallback for callers that pass summary only.
    signed = bool(summary.get("signed_receipts", False))
    return signed, f"summary.signed_receipts={signed}"


def evaluate(
    summary: dict[str, Any],
    cfg: dict[str, Any],
    *,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gates = cfg["quality_gates"]["required"]
    weights = cfg["quality_gates"]["scoring"]["weights"]
    min_snr = float(cfg["quality_gates"]["scoring"]["min_snr_score"])
    signed_ok, signed_detail = _check_signed_receipts(payload or {}, summary)

    checks = [
        CheckResult(
            name="final_state",
            stream="operations",
            passed=summary.get("final_state") == gates["final_state"],
            actual=summary.get("final_state"),
            expected=gates["final_state"],
            weight=float(weights["operations"]),
        ),
        CheckResult(
            name="ledger_chain_valid",
            stream="security",
            passed=bool(summary.get("ledger_chain_valid"))
            is bool(gates["ledger_chain_valid"]),
            actual=summary.get("ledger_chain_valid"),
            expected=gates["ledger_chain_valid"],
            weight=float(weights["security"]),
        ),
        CheckResult(
            name="avg_ihsan",
            stream="quality",
            passed=float(summary.get("avg_ihsan", 0.0))
            >= float(gates["min_avg_ihsan"]),
            actual=summary.get("avg_ihsan"),
            expected=f">={gates['min_avg_ihsan']}",
            weight=float(weights["quality"]),
        ),
        CheckResult(
            name="speedup_system1_vs_system2",
            stream="architecture",
            passed=float(summary.get("speedup_system1_vs_system2", 0.0))
            >= float(gates["min_speedup_system1_vs_system2"]),
            actual=summary.get("speedup_system1_vs_system2"),
            expected=f">={gates['min_speedup_system1_vs_system2']}",
            weight=float(weights["architecture"]),
        ),
        CheckResult(
            name="avg_latency_ms",
            stream="performance",
            passed=float(summary.get("avg_latency_ms", 9e9))
            <= float(gates["max_avg_latency_ms"]),
            actual=summary.get("avg_latency_ms"),
            expected=f"<={gates['max_avg_latency_ms']}",
            weight=float(weights["performance"]),
        ),
        CheckResult(
            name="impt_balance",
            stream="economics",
            passed=float(summary.get("impt_balance", 0.0))
            >= float(gates["min_impt_balance"]),
            actual=summary.get("impt_balance"),
            expected=f">={gates['min_impt_balance']}",
            weight=float(weights["economics"]),
        ),
        CheckResult(
            name="signed_receipts",
            stream="trust",
            passed=(
                signed_ok if bool(gates.get("signed_receipts_required", True)) else True
            ),
            actual=signed_detail,
            expected="all receipts signed",
            weight=float(weights.get("trust", 0.0)),
        ),
    ]

    raw_score = sum(c.score() for c in checks)
    total_weight = sum(c.weight for c in checks)
    snr_score = (raw_score / total_weight) if total_weight > 0 else 0.0
    hard_fail = not all(c.passed for c in checks)
    pass_snr = snr_score >= min_snr
    gate_passed = (not hard_fail) and pass_snr

    return {
        "program_id": cfg["program"]["id"],
        "program_version": cfg["program"]["version"],
        "gate_passed": gate_passed,
        "snr_score": round(snr_score, 4),
        "min_snr_score": min_snr,
        "hard_fail": hard_fail,
        "checks": [c.to_dict() for c in checks],
        "summary_snapshot": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase65 blueprint quality gates."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        required=True,
        help="Lifecycle emulation summary JSON path.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/phase65_masterpiece_roadmap.yaml"),
        help="Blueprint config YAML path.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional report JSON output path.",
    )
    args = parser.parse_args()

    summary_payload = _load_json(args.summary)
    summary = summary_payload.get("summary", summary_payload)
    config = _load_yaml(args.config)
    report = evaluate(summary, config, payload=summary_payload)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
