#!/usr/bin/env python3
"""Claim-discipline drift probe for BIZRA omnidirectional audit.

Read-only. Scans target markdown files for documented claim-discipline
violations and, in ``--ci`` mode, fails closed on any CLEAN_SET violation.

Two operating modes:

1. Debug mode (default): emits NDJSON findings to a log path so an operator
   or debugger can inspect every hit. Always exits 0.

2. CI-gate mode (``--ci``): additionally treats any H1 / H4 hit inside the
   CLEAN_SET as a failure and exits non-zero. The CLEAN_SET is the subset
   of files asserted to be claim-clean today; the WATCH_SET is monitored
   but not gated (those files may legitimately *quote* prohibited phrases
   while explaining the rule).

Suppression: any line containing ``claim-probe: allow`` (case-sensitive)
is exempted from gating. Use this sparingly and only when a prohibited
phrase is discussed in meta-context (e.g. inside a code block or a
register that names the phrase verbatim).

This probe never mutates any file, never opens network connections, and
does not ingest any document into runtime canon.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

REPO = Path(__file__).resolve().parents[2]
DEFAULT_LOG_PATH = REPO / ".cursor" / "debug-c98f9f.log"
SESSION_ID = "c98f9f"

ALLOW_MARKER = "claim-probe: allow"

# Files asserted claim-clean *today*. CI gate fails on any H1/H4 hit here.
CLEAN_SET: Tuple[str, ...] = (
    "ULTIMATE_MASTERPIECE_EXECUTIVE_BRIEF.md",
    "ULTIMATE_MASTERPIECE_MANIFESTO.md",
    "ULTIMATE_MASTERPIECE_POLYMATH_SYNTHESIS.md",
)

# Files monitored but not gated. May legitimately discuss prohibited phrases
# in a meta/quoting context.
WATCH_SET: Tuple[str, ...] = (
    "STATUS.md",
    "TMP_v1.0.md",
    "docs/ORGANISM_STATE_v0.88.1.md",
    "docs/ROLLBACK-RUNBOOK-Cycle-5.md",
    "docs/TOOL_TECHNOLOGY_MATRIX.md",
    "docs/WEBSITE_PLAN.md",
    "docs/business/BUSINESS_PLAN.md",
    "docs/business/EARLY_CUSTOMERS_OUTREACH.md",
    "docs/business/INVESTOR_PACKAGE.md",
    "docs/business/ONE_PAGE_PITCH.md",
    "docs/architecture/BIZRA_NODE0_TO_URP_ECOSYSTEM_TRANSITION_v0_1.md",
    "docs/gtm/node0_activation_go_to_market_v0_1/README.md",
    "docs/gtm/node0_activation_go_to_market_v0_1/GO_TO_MARKET_PLAN.md",
    "docs/gtm/node0_activation_go_to_market_v0_1/PRODUCTION_READINESS_AND_GTM_CLOSURE_SPRINT.md",
    "docs/gtm/node0_activation_go_to_market_v0_1/PILOT_EVIDENCE_REGISTER.md",
    "docs/gtm/node0_activation_go_to_market_v0_1/INVESTOR_OPERATOR_HANDOVER.md",
    "docs/gtm/node0_activation_go_to_market_v0_1/EXECUTIVE_STRATEGY_MEMO.md",
    "docs/gtm/node0_activation_go_to_market_v0_1/BUSINESS_MODEL_AND_PRICING_OPTIONS.md",
    "docs/gtm/node0_activation_go_to_market_v0_1/CLAIM_DISCIPLINE_FOR_NODE0_AND_URP.md",
    "docs/audits/omnidirectional_hyperdimensional_audit_v0_1/CANON_STORE_INGESTION_GATE_DESIGN.md",
    "docs/audits/omnidirectional_hyperdimensional_audit_v0_1/NODE0_ACTIVATION_READINESS_AUDIT.md",
)

H1_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"\bREADY\s+FOR\s+PRODUCTION\b", "explicit production-ready overclaim"),
    (
        r"\bproduction[- ]ready\b(?!.{0,80}(?:PARTIAL|PLANNED|CANDIDATE|NOT\s+YET|PREPARATION|NO-GO))",
        "production-ready without truth-label qualifier",
    ),
    (r"\btrustless\b", "trustless claim; BIZRA operates fail-closed"),
    (r"\bAGI\b", "PROHIBITED AGI claim"),
    (r"\bworld[- ]first\b", "PROHIBITED world-first claim"),
    (r"\bfirst[- ]in[- ]the[- ]world\b", "PROHIBITED first-in-the-world claim"),
    (r"\bguaranteed\b", "PROHIBITED marketing absolute"),
    (r"\bno\s+risk\b", "PROHIBITED marketing absolute"),
    (r"\bsave\s+you\s+money\b", "PROHIBITED marketing absolute"),
)

H2_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (r"SNR[^\n]{0,40}0\.974", "C-class numeric SNR 0.974 without receipt link"),
    (r"\$0\.10\s*[\u2192\-\>]+\s*\$0\.008", "C-class cost-drop claim"),
    (r"100%\s+pass", "100% pass claim"),
    (r"\b73\s*/\s*100\b", "73/100 nodes claim"),
)

H3_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (
        r"Origin\s+Kernel[^\n]{0,120}(ingested|committed\s+to\s+main|runtime\s+canon)",
        "Origin Kernel section 6.3 discipline drift",
    ),
)

H5_PATTERNS: Tuple[Tuple[str, str], ...] = (
    (
        r"(cross[- ]device|multi[- ]node)[^\n]{0,80}(MEASURED|proven|production)",
        "pilot scope creep beyond local artifact",
    ),
)

H4_A = re.compile(r"Node0 proves the seed can live alone", re.IGNORECASE)
H4_B = re.compile(r"Each human node mints\s*PAT[- ]?7", re.IGNORECASE)


class LogSink:
    """Append-only NDJSON sink; inert when path is None."""

    def __init__(self, path: Optional[Path], run_id: str) -> None:
        self.path = path
        self.run_id = run_id
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, **kwargs: object) -> None:
        if self.path is None:
            return
        payload = {
            "sessionId": SESSION_ID,
            "runId": self.run_id,
            "timestamp": int(time.time() * 1000),
        }
        payload.update(kwargs)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def scan_file(rel_path: str, sink: LogSink) -> List[dict]:
    """Return findings for a single file. Emits to sink as a side-effect."""
    path = REPO / rel_path
    if not path.exists():
        sink.emit(
            hypothesisId="H0",
            location=f"{rel_path}:0",
            message="target file missing",
            data={"path": str(path)},
        )
        return [{"hypothesisId": "H0", "path": rel_path, "line": 0, "missing": True}]

    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        sink.emit(
            hypothesisId="H0",
            location=f"{rel_path}:0",
            message="target file unreadable",
            data={"error": str(exc)},
        )
        return [{"hypothesisId": "H0", "path": rel_path, "line": 0, "error": str(exc)}]

    lines = text.splitlines()
    findings: List[dict] = []

    for hid, patterns in (
        ("H1", H1_PATTERNS),
        ("H2", H2_PATTERNS),
        ("H3", H3_PATTERNS),
        ("H5", H5_PATTERNS),
    ):
        for pattern, why in patterns:
            rx = re.compile(pattern, re.IGNORECASE)
            for lineno, line in enumerate(lines, 1):
                if ALLOW_MARKER in line:
                    continue
                match = rx.search(line)
                if not match:
                    continue
                finding = {
                    "hypothesisId": hid,
                    "path": rel_path,
                    "line": lineno,
                    "matched": match.group(0)[:200],
                    "why": why,
                    "snippet": line.strip()[:300],
                }
                findings.append(finding)
                sink.emit(
                    hypothesisId=hid,
                    location=f"{rel_path}:{lineno}",
                    message=why,
                    data={
                        "pattern": pattern,
                        "matched": match.group(0)[:200],
                        "line": line.strip()[:300],
                    },
                )

    has_legacy = bool(H4_A.search(text)) and ALLOW_MARKER not in text
    has_canonical = bool(H4_B.search(text)) and ALLOW_MARKER not in text
    if has_legacy and has_canonical:
        findings.append(
            {
                "hypothesisId": "H4",
                "path": rel_path,
                "line": 0,
                "matched": "double-canonical",
                "why": "document carries BOTH legacy and Topology Canon canonical sentences",
                "snippet": "",
            }
        )
        sink.emit(
            hypothesisId="H4",
            location=f"{rel_path}:0",
            message="document carries BOTH legacy and Topology Canon canonical sentences",
            data={"legacy_seed": has_legacy, "topology_canon": has_canonical},
        )

    return findings


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BIZRA claim-discipline drift probe (read-only).",
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI-gate mode: exit non-zero on any CLEAN_SET H1/H4 finding.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help=(
            "NDJSON output path. Default: debug session log for local runs, "
            "suppressed in CI mode unless explicitly set."
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Write a machine-readable JSON summary to this path.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print all findings (default: CLEAN_SET only).",
    )
    parser.add_argument(
        "--run-id",
        default="scan",
        help="Run identifier written into each NDJSON record.",
    )
    return parser


def _resolve_log_path(arg: Optional[Path], ci_mode: bool) -> Optional[Path]:
    if arg is not None:
        return arg
    if ci_mode:
        # In CI we prefer stdout/summary; NDJSON is opt-in via --log-path.
        return None
    return DEFAULT_LOG_PATH


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    log_path = _resolve_log_path(args.log_path, args.ci)
    sink = LogSink(log_path, args.run_id)
    sink.emit(
        hypothesisId="meta",
        location="probe:start",
        message="claim drift probe start",
        data={
            "clean_set": len(CLEAN_SET),
            "watch_set": len(WATCH_SET),
            "repo": str(REPO),
            "ci": args.ci,
        },
    )

    clean_findings: List[dict] = []
    watch_findings: List[dict] = []

    for rel in CLEAN_SET:
        clean_findings.extend(scan_file(rel, sink))
    for rel in WATCH_SET:
        watch_findings.extend(scan_file(rel, sink))

    sink.emit(
        hypothesisId="meta",
        location="probe:end",
        message="claim drift probe end",
        data={
            "clean_findings": len(clean_findings),
            "watch_findings": len(watch_findings),
        },
    )

    clean_gating = [
        f for f in clean_findings if f["hypothesisId"] in {"H1", "H4"}
    ]

    verdict = "PASS" if not clean_gating else "FAIL"
    summary = {
        "verdict": verdict,
        "clean_set": {
            "files": len(CLEAN_SET),
            "findings": len(clean_findings),
            "gating_findings": len(clean_gating),
        },
        "watch_set": {
            "files": len(WATCH_SET),
            "findings": len(watch_findings),
        },
        "ci_mode": args.ci,
        "log_path": str(log_path) if log_path else None,
    }

    print(f"[claim-drift-probe] verdict={verdict}")
    print(
        f"  CLEAN_SET: {summary['clean_set']['files']} files, "
        f"{summary['clean_set']['findings']} findings, "
        f"{summary['clean_set']['gating_findings']} gating (H1/H4)"
    )
    print(
        f"  WATCH_SET: {summary['watch_set']['files']} files, "
        f"{summary['watch_set']['findings']} findings (report-only)"
    )
    if log_path:
        print(f"  NDJSON log: {log_path}")

    if clean_gating or args.verbose:
        print()
        print("  Gating findings (CLEAN_SET):" if clean_gating else "  No gating findings.")
        for f in clean_gating[:25]:
            print(
                f"    {f['path']}:{f['line']} "
                f"[{f['hypothesisId']}] {f['why']} — \"{f['matched']}\""
            )
        if len(clean_gating) > 25:
            print(f"    ... and {len(clean_gating) - 25} more")

    if args.verbose and watch_findings:
        print()
        print("  Watch findings (report-only):")
        for f in watch_findings[:50]:
            print(
                f"    {f['path']}:{f['line']} "
                f"[{f['hypothesisId']}] {f['why']} — \"{f.get('matched','')}\""
            )
        if len(watch_findings) > 50:
            print(f"    ... and {len(watch_findings) - 50} more")

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    if args.ci and clean_gating:
        print(
            "\n[claim-drift-probe] CI gate FAILED: "
            f"{len(clean_gating)} gating finding(s) in CLEAN_SET.",
            file=sys.stderr,
        )
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
