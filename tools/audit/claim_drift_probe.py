#!/usr/bin/env python3
"""Claim-discipline drift probe for BIZRA omnidirectional audit (debug session c98f9f).

Read-only. Emits NDJSON log lines to the debug session log path for each finding.

The probe implements the hypothesis battery:

- H1: overclaim drift in top-level briefs and status docs.
- H2: C-class numeric claims without receipts.
- H3: Origin Kernel section 6.3 discipline drift.
- H4: canonical-sentence ambiguity (both canonical lines coexisting).
- H5: pilot-evidence scope creep beyond MEASURED_LOCAL_ARTIFACT.

It does not mutate any file, does not open network connections, and does not
ingest any document into runtime canon. It only reads files inside the repo
and appends NDJSON lines to the debug log path.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LOG_PATH = REPO / ".cursor" / "debug-c98f9f.log"
SESSION_ID = "c98f9f"
RUN_ID = "scan"


def emit(**kwargs) -> None:
    """Append a single NDJSON line to the debug log path."""
    payload = {
        "sessionId": SESSION_ID,
        "runId": RUN_ID,
        "timestamp": int(time.time() * 1000),
    }
    payload.update(kwargs)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


TARGETS = [
    "ULTIMATE_MASTERPIECE_EXECUTIVE_BRIEF.md",
    "ULTIMATE_MASTERPIECE_MANIFESTO.md",
    "ULTIMATE_MASTERPIECE_POLYMATH_SYNTHESIS.md",
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
]


H1_PATTERNS = [
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
]

H2_PATTERNS = [
    (r"SNR[^\n]{0,40}0\.974", "C-class numeric SNR 0.974 without receipt link"),
    (r"\$0\.10\s*[\u2192\-\>]+\s*\$0\.008", "C-class cost-drop claim"),
    (r"100%\s+pass", "100% pass claim"),
    (r"\b73\s*/\s*100\b", "73/100 nodes claim"),
]

H3_PATTERNS = [
    (
        r"Origin\s+Kernel[^\n]{0,120}(ingested|committed\s+to\s+main|runtime\s+canon)",
        "Origin Kernel section 6.3 discipline drift",
    ),
]

H5_PATTERNS = [
    (
        r"(cross[- ]device|multi[- ]node)[^\n]{0,80}(MEASURED|proven|production)",
        "pilot scope creep beyond local artifact",
    ),
]

H4_A = re.compile(r"Node0 proves the seed can live alone", re.IGNORECASE)
H4_B = re.compile(r"Each human node mints\s*PAT[- ]?7", re.IGNORECASE)


def scan_file(rel_path: str) -> None:
    path = REPO / rel_path
    if not path.exists():
        emit(
            hypothesisId="H0",
            location=f"{rel_path}:0",
            message="target file missing",
            data={"path": str(path)},
        )
        return
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        emit(
            hypothesisId="H0",
            location=f"{rel_path}:0",
            message="target file unreadable",
            data={"error": str(exc)},
        )
        return

    lines = text.splitlines()
    for hid, patterns in (
        ("H1", H1_PATTERNS),
        ("H2", H2_PATTERNS),
        ("H3", H3_PATTERNS),
        ("H5", H5_PATTERNS),
    ):
        for pattern, why in patterns:
            rx = re.compile(pattern, re.IGNORECASE)
            for lineno, line in enumerate(lines, 1):
                match = rx.search(line)
                if match:
                    emit(
                        hypothesisId=hid,
                        location=f"{rel_path}:{lineno}",
                        message=why,
                        data={
                            "pattern": pattern,
                            "matched": match.group(0)[:200],
                            "line": line.strip()[:300],
                        },
                    )

    has_legacy = bool(H4_A.search(text))
    has_canonical = bool(H4_B.search(text))
    if has_legacy and has_canonical:
        emit(
            hypothesisId="H4",
            location=f"{rel_path}:0",
            message="document carries BOTH legacy and Topology Canon canonical sentences",
            data={"legacy_seed": has_legacy, "topology_canon": has_canonical},
        )


def main() -> None:
    emit(
        hypothesisId="meta",
        location="probe:start",
        message="claim drift probe start",
        data={"targets": len(TARGETS), "repo": str(REPO)},
    )
    for target in TARGETS:
        scan_file(target)
    emit(
        hypothesisId="meta",
        location="probe:end",
        message="claim drift probe end",
        data={},
    )


if __name__ == "__main__":
    main()
