#!/usr/bin/env python3
"""
SAP v0 Internal Release Gate

Hard checks for spec integrity and claim discipline:
1) Required SAP v0 artifacts exist.
2) Conformance validator passes.
3) Forbidden phrasing is absent from SAP artifacts.
4) STATUS.md contains SAP rows.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

REQUIRED_FILES = [
    "specs/sap-v0/README.md",
    "specs/sap-v0/01-core-primitives.md",
    "specs/sap-v0/02-sovereignty-constraints.md",
    "specs/sap-v0/03-wire-mapping-local-first.md",
    "specs/sap-v0/04-conformance.md",
    "specs/sap-v0/profiles/agentic-ads-retail-v0.md",
    "schemas/sap/v0/agent_card.schema.json",
    "schemas/sap/v0/permit_envelope.schema.json",
    "schemas/sap/v0/meet_open.schema.json",
    "schemas/sap/v0/meet_message.schema.json",
    "schemas/sap/v0/offer.schema.json",
    "schemas/sap/v0/disclosure.schema.json",
    "schemas/sap/v0/consent_receipt.schema.json",
    "schemas/sap/v0/outcome_receipt.schema.json",
    "schemas/sap/v0/redline_violation.schema.json",
    "scripts/spec/validate_sap_v0.py",
    "docs/internal/SAP_V0_EVIDENCE_MATRIX.md",
    "docs/internal/SAP_AGENTIC_ADS_PILOT_KPIS.md",
    "STATUS.md",
]

FORBIDDEN_PHRASES = [
    "externally audited benchmark",
    "independently verified measurement",
]

SAP_SCAN_GLOBS = [
    "specs/sap-v0/**/*.md",
    "docs/internal/SAP_V0_EVIDENCE_MATRIX.md",
    "docs/internal/SAP_AGENTIC_ADS_PILOT_KPIS.md",
]

STATUS_REQUIRED_SNIPPETS = [
    "SAP v0 protocol specification package",
    "SAP v0 schema + conformance fixture pack",
    "Agentic Ads Retail profile v0",
]


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")


def ok(msg: str) -> None:
    print(f"[PASS] {msg}")


def check_required_files() -> int:
    missing = []
    for rel in REQUIRED_FILES:
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
    if missing:
        for item in missing:
            fail(f"missing required file: {item}")
        return 1
    ok(f"required files present ({len(REQUIRED_FILES)})")
    return 0


def check_forbidden_phrases() -> int:
    errors = 0
    scan_paths: list[Path] = []
    for pattern in SAP_SCAN_GLOBS:
        scan_paths.extend(sorted(ROOT.glob(pattern)))

    seen = set()
    for path in scan_paths:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        content = path.read_text(encoding="utf-8").lower()
        for phrase in FORBIDDEN_PHRASES:
            if phrase in content:
                fail(f"forbidden phrase '{phrase}' found in {path.relative_to(ROOT)}")
                errors += 1
    if errors == 0:
        ok("no forbidden external-audit phrasing in SAP artifacts")
    return 1 if errors else 0


def check_status_rows() -> int:
    status_path = ROOT / "STATUS.md"
    content = status_path.read_text(encoding="utf-8")
    missing = [s for s in STATUS_REQUIRED_SNIPPETS if s not in content]
    if missing:
        for item in missing:
            fail(f"STATUS.md missing SAP row snippet: {item}")
        return 1
    ok("STATUS.md contains SAP rows")
    return 0


def run_conformance() -> int:
    cmd = [sys.executable, str(ROOT / "scripts/spec/validate_sap_v0.py")]
    result = subprocess.run(cmd, cwd=ROOT, check=False)
    if result.returncode != 0:
        fail("SAP conformance validation failed")
        return 1
    ok("SAP conformance validation passed")
    return 0


def main() -> int:
    checks = [
        check_required_files,
        check_forbidden_phrases,
        check_status_rows,
        run_conformance,
    ]
    failures = 0
    for check in checks:
        failures += check()
    if failures:
        print(f"\nSAP release gate FAILED ({failures} check failure(s))")
        return 1
    print("\nSAP release gate PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

