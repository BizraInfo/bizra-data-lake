"""MVDA CLI entrypoint — run a single dual-agentic proof cycle."""

import json
import sys
import time

from mvda.config import LEDGER_PATH
from mvda.fate_crossing import execute_mvda
from mvda.ledger import MvdaLedger


def main():
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else (
        "What is the Spearpoint seal (commit b08f2208) and why does it "
        "matter for BIZRA's constitutional integrity?"
    )

    print(f"=== MVDA v0.1 — Minimum Viable Dual-Agentic Proof ===")
    print(f"Question: {question}")
    print(f"Ledger: {LEDGER_PATH}")
    print()

    ledger = MvdaLedger()
    t0 = time.time()
    result = execute_mvda(question, ledger)
    elapsed = time.time() - t0

    print(f"── PAT Researcher ({result['pat_model']}) ──")
    print(f"Evidence refs: {result['pat_evidence_refs']}")
    print(f"Confidence: {result['pat_confidence']}")
    print(f"Answer: {result['pat_answer'][:500]}")
    print()
    print(f"── SAT Validator ({result['sat_model']}) ──")
    print(f"Verdict: {result['sat_verdict']}")
    print(f"Reason: {result['sat_reason']}")
    print(f"Ihsan score: {result['sat_ihsan_score']}")
    print(f"Evidence sufficient: {result['sat_evidence_sufficient']}")
    print()
    print(f"── Summary ──")
    print(f"Receipts emitted: {result['receipts_emitted']}")
    print(f"Elapsed: {elapsed:.1f}s")

    valid, count = ledger.verify_chain()
    print(f"Ledger entries: {count}, chain valid: {valid}")
    print()

    sys.exit(0 if result["sat_verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
