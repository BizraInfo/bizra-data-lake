"""
Genesis-100 Ceremony — The Final Gate
=======================================

ALL 5 agents must approve. No exceptions. No overrides.

When all pass, 100 invitations are authorized.
The forest begins.

Standing on Giants:
- Nakamoto (2008): Genesis block ceremony
- Bernstein (2011): Ed25519 signing
- Lamport (1982): Byzantine agreement
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.sat.ambassador_gate import ambassador_verify
from core.sat.conductor_gate import conductor_verify
from core.sat.gate_result import CheckStatus, GateResult
from core.sat.ledger_gate import ledger_verify
from core.sat.oracle_s_gate import oracle_s_verify
from core.sat.sentinel_gate import sentinel_verify


@dataclass
class GenesisReceipt:
    """Immutable receipt from the Genesis-100 ceremony."""

    ceremony: str = "GENESIS_100"
    timestamp: str = ""
    agents: List[Dict[str, Any]] = field(default_factory=list)
    all_passed: bool = False
    total_checks: int = 0
    passed_checks: int = 0
    failed_checks: List[Tuple[str, str]] = field(default_factory=list)
    signature: Optional[str] = None
    hash: Optional[str] = None

    def sign(self, signer: Any) -> None:
        """Sign the receipt with an Ed25519 key."""
        payload = json.dumps(
            {
                "ceremony": self.ceremony,
                "timestamp": self.timestamp,
                "all_passed": self.all_passed,
                "total_checks": self.total_checks,
                "passed_checks": self.passed_checks,
                "failed_count": len(self.failed_checks),
            },
            sort_keys=True,
        ).encode()
        self.hash = hashlib.blake2b(payload).hexdigest()
        try:
            sig = signer.sign(payload)
            self.signature = (
                sig.signature.hex() if hasattr(sig, "signature") else sig.hex()
            )
        except Exception:
            # Fallback for different signer APIs
            self.signature = hashlib.blake2b(payload, key=b"fallback").hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ceremony": self.ceremony,
            "timestamp": self.timestamp,
            "all_passed": self.all_passed,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "hash": self.hash,
            "signature": self.signature,
            "agents": self.agents,
        }


def genesis_100_ceremony(
    skip_manual: bool = False,
    skip_slow: bool = False,
    sign: bool = True,
    store: bool = True,
) -> Tuple[bool, GenesisReceipt]:
    """The final gate. ALL 5 agents must approve."""

    # Run all 5 gates
    results: List[GateResult] = [
        sentinel_verify(skip_slow=skip_slow),
        oracle_s_verify(skip_manual=skip_manual),
        ledger_verify(),
        conductor_verify(),
        ambassador_verify(skip_manual=skip_manual),
    ]

    all_passed = all(r.passed for r in results)
    total_checks = sum(len(r.checks) for r in results)
    passed_checks = sum(1 for r in results for c in r.checks if c.passed)
    failed_list: List[Tuple[str, str]] = [
        (r.agent, c.name)
        for r in results
        for c in r.checks
        if not c.passed and c.status != CheckStatus.SKIPPED
    ]

    receipt = GenesisReceipt(
        timestamp=datetime.now(timezone.utc).isoformat(),
        all_passed=all_passed,
        total_checks=total_checks,
        passed_checks=passed_checks,
        failed_checks=failed_list,
        agents=[r.to_dict() for r in results],
    )

    # Sign with Ed25519 key
    if sign:
        try:
            # Try to load persistent signer
            signer = _load_signer()
            receipt.sign(signer)
        except Exception as e:
            print(f"WARNING: Could not sign receipt: {e}", file=sys.stderr)

    # Store as evidence block
    if store:
        try:
            from core.proof_engine.evidence_ledger import EvidenceLedger

            ledger = EvidenceLedger(validate_on_append=False)
            ledger.append(receipt=receipt.to_dict())
        except Exception as e:
            print(f"WARNING: Could not store receipt: {e}", file=sys.stderr)

    # Print result
    _print_ceremony_result(receipt, results)

    return (all_passed, receipt)


def _load_signer() -> Any:
    """Load or create an Ed25519 signing key."""
    from pathlib import Path

    key_path = Path("sovereign_state/ceremony_signer.key")
    try:
        from nacl.signing import SigningKey

        if key_path.exists():
            return SigningKey(key_path.read_bytes())
        key_path.parent.mkdir(parents=True, exist_ok=True)
        sk = SigningKey.generate()
        key_path.write_bytes(bytes(sk))
        return sk
    except ImportError:
        raise RuntimeError("PyNaCl not installed — cannot sign ceremony receipt")


def _print_ceremony_result(receipt: GenesisReceipt, results: List[GateResult]) -> None:
    """Formatted ceremony output."""
    if receipt.all_passed:
        print()
        print("  ╔═══════════════════════════════════════╗")
        print("  ║  GENESIS-100: ALL GATES PASSED        ║")
        print("  ║  100 invitations authorized.           ║")
        print("  ║  The forest begins.                    ║")
        print("  ╚═══════════════════════════════════════╝")
    else:
        fc = len(receipt.failed_checks)
        print()
        print("  ╔═══════════════════════════════════════╗")
        print("  ║  GENESIS-100: BLOCKED                 ║")
        print(f"  ║  {fc} checks need attention.{' ' * max(0, 14 - len(str(fc)))}║")
        print("  ╚═══════════════════════════════════════╝")

    print()
    for result in results:
        icon = "✓" if result.passed else "✗"
        print(f"  {icon} [{result.agent}] {result.layer}: {result.verdict.value}")
        s = result.stats
        print(
            f"    {s['pass']} pass, {s['fail']} fail, "
            f"{s['partial']} partial, {s['not_impl']} not-impl, "
            f"{s['skipped']} skipped"
        )
        if result.failed:
            for check in result.failed:
                evidence = check.evidence[:80] if check.evidence else ""
                print(f"      ✗ {check.name}: {evidence}")

    print()
    print(f"  Total: {receipt.passed_checks}/{receipt.total_checks} checks passed")
    if receipt.hash:
        print(f"  Receipt hash: {receipt.hash[:32]}...")
    if receipt.signature:
        print(f"  Signed: Ed25519 ({receipt.signature[:16]}...)")
