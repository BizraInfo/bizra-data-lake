"""
Economic Equilibrium — Riba Zero (R = 0) Auditor.

Audits the SEED ledger for constitutional compliance:
1. All amounts are exact integers (no floating-point drift)
2. No negative balances (no hidden lending)
3. Zakat deduction is exactly floor(gross * 25 / 1000)
4. No interest transactions exist

THEOREM (Zero Drift):
    For all x, y in Z_s: Error(x op y) = 0 for op in {+, -, *, div_exact}
    because all operations are integer, no IEEE 754 involved.

Standing on Giants:
- Babylonian scribes (1900 BCE): Regular numbers guarantee exact reciprocals
- Mansfield (2021): Si.427 — systematic exact arithmetic
- Ramanujan (1919): Taxicab numbers — number-theoretic structure
- BIZRA Constitution: Riba Zero — no extractive economics
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger("bizra.treasury.riba_zero")


@dataclass
class Violation:
    """A single Riba Zero violation."""

    tx_id: str
    rule: str
    detail: str


@dataclass
class AuditResult:
    """Result of a Riba Zero audit."""

    total_transactions: int
    violations: List[Violation] = field(default_factory=list)

    @property
    def riba_zero(self) -> bool:
        return len(self.violations) == 0


def is_regular(n: int) -> bool:
    """Check if n is 2,3,5-smooth (Hamming number / Regular number).

    These are the numbers the Babylonian scribes could divide exactly
    in base-60 arithmetic. BIZRA uses them to guarantee zero drift.
    """
    if n <= 0:
        return False
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1


@dataclass
class ClosureResult:
    """Result of Sippar closure verification."""

    a: tuple[int, int, int]
    b: tuple[int, int, int]
    product: tuple[int, int, int]
    is_regular: bool


@dataclass
class AdditionResult:
    """Result of addition regularity check."""

    a: int
    b: int
    total: int
    is_regular: bool
    requires_promotion: bool


def verify_sippar_closure(
    a: tuple[int, int, int],
    b: tuple[int, int, int],
) -> ClosureResult:
    """Verify that multiplication of two regular numbers is regular.

    RegularNumber = 2^e2 * 3^e3 * 5^e5
    Product exponents: (e2+e2', e3+e3', e5+e5') — always regular.
    """
    product = (a[0] + b[0], a[1] + b[1], a[2] + b[2])
    value = (2 ** product[0]) * (3 ** product[1]) * (5 ** product[2])
    return ClosureResult(a=a, b=b, product=product, is_regular=is_regular(value))


def verify_addition_safety(a_val: int, b_val: int) -> AdditionResult:
    """Addition may produce irregular numbers — detect and flag."""
    total = a_val + b_val
    regular = is_regular(total) if total > 0 else False
    return AdditionResult(
        a=a_val,
        b=b_val,
        total=total,
        is_regular=regular,
        requires_promotion=not regular,
    )


class RibaZeroAuditor:
    """Audit the SEED ledger for Riba Zero compliance.

    Walks every transaction and verifies:
    1. Exact integer amounts (no float contamination)
    2. No negative balances
    3. Correct zakat deduction (floor(gross * 25 / 1000))
    4. No interest transactions
    """

    def __init__(self, ledger_path: Path) -> None:
        self._ledger_path = ledger_path

    def audit(self) -> AuditResult:
        """Run the full Riba Zero audit."""
        violations: List[Violation] = []
        running_balance: Dict[str, int] = {}
        tx_count = 0

        for tx in self._read_transactions():
            tx_count += 1
            tx_id = tx.get("tx_id", f"line_{tx_count}")

            # Rule 1: amount must be integer
            amount = tx.get("amount", 0)
            if not isinstance(amount, int):
                violations.append(
                    Violation(
                        tx_id=tx_id,
                        rule="EXACT_AMOUNT",
                        detail=f"amount is {type(amount).__name__}, expected int",
                    )
                )
                continue

            # Rule 2: no negative balances
            recipient = tx.get("recipient", tx.get("node_id", "unknown"))
            running_balance[recipient] = running_balance.get(recipient, 0) + amount
            if running_balance[recipient] < 0:
                violations.append(
                    Violation(
                        tx_id=tx_id,
                        rule="NO_NEGATIVE_BALANCE",
                        detail=f"balance would be {running_balance[recipient]}",
                    )
                )

            # Rule 3: zakat exactness on mint transactions
            if tx.get("tx_type") == "mint":
                gross = tx.get("gross_amount", 0)
                expected_zakat = gross * 25 // 1000
                actual_zakat = tx.get("zakat_deducted", 0)
                if actual_zakat != expected_zakat:
                    violations.append(
                        Violation(
                            tx_id=tx_id,
                            rule="ZAKAT_EXACT",
                            detail=f"expected zakat {expected_zakat}, got {actual_zakat}",
                        )
                    )

            # Rule 4: no interest transactions
            if tx.get("tx_type") == "interest":
                violations.append(
                    Violation(
                        tx_id=tx_id,
                        rule="RIBA_ZERO",
                        detail="interest transaction detected — constitutional violation",
                    )
                )

        return AuditResult(total_transactions=tx_count, violations=violations)

    def _read_transactions(self) -> List[Dict[str, Any]]:
        """Read all transactions from the JSONL ledger."""
        if not self._ledger_path.exists():
            return []
        txs: List[Dict[str, Any]] = []
        with open(self._ledger_path, "r", encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    txs.append(json.loads(stripped))
                except json.JSONDecodeError:
                    logger.warning("Corrupt ledger line skipped: %s", stripped[:80])
        return txs
