# Phase 4: Economic Equilibrium — Riba Zero (R = 0)

**Spec:** CMN-004
**Status:** Verification layer (existing: IMPLEMENTED)
**Formal Property:** All URP arithmetic uses Sippar exact (2,3,5-smooth); Error(x + y) = 0
**Existing Code:** `bizra-sippar/lib.rs`, `bizra-core/islamic_finance`, `core/proof_engine/seed_ledger.py`

---

## 1. Objective

Sippar exact arithmetic is implemented in Rust. This spec formalizes the **closure theorem**
(operations on regular numbers produce regular numbers) and adds a Python verification layer
that audits the SEED ledger for floating-point contamination.

---

## 2. Definitions

```
RegularNumber := n in Z+ where prime_factors(n) subset {2, 3, 5}
    Representation: (exp2, exp3, exp5) where n = 2^exp2 * 3^exp3 * 5^exp5

ExactAmount := RegularNumber with unit label (SEED, zakat, fee)

INVARIANT (Riba Zero):
    For all transactions T in SEED_Ledger:
        T.amount is ExactAmount
        AND T.fee is ExactAmount (or zero)
        AND accumulate(T) produces no interest, no drift

THEOREM (Closure):
    If a, b are RegularNumber:
        a * b is RegularNumber  (exp2+exp2', exp3+exp3', exp5+exp5')
        a / b is RegularNumber iff b divides a exactly
        a + b may NOT be RegularNumber (e.g., 2 + 3 = 5, which is regular;
              but 4 + 9 = 13, which is not)
    => Addition requires promotion to common base or rejection

THEOREM (Zero Drift):
    For all x, y in Z_s (Sippar integers):
        Error(x op y) = 0 for op in {+, -, *, div_exact}
    This holds because all operations are integer, no IEEE 754 involved.
```

---

## 3. Pseudocode

### 3.1 Ledger Auditor (new: `core/treasury/riba_zero_auditor.py`)

```python
class RibaZeroAuditor:
    """Audit the SEED ledger for Riba Zero compliance."""

    def __init__(self, ledger_path: Path):
        self._ledger_path = ledger_path

    def audit(self) -> AuditResult:
        """Walk every transaction and verify exact arithmetic."""
        violations = []
        running_balance = {}  # node_id -> int (exact)

        for tx in self._read_transactions():
            # Check 1: amount must be integer (no float)
            if not isinstance(tx.amount, int):
                violations.append(Violation(
                    tx_id=tx.tx_id,
                    rule="EXACT_AMOUNT",
                    detail=f"amount is {type(tx.amount).__name__}, expected int"
                ))

            # Check 2: no negative balances (no lending/borrowing)
            node_bal = running_balance.get(tx.recipient, 0) + tx.amount
            if node_bal < 0:
                violations.append(Violation(
                    tx_id=tx.tx_id,
                    rule="NO_NEGATIVE_BALANCE",
                    detail=f"balance would be {node_bal}"
                ))
            running_balance[tx.recipient] = node_bal

            # Check 3: zakat deduction is exactly 2.5% (integer floor)
            if tx.tx_type == "mint":
                expected_zakat = tx.gross_amount * 25 // 1000
                if tx.zakat_deducted != expected_zakat:
                    violations.append(Violation(
                        tx_id=tx.tx_id,
                        rule="ZAKAT_EXACT",
                        detail=f"expected {expected_zakat}, got {tx.zakat_deducted}"
                    ))

            # Check 4: no interest accumulation
            if tx.tx_type == "interest":
                violations.append(Violation(
                    tx_id=tx.tx_id,
                    rule="RIBA_ZERO",
                    detail="interest transaction detected"
                ))

        return AuditResult(
            total_transactions=len(list(self._read_transactions())),
            violations=violations,
            riba_zero=len(violations) == 0,
        )
```

### 3.2 Sippar Arithmetic Verifier (Python bridge to Rust)

```python
def verify_sippar_closure(a: tuple[int,int,int], b: tuple[int,int,int]) -> ClosureResult:
    """Verify that multiplication of two regular numbers is regular."""
    # Multiplication: exponents add
    result = (a[0]+b[0], a[1]+b[1], a[2]+b[2])
    value = (2**result[0]) * (3**result[1]) * (5**result[2])
    # Verify reconstruction
    assert is_regular(value), "closure violation"
    return ClosureResult(a=a, b=b, product=result, is_regular=True)

def verify_addition_safety(a_val: int, b_val: int) -> AdditionResult:
    """Addition may produce irregular numbers — detect and flag."""
    total = a_val + b_val
    regular = is_regular(total)
    return AdditionResult(
        a=a_val, b=b_val, sum=total,
        is_regular=regular,
        requires_promotion=not regular,
    )

def is_regular(n: int) -> bool:
    """Check if n is 2,3,5-smooth (Hamming number)."""
    if n <= 0:
        return False
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1
```

---

## 4. TDD Anchors

```python
# tests/core/test_riba_zero.py

def test_sippar_multiplication_closure():
    """RegularNumber * RegularNumber => RegularNumber."""
    # 12 = 2^2 * 3^1, 15 = 3^1 * 5^1 => 180 = 2^2 * 3^2 * 5^1
    result = verify_sippar_closure((2,1,0), (0,1,1))
    assert result.is_regular is True
    assert result.product == (2, 2, 1)

def test_sippar_addition_irregular_detected():
    """4 + 9 = 13 — not regular, flagged."""
    result = verify_addition_safety(4, 9)
    assert result.is_regular is False
    assert result.requires_promotion is True

def test_sippar_addition_regular_passes():
    """2 + 3 = 5 — still regular."""
    result = verify_addition_safety(2, 3)
    assert result.is_regular is True

def test_ledger_no_float_amounts():
    """Every transaction amount must be integer."""
    auditor = RibaZeroAuditor(mock_ledger_path)
    inject_float_transaction(mock_ledger_path, amount=1.5)
    result = auditor.audit()
    assert result.riba_zero is False
    assert any(v.rule == "EXACT_AMOUNT" for v in result.violations)

def test_ledger_no_interest_transactions():
    """Interest transactions are constitutional violations."""
    auditor = RibaZeroAuditor(mock_ledger_path)
    inject_interest_transaction(mock_ledger_path)
    result = auditor.audit()
    assert any(v.rule == "RIBA_ZERO" for v in result.violations)

def test_zakat_exact_deduction():
    """Zakat must be exactly floor(gross * 25 / 1000)."""
    auditor = RibaZeroAuditor(mock_ledger_path)
    # Mint 1000 SEED => zakat = 25
    inject_mint(mock_ledger_path, gross=1000, zakat=25)
    result = auditor.audit()
    assert result.riba_zero is True

def test_zakat_wrong_deduction_caught():
    """Zakat of 24 on 1000 gross => violation."""
    auditor = RibaZeroAuditor(mock_ledger_path)
    inject_mint(mock_ledger_path, gross=1000, zakat=24)
    result = auditor.audit()
    assert any(v.rule == "ZAKAT_EXACT" for v in result.violations)

def test_is_regular_hamming_numbers():
    """Verify Hamming number detection."""
    assert is_regular(1) is True      # 2^0 * 3^0 * 5^0
    assert is_regular(60) is True     # 2^2 * 3 * 5
    assert is_regular(7) is False     # prime, not 2/3/5
    assert is_regular(13) is False
    assert is_regular(1080) is True   # 2^3 * 3^3 * 5
```

---

## 5. Integration Points

| Existing Module | Integration |
|----------------|-------------|
| `bizra-sippar/lib.rs` | Source of truth for RegularNumber; Python auditor verifies ledger |
| `core/proof_engine/seed_ledger.py` | Auditor reads this JSONL ledger |
| `core/urp/constitution.py` | `riba_zero: bool` and `zakat_rate: 0.025` |
| `core/treasury/` | Future: RibaZeroAuditor lives here |
| `bizra-core/islamic_finance` | Rust-side Riba/Zakat/Halal enforcement |
