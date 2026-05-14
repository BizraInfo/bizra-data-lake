"""Economic Constitution v1.0 contract tests."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

import pytest

from core.dema.semantic_transducer import (
    ConstitutionalPolicy,
    GateVerdict,
    IntentType,
    RawParsedClaim,
    ResourceScope,
    ResourceType,
    StepDescriptor,
    validate_raw_claim,
)
from core.economy.ledger import (
    NISAB_THRESHOLD_NC,
    ZAKAT_RATE_BPS,
    Identity,
    InMemoryIdentityRegistry,
    LedgerEntry,
    LedgerState,
    MockRegistry,
    RibaDetector,
    RibaPattern,
    TransactionType,
    ZakatAssessment,
    assess_zakat,
    build_entry,
    economic_fate_gate,
    enforce,
    gini,
    simulate_gini,
)
from core.integration.constants import CONSTITUTIONAL_GINI_THRESHOLD


@pytest.fixture
def amina() -> Identity:
    return Identity(node_id=uuid4(), public_key=b"\x01" * 32, label="Amina")


@pytest.fixture
def youssef() -> Identity:
    return Identity(node_id=uuid4(), public_key=b"\x02" * 32, label="Youssef")


@pytest.fixture
def registry(amina: Identity, youssef: Identity) -> InMemoryIdentityRegistry:
    return InMemoryIdentityRegistry((amina, youssef))


@pytest.fixture
def policy() -> ConstitutionalPolicy:
    return ConstitutionalPolicy(
        version="economic-v1.0",
        ihsan_floor=0.9,
        gini_threshold=CONSTITUTIONAL_GINI_THRESHOLD,
        riba_zero=True,
        zann_zero=True,
    )


def _claim(
    evidence: dict[str, object],
    *,
    floor_ready: bool = True,
):
    payload = {
        "purpose": "economic invariant test",
        "receipt_ref": "local-dev",
        "operator_request": "economic transfer",
    }
    payload.update(evidence)
    if floor_ready:
        payload.setdefault("policy_ref", "economic-v1")
        payload.setdefault("context_ref", "unit-test")
    raw = RawParsedClaim(
        intent_type=IntentType.ECONOMIC_TRANSFER.value,
        evidence=payload,
        proposed_steps=(
            StepDescriptor(
                tool_id="ledger.preview",
                resource_type=ResourceType.RECEIPT_WRITE,
            ),
        ),
        requested_scope=ResourceScope.of(ResourceType.RECEIPT_WRITE),
    )
    return validate_raw_claim(raw, mission_id=uuid4(), parser_id="system.test")


def _entry(
    source: Identity,
    destination: Identity,
    amount_nc: int,
    tx_type: TransactionType = TransactionType.TRANSFER,
) -> LedgerEntry:
    return LedgerEntry(
        entry_id=uuid4(),
        timestamp=datetime.now(timezone.utc),
        source=source,
        destination=destination,
        amount_nc=amount_nc,
        tx_type=tx_type,
        claim_id=uuid4(),
        signature=b"\x03" * 64,
        signature_status="SIGNED",
    )


class TestGini:
    def test_equal_distribution_is_zero(self) -> None:
        assert gini({"a": 100, "b": 100}) == 0.0

    def test_extreme_two_holder_distribution(self) -> None:
        assert gini({"rich": 1000, "poor": 0}) == pytest.approx(0.5)

    def test_single_holder_is_zero(self) -> None:
        assert gini({"a": 500}) == 0.0

    def test_empty_distribution_is_zero(self) -> None:
        assert gini({}) == 0.0

    def test_many_zero_balances_approach_extreme_inequality(self) -> None:
        balances = {"rich": 1_000_000, **{f"poor_{i}": 0 for i in range(100)}}

        assert gini(balances) > 0.95

    def test_negative_balances_are_treated_as_zero_for_gini(self) -> None:
        assert gini({"creditor": 1000, "debtor": -500}) == gini(
            {"creditor": 1000, "debtor": 0}
        )


class TestZakat:
    def test_below_nisab_has_no_obligation(self, amina: Identity) -> None:
        ledger = LedgerState.genesis({str(amina.node_id): NISAB_THRESHOLD_NC - 1})

        assessment = assess_zakat(amina, ledger)

        assert assessment.eligible is False
        assert assessment.obligation_nc == 0

    def test_exact_integer_basis_point_obligation(self, amina: Identity) -> None:
        balance = NISAB_THRESHOLD_NC * 2
        ledger = LedgerState.genesis({str(amina.node_id): balance})

        assessment = assess_zakat(amina, ledger)

        assert assessment.eligible is True
        assert assessment.obligation_nc == balance * ZAKAT_RATE_BPS // 10_000

    def test_zakat_assessment_rejects_inconsistent_arithmetic(
        self, amina: Identity
    ) -> None:
        with pytest.raises(ValueError, match="zakat arithmetic error"):
            ZakatAssessment(amina, 1000, 100, 250, True, 999)


class TestRiba:
    def test_interest_rate_is_detected(self) -> None:
        assert RibaPattern.FIXED_INTEREST in RibaDetector.scan(
            {"desc": "loan at 5% interest rate"}
        )

    def test_late_fee_alone_does_not_trigger_riba(self) -> None:
        assert RibaDetector.scan({"desc": "$10 late fee"}) == frozenset()

    def test_late_fee_with_compounding_interest_triggers(self) -> None:
        assert RibaPattern.COMPOUNDING in RibaDetector.scan(
            {"desc": "late fee with compounding interest"}
        )

    def test_charity_statement_is_clean(self) -> None:
        assert RibaDetector.is_clean({"desc": "gift of 50 CAP, no interest charged"})

    def test_leverage_is_detected(self) -> None:
        assert RibaPattern.LEVERAGE in RibaDetector.scan(
            {"desc": "leveraged margin structure"}
        )

    def test_unicode_interest_lookalike_is_normalized(self) -> None:
        assert RibaPattern.FIXED_INTEREST in RibaDetector.scan(
            {"desc": "loan at 5% ınterest rate"}
        )


class TestLedger:
    def test_functional_update_preserves_previous_state(
        self, amina: Identity, youssef: Identity
    ) -> None:
        state = LedgerState.genesis({str(amina.node_id): 500})
        updated = state.apply(_entry(amina, youssef, 100))

        assert state.balance(amina) == 500
        assert updated.balance(amina) == 400
        assert updated.balance(youssef) == 100

    def test_insufficient_funds_raise(self, amina: Identity, youssef: Identity) -> None:
        state = LedgerState.genesis({str(amina.node_id): 50})

        with pytest.raises(ValueError, match="insufficient balance"):
            state.apply(_entry(amina, youssef, 100))

    def test_total_issued_is_conserved(
        self, amina: Identity, youssef: Identity
    ) -> None:
        state = LedgerState.genesis(
            {str(amina.node_id): 500, str(youssef.node_id): 500}
        )

        for _ in range(10):
            state = state.apply(_entry(amina, youssef, 10))

        assert sum(state.balances.values()) == 1000
        assert state.total_issued_nc == 1000

    def test_balances_are_immutable(self, amina: Identity) -> None:
        state = LedgerState.genesis({str(amina.node_id): 100})

        with pytest.raises(TypeError):
            state.balances[str(amina.node_id)] = 0  # type: ignore[index]

    def test_self_transfer_preserves_balance(self, amina: Identity) -> None:
        state = LedgerState.genesis({str(amina.node_id): 100})

        updated = state.apply(_entry(amina, amina, 30))

        assert updated.balance(amina) == 100
        assert updated.total_issued_nc == 100

    def test_placeholder_identity_and_signature_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="Placeholder public key"):
            Identity(node_id=uuid4(), public_key=b"\x00" * 32)

        good = Identity(node_id=uuid4(), public_key=b"\x01" * 32)
        with pytest.raises(ValueError, match="Placeholder signatures"):
            LedgerEntry(
                entry_id=uuid4(),
                timestamp=datetime.now(timezone.utc),
                source=good,
                destination=good,
                amount_nc=1,
                tx_type=TransactionType.TRANSFER,
                claim_id=uuid4(),
                signature=b"\x00" * 64,
                signature_status="SIGNED",
            )

    def test_entry_hash_is_deterministic(
        self, amina: Identity, youssef: Identity
    ) -> None:
        entry = _entry(amina, youssef, 100)

        assert entry.entry_hash() == entry.entry_hash()
        assert len(entry.entry_hash()) == 64


class TestSimulateGini:
    def test_transfer_from_rich_to_poor_reduces_inequality(
        self, amina: Identity, youssef: Identity
    ) -> None:
        state = LedgerState.genesis({str(amina.node_id): 1000, str(youssef.node_id): 0})

        post = simulate_gini(state, str(amina.node_id), str(youssef.node_id), 200)

        assert post < gini(state.balances)

    def test_unknown_source_raises(self) -> None:
        state = LedgerState.genesis({"amina": 100})

        with pytest.raises(ValueError, match="source not in ledger"):
            simulate_gini(state, "ghost", "amina", 10)

    def test_unknown_destination_raises(self, amina: Identity) -> None:
        state = LedgerState.genesis({str(amina.node_id): 100})

        with pytest.raises(ValueError, match="destination not in ledger"):
            simulate_gini(state, str(amina.node_id), str(uuid4()), 25)

    def test_insufficient_source_funds_raise(
        self, amina: Identity, youssef: Identity
    ) -> None:
        state = LedgerState.genesis({str(amina.node_id): 50, str(youssef.node_id): 0})

        with pytest.raises(ValueError, match="insufficient funds"):
            simulate_gini(state, str(amina.node_id), str(youssef.node_id), 999)


class TestGate:
    def test_riba_claim_is_rejected(
        self, policy: ConstitutionalPolicy, amina: Identity, youssef: Identity
    ) -> None:
        claim = _claim(
            {
                "desc": "loan at 5% interest rate",
                "source_node_id": str(amina.node_id),
                "dest_node_id": str(youssef.node_id),
                "amount_nc": 100,
            }
        )
        ledger = LedgerState.genesis({str(amina.node_id): 500, str(youssef.node_id): 0})

        decision = enforce(claim, ledger, policy)

        assert decision.verdict is GateVerdict.REJECT
        assert decision.rule_id == "economic.riba_detected"

    def test_gini_worsening_above_threshold_escalates(
        self, policy: ConstitutionalPolicy, amina: Identity, youssef: Identity
    ) -> None:
        ledger = LedgerState.genesis(
            {str(amina.node_id): 1000, str(youssef.node_id): 0}
        )
        claim = _claim(
            {
                "transaction_type": TransactionType.TRANSFER.value,
                "source_node_id": str(youssef.node_id),
                "dest_node_id": str(amina.node_id),
                "amount_nc": 1,
            }
        )

        decision = enforce(claim, ledger, policy)

        assert decision.verdict is GateVerdict.ESCALATE
        assert decision.rule_id in {
            "economic.gini_sim_error",
            "economic.gini_worsening",
        }

    def test_zakat_like_redistribution_is_permitted(
        self, policy: ConstitutionalPolicy, amina: Identity, youssef: Identity
    ) -> None:
        ledger = LedgerState.genesis(
            {str(amina.node_id): 1000, str(youssef.node_id): 0}
        )
        claim = _claim(
            {
                "transaction_type": TransactionType.ZAKAT.value,
                "source_node_id": str(amina.node_id),
                "dest_node_id": str(youssef.node_id),
                "amount_nc": 250,
            }
        )

        decision = enforce(claim, ledger, policy)

        assert decision.verdict is GateVerdict.PERMIT

    def test_incomplete_economic_transfer_escalates(
        self, policy: ConstitutionalPolicy, amina: Identity
    ) -> None:
        claim = _claim({"amount_nc": 100})
        ledger = LedgerState.genesis({str(amina.node_id): 1000})

        decision = enforce(claim, ledger, policy)

        assert decision.verdict is GateVerdict.ESCALATE
        assert decision.rule_id == "economic.transfer_incomplete"

    def test_unified_gate_preserves_semantic_failure(
        self, policy: ConstitutionalPolicy, amina: Identity
    ) -> None:
        claim = _claim({"purpose": "thin"}, floor_ready=False)
        ledger = LedgerState.genesis({str(amina.node_id): 1000})

        decision = economic_fate_gate(claim, policy, ledger)

        assert decision.verdict is GateVerdict.ESCALATE
        assert decision.rule_id == "ihsan.floor"

    def test_unified_gate_permits_clean_balanced_transfer(
        self, policy: ConstitutionalPolicy, amina: Identity, youssef: Identity
    ) -> None:
        ledger = LedgerState.genesis(
            {str(amina.node_id): 500, str(youssef.node_id): 500}
        )
        claim = _claim(
            {
                "transaction_type": TransactionType.TRANSFER.value,
                "source_node_id": str(amina.node_id),
                "dest_node_id": str(youssef.node_id),
                "amount_nc": 100,
            }
        )

        decision = economic_fate_gate(claim, policy, ledger)

        assert decision.verdict is GateVerdict.PERMIT


class TestBuildEntry:
    def test_build_entry_from_claim(
        self,
        amina: Identity,
        youssef: Identity,
        registry: InMemoryIdentityRegistry,
    ) -> None:
        claim = _claim(
            {
                "source_node_id": str(amina.node_id),
                "dest_node_id": str(youssef.node_id),
                "amount_nc": 50,
                "tx_type": TransactionType.GIFT.value,
            }
        )

        entry = build_entry(claim, registry)

        assert entry.source.node_id == amina.node_id
        assert entry.destination.node_id == youssef.node_id
        assert entry.amount_nc == 50
        assert entry.tx_type is TransactionType.GIFT
        assert entry.signature == b""
        assert entry.signature_status == "LOCAL_UNSIGNED_DEV"

    def test_missing_identity_is_rejected(
        self, amina: Identity, registry: InMemoryIdentityRegistry
    ) -> None:
        claim = _claim(
            {
                "source_node_id": str(amina.node_id),
                "dest_node_id": str(uuid4()),
                "amount_nc": 50,
            }
        )

        with pytest.raises(ValueError, match="identity not found"):
            build_entry(claim, registry)

    def test_invalid_amount_is_rejected(
        self,
        amina: Identity,
        youssef: Identity,
        registry: InMemoryIdentityRegistry,
    ) -> None:
        claim = _claim(
            {
                "source_node_id": str(amina.node_id),
                "dest_node_id": str(youssef.node_id),
                "amount_nc": 0,
            }
        )

        with pytest.raises(ValueError, match="amount_nc must be > 0"):
            build_entry(claim, registry)

    def test_mock_registry_alias_matches_requested_contract(
        self, amina: Identity, youssef: Identity
    ) -> None:
        registry = MockRegistry((amina, youssef))

        assert registry.get(amina.node_id) == amina
        assert registry.public_key(youssef.node_id) == youssef.public_key
