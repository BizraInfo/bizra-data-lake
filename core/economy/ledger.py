"""Economic Constitution v1.0 ledger and gate primitives.

[ENFORCEMENT: WIRED] This module activates deterministic economic invariants
that were schema-visible in the Semantic Transducer contract: riba detection,
zakat arithmetic, Gini simulation, immutable ledger transitions, and a unified
semantic-then-economic gate. It does not execute real transfers, start daemons,
or claim cryptographic signing is complete.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Final, Protocol
from uuid import UUID, uuid4

from core.dema.semantic_transducer import (
    Claim,
    ConstitutionalPolicy,
    GateDecision,
    GateVerdict,
    fate_gate,
)
from core.integration.constants import CONSTITUTIONAL_GINI_THRESHOLD
from core.proof_engine.canonical import canonical_bytes, hex_digest

CAP_PRECISION: Final[int] = 9
NANOCAPS_PER_CAP: Final[int] = 10**CAP_PRECISION
NISAB_THRESHOLD_NC: Final[int] = 85 * NANOCAPS_PER_CAP
ZAKAT_RATE_BPS: Final[int] = 250
GINI_EPSILON: Final[float] = 1e-12


class RibaPattern(str, Enum):
    """[ENFORCEMENT: WIRED] Riba patterns recognized by the v1 detector."""

    FIXED_INTEREST = "riba.fixed_interest"
    COMPOUNDING = "riba.compounding"
    DISCOUNT_DISTORTION = "riba.discount_distortion"
    LEVERAGE = "riba.leverage"


class TransactionType(str, Enum):
    """[ENFORCEMENT: WIRED] Economic transaction vocabulary."""

    TRANSFER = "economic.transfer"
    ZAKAT = "economic.zakat"
    GIFT = "economic.gift"
    TRADE = "economic.trade"


class IdentityRegistry(Protocol):
    """[ENFORCEMENT: WIRED] Read-only identity lookup protocol."""

    def get(self, node_id: UUID) -> "Identity | None":
        """Return an identity by node UUID."""
        ...

    def public_key(self, node_id: UUID) -> bytes | None:
        """Return a public key by node UUID."""
        ...


@dataclass(frozen=True)
class Identity:
    """[ENFORCEMENT: WIRED] Sovereign identity with Ed25519 public key shape."""

    node_id: UUID
    public_key: bytes
    label: str = ""

    def __post_init__(self) -> None:
        if len(self.public_key) != 32:
            raise ValueError("Ed25519 public key must be 32 bytes")
        if self.public_key == b"\x00" * 32:
            raise ValueError("placeholder public key is not a sovereign identity")


@dataclass(frozen=True)
class LedgerEntry:
    """[ENFORCEMENT: WIRED] Immutable economic transaction record."""

    entry_id: UUID
    timestamp: datetime
    source: Identity
    destination: Identity
    amount_nc: int
    tx_type: TransactionType
    claim_id: UUID
    signature: bytes = b""
    signature_status: str = "LOCAL_UNSIGNED_DEV"

    def __post_init__(self) -> None:
        if self.amount_nc <= 0:
            raise ValueError("amount_nc must be > 0")
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be timezone-aware")
        if not isinstance(self.tx_type, TransactionType):
            raise ValueError("tx_type must be a TransactionType")
        if len(self.signature) not in (0, 64):
            raise ValueError("signature must be empty or 64 bytes")
        if self.signature == b"\x00" * 64:
            raise ValueError("placeholder signatures are forbidden")
        expected_status = "SIGNED" if self.signature else "LOCAL_UNSIGNED_DEV"
        if self.signature_status != expected_status:
            raise ValueError("signature_status does not match signature material")

    def canonical_payload(self) -> Mapping[str, object]:
        """[ENFORCEMENT: WIRED] Return deterministic payload for hashing."""
        return {
            "entry_id": str(self.entry_id),
            "timestamp": self.timestamp.astimezone(timezone.utc).isoformat(),
            "source": str(self.source.node_id),
            "destination": str(self.destination.node_id),
            "amount_nc": self.amount_nc,
            "tx_type": self.tx_type.value,
            "claim_id": str(self.claim_id),
            "signature_status": self.signature_status,
        }

    def entry_hash(self) -> str:
        """[ENFORCEMENT: WIRED] Return BLAKE3 hash of canonical entry payload."""
        return hex_digest(canonical_bytes(self.canonical_payload()))


@dataclass(frozen=True)
class LedgerState:
    """[ENFORCEMENT: WIRED] Functional immutable ledger snapshot."""

    __hash__ = None

    entries: tuple[LedgerEntry, ...] = field(default_factory=tuple)
    balances: Mapping[str, int] = field(default_factory=dict)
    total_issued_nc: int = 0

    def __post_init__(self) -> None:
        frozen_balances = _freeze_balances(self.balances)
        object.__setattr__(self, "balances", frozen_balances)
        object.__setattr__(self, "entries", tuple(self.entries))
        if self.total_issued_nc < 0:
            raise ValueError("total_issued_nc must be non-negative")
        if (
            self.total_issued_nc
            and sum(frozen_balances.values()) != self.total_issued_nc
        ):
            raise ValueError("balances must conserve total_issued_nc")

    @classmethod
    def genesis(cls, allocations: Mapping[str, int]) -> "LedgerState":
        """[ENFORCEMENT: WIRED] Build a conserved genesis state."""
        balances = _freeze_balances(allocations)
        return cls(balances=balances, total_issued_nc=sum(balances.values()))

    def apply(self, entry: LedgerEntry) -> "LedgerState":
        """[ENFORCEMENT: WIRED] Return a new state after applying an entry."""
        source_id = str(entry.source.node_id)
        destination_id = str(entry.destination.node_id)
        current = dict(self.balances)
        if current.get(source_id, 0) < entry.amount_nc:
            raise ValueError(f"insufficient balance: {source_id}")
        current[source_id] = current[source_id] - entry.amount_nc
        current[destination_id] = current.get(destination_id, 0) + entry.amount_nc
        return LedgerState(
            entries=self.entries + (entry,),
            balances=current,
            total_issued_nc=self.total_issued_nc,
        )

    def balance(self, identity: Identity) -> int:
        """[ENFORCEMENT: WIRED] Return an identity balance in nanocaps."""
        return self.balances.get(str(identity.node_id), 0)

    def has(self, node_id: str) -> bool:
        """[ENFORCEMENT: WIRED] Return whether a node has a ledger account."""
        return node_id in self.balances


@dataclass(frozen=True)
class ZakatAssessment:
    """[ENFORCEMENT: WIRED] Deterministic zakat obligation assessment."""

    holder: Identity
    balance_nc: int
    nisab_nc: int
    rate_bps: int
    eligible: bool
    obligation_nc: int

    def __post_init__(self) -> None:
        expected = self.balance_nc * self.rate_bps // 10_000 if self.eligible else 0
        if self.obligation_nc != expected:
            raise ValueError("zakat arithmetic error")


@dataclass(frozen=True)
class EconomicPolicyView:
    """[ENFORCEMENT: WIRED] Resolved economic knobs from a semantic policy."""

    version: str
    zann_zero: bool
    riba_zero: bool
    gini_threshold: float


class RibaDetector:
    """[OPTIMIZATION: PARTIAL] Deterministic text-pattern riba detector."""

    _FIXED_INTEREST_PATTERNS: Final[tuple[re.Pattern[str], ...]] = (
        re.compile(r"\binterest\s+rate\b"),
        re.compile(r"\bapr\b"),
        re.compile(r"\bapy\b"),
        re.compile(r"\b\d+(?:\.\d+)?\s*%\s+interest\b"),
    )

    @classmethod
    def scan(cls, evidence: Mapping[str, object]) -> frozenset[RibaPattern]:
        """[ENFORCEMENT: WIRED] Return recognized riba patterns in evidence."""
        text = _evidence_text(evidence)
        found: set[RibaPattern] = set()
        if any(pattern.search(text) for pattern in cls._FIXED_INTEREST_PATTERNS):
            found.add(RibaPattern.FIXED_INTEREST)
        if ("compounding" in text and "interest" in text) or (
            "late fee" in text and ("interest" in text or "compounding" in text)
        ):
            found.add(RibaPattern.COMPOUNDING)
        if "discount" in text and ("early" in text or "now" in text):
            found.add(RibaPattern.DISCOUNT_DISTORTION)
        if "leverage" in text or "leveraged" in text or "margin" in text:
            found.add(RibaPattern.LEVERAGE)
        return frozenset(found)

    @classmethod
    def is_clean(cls, evidence: Mapping[str, object]) -> bool:
        """[ENFORCEMENT: WIRED] Return whether evidence is riba-clean."""
        return not cls.scan(evidence)


class InMemoryIdentityRegistry:
    """[ENFORCEMENT: WIRED] Read-only in-memory identity registry for tests/tools."""

    def __init__(self, identities: tuple[Identity, ...] | None = None) -> None:
        self._store = MappingProxyType(
            {identity.node_id: identity for identity in (identities or ())}
        )

    def get(self, node_id: UUID) -> Identity | None:
        """Return an identity by node UUID."""
        return self._store.get(node_id)

    def public_key(self, node_id: UUID) -> bytes | None:
        """Return a public key by node UUID."""
        identity = self.get(node_id)
        return identity.public_key if identity is not None else None


def gini(balances: Mapping[str, int]) -> float:
    """[ENFORCEMENT: WIRED] Compute non-negative balance Gini coefficient."""
    values = [max(0, int(balance)) for balance in balances.values()]
    count = len(values)
    if count == 0:
        return 0.0
    total = sum(values)
    if total < GINI_EPSILON:
        return 0.0
    weighted_sum = 0
    for index, value in enumerate(sorted(values), start=1):
        weighted_sum += (2 * index - count - 1) * value
    return weighted_sum / (count * total)


def assess_zakat(holder: Identity, ledger: LedgerState) -> ZakatAssessment:
    """[ENFORCEMENT: WIRED] Assess zakat using integer basis-point math."""
    balance = ledger.balance(holder)
    eligible = balance >= NISAB_THRESHOLD_NC
    obligation = balance * ZAKAT_RATE_BPS // 10_000 if eligible else 0
    return ZakatAssessment(
        holder=holder,
        balance_nc=balance,
        nisab_nc=NISAB_THRESHOLD_NC,
        rate_bps=ZAKAT_RATE_BPS,
        eligible=eligible,
        obligation_nc=obligation,
    )


def simulate_gini(
    ledger: LedgerState, source_id: str, destination_id: str, amount: int
) -> float:
    """[ENFORCEMENT: WIRED] Simulate post-transfer Gini without mutation."""
    if amount <= 0:
        raise ValueError("amount must be > 0")
    if not ledger.has(source_id):
        raise ValueError(f"source not in ledger: {source_id}")
    if ledger.balances[source_id] < amount:
        raise ValueError(f"insufficient funds: {source_id}")
    simulated = dict(ledger.balances)
    simulated[source_id] -= amount
    simulated[destination_id] = simulated.get(destination_id, 0) + amount
    return gini(simulated)


def enforce(
    claim: Claim,
    ledger: LedgerState,
    policy: ConstitutionalPolicy,
) -> GateDecision:
    """[ENFORCEMENT: WIRED] Apply economic invariants to a validated claim."""
    economic_policy = _resolve_policy(policy)
    if economic_policy.zann_zero and claim.evidence_weight <= 0.0:
        return GateDecision(
            verdict=GateVerdict.ESCALATE,
            rule_id="economic.zann_zero",
            evidence_weight=claim.evidence_weight,
            gate_version=economic_policy.version,
            reason_code="ECONOMIC_EVIDENCE_INSUFFICIENT",
        )

    patterns = (
        RibaDetector.scan(claim.evidence) if economic_policy.riba_zero else frozenset()
    )
    if patterns:
        return GateDecision(
            verdict=GateVerdict.REJECT,
            rule_id="economic.riba_detected",
            evidence_weight=claim.evidence_weight,
            gate_version=economic_policy.version,
            reason_code="RIBA:"
            + ",".join(sorted(pattern.value for pattern in patterns)),
        )

    try:
        transfer = _extract_transfer(claim.evidence)
    except ValueError as exc:
        return GateDecision(
            verdict=GateVerdict.ESCALATE,
            rule_id="economic.transfer_incomplete",
            evidence_weight=claim.evidence_weight,
            gate_version=economic_policy.version,
            reason_code=str(exc)[:120],
        )
    if transfer is None:
        if _looks_economic(claim.evidence):
            return GateDecision(
                verdict=GateVerdict.ESCALATE,
                rule_id="economic.transfer_incomplete",
                evidence_weight=claim.evidence_weight,
                gate_version=economic_policy.version,
                reason_code="ECONOMIC_TRANSFER_FIELDS_REQUIRED",
            )
        return _permit(claim, economic_policy)

    tx_type, source_id, destination_id, amount = transfer
    current_gini = gini(ledger.balances)
    try:
        post_gini = simulate_gini(ledger, source_id, destination_id, amount)
    except ValueError as exc:
        return GateDecision(
            verdict=GateVerdict.ESCALATE,
            rule_id="economic.gini_sim_error",
            evidence_weight=current_gini,
            gate_version=economic_policy.version,
            reason_code=str(exc)[:120],
        )

    if _gini_worsens_above_policy(current_gini, post_gini, economic_policy):
        return GateDecision(
            verdict=GateVerdict.ESCALATE,
            rule_id="economic.gini_worsening",
            evidence_weight=current_gini,
            gate_version=economic_policy.version,
            reason_code=(
                f"Gini {current_gini:.4f}->{post_gini:.4f} "
                f"(threshold {economic_policy.gini_threshold})"
            ),
        )
    if (
        tx_type not in _redistributive_types()
        and current_gini > economic_policy.gini_threshold
        and post_gini > current_gini + GINI_EPSILON
    ):
        return GateDecision(
            verdict=GateVerdict.ESCALATE,
            rule_id="economic.gini_ceiling",
            evidence_weight=current_gini,
            gate_version=economic_policy.version,
            reason_code=f"Gini {current_gini:.4f} > {economic_policy.gini_threshold}",
        )
    return _permit(claim, economic_policy)


def economic_fate_gate(
    claim: Claim,
    policy: ConstitutionalPolicy,
    ledger: LedgerState | None = None,
) -> GateDecision:
    """[ENFORCEMENT: WIRED] Gate semantic admissibility before economics."""
    decision = fate_gate(claim, policy)
    if decision.verdict is not GateVerdict.PERMIT or ledger is None:
        return decision
    return enforce(claim, ledger, policy)


def build_entry(claim: Claim, registry: IdentityRegistry) -> LedgerEntry:
    """[ENFORCEMENT: WIRED] Build an unsigned ledger entry from a claim."""
    source_id = _required_uuid(claim.evidence, "source_node_id")
    destination_id = _required_uuid(claim.evidence, "dest_node_id")
    source = registry.get(source_id)
    destination = registry.get(destination_id)
    if source is None or destination is None:
        raise ValueError("identity not found in registry")
    amount = _required_positive_int(claim.evidence, "amount_nc")
    tx_type = _parse_transaction_type(
        str(
            claim.evidence.get(
                "tx_type",
                claim.evidence.get("transaction_type", TransactionType.TRANSFER.value),
            )
        )
    )
    return LedgerEntry(
        entry_id=uuid4(),
        timestamp=datetime.now(timezone.utc),
        source=source,
        destination=destination,
        amount_nc=amount,
        tx_type=tx_type,
        claim_id=claim.mission_id,
    )


def _freeze_balances(balances: Mapping[str, int]) -> Mapping[str, int]:
    frozen: dict[str, int] = {}
    for node_id, balance in balances.items():
        normalized = int(balance)
        if normalized < 0:
            raise ValueError("ledger balances must be non-negative")
        frozen[str(node_id)] = normalized
    return MappingProxyType(frozen)


def _resolve_policy(policy: ConstitutionalPolicy) -> EconomicPolicyView:
    return EconomicPolicyView(
        version=policy.version,
        zann_zero=bool(getattr(policy, "zann_zero", True)),
        riba_zero=bool(getattr(policy, "riba_zero", True)),
        gini_threshold=float(
            getattr(policy, "gini_threshold", CONSTITUTIONAL_GINI_THRESHOLD)
        ),
    )


def _permit(claim: Claim, policy: EconomicPolicyView) -> GateDecision:
    return GateDecision(
        verdict=GateVerdict.PERMIT,
        rule_id="economic.permit",
        evidence_weight=claim.evidence_weight,
        gate_version=policy.version,
        reason_code="ECONOMIC_INVARIANTS_SATISFIED",
    )


def _gini_worsens_above_policy(
    current_gini: float,
    post_gini: float,
    policy: EconomicPolicyView,
) -> bool:
    return post_gini > policy.gini_threshold and post_gini > current_gini + GINI_EPSILON


def _redistributive_types() -> frozenset[TransactionType]:
    return frozenset({TransactionType.ZAKAT, TransactionType.GIFT})


def _extract_transfer(
    evidence: Mapping[str, object],
) -> tuple[TransactionType, str, str, int] | None:
    source = evidence.get("source_node_id")
    destination = evidence.get("dest_node_id")
    amount = evidence.get("amount_nc")
    if source is None and destination is None and amount is None:
        return None
    if source is None or destination is None or amount is None:
        raise ValueError("source_node_id, dest_node_id, and amount_nc are required")
    tx_type = _parse_transaction_type(
        str(evidence.get("transaction_type", evidence.get("tx_type", "")))
    )
    return (tx_type, str(source), str(destination), _coerce_positive_int(amount))


def _parse_transaction_type(value: str) -> TransactionType:
    if not value:
        return TransactionType.TRANSFER
    try:
        return TransactionType(value)
    except ValueError:
        return TransactionType.TRANSFER


def _required_uuid(evidence: Mapping[str, object], key: str) -> UUID:
    raw = evidence.get(key)
    if raw is None:
        raise ValueError(f"{key} required")
    try:
        return UUID(str(raw))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {key}: {exc}") from exc


def _required_positive_int(evidence: Mapping[str, object], key: str) -> int:
    raw = evidence.get(key)
    if raw is None:
        raise ValueError(f"{key} required")
    return _coerce_positive_int(raw)


def _coerce_positive_int(value: object) -> int:
    try:
        amount = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid amount_nc: {exc}") from exc
    if amount <= 0:
        raise ValueError("amount_nc must be > 0")
    return amount


def _looks_economic(evidence: Mapping[str, object]) -> bool:
    economic_keys = {
        "amount_nc",
        "dest_node_id",
        "source_node_id",
        "transaction_type",
        "tx_type",
    }
    return any(key in evidence for key in economic_keys)


def _evidence_text(value: object) -> str:
    if isinstance(value, Mapping):
        return " ".join(_evidence_text(item) for item in value.values()).lower()
    if isinstance(value, (list, tuple, set, frozenset)):
        return " ".join(_evidence_text(item) for item in value).lower()
    return _normalize_economic_text(str(value))


def _normalize_economic_text(value: str) -> str:
    translated = value.translate(str.maketrans({"ı": "i", "İ": "I"}))
    decomposed = unicodedata.normalize("NFKD", translated)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return stripped.casefold()


__all__ = [
    "CAP_PRECISION",
    "NANOCAPS_PER_CAP",
    "NISAB_THRESHOLD_NC",
    "ZAKAT_RATE_BPS",
    "EconomicPolicyView",
    "Identity",
    "IdentityRegistry",
    "InMemoryIdentityRegistry",
    "LedgerEntry",
    "LedgerState",
    "RibaDetector",
    "RibaPattern",
    "TransactionType",
    "ZakatAssessment",
    "assess_zakat",
    "build_entry",
    "economic_fate_gate",
    "enforce",
    "gini",
    "simulate_gini",
]
