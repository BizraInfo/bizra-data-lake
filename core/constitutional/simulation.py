from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from typing import Any

from core.constitutional.algorithms import (
    IHSAN_FLOOR,
    compute_gini,
    full_ihsan_check,
    ghazali_equity_factor,
    khaldunian_throttle,
    network_asabiyyah,
    shura_vote,
    verify_event_chain,
)
from core.constitutional.declaration import (
    DECLARATION_BLAKE2B_256,
    create_genesis_event,
    load_declaration,
    verify_covenant_chain,
    verify_declaration_hash,
)
from core.constitutional.fixed_point import FP_ZERO, fp, fp_div, fp_float
from core.constitutional.ticker import process_tick
from core.constitutional.types import ActionReceipt, Event, Proposal, Reflex, WalletState

DEFAULT_START_TIME_MS = 1_741_392_000_000
MILESTONE_DAYS = (1, 30, 90, 180, 365)


@dataclass(frozen=True)
class SimulationConfig:
    num_nodes: int = 100
    days: int = 548
    seed: int = 42
    active_ratio: float = 0.70
    wealthy_ratio: float = 0.05
    medium_ratio: float = 0.15
    milestone_days: tuple[int, ...] = MILESTONE_DAYS
    start_time_ms: int = DEFAULT_START_TIME_MS


@dataclass(frozen=True)
class SimulationMilestone:
    day: int
    network_gini: int
    network_asabiyyah: int
    total_seed: int
    total_bloom: int
    total_actions: int
    intent_rejected: int
    ihsan_rejected: int
    attestations: int
    reflexes_compiled: int
    proposals_passed: int
    khaldun_throttle: int
    newcomer_equity_factor: int
    newcomer_months_to_median: float


@dataclass(frozen=True)
class SimulationReport:
    config: SimulationConfig
    declaration_hash: str
    declaration_verified: bool
    covenant_chain_valid: bool
    covenant_chain_errors: tuple[str, ...]
    event_chain_valid: bool
    event_chain_errors: tuple[str, ...]
    genesis_event_hash: str
    final_event_hash: str
    event_count: int
    reflexes_compiled: int
    proposals_passed: int
    total_seed: int
    total_bloom: int
    total_actions: int
    total_attestations: int
    reciprocal_bonds: int
    network_gini: int
    network_asabiyyah: int
    mean_balance: int
    median_balance: int
    newcomer_equity_factor: int
    wealthy_equity_factor: int
    newcomer_advantage_ratio: float
    newcomer_months_to_median: float
    milestone_reports: tuple[SimulationMilestone, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["milestone_reports"] = [asdict(item) for item in self.milestone_reports]
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


class SovereignNetworkSimulation:
    def __init__(self, config: SimulationConfig):
        if config.num_nodes < 2:
            raise ValueError("num_nodes must be >= 2")
        if config.days < 1:
            raise ValueError("days must be >= 1")
        if not (0 < config.active_ratio <= 1.0):
            raise ValueError("active_ratio must be in (0, 1]")

        self.config = config
        self.rng = random.Random(config.seed)
        self.wallets: list[WalletState] = []
        self.event_log: list[Event] = []
        self.reflex_cache: dict[bytes, Reflex] = {}
        self.proposals: list[Proposal] = []
        self.milestones: list[SimulationMilestone] = []
        self.total_attestations = 0
        self._passed_proposals = 0

        self._genesis()

    def _genesis(self) -> None:
        declaration_text = load_declaration()
        declaration_verified = verify_declaration_hash(declaration_text)
        if not declaration_verified:
            raise ValueError("declaration hash mismatch")

        genesis = create_genesis_event(declaration_text)
        canonical_genesis = json.dumps(
            {
                "id": genesis.event_id,
                "type": genesis.event_type,
                "data": genesis.data,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        genesis.hash = hashlib.blake2b(
            genesis.prev_hash + canonical_genesis,
            digest_size=32,
        ).digest()
        self.event_log.append(genesis)

        wealthy_cutoff = max(1, int(self.config.num_nodes * self.config.wealthy_ratio))
        medium_cutoff = wealthy_cutoff + max(
            1, int(self.config.num_nodes * self.config.medium_ratio)
        )

        for index in range(self.config.num_nodes):
            node_id = self._digest(f"node:{self.config.seed}:{index}")
            if index < wealthy_cutoff:
                balance = fp(self.rng.uniform(500.0, 2000.0))
            elif index < medium_cutoff:
                balance = fp(self.rng.uniform(50.0, 500.0))
            else:
                balance = fp(self.rng.uniform(0.0, 50.0))

            self.wallets.append(
                WalletState(
                    node_id=node_id,
                    seed_balance=balance,
                    bloom_balance=FP_ZERO,
                    last_active=self.config.start_time_ms,
                    total_actions=0,
                    created_at=self.config.start_time_ms,
                )
            )

    def run(self) -> SimulationReport:
        for day in range(1, self.config.days + 1):
            current_time = self.config.start_time_ms + (day * 86_400_000)
            active_wallets = self._sample_active_wallets()
            receipts = self._build_receipts(active_wallets, day, current_time)
            intent_rejected, ihsan_rejected = self._precompute_rejections(receipts)

            if day > 30:
                self._apply_attestations(active_wallets, receipts)
            if day % 30 == 0:
                self._run_governance_round(active_wallets, day, current_time)

            process_tick(
                wallets=self.wallets,
                receipts=receipts,
                proposals=self.proposals,
                event_log=self.event_log,
                reflex_cache=self.reflex_cache,
                current_time=current_time,
                is_zakat_cycle=(day % 365 == 0),
            )
            self._passed_proposals = sum(
                1 for proposal in self.proposals if proposal.status == "passed"
            )

            if day in self.config.milestone_days or day == self.config.days:
                self.milestones.append(
                    self._build_milestone(
                        day=day,
                        intent_rejected=intent_rejected,
                        ihsan_rejected=ihsan_rejected,
                        attestations=self.total_attestations,
                    )
                )

        covenant_ok, covenant_errors = verify_covenant_chain(self.event_log)
        event_ok, event_errors = verify_event_chain(self.event_log)
        balances = [wallet.seed_balance for wallet in self.wallets]
        mean_balance = self._mean_balance(balances)
        median_balance = sorted(balances)[len(balances) // 2]
        newcomer = min(self.wallets, key=lambda wallet: wallet.seed_balance)
        wealthy = max(self.wallets, key=lambda wallet: wallet.seed_balance)
        newcomer_equity = ghazali_equity_factor(newcomer, mean_balance)
        wealthy_equity = ghazali_equity_factor(wealthy, mean_balance)
        ratio = (
            fp_float(fp_div(newcomer_equity, wealthy_equity))
            if wealthy_equity > 0
            else 0.0
        )

        return SimulationReport(
            config=self.config,
            declaration_hash=DECLARATION_BLAKE2B_256,
            declaration_verified=True,
            covenant_chain_valid=covenant_ok,
            covenant_chain_errors=tuple(covenant_errors),
            event_chain_valid=event_ok,
            event_chain_errors=tuple(event_errors),
            genesis_event_hash=self.event_log[0].hash.hex(),
            final_event_hash=self.event_log[-1].hash.hex(),
            event_count=len(self.event_log),
            reflexes_compiled=len(self.reflex_cache),
            proposals_passed=self._passed_proposals,
            total_seed=sum(wallet.seed_balance for wallet in self.wallets),
            total_bloom=sum(wallet.bloom_balance for wallet in self.wallets),
            total_actions=sum(wallet.total_actions for wallet in self.wallets),
            total_attestations=self.total_attestations,
            reciprocal_bonds=sum(
                len(wallet.attestations_given & wallet.attestations_received)
                for wallet in self.wallets
            ),
            network_gini=compute_gini(balances),
            network_asabiyyah=network_asabiyyah(self.wallets),
            mean_balance=mean_balance,
            median_balance=median_balance,
            newcomer_equity_factor=newcomer_equity,
            wealthy_equity_factor=wealthy_equity,
            newcomer_advantage_ratio=ratio,
            newcomer_months_to_median=self._months_to_median(newcomer, median_balance),
            milestone_reports=tuple(self.milestones),
        )

    def _sample_active_wallets(self) -> list[WalletState]:
        count = max(1, int(round(self.config.num_nodes * self.config.active_ratio)))
        return self.rng.sample(self.wallets, count)

    def _build_receipts(
        self, active_wallets: list[WalletState], day: int, current_time: int
    ) -> list[ActionReceipt]:
        receipts: list[ActionReceipt] = []
        action_types = ("contribution", "research", "review", "governance")

        for ordinal, wallet in enumerate(active_wallets):
            quality = self.rng.choices(
                ("high", "medium", "low"),
                weights=(0.85, 0.10, 0.05),
                k=1,
            )[0]
            intent, efficiency, impact, reproducibility = self._quality_scores(quality)
            action_type = action_types[(day + ordinal + self.config.seed) % len(action_types)]
            receipt_id = self._digest(
                f"receipt:{self.config.seed}:{day}:{ordinal}:{wallet.node_id.hex()}"
            )
            metadata_hash = self._digest(
                f"metadata:{action_type}:{quality}:{day}:{wallet.node_id.hex()}"
            )
            oracle_signature = self._digest(
                f"oracle:{self.config.seed}:{day}:{ordinal}"
            ) + self._digest(f"oracle-sig:{day}:{ordinal}")

            receipts.append(
                ActionReceipt(
                    receipt_id=receipt_id,
                    actor_id=wallet.node_id,
                    action_type=action_type,
                    timestamp=current_time,
                    intent_score=intent,
                    efficiency_score=efficiency,
                    impact_score=impact,
                    reproducibility_score=reproducibility,
                    oracle_signature=oracle_signature,
                    metadata_hash=metadata_hash,
                    co_actors=(),
                )
            )

        return receipts

    def _quality_scores(self, quality: str) -> tuple[int, int, int, int]:
        if quality == "high":
            ranges = ((0.92, 1.00), (0.90, 1.00), (0.88, 1.00), (0.90, 1.00))
        elif quality == "medium":
            ranges = ((0.85, 0.95), (0.80, 0.95), (0.80, 0.95), (0.85, 0.95))
        else:
            ranges = ((0.70, 0.90), (0.70, 0.90), (0.70, 0.90), (0.70, 0.90))

        return tuple(fp(self.rng.uniform(low, high)) for low, high in ranges)  # type: ignore[return-value]

    def _precompute_rejections(self, receipts: list[ActionReceipt]) -> tuple[int, int]:
        intent_rejected = 0
        ihsan_rejected = 0
        for receipt in receipts:
            passed, _ = full_ihsan_check(receipt)
            if receipt.intent_score < fp(0.90):
                intent_rejected += 1
            elif not passed:
                ihsan_rejected += 1
        return intent_rejected, ihsan_rejected

    def _apply_attestations(
        self, active_wallets: list[WalletState], receipts: list[ActionReceipt]
    ) -> None:
        if len(active_wallets) < 2 or not receipts:
            return

        attempts = self.rng.randint(5, min(15, len(receipts) + 5))
        receipt_by_actor = {receipt.actor_id: receipt for receipt in receipts}
        active_set = {wallet.node_id: wallet for wallet in active_wallets}

        for _ in range(attempts):
            receipt = self.rng.choice(receipts)
            attestee = active_set.get(receipt.actor_id)
            if attestee is None:
                continue
            attester = self.rng.choice(active_wallets)
            if attester.node_id == attestee.node_id:
                continue
            if attester.ihsan_history:
                avg = sum(attester.ihsan_history[-10:]) // len(attester.ihsan_history[-10:])
                if avg < IHSAN_FLOOR:
                    continue
            if receipt_by_actor.get(attestee.node_id) is None:
                continue
            if attestee.node_id in attester.attestations_given:
                continue
            attester.attestations_given.add(attestee.node_id)
            attestee.attestations_received.add(attester.node_id)
            attester.cooperative_actions += 1
            self.total_attestations += 1

    def _run_governance_round(
        self, active_wallets: list[WalletState], day: int, current_time: int
    ) -> None:
        proposal = Proposal(
            proposal_id=self._digest(f"proposal:{self.config.seed}:{day}"),
            proposer=active_wallets[0].node_id,
            description=f"Simulation proposal day {day}",
            created_at=current_time,
        )
        for wallet in active_wallets:
            wallet.governance_votes += 1
            approve = self.rng.random() >= 0.25
            shura_vote(proposal, wallet, approve)
        self.proposals.append(proposal)

    def _build_milestone(
        self,
        *,
        day: int,
        intent_rejected: int,
        ihsan_rejected: int,
        attestations: int,
    ) -> SimulationMilestone:
        balances = [wallet.seed_balance for wallet in self.wallets]
        mean_balance = self._mean_balance(balances)
        newcomer = min(self.wallets, key=lambda wallet: wallet.seed_balance)
        gini = compute_gini(balances)
        asabiyyah = network_asabiyyah(self.wallets)
        return SimulationMilestone(
            day=day,
            network_gini=gini,
            network_asabiyyah=asabiyyah,
            total_seed=sum(balances),
            total_bloom=sum(wallet.bloom_balance for wallet in self.wallets),
            total_actions=sum(wallet.total_actions for wallet in self.wallets),
            intent_rejected=intent_rejected,
            ihsan_rejected=ihsan_rejected,
            attestations=attestations,
            reflexes_compiled=len(self.reflex_cache),
            proposals_passed=self._passed_proposals,
            khaldun_throttle=khaldunian_throttle(gini, asabiyyah),
            newcomer_equity_factor=ghazali_equity_factor(newcomer, mean_balance),
            newcomer_months_to_median=self._months_to_median(
                newcomer,
                sorted(balances)[len(balances) // 2],
            ),
        )

    def _months_to_median(self, wallet: WalletState, median_balance: int) -> float:
        if wallet.total_actions <= 0 or wallet.seed_balance >= median_balance:
            return 0.0
        avg_mint = wallet.seed_balance // wallet.total_actions
        if avg_mint <= 0:
            return 0.0
        remaining = max(0, median_balance - wallet.seed_balance)
        actions_to_median = remaining / avg_mint
        return actions_to_median / 30.0

    @staticmethod
    def _mean_balance(balances: list[int]) -> int:
        positive = [balance for balance in balances if balance > 0]
        if not positive:
            return FP_ZERO
        return fp_div(sum(positive), fp(len(positive)))

    @staticmethod
    def _digest(value: str) -> bytes:
        return hashlib.blake2b(value.encode("utf-8"), digest_size=32).digest()


def run_simulation(
    num_nodes: int = 100,
    days: int = 548,
    seed: int = 42,
) -> SimulationReport:
    simulation = SovereignNetworkSimulation(
        SimulationConfig(num_nodes=num_nodes, days=days, seed=seed)
    )
    return simulation.run()


def render_simulation_report(report: SimulationReport) -> str:
    lines = [
        "BIZRA Sovereign Network Emulation",
        f"Declaration hash verified: {'yes' if report.declaration_verified else 'no'}",
        f"Covenant chain valid:     {'yes' if report.covenant_chain_valid else 'no'}",
        f"Event chain valid:        {'yes' if report.event_chain_valid else 'no'}",
        "",
        "Final State",
        f"  Nodes:                  {report.config.num_nodes}",
        f"  Days:                   {report.config.days}",
        f"  Total actions:          {report.total_actions}",
        f"  Total SEED:             {fp_float(report.total_seed):,.2f}",
        f"  Total BLOOM:            {fp_float(report.total_bloom):,.2f}",
        f"  Gini:                   {fp_float(report.network_gini):.4f}",
        f"  Asabiyyah:              {fp_float(report.network_asabiyyah):.4f}",
        f"  Reflexes compiled:      {report.reflexes_compiled}",
        f"  Proposals passed:       {report.proposals_passed}",
        f"  Total attestations:     {report.total_attestations}",
        f"  Newcomer multiplier:    {fp_float(report.newcomer_equity_factor):.2f}x",
        f"  Newcomer advantage:     {report.newcomer_advantage_ratio:.2f}x",
        f"  Months to median:       {report.newcomer_months_to_median:.1f}",
        "",
        "Milestones",
    ]
    for milestone in report.milestone_reports:
        lines.append(
            "  Day {day}: gini={gini:.4f} asab={asab:.4f} seed={seed:,.2f} actions={actions} reflexes={reflexes} passed={passed}".format(
                day=milestone.day,
                gini=fp_float(milestone.network_gini),
                asab=fp_float(milestone.network_asabiyyah),
                seed=fp_float(milestone.total_seed),
                actions=milestone.total_actions,
                reflexes=milestone.reflexes_compiled,
                passed=milestone.proposals_passed,
            )
        )
    return "\n".join(lines)
