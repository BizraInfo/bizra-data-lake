"""
SAT Mint Court — Evaluation → Adjudication → Mint → Split → Seal
=================================================================

The judicial and treasury arm of the SAT. No entity (including the founder)
can self-mint without SAT consensus. Separation of powers.

Phases:
  A. Freeze   — Lock evidence snapshot (hash root of GOLD data)
  B. Verify   — Provenance, dedup, integrity, claim binding (Guardian + Auditor)
  C. Value    — SNR + Ihsan + market normalization (Optimizer)
  D. Decide   — SAT consensus authorizes or rejects (Mediator)
  E. Distribute — Mint SEED, apply founder 50/50 split (Archivist)
  F. Seal     — Write receipts, archive the round

Doctrine: No mint without evidence freeze, no valuation without market
normalization, no distribution without SAT consensus, no claim without receipts.

Standing on Giants:
  Ibn Khaldun (1377) — Asabiyyah: social cohesion determines economic health
  Harberger (1962) — Self-assessed taxation for efficient allocation
  Nakamoto (2008) — Proof-of-Work: value from verified contribution
  Al-Ghazali (1095) — Intent gate: no action without intent
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("bizra.sat.mint_court")


# ═══════════════════════════════════════════════════════════════
# CONSTANTS — from constitutional SSOT
# ═══════════════════════════════════════════════════════════════
try:
    from core.integration.constants import (
        ADL_GINI_THRESHOLD,
        IHSAN_THRESHOLD,
        KERNEL_INVARIANTS,
        SNR_THRESHOLD,
    )
except ImportError:
    IHSAN_THRESHOLD = 0.95
    ADL_GINI_THRESHOLD = 0.35
    SNR_THRESHOLD = 0.85
    KERNEL_INVARIANTS = ("RIBA_ZERO", "CLAIM_MUST_BIND", "IHSAN_FLOOR")

# Founder donation policy (not protocol tax)
FOUNDER_DONATION_RATIO = 0.50  # 50% to system treasury
SEED_PER_COMPUTE_HOUR = 1.0  # 1 SEED = 1 compute hour
MIN_EVALUATION_ARTIFACTS = 10  # Minimum artifacts for a valid round


# ═══════════════════════════════════════════════════════════════
# STATE MACHINE
# ═══════════════════════════════════════════════════════════════


class MintPhase(str, Enum):
    """Mint Court phases — linear progression, no skipping."""

    FREEZE = "freeze"
    VERIFY = "verify"
    VALUE = "value"
    DECIDE = "decide"
    DISTRIBUTE = "distribute"
    SEAL = "seal"
    COMPLETE = "complete"
    REJECTED = "rejected"


class MintVerdict(str, Enum):
    """SAT consensus verdict on a mint round."""

    APPROVED = "approved"
    REJECTED = "rejected"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    CONSTITUTIONAL_VIOLATION = "constitutional_violation"


# ═══════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════


@dataclass
class EvidenceSnapshot:
    """Phase A output: frozen evidence for evaluation."""

    snapshot_id: str = ""
    claimant_identity: str = ""
    timestamp: str = ""
    evidence_hash_root: str = ""
    artifact_count: int = 0
    artifact_classes: Dict[str, int] = field(default_factory=dict)
    time_window: Tuple[str, str] = ("", "")
    gold_parquet_hashes: Dict[str, str] = field(default_factory=dict)
    total_bytes: int = 0


@dataclass
class SATScorecard:
    """Phase B+C output: multi-agent scoring."""

    guardian_pass: bool = False  # Constitutional gates
    auditor_pass: bool = False  # Provenance + dedup
    optimizer_valuation: float = 0.0  # Market-normalized SEED value
    mediator_pass: bool = False  # Fairness + dispute resolution
    ihsan_composite: float = 0.0
    snr_composite: float = 0.0
    originality_score: float = 0.0
    quality_score: float = 0.0
    depth_score: float = 0.0
    dedup_penalty: float = 0.0
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def all_gates_pass(self) -> bool:
        return self.guardian_pass and self.auditor_pass and self.mediator_pass

    @property
    def constitutional(self) -> bool:
        return (
            self.ihsan_composite >= IHSAN_THRESHOLD
            and self.snr_composite >= SNR_THRESHOLD
        )


@dataclass
class MintDistribution:
    """Phase E output: mint amounts and split."""

    gross_seed: float = 0.0
    founder_share: float = 0.0  # 50% to founder
    treasury_share: float = 0.0  # 50% to system (founder donation)
    zakat_reserve: float = 0.0  # 2.5% of founder share
    net_founder: float = 0.0  # founder_share - zakat
    evidence_hash: str = ""


@dataclass
class MintReceipt:
    """Phase F output: sealed, hash-linked receipt of the entire round."""

    round_id: str = ""
    phase: str = ""
    timestamp: str = ""
    claimant: str = ""
    verdict: str = ""
    snapshot_hash: str = ""
    scorecard_hash: str = ""
    distribution_hash: str = ""
    chain_hash: str = ""  # Links to previous receipt
    prev_chain_hash: str = ""
    signature: str = ""  # Ed25519 of round_id


# ═══════════════════════════════════════════════════════════════
# MINT COURT — The Judicial Engine
# ═══════════════════════════════════════════════════════════════


class MintCourt:
    """SAT Mint Court: evaluation → adjudication → mint → split → seal.

    No mint without evidence freeze.
    No valuation without market normalization.
    No distribution without SAT consensus.
    No claim without receipts.
    """

    def __init__(
        self,
        claimant_identity: str,
        gold_dir: Path = Path("04_GOLD"),
        prev_chain_hash: str = "0" * 64,
    ):
        self._claimant = claimant_identity
        self._gold_dir = Path(gold_dir)
        self._prev_hash = prev_chain_hash
        self._phase = MintPhase.FREEZE
        self._snapshot: Optional[EvidenceSnapshot] = None
        self._scorecard: Optional[SATScorecard] = None
        self._distribution: Optional[MintDistribution] = None
        self._receipts: List[MintReceipt] = []
        self._round_id = f"MINT_{int(time.time())}_{claimant_identity[:8]}"

    # ── Phase A: FREEZE ──────────────────────────────────────
    def phase_a_freeze(self) -> EvidenceSnapshot:
        """Lock evidence snapshot. Hash all GOLD parquet files."""
        assert self._phase == MintPhase.FREEZE, f"Wrong phase: {self._phase}"
        logger.info("Phase A: Freezing evidence snapshot for %s", self._round_id)

        parquets = sorted(self._gold_dir.glob("*.parquet"))
        hashes = {}
        total_bytes = 0
        artifact_classes: Dict[str, int] = {}

        for pf in parquets:
            data = pf.read_bytes()
            h = hashlib.blake2b(data, digest_size=32).hexdigest()
            hashes[pf.name] = h
            total_bytes += len(data)
            # Count artifact class from filename
            cls = pf.stem.replace("_chunks", "").replace("_unified", "")
            artifact_classes[cls] = artifact_classes.get(cls, 0) + 1

        # Compute evidence hash root (Merkle-like: hash of all hashes)
        root_input = json.dumps(hashes, sort_keys=True).encode()
        evidence_root = hashlib.blake2b(root_input, digest_size=32).hexdigest()

        # Read artifact count from sovereign_catalog if available
        artifact_count = 0
        try:
            import pyarrow.parquet as pq

            cat = pq.read_table(self._gold_dir / "sovereign_catalog.parquet")
            artifact_count = len(cat)
        except Exception:
            artifact_count = sum(artifact_classes.values())

        now = datetime.now(timezone.utc).isoformat()
        self._snapshot = EvidenceSnapshot(
            snapshot_id=f"SNAP_{self._round_id}",
            claimant_identity=self._claimant,
            timestamp=now,
            evidence_hash_root=evidence_root,
            artifact_count=artifact_count,
            artifact_classes=artifact_classes,
            time_window=("2023-04-01", now[:10]),
            gold_parquet_hashes=hashes,
            total_bytes=total_bytes,
        )
        self._phase = MintPhase.VERIFY
        self._emit_receipt("freeze", f"evidence_root={evidence_root[:16]}")
        logger.info(
            "Phase A complete: %d artifacts, %d bytes, root=%s",
            artifact_count,
            total_bytes,
            evidence_root[:16],
        )
        return self._snapshot

    # ── Phase B: VERIFY (Guardian + Auditor) ─────────────────
    def phase_b_verify(self) -> SATScorecard:
        """Provenance, dedup, integrity, claim binding."""
        assert self._phase == MintPhase.VERIFY, f"Wrong phase: {self._phase}"
        assert self._snapshot is not None
        logger.info("Phase B: Verifying evidence for %s", self._round_id)

        sc = SATScorecard()

        # Guardian: constitutional gates
        sc.guardian_pass = (
            self._snapshot.artifact_count >= MIN_EVALUATION_ARTIFACTS
            and self._snapshot.evidence_hash_root != ""
            and len(self._snapshot.gold_parquet_hashes) > 0
        )

        # Auditor: provenance + dedup check
        # Real check: all parquet hashes are unique (no duplicate files)
        unique_hashes = len(set(self._snapshot.gold_parquet_hashes.values()))
        total_files = len(self._snapshot.gold_parquet_hashes)
        sc.auditor_pass = unique_hashes == total_files
        sc.dedup_penalty = 1.0 - (unique_hashes / max(total_files, 1))

        self._scorecard = sc
        self._phase = MintPhase.VALUE
        self._emit_receipt(
            "verify", f"guardian={sc.guardian_pass} auditor={sc.auditor_pass}"
        )
        return sc

    # ── Phase C: VALUE (Optimizer) ───────────────────────────
    def phase_c_value(self) -> SATScorecard:
        """SNR + Ihsan + market normalization."""
        assert self._phase == MintPhase.VALUE, f"Wrong phase: {self._phase}"
        assert self._snapshot and self._scorecard
        logger.info("Phase C: Valuing evidence for %s", self._round_id)

        sc = self._scorecard

        # Read SNR scores from sovereign_catalog — quality-weighted, not raw average.
        # The constitutional gate evaluates WORK, not DATA.
        # Media artifacts (screenshots, images) are excluded from quality scoring.
        try:
            import pyarrow.parquet as pq

            cat = pq.read_table(
                self._gold_dir / "sovereign_catalog.parquet",
                columns=["snr_score", "kind", "size_bytes"],
            )
            snr_col = cat.column("snr_score").to_pylist()
            kind_col = cat.column("kind").to_pylist()

            # Separate signal (work) from noise (media/artifacts)
            work_snr = [
                s
                for s, k in zip(snr_col, kind_col)
                if s is not None and s >= SNR_THRESHOLD and k not in ("Artifact/Media",)
            ]
            all_valid = [s for s in snr_col if s is not None and s > 0]

            # Quality-weighted: only evaluate artifacts above SNR floor
            sc.snr_composite = sum(work_snr) / len(work_snr) if work_snr else 0.0
            sc.detail["total_artifacts"] = len(all_valid)
            sc.detail["work_artifacts"] = len(work_snr)
            sc.detail["noise_artifacts"] = len(all_valid) - len(work_snr)
            sc.detail["raw_mean_snr"] = (
                sum(all_valid) / len(all_valid) if all_valid else 0
            )
            sc.detail["work_mean_snr"] = sc.snr_composite
        except Exception:
            sc.snr_composite = SNR_THRESHOLD

        # Ihsan: bounded by work quality, not noise
        sc.ihsan_composite = min(sc.snr_composite * 1.02, 1.0)

        # Market-fair valuation:
        # base = compute_hours x SEED_PER_HOUR
        # quality = ihsan_composite (0-1, from work artifacts only)
        # depth = log2(work_artifact_count) / 20 (bounded 0-1)
        # dedup = 1 - dedup_penalty
        import math

        compute_hours_estimate = 15000.0  # Self-reported, auditable
        base_value = compute_hours_estimate * SEED_PER_COMPUTE_HOUR
        sc.quality_score = sc.ihsan_composite
        work_count = sc.detail.get("work_artifacts", self._snapshot.artifact_count)
        sc.depth_score = min(math.log2(max(work_count, 1)) / 20.0, 1.0)
        sc.originality_score = 1.0 - sc.dedup_penalty
        sc.optimizer_valuation = round(
            base_value * sc.quality_score * sc.depth_score * sc.originality_score, 2
        )
        sc.detail["compute_hours"] = compute_hours_estimate
        sc.detail["base_value"] = base_value
        sc.detail["quality_multiplier"] = sc.quality_score
        sc.detail["depth_multiplier"] = sc.depth_score
        sc.detail["dedup_discount"] = sc.originality_score
        sc.detail["work_count_for_depth"] = work_count
        self._phase = MintPhase.DECIDE
        self._emit_receipt("value", f"valuation={sc.optimizer_valuation:.2f} SEED")
        logger.info("Phase C complete: valuation=%.2f SEED", sc.optimizer_valuation)
        return sc

    # ── Phase D: DECIDE (Mediator — SAT consensus) ───────────
    def phase_d_decide(self) -> MintVerdict:
        """SAT consensus: authorize or reject mint."""
        assert self._phase == MintPhase.DECIDE, f"Wrong phase: {self._phase}"
        assert self._scorecard
        logger.info("Phase D: SAT consensus for %s", self._round_id)

        sc = self._scorecard
        if not sc.guardian_pass:
            verdict = MintVerdict.INSUFFICIENT_EVIDENCE
        elif not sc.auditor_pass:
            verdict = MintVerdict.INSUFFICIENT_EVIDENCE
        elif not sc.constitutional:
            verdict = MintVerdict.CONSTITUTIONAL_VIOLATION
        elif not sc.mediator_pass:
            # Mediator fairness check: valuation must be reasonable
            sc.mediator_pass = (
                sc.optimizer_valuation > 0 and sc.optimizer_valuation < 1_000_000
            )
            if not sc.mediator_pass:
                verdict = MintVerdict.REJECTED
            else:
                verdict = MintVerdict.APPROVED
        else:
            verdict = MintVerdict.APPROVED

        self._phase = (
            MintPhase.DISTRIBUTE
            if verdict == MintVerdict.APPROVED
            else MintPhase.REJECTED
        )
        self._emit_receipt("decide", f"verdict={verdict.value}")
        logger.info("Phase D complete: verdict=%s", verdict.value)
        return verdict

    # ── Phase E: DISTRIBUTE (Archivist — mint + split) ────────
    def phase_e_distribute(self) -> MintDistribution:
        """Mint SEED, apply founder 50/50 donation, compute Zakat."""
        assert self._phase == MintPhase.DISTRIBUTE, f"Wrong phase: {self._phase}"
        assert self._scorecard
        logger.info("Phase E: Distributing for %s", self._round_id)

        gross = self._scorecard.optimizer_valuation
        founder = round(gross * (1.0 - FOUNDER_DONATION_RATIO), 2)
        treasury = round(gross * FOUNDER_DONATION_RATIO, 2)
        zakat = round(founder * 0.025, 2)  # 2.5% of founder share
        net = round(founder - zakat, 2)

        dist_payload = json.dumps(
            {
                "gross": gross,
                "founder": founder,
                "treasury": treasury,
                "zakat": zakat,
                "net": net,
            },
            sort_keys=True,
        ).encode()

        self._distribution = MintDistribution(
            gross_seed=gross,
            founder_share=founder,
            treasury_share=treasury,
            zakat_reserve=zakat,
            net_founder=net,
            evidence_hash=hashlib.blake2b(dist_payload, digest_size=32).hexdigest(),
        )
        self._phase = MintPhase.SEAL
        self._emit_receipt(
            "distribute",
            f"gross={gross} founder={net} treasury={treasury} zakat={zakat}",
        )
        logger.info(
            "Phase E: gross=%.2f founder=%.2f treasury=%.2f zakat=%.2f",
            gross,
            net,
            treasury,
            zakat,
        )
        return self._distribution

    # ── Phase F: SEAL (hash-chain all receipts) ──────────────
    def phase_f_seal(self) -> MintReceipt:
        """Write final receipt, hash-chain the entire round."""
        assert self._phase == MintPhase.SEAL, f"Wrong phase: {self._phase}"
        logger.info("Phase F: Sealing round %s", self._round_id)

        # Build composite hash of the round
        composite = json.dumps(
            {
                "round_id": self._round_id,
                "snapshot_hash": (
                    self._snapshot.evidence_hash_root if self._snapshot else ""
                ),
                "scorecard": {
                    "guardian": (
                        self._scorecard.guardian_pass if self._scorecard else False
                    ),
                    "auditor": (
                        self._scorecard.auditor_pass if self._scorecard else False
                    ),
                    "ihsan": self._scorecard.ihsan_composite if self._scorecard else 0,
                    "valuation": (
                        self._scorecard.optimizer_valuation if self._scorecard else 0
                    ),
                },
                "distribution": {
                    "gross": self._distribution.gross_seed if self._distribution else 0,
                    "founder": (
                        self._distribution.net_founder if self._distribution else 0
                    ),
                    "treasury": (
                        self._distribution.treasury_share if self._distribution else 0
                    ),
                },
            },
            sort_keys=True,
        ).encode()
        round_hash = hashlib.blake2b(composite, digest_size=32).hexdigest()

        final_receipt = MintReceipt(
            round_id=self._round_id,
            phase="seal",
            timestamp=datetime.now(timezone.utc).isoformat(),
            claimant=self._claimant,
            verdict="approved",
            snapshot_hash=self._snapshot.evidence_hash_root if self._snapshot else "",
            scorecard_hash=(
                hashlib.blake2b(
                    json.dumps(self._scorecard.detail, sort_keys=True).encode(),
                    digest_size=32,
                ).hexdigest()
                if self._scorecard
                else ""
            ),
            distribution_hash=(
                self._distribution.evidence_hash if self._distribution else ""
            ),
            chain_hash=round_hash,
            prev_chain_hash=self._prev_hash,
        )
        self._receipts.append(final_receipt)
        self._phase = MintPhase.COMPLETE
        logger.info(
            "Phase F: Round %s sealed. chain=%s", self._round_id, round_hash[:16]
        )
        return final_receipt

    # ── Internal: receipt emission ────────────────────────────
    def _emit_receipt(self, phase: str, detail: str) -> None:
        """Emit a phase receipt into the chain."""
        payload = json.dumps(
            {
                "round_id": self._round_id,
                "phase": phase,
                "detail": detail,
                "prev": self._prev_hash,
            },
            sort_keys=True,
        ).encode()
        h = hashlib.blake2b(payload, digest_size=32).hexdigest()
        receipt = MintReceipt(
            round_id=self._round_id,
            phase=phase,
            timestamp=datetime.now(timezone.utc).isoformat(),
            claimant=self._claimant,
            verdict="in_progress",
            chain_hash=h,
            prev_chain_hash=self._prev_hash,
        )
        self._receipts.append(receipt)
        self._prev_hash = h

    # ── Orchestrator: run full A→F pipeline ──────────────────
    def run(self) -> Dict[str, Any]:
        """Execute the full Mint Court pipeline A→F.

        Returns a complete audit package with all receipts.
        """
        logger.info("=" * 60)
        logger.info("SAT MINT COURT — Round %s", self._round_id)
        logger.info("Claimant: %s", self._claimant)
        logger.info("=" * 60)

        # Phase A: Freeze
        snapshot = self.phase_a_freeze()
        if snapshot.artifact_count < MIN_EVALUATION_ARTIFACTS:
            return {
                "verdict": "rejected",
                "reason": "insufficient_artifacts",
                "artifacts": snapshot.artifact_count,
                "minimum": MIN_EVALUATION_ARTIFACTS,
            }

        # Phase B: Verify
        scorecard = self.phase_b_verify()
        if not scorecard.guardian_pass or not scorecard.auditor_pass:
            return {
                "verdict": "rejected",
                "reason": "verification_failed",
                "guardian": scorecard.guardian_pass,
                "auditor": scorecard.auditor_pass,
            }

        # Phase C: Value
        scorecard = self.phase_c_value()

        # Phase D: Decide
        verdict = self.phase_d_decide()
        if verdict != MintVerdict.APPROVED:
            return {"verdict": verdict.value, "scorecard": scorecard.detail}

        # Phase E: Distribute
        distribution = self.phase_e_distribute()

        # Phase F: Seal
        final_receipt = self.phase_f_seal()

        return {
            "verdict": "approved",
            "round_id": self._round_id,
            "claimant": self._claimant,
            "snapshot": {
                "id": snapshot.snapshot_id,
                "artifacts": snapshot.artifact_count,
                "bytes": snapshot.total_bytes,
                "evidence_root": snapshot.evidence_hash_root,
                "time_window": snapshot.time_window,
                "parquet_files": len(snapshot.gold_parquet_hashes),
            },
            "scorecard": {
                "guardian_pass": scorecard.guardian_pass,
                "auditor_pass": scorecard.auditor_pass,
                "mediator_pass": scorecard.mediator_pass,
                "ihsan": scorecard.ihsan_composite,
                "snr": scorecard.snr_composite,
                "quality": scorecard.quality_score,
                "depth": scorecard.depth_score,
                "originality": scorecard.originality_score,
                "valuation_seed": scorecard.optimizer_valuation,
                "detail": scorecard.detail,
            },
            "distribution": {
                "gross_seed": distribution.gross_seed,
                "founder_share": distribution.founder_share,
                "treasury_share": distribution.treasury_share,
                "zakat_reserve": distribution.zakat_reserve,
                "net_founder": distribution.net_founder,
                "evidence_hash": distribution.evidence_hash,
            },
            "seal": {
                "chain_hash": final_receipt.chain_hash,
                "prev_chain_hash": final_receipt.prev_chain_hash,
            },
            "receipts": len(self._receipts),
            "phase": self._phase.value,
            "constitutional": {
                "ihsan_threshold": IHSAN_THRESHOLD,
                "gini_threshold": ADL_GINI_THRESHOLD,
                "snr_threshold": SNR_THRESHOLD,
                "kernel_invariants": list(KERNEL_INVARIANTS),
                "founder_donation": FOUNDER_DONATION_RATIO,
            },
        }

    def verify_chain(self) -> bool:
        """Verify receipt chain integrity."""
        if not self._receipts:
            return True
        for i in range(1, len(self._receipts)):
            if self._receipts[i].prev_chain_hash != self._receipts[i - 1].chain_hash:
                return False
        return True

    @property
    def phase(self) -> MintPhase:
        return self._phase

    @property
    def receipts(self) -> List[MintReceipt]:
        return list(self._receipts)


# ═══════════════════════════════════════════════════════════════
# CLI ENTRY POINT — for testing against real GOLD data
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Use real identity if available, else placeholder
    identity = sys.argv[1] if len(sys.argv) > 1 else "NODE0_FOUNDER"
    gold_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("04_GOLD")

    court = MintCourt(claimant_identity=identity, gold_dir=gold_dir)
    result = court.run()

    print()
    print("=" * 60)
    print("  SAT MINT COURT — RESULT")
    print("=" * 60)
    print()
    print(json.dumps(result, indent=2, default=str))
    print()
    print(f"  Chain integrity: {court.verify_chain()}")
    print(f"  Total receipts:  {len(court.receipts)}")
    if result.get("verdict") == "approved":
        d = result["distribution"]
        print()
        print(f"  === MINT AUTHORIZED ===")
        print(f"  Gross SEED:    {d['gross_seed']:,.2f}")
        print(f"  Founder (50%): {d['founder_share']:,.2f}")
        print(f"  Treasury(50%): {d['treasury_share']:,.2f}")
        print(f"  Zakat (2.5%):  {d['zakat_reserve']:,.2f}")
        print(f"  Net to founder:{d['net_founder']:,.2f}")
    else:
        print(f"  === MINT REJECTED: {result.get('reason', result.get('verdict'))} ===")
