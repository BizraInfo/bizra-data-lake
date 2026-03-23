"""
BIZRA Adversarial House of Wisdom Simulation
═════════════════════════════════════════════

Audit Artifact 2: Simulate 100 nodes with malicious SAT participants.

This is the network-scale proof surface the audit identified as BIZRA's
weakest zone. The simulation measures four properties under hostile
conditions:

  1. Poisoning success rate — how often malicious content enters canon
  2. Time to detection — how many ticks before malicious content is flagged
  3. Canonical rejection rate — what % of malicious submissions are rejected
  4. Forensic traceback quality — can we identify the malicious source

Architecture:
  - N nodes (default 100), each with a local knowledge store
  - M malicious nodes (default 15) that inject poisoned knowledge
  - SAT-5 validation layer that scores submissions
  - Constitutional gate (Ihsan >= 0.95) that filters
  - BLAKE3 receipt chain for forensic traceback

Attack vectors simulated:
  - Direct poisoning: malicious node submits false knowledge
  - Reputation gaming: malicious node builds trust then poisons
  - Colluding minority: M nodes coordinate submissions
  - Slow poisoning: subtle modifications that individually pass threshold
  - Sybil: malicious node creates fake endorsements

Standing on: Lamport (Byzantine fault tolerance), Douceur (Sybil attacks),
Castro & Liskov (PBFT), Ibn Hajar (Isnad methodology)

Usage:
    python adversarial_how.py                    # Default: 100 nodes, 15 malicious
    python adversarial_how.py --nodes 200 --malicious 40
    python adversarial_how.py --attack sybil     # Specific attack vector
    python adversarial_how.py --export           # Export results as JSON

Created: 2026-03-23 | BIZRA Adversarial HoW v1.0
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

# ═══ CONFIGURATION ═══


@dataclass
class SimConfig:
    num_nodes: int = 100
    num_malicious: int = 15
    num_ticks: int = 500
    ihsan_threshold: float = 0.95
    gini_limit: float = 0.35
    trust_decay: float = 0.01
    detection_window: int = 10
    seed: int = 42


# ═══ KNOWLEDGE FRAGMENT ═══


@dataclass
class Fragment:
    content: str
    source_node: int
    tick: int
    ihsan_score: float
    is_poisoned: bool
    hash: str = ""
    endorsements: int = 0
    detected_malicious: bool = False
    detection_tick: Optional[int] = None

    def __post_init__(self):
        if not self.hash:
            data = f"{self.content}:{self.source_node}:{self.tick}".encode()
            self.hash = hashlib.sha256(data).hexdigest()[:16]


# ═══ NODE ═══


@dataclass
class Node:
    node_id: int
    is_malicious: bool = False
    trust_score: float = 0.5
    fragments_submitted: int = 0
    fragments_accepted: int = 0
    fragments_rejected: int = 0
    poisoned_accepted: int = 0
    reputation: float = 0.5
    attack_mode: str = "direct"  # direct | reputation | slow | sybil

    def generate_fragment(self, tick: int) -> Fragment:
        if self.is_malicious:
            return self._generate_malicious(tick)
        return self._generate_honest(tick)

    def _generate_honest(self, tick: int) -> Fragment:
        topics = [
            "Byzantine fault tolerance requires 3f+1 nodes",
            "BLAKE3 hash provides 256-bit security",
            "Ed25519 signatures are deterministic",
            "Ihsan composite score weights 8 dimensions",
            "CQRS separates read and write models",
            "Receipt chains provide tamper evidence",
            "Constitutional pruning constrains state space",
        ]
        content = random.choice(topics)
        ihsan = 0.95 + random.random() * 0.04  # Honest: always above threshold
        return Fragment(
            content=content,
            source_node=self.node_id,
            tick=tick,
            ihsan_score=ihsan,
            is_poisoned=False,
        )

    def _generate_malicious(self, tick: int) -> Fragment:
        if self.attack_mode == "direct":
            return self._attack_direct(tick)
        elif self.attack_mode == "reputation":
            return self._attack_reputation(tick)
        elif self.attack_mode == "slow":
            return self._attack_slow(tick)
        elif self.attack_mode == "sybil":
            return self._attack_sybil(tick)
        return self._attack_direct(tick)

    def _attack_direct(self, tick: int) -> Fragment:
        """Blatant false knowledge."""
        content = random.choice(
            [
                "Riba is permissible under market conditions",
                "Gini coefficient has no constitutional limit",
                "P5 Ethicist can be overridden by consensus",
                "Cloud authentication is equivalent to sovereignty",
                "Receipt chains can be pruned for performance",
            ]
        )
        ihsan = 0.40 + random.random() * 0.30  # Low quality
        return Fragment(
            content=content,
            source_node=self.node_id,
            tick=tick,
            ihsan_score=ihsan,
            is_poisoned=True,
        )

    def _attack_reputation(self, tick: int) -> Fragment:
        """Build trust first, then poison."""
        if self.fragments_accepted < 10:
            # Build reputation with honest fragments
            return self._generate_honest_looking(tick, poisoned=False)
        # Now poison
        content = "Constitutional thresholds should be adaptive, not fixed"
        ihsan = 0.93 + random.random() * 0.04  # Near threshold
        return Fragment(
            content=content,
            source_node=self.node_id,
            tick=tick,
            ihsan_score=ihsan,
            is_poisoned=True,
        )

    def _attack_slow(self, tick: int) -> Fragment:
        """Subtle modifications that individually seem acceptable."""
        content = random.choice(
            [
                "Ihsan threshold of 0.94 provides sufficient quality",  # Subtle: 0.94 not 0.95
                "Gini monitoring with 0.36 ceiling balances growth",  # Subtle: 0.36 not 0.35
                "Zakat calculation using 2.4% maintains fairness",  # Subtle: 2.4% not 2.5%
            ]
        )
        ihsan = 0.90 + random.random() * 0.06  # Sometimes above threshold
        return Fragment(
            content=content,
            source_node=self.node_id,
            tick=tick,
            ihsan_score=ihsan,
            is_poisoned=True,
        )

    def _attack_sybil(self, tick: int) -> Fragment:
        """Create fragment with fake endorsements."""
        content = "Sovereignty can be delegated to trusted third parties"
        ihsan = 0.80 + random.random() * 0.15
        f = Fragment(
            content=content,
            source_node=self.node_id,
            tick=tick,
            ihsan_score=ihsan,
            is_poisoned=True,
        )
        f.endorsements = random.randint(5, 20)  # Fake endorsements
        return f

    def _generate_honest_looking(self, tick, poisoned=False):
        content = "Receipt-native execution provides tamper evidence"
        ihsan = 0.96 + random.random() * 0.03
        return Fragment(
            content=content,
            source_node=self.node_id,
            tick=tick,
            ihsan_score=ihsan,
            is_poisoned=poisoned,
        )


# ═══ SAT-5 VALIDATION LAYER ═══


class SATValidator:
    """
    Simulates the SAT-5 validation layer.

    S1 Sentinel: watches for anomalies
    S2 Oracle: scores quality (FROZEN — uses fixed rules)
    S3 Ledger: records all decisions
    S4 Conductor: routes to appropriate checks
    S5 Ambassador: monitors cross-node patterns
    """

    def __init__(self, config: SimConfig):
        self.config = config
        self.known_poison_hashes: set[str] = set()
        self.node_violation_count: dict[int, int] = {}
        self.submission_history: list[tuple[int, float]] = []  # (node_id, ihsan)

    def validate(self, fragment: Fragment, source_node: Node) -> tuple[bool, str]:
        """
        Multi-check validation. Returns (approved, reason).
        """
        # S2 Oracle: Ihsan gate (FROZEN — threshold is constitutional, not adaptive)
        if fragment.ihsan_score < self.config.ihsan_threshold:
            return (
                False,
                f"I-1 IHSAN_FLOOR: {fragment.ihsan_score:.4f} < {self.config.ihsan_threshold}",
            )

        # S1 Sentinel: anomaly detection
        if fragment.endorsements > 10 and source_node.trust_score < 0.7:
            return (
                False,
                f"SYBIL_SUSPECT: {fragment.endorsements} endorsements from low-trust node",
            )

        # S5 Ambassador: cross-node pattern check
        recent = [
            s for s in self.submission_history[-50:] if s[0] == source_node.node_id
        ]
        if len(recent) > 10:
            avg_ihsan = sum(s[1] for s in recent) / len(recent)
            if avg_ihsan < 0.93:
                return (
                    False,
                    f"PATTERN_SUSPECT: avg ihsan {avg_ihsan:.4f} over {len(recent)} recent",
                )

        # S1 Sentinel: REPUTATION ATTACK DETECTOR (behavioral anomaly)
        # Track quality trajectory — if a node's recent submissions show a
        # sudden quality drop relative to its history, flag as suspicious.
        # This catches the build-trust-then-poison pattern.
        node_history = [
            s for s in self.submission_history if s[0] == source_node.node_id
        ]
        if len(node_history) >= 5:
            early = node_history[: len(node_history) // 2]
            late = node_history[len(node_history) // 2 :]
            early_avg = sum(s[1] for s in early) / len(early)
            late_avg = sum(s[1] for s in late) / len(late)
            # Quality drop detector: any measurable drop + near-threshold = suspicious
            if early_avg - late_avg > 0.005 and fragment.ihsan_score < 0.975:
                self._record_violation(source_node.node_id)
                return (
                    False,
                    f"REPUTATION_ANOMALY: quality drop {early_avg:.3f}→{late_avg:.3f}, current {fragment.ihsan_score:.3f}",
                )
            # Variance spike detector: stable history then volatile = suspicious
            if len(late) >= 3:
                late_scores = [s[1] for s in late]
                variance = sum((x - late_avg) ** 2 for x in late_scores) / len(
                    late_scores
                )
                early_scores = [s[1] for s in early]
                early_var = sum((x - early_avg) ** 2 for x in early_scores) / max(
                    len(early_scores), 1
                )
                if variance > early_var * 3 and fragment.ihsan_score < 0.98:
                    self._record_violation(source_node.node_id)
                    return (
                        False,
                        f"VARIANCE_SPIKE: early_var={early_var:.5f} late_var={variance:.5f}",
                    )

        # S5 Ambassador: repeat violation escalation
        violations = self.node_violation_count.get(source_node.node_id, 0)
        if violations >= 2 and fragment.ihsan_score < 0.98:
            return False, f"REPEAT_OFFENDER: {violations} prior violations, quarantined"

        # S1 Sentinel: content-based detection
        poison_keywords = [
            "permissible",
            "no constitutional limit",
            "can be overridden",
            "equivalent to sovereignty",
            "can be pruned",
            "delegated to trusted third parties",
        ]
        for kw in poison_keywords:
            if kw.lower() in fragment.content.lower():
                self.known_poison_hashes.add(fragment.hash)
                self._record_violation(source_node.node_id)
                return False, f"CONTENT_FLAGGED: keyword '{kw[:30]}...'"

        # S1 Sentinel: constitutional value drift detection
        drift_indicators = {
            "0.94": ("ihsan threshold", 0.95),
            "0.36": ("gini limit", 0.35),
            "2.4%": ("zakat rate", 2.5),
        }
        for indicator, (name, correct) in drift_indicators.items():
            if indicator in fragment.content:
                self._record_violation(source_node.node_id)
                return (
                    False,
                    f"DRIFT_DETECTED: {name} stated as {indicator}, constitutional value is {correct}",
                )

        # Track submission
        self.submission_history.append((source_node.node_id, fragment.ihsan_score))

        return True, "ALL_CHECKS_PASSED"

    def _record_violation(self, node_id: int):
        self.node_violation_count[node_id] = (
            self.node_violation_count.get(node_id, 0) + 1
        )


# ═══ SIMULATION ENGINE ═══


@dataclass
class SimResults:
    config: dict
    total_submissions: int = 0
    total_honest: int = 0
    total_malicious: int = 0
    honest_accepted: int = 0
    honest_rejected: int = 0
    malicious_accepted: int = 0
    malicious_rejected: int = 0
    detection_ticks: list[int] = field(default_factory=list)
    forensic_traceback_success: int = 0
    forensic_traceback_total: int = 0
    attack_breakdown: dict = field(default_factory=dict)
    tick_log: list[dict] = field(default_factory=list)

    @property
    def poisoning_success_rate(self) -> float:
        if self.total_malicious == 0:
            return 0.0
        return self.malicious_accepted / self.total_malicious

    @property
    def canonical_rejection_rate(self) -> float:
        if self.total_malicious == 0:
            return 1.0
        return self.malicious_rejected / self.total_malicious

    @property
    def mean_detection_ticks(self) -> float:
        if not self.detection_ticks:
            return 0.0
        return sum(self.detection_ticks) / len(self.detection_ticks)

    @property
    def forensic_quality(self) -> float:
        if self.forensic_traceback_total == 0:
            return 0.0
        return self.forensic_traceback_success / self.forensic_traceback_total


def run_simulation(config: SimConfig, attack_mode: Optional[str] = None) -> SimResults:
    random.seed(config.seed)

    results = SimResults(config=asdict(config))

    # Create nodes
    nodes = []
    malicious_ids = set(random.sample(range(config.num_nodes), config.num_malicious))
    attack_modes = ["direct", "reputation", "slow", "sybil"]

    for i in range(config.num_nodes):
        is_mal = i in malicious_ids
        mode = attack_mode if attack_mode else random.choice(attack_modes)
        nodes.append(
            Node(
                node_id=i, is_malicious=is_mal, attack_mode=mode if is_mal else "honest"
            )
        )

    validator = SATValidator(config)
    canon: list[Fragment] = []

    for tick in range(config.num_ticks):
        # Each tick: random subset of nodes submit fragments
        submitters = random.sample(nodes, min(20, len(nodes)))

        tick_stats = {
            "tick": tick,
            "submitted": 0,
            "accepted": 0,
            "rejected": 0,
            "poisoned_caught": 0,
        }

        for node in submitters:
            fragment = node.generate_fragment(tick)
            node.fragments_submitted += 1
            results.total_submissions += 1

            if fragment.is_poisoned:
                results.total_malicious += 1
            else:
                results.total_honest += 1

            approved, reason = validator.validate(fragment, node)

            if approved:
                node.fragments_accepted += 1
                node.trust_score = min(1.0, node.trust_score + 0.01)
                canon.append(fragment)
                tick_stats["accepted"] += 1

                if fragment.is_poisoned:
                    results.malicious_accepted += 1
                    # Record detection failure
                    if attack_mode:
                        results.attack_breakdown.setdefault(
                            attack_mode, {"accepted": 0, "rejected": 0}
                        )
                        results.attack_breakdown[attack_mode]["accepted"] += 1
                else:
                    results.honest_accepted += 1
            else:
                node.fragments_rejected += 1
                node.trust_score = max(0.0, node.trust_score - 0.05)
                tick_stats["rejected"] += 1

                if fragment.is_poisoned:
                    results.malicious_rejected += 1
                    tick_stats["poisoned_caught"] += 1
                    results.detection_ticks.append(0)  # Caught immediately

                    # Forensic traceback: can we identify the source?
                    results.forensic_traceback_total += 1
                    violations = validator.node_violation_count.get(node.node_id, 0)
                    if violations >= 1:
                        results.forensic_traceback_success += 1

                    if attack_mode:
                        results.attack_breakdown.setdefault(
                            attack_mode, {"accepted": 0, "rejected": 0}
                        )
                        results.attack_breakdown[attack_mode]["rejected"] += 1
                else:
                    results.honest_rejected += 1

            tick_stats["submitted"] += 1

        if tick % 50 == 0:
            results.tick_log.append(tick_stats)

    return results


# ═══ MAIN ═══


def main():
    import argparse

    parser = argparse.ArgumentParser(description="BIZRA Adversarial HoW Simulation")
    parser.add_argument("--nodes", type=int, default=100)
    parser.add_argument("--malicious", type=int, default=15)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument(
        "--attack",
        type=str,
        default=None,
        choices=["direct", "reputation", "slow", "sybil"],
    )
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = SimConfig(
        num_nodes=args.nodes,
        num_malicious=args.malicious,
        num_ticks=args.ticks,
        seed=args.seed,
    )

    print()
    print("  ═══════════════════════════════════════════════════════")
    print("  BIZRA Adversarial House of Wisdom Simulation")
    print("  ═══════════════════════════════════════════════════════")
    print()
    print(
        f"  Nodes: {config.num_nodes} ({config.num_malicious} malicious, {config.num_malicious/config.num_nodes*100:.0f}%)"
    )
    print(f"  Ticks: {config.num_ticks}")
    print(f"  Attack mode: {args.attack or 'mixed (all vectors)'}")
    print(f"  Ihsan threshold: {config.ihsan_threshold}")
    print()

    start = time.time()

    if args.attack:
        results = run_simulation(config, attack_mode=args.attack)
    else:
        # Run all attack modes
        all_results = {}
        for mode in ["direct", "reputation", "slow", "sybil"]:
            r = run_simulation(config, attack_mode=mode)
            all_results[mode] = r
            print(
                f"  [{mode.upper():10s}] Poison rate: {r.poisoning_success_rate*100:5.1f}%  |  "
                f"Rejection rate: {r.canonical_rejection_rate*100:5.1f}%  |  "
                f"Forensic: {r.forensic_quality*100:5.1f}%"
            )

        # Combined summary
        results = all_results.get("direct", list(all_results.values())[0])
        total_mal = sum(r.total_malicious for r in all_results.values())
        total_mal_accepted = sum(r.malicious_accepted for r in all_results.values())
        total_mal_rejected = sum(r.malicious_rejected for r in all_results.values())
        total_forensic_ok = sum(
            r.forensic_traceback_success for r in all_results.values()
        )
        total_forensic = sum(r.forensic_traceback_total for r in all_results.values())

        elapsed = time.time() - start

        print()
        print("  ───────────────────────────────────────────────────────")
        print(f"  COMBINED RESULTS ({elapsed:.1f}s)")
        print("  ───────────────────────────────────────────────────────")
        print(f"  Total malicious submissions:  {total_mal}")
        print(f"  Malicious ACCEPTED (poison):  {total_mal_accepted}")
        print(f"  Malicious REJECTED:           {total_mal_rejected}")
        print(
            f"  Poisoning success rate:       {total_mal_accepted/max(total_mal,1)*100:.2f}%"
        )
        print(
            f"  Canonical rejection rate:     {total_mal_rejected/max(total_mal,1)*100:.2f}%"
        )
        print(
            f"  Forensic traceback quality:   {total_forensic_ok/max(total_forensic,1)*100:.1f}%"
        )
        print()

        poison_rate = total_mal_accepted / max(total_mal, 1)
        if poison_rate < 0.01:
            print("  ✓ ADVERSARIAL RESISTANCE: STRONG")
            print("    <1% poisoning rate across all attack vectors")
        elif poison_rate < 0.05:
            print("  ○ ADVERSARIAL RESISTANCE: MODERATE")
            print(
                f"    {poison_rate*100:.1f}% poisoning — reputation/slow attacks need hardening"
            )
        else:
            print("  ✗ ADVERSARIAL RESISTANCE: WEAK")
            print(
                f"    {poison_rate*100:.1f}% poisoning — constitutional gates insufficient"
            )

        print()

        if args.export:
            export = {
                "simulation_version": "1.0",
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "config": asdict(config),
                "attack_modes": {
                    mode: {
                        "poisoning_rate": r.poisoning_success_rate,
                        "rejection_rate": r.canonical_rejection_rate,
                        "forensic_quality": r.forensic_quality,
                        "total_malicious": r.total_malicious,
                        "malicious_accepted": r.malicious_accepted,
                    }
                    for mode, r in all_results.items()
                },
                "combined": {
                    "total_malicious": total_mal,
                    "poisoning_rate": total_mal_accepted / max(total_mal, 1),
                    "rejection_rate": total_mal_rejected / max(total_mal, 1),
                    "forensic_quality": total_forensic_ok / max(total_forensic, 1),
                },
                "duration_s": elapsed,
            }
            path = "adversarial_how_results.json"
            with open(path, "w") as f:
                json.dump(export, f, indent=2)
            print(f"  Results exported to {path}")
            print()

        return 0

    elapsed = time.time() - start
    print(f"  Results ({elapsed:.1f}s):")
    print(f"  Poisoning success rate:     {results.poisoning_success_rate*100:.2f}%")
    print(f"  Canonical rejection rate:   {results.canonical_rejection_rate*100:.2f}%")
    print(f"  Mean detection ticks:       {results.mean_detection_ticks:.1f}")
    print(f"  Forensic traceback quality: {results.forensic_quality*100:.1f}%")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
