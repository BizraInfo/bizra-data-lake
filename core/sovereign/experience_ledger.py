"""
Sovereign Experience Ledger (SEL) — Python Bridge
===================================================
Content-addressed, hash-chained episodic memory store.
Auto-commits episodes on every SNR_OK query verdict.

Uses the Rust-native implementation via PyO3 when available,
falling back to a pure-Python implementation for portability.

Standing on Giants:
  - Tulving (1972): Episodic vs semantic memory distinction
  - Park et al. (2023): Generative agent memory architecture
  - Besta et al. (2024): Graph-of-Thoughts as first-class artifact
  - Shannon (1948): Information-theoretic SNR measurement
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, Sequence

logger = logging.getLogger("sovereign.experience_ledger")

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

SEL_DOMAIN = "bizra-sel-v1"
CHAIN_DOMAIN = "bizra-sel-chain-v1"
DEFAULT_WEIGHT_RECENCY = 0.3
DEFAULT_WEIGHT_IMPORTANCE = 0.3
DEFAULT_WEIGHT_RELEVANCE = 0.4
DEFAULT_DECAY_LAMBDA = 0.001
DEFAULT_MAX_EPISODES = 10_000
EFFICIENCY_PRECISION = 1_000_000  # P = 10^6 fixed-point

# ═══════════════════════════════════════════════════════════════════════════════
# Deterministic Integer Arithmetic
# ═══════════════════════════════════════════════════════════════════════════════


def _integer_log2(n: int) -> int:
    """Integer floor(log2(n)) using bit-length. No floating-point.

    Returns 0 for n <= 1.  Deterministic across all platforms.
    """
    if n <= 1:
        return 0
    return n.bit_length() - 1


def _compute_efficiency_score(snr: float, ihsan: float, tokens_used: int) -> float:
    """Efficiency_k = (SNR_k * Ihsan_k) / max(1, floor(log2(tokens_used + 2))).

    Uses integer log2 approximation and fixed-point scaling for
    cross-platform determinism.
    """
    numerator = _quantize(snr) * _quantize(ihsan)
    log_val = max(1, _integer_log2(tokens_used + 2))
    efficiency_fp = numerator // log_val
    return efficiency_fp / (EFFICIENCY_PRECISION * EFFICIENCY_PRECISION)


# ═══════════════════════════════════════════════════════════════════════════════
# Episode Schema
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class EpisodeAction:
    """An action recorded within an episode."""

    action_type: str
    description: str
    success: bool
    duration_us: int


@dataclass
class EpisodeImpact:
    """Impact measurement for an episode."""

    snr_score: float
    ihsan_score: float
    snr_ok: bool
    user_feedback: Optional[float] = None
    tokens_used: int = 0
    efficiency_score: float = 0.0

    def importance(self) -> float:
        """Compute importance score for RIR retrieval.

        I_k = SNR_k * Ihsan_k * Efficiency_k (when efficiency available)
        Falls back to SNR_k * Ihsan_k when tokens_used == 0.
        """
        base = self.snr_score * self.ihsan_score
        if self.tokens_used > 0 and self.efficiency_score > 0.0:
            return base * self.efficiency_score
        return base


@dataclass
class Episode:
    """A single episode in the Sovereign Experience Ledger."""

    sequence: int
    timestamp_secs: int
    context: str
    graph_hash: str
    graph_node_count: int
    actions: list[EpisodeAction]
    impact: EpisodeImpact
    episode_hash: str
    prev_hash: str
    chain_hash: str
    context_embedding: Optional[list[float]] = None
    response_summary: Optional[str] = None

    def verify_hash(self) -> bool:
        """Verify the content-address hash."""
        computed = _compute_episode_hash(
            self.sequence,
            self.timestamp_secs,
            self.context,
            self.graph_hash,
            self.graph_node_count,
            self.actions,
            self.impact,
        )
        return computed == self.episode_hash

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        d: dict[str, Any] = {
            "sequence": self.sequence,
            "timestamp_secs": self.timestamp_secs,
            "context": self.context,
            "graph_hash": self.graph_hash,
            "graph_node_count": self.graph_node_count,
            "snr_score": self.impact.snr_score,
            "ihsan_score": self.impact.ihsan_score,
            "snr_ok": self.impact.snr_ok,
            "episode_hash": self.episode_hash,
            "chain_hash": self.chain_hash,
        }
        if self.impact.tokens_used > 0:
            d["tokens_used"] = self.impact.tokens_used
            d["efficiency_score"] = self.impact.efficiency_score
        if self.response_summary:
            d["response_summary"] = self.response_summary
        return d


# ═══════════════════════════════════════════════════════════════════════════════
# Hash Functions (BLAKE3 via canonical — SEC-001)
# ═══════════════════════════════════════════════════════════════════════════════

from core.proof_engine.canonical import hex_digest


def _blake3_hash(data: bytes) -> str:
    """Content hash using BLAKE3 canonical digest (SHA-256 fallback inside canonical)."""
    return hex_digest(data)  # SEC-001: BLAKE3 for Rust interop


def _domain_hash(domain: str, data: bytes) -> str:
    """Domain-separated hash."""
    return _blake3_hash(domain.encode("utf-8") + b":" + data)


def _quantize(value: float) -> int:
    """Quantize f64 to deterministic u32 (6 decimal places)."""
    return int(max(0.0, min(1.0, value)) * 1_000_000)


def _compute_episode_hash(
    sequence: int,
    timestamp_secs: int,
    context: str,
    graph_hash: str,
    graph_node_count: int,
    actions: list[EpisodeAction],
    impact: EpisodeImpact,
) -> str:
    """Compute content-address hash for an episode."""
    parts = [
        SEL_DOMAIN.encode("utf-8"),
        b":",
        sequence.to_bytes(8, "little"),
        timestamp_secs.to_bytes(8, "little"),
        len(context).to_bytes(4, "little"),
        context.encode("utf-8"),
        len(graph_hash).to_bytes(4, "little"),
        graph_hash.encode("utf-8"),
        graph_node_count.to_bytes(4, "little"),
        len(actions).to_bytes(4, "little"),
    ]

    for action in actions:
        parts.extend(
            [
                len(action.action_type).to_bytes(4, "little"),
                action.action_type.encode("utf-8"),
                len(action.description).to_bytes(4, "little"),
                action.description.encode("utf-8"),
                bytes([1 if action.success else 0]),
                action.duration_us.to_bytes(8, "little"),
            ]
        )

    # Impact (fixed-point)
    impact_bytes = bytearray()
    impact_bytes.extend(_quantize(impact.snr_score).to_bytes(4, "little"))
    impact_bytes.extend(_quantize(impact.ihsan_score).to_bytes(4, "little"))
    impact_bytes.append(1 if impact.snr_ok else 0)
    if impact.user_feedback is not None:
        impact_bytes.append(1)
        shifted = (max(-1.0, min(1.0, impact.user_feedback)) + 1.0) / 2.0
        impact_bytes.extend(_quantize(shifted).to_bytes(4, "little"))
    else:
        impact_bytes.append(0)
    # Efficiency_k (deterministic fixed-point)
    impact_bytes.extend(impact.tokens_used.to_bytes(8, "little"))
    impact_bytes.extend(_quantize(impact.efficiency_score).to_bytes(4, "little"))

    parts.append(bytes(impact_bytes))
    return _blake3_hash(b"".join(parts))


def _compute_chain_hash(prev_chain_hash: str, episode_hash: str) -> str:
    """Compute chain hash: domain_hash(prev || episode)."""
    combined = f"{prev_chain_hash}:{episode_hash}"
    return _domain_hash(CHAIN_DOMAIN, combined.encode("utf-8"))


# ═══════════════════════════════════════════════════════════════════════════════
# Sovereign Experience Ledger
# ═══════════════════════════════════════════════════════════════════════════════


class SovereignExperienceLedger:
    """Content-addressed, hash-chained episodic memory store.

    Episodes are committed on every SNR_OK query verdict, building
    an auditable chain of reasoning experiences.
    """

    def __init__(
        self,
        max_episodes: int = DEFAULT_MAX_EPISODES,
        weight_recency: float = DEFAULT_WEIGHT_RECENCY,
        weight_importance: float = DEFAULT_WEIGHT_IMPORTANCE,
        weight_relevance: float = DEFAULT_WEIGHT_RELEVANCE,
        decay_lambda: float = DEFAULT_DECAY_LAMBDA,
    ) -> None:
        self._episodes: Deque[Episode] = deque()
        self._hash_index: dict[str, Episode] = {}
        self._seq_index: dict[int, Episode] = {}
        self._next_sequence: int = 0
        self._chain_head: str = "genesis"
        self._max_episodes: int = max_episodes
        self._distillation_count: int = 0

        # RIR weights
        self._w_recency = weight_recency
        self._w_importance = weight_importance
        self._w_relevance = weight_relevance
        self._decay_lambda = decay_lambda

    # ─────────────────────────────────────────────────────────────────────────
    # Commit
    # ─────────────────────────────────────────────────────────────────────────

    def commit(
        self,
        context: str,
        graph_hash: str,
        graph_node_count: int,
        actions: list[tuple[str, str, bool, int]],
        snr_score: float,
        ihsan_score: float,
        snr_ok: bool,
        context_embedding: Optional[list[float]] = None,
        response_summary: Optional[str] = None,
        tokens_used: int = 0,
    ) -> str:
        """Commit a new episode. Returns the episode hash."""
        sequence = self._next_sequence
        timestamp_secs = int(time.time())

        episode_actions = [
            EpisodeAction(action_type=at, description=desc, success=ok, duration_us=dur)
            for at, desc, ok, dur in actions
        ]

        # Compute Efficiency_k deterministically (integer log2, fixed-point)
        efficiency = 0.0
        if tokens_used > 0:
            efficiency = _compute_efficiency_score(snr_score, ihsan_score, tokens_used)

        impact = EpisodeImpact(
            snr_score=snr_score,
            ihsan_score=ihsan_score,
            snr_ok=snr_ok,
            tokens_used=tokens_used,
            efficiency_score=efficiency,
        )

        episode_hash = _compute_episode_hash(
            sequence,
            timestamp_secs,
            context,
            graph_hash,
            graph_node_count,
            episode_actions,
            impact,
        )
        chain_hash = _compute_chain_hash(self._chain_head, episode_hash)

        episode = Episode(
            sequence=sequence,
            timestamp_secs=timestamp_secs,
            context=context,
            graph_hash=graph_hash,
            graph_node_count=graph_node_count,
            actions=episode_actions,
            impact=impact,
            episode_hash=episode_hash,
            prev_hash=self._chain_head,
            chain_hash=chain_hash,
            context_embedding=context_embedding,
            response_summary=response_summary,
        )

        self._chain_head = chain_hash
        self._next_sequence += 1
        self._episodes.append(episode)
        self._hash_index[episode_hash] = episode
        self._seq_index[sequence] = episode

        if len(self._episodes) > self._max_episodes:
            self._distill()

        logger.debug(
            f"SEL commit: seq={sequence} hash={episode_hash[:16]}... "
            f"SNR={snr_score:.3f} Ihsan={ihsan_score:.3f}"
        )
        return episode_hash

    # ─────────────────────────────────────────────────────────────────────────
    # Retrieve (RIR Algorithm)
    # ─────────────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
        query_embedding: Optional[list[float]] = None,
    ) -> list[Episode]:
        """Retrieve top-K episodes using RIR algorithm."""
        if not self._episodes or top_k <= 0:
            return []

        now_secs = int(time.time())

        scored = []
        for ep in self._episodes:
            recency = math.exp(
                -self._decay_lambda * max(0, now_secs - ep.timestamp_secs)
            )
            importance = ep.impact.importance()
            relevance = self._compute_relevance(ep, query_text, query_embedding)

            score = (
                self._w_recency * recency
                + self._w_importance * importance
                + self._w_relevance * relevance
            )
            scored.append((ep, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [ep for ep, _ in scored[:top_k]]

    def _compute_relevance(
        self,
        episode: Episode,
        query_text: str,
        query_embedding: Optional[list[float]],
    ) -> float:
        """Compute relevance via cosine similarity or keyword overlap."""
        if query_embedding and episode.context_embedding:
            return _cosine_similarity(query_embedding, episode.context_embedding)
        return _keyword_similarity(query_text, episode.context)

    # ─────────────────────────────────────────────────────────────────────────
    # Verification
    # ─────────────────────────────────────────────────────────────────────────

    def verify_chain_integrity(self) -> bool:
        """Verify the entire chain integrity."""
        prev_chain = "genesis"
        for ep in self._episodes:
            if not ep.verify_hash():
                return False
            if ep.prev_hash != prev_chain:
                return False
            expected_chain = _compute_chain_hash(prev_chain, ep.episode_hash)
            if ep.chain_hash != expected_chain:
                return False
            prev_chain = ep.chain_hash
        return prev_chain == self._chain_head

    # ─────────────────────────────────────────────────────────────────────────
    # Accessors
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def chain_head(self) -> str:
        return self._chain_head

    @property
    def sequence(self) -> int:
        return self._next_sequence

    @property
    def distillation_count(self) -> int:
        return self._distillation_count

    def __len__(self) -> int:
        return len(self._episodes)

    def get_by_hash(self, hash_: str) -> Optional[Episode]:
        return self._hash_index.get(hash_)

    def get_by_sequence(self, seq: int) -> Optional[Episode]:
        return self._seq_index.get(seq)

    # ─────────────────────────────────────────────────────────────────────────
    # Distillation
    # ─────────────────────────────────────────────────────────────────────────

    def _distill(self) -> None:
        """Remove low-importance episodes from the oldest half."""
        half = len(self._episodes) // 2
        if half == 0:
            return

        importances = sorted(
            ep.impact.importance() for ep in list(self._episodes)[:half]
        )
        median = importances[len(importances) // 2]

        # Keep only above-median from the oldest half
        kept = deque(
            ep for ep in list(self._episodes)[:half] if ep.impact.importance() >= median
        )
        rest = deque(list(self._episodes)[half:])
        kept.extend(rest)
        self._episodes = kept
        self._distillation_count += 1

        # Rebuild indexes after distillation
        self._hash_index = {ep.episode_hash: ep for ep in self._episodes}
        self._seq_index = {ep.sequence: ep for ep in self._episodes}

    # ─────────────────────────────────────────────────────────────────────────
    # Serialization
    # ─────────────────────────────────────────────────────────────────────────

    def export_jsonl(self, path: str) -> int:
        """Export all episodes to a JSONL file. Returns count written."""
        count = 0
        with open(path, "w", encoding="utf-8") as f:
            for ep in self._episodes:
                record = ep.to_dict()
                record["prev_hash"] = ep.prev_hash
                record["actions"] = [
                    {
                        "action_type": a.action_type,
                        "description": a.description,
                        "success": a.success,
                        "duration_us": a.duration_us,
                    }
                    for a in ep.actions
                ]
                if ep.context_embedding:
                    record["context_embedding"] = ep.context_embedding
                f.write(json.dumps(record, separators=(",", ":")) + "\n")
                count += 1
        return count

    @classmethod
    def import_jsonl(
        cls, path: str, verify: bool = True
    ) -> "SovereignExperienceLedger":
        """Import episodes from JSONL, rebuilding chain state.

        Raises SELIntegrityError if verify=True and chain fails verification.
        """
        sel = cls()
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                actions = [
                    EpisodeAction(
                        action_type=a["action_type"],
                        description=a["description"],
                        success=a["success"],
                        duration_us=a["duration_us"],
                    )
                    for a in d.get("actions", [])
                ]
                impact = EpisodeImpact(
                    snr_score=d["snr_score"],
                    ihsan_score=d["ihsan_score"],
                    snr_ok=d["snr_ok"],
                    tokens_used=d.get("tokens_used", 0),
                    efficiency_score=d.get("efficiency_score", 0.0),
                )
                episode = Episode(
                    sequence=d["sequence"],
                    timestamp_secs=d["timestamp_secs"],
                    context=d["context"],
                    graph_hash=d["graph_hash"],
                    graph_node_count=d["graph_node_count"],
                    actions=actions,
                    impact=impact,
                    episode_hash=d["episode_hash"],
                    prev_hash=d["prev_hash"],
                    chain_hash=d["chain_hash"],
                    context_embedding=d.get("context_embedding"),
                    response_summary=d.get("response_summary"),
                )
                sel._episodes.append(episode)
                sel._hash_index[episode.episode_hash] = episode
                sel._seq_index[episode.sequence] = episode

        if sel._episodes:
            sel._chain_head = sel._episodes[-1].chain_hash
            sel._next_sequence = sel._episodes[-1].sequence + 1

        if verify and not sel.verify_chain_integrity():
            raise SELIntegrityError(f"Chain verification failed on import from {path}")

        return sel


class SELIntegrityError(Exception):
    """Raised when SEL chain verification fails on import."""


# ═══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ═══════════════════════════════════════════════════════════════════════════════


def _cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two float vectors."""
    if len(a) != len(b) or not a:
        return 0.0

    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai * ai for ai in a))
    norm_b = math.sqrt(sum(bi * bi for bi in b))
    denom = norm_a * norm_b

    if denom < 1e-12:
        return 0.0
    return max(0.0, min(1.0, dot / denom))


def _keyword_similarity(a: str, b: str) -> float:
    """Jaccard similarity on words (> 2 chars)."""
    words_a = {
        w.strip(".,!?;:()[]{}'\"")
        for w in a.lower().split()
        if len(w.strip(".,!?;:()[]{}'\"")) > 2
    }
    words_b = {
        w.strip(".,!?;:()[]{}'\"")
        for w in b.lower().split()
        if len(w.strip(".,!?;:()[]{}'\"")) > 2
    }

    if not words_a or not words_b:
        return 0.0

    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union if union > 0 else 0.0
