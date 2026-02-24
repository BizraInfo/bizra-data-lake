"""Autonomous graph-of-thoughts compiler for stereoscopic fragment intelligence."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

from normalizers import CORE8, CONVERSATION_PLATFORMS, detect_provider, parse_file
from normalizers.base import apply_cross_platform_boost, stable_turn_id
from schemas import ConversationTurn, FragmentHint, FragmentKind

_ROLE_WEIGHT = {
    "user": 1.00,
    "assistant": 0.95,
    "tool": 0.80,
    "system": 0.75,
    "unknown": 0.60,
}


@dataclass(frozen=True)
class GiantPrinciple:
    """Foundational principle encoded in the scoring/report protocol."""

    name: str
    principle: str
    implementation: str


GIANTS_PROTOCOL: tuple[GiantPrinciple, ...] = (
    GiantPrinciple(
        name="Claude Shannon",
        principle="Signal-to-noise discipline",
        implementation="Composite SNR scoring and threshold gating",
    ),
    GiantPrinciple(
        name="Ibn al-Haytham",
        principle="Evidence-driven reasoning",
        implementation="Per-node provenance from source hints and providers",
    ),
    GiantPrinciple(
        name="Al-Ghazali",
        principle="Ihsan through rigor",
        implementation="Fail-closed confidence bounds and deterministic normalization",
    ),
    GiantPrinciple(
        name="Isaac Newton",
        principle="Standing on shoulders of giants",
        implementation="Cross-platform corroboration boost at 3+ independent providers",
    ),
)


@dataclass
class SignalNode:
    """A consolidated fragment signal in the graph-of-thoughts."""

    node_id: str
    kind: FragmentKind
    signal: str
    evidence_count: int
    provider_count: int
    providers: list[str]
    source_tags: list[str]
    role_mix: dict[str, int]
    conversations: list[str]
    base_confidence: float
    boosted_confidence: float
    snr_score: float
    first_seen: int
    last_seen: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "signal": self.signal,
            "evidence_count": self.evidence_count,
            "provider_count": self.provider_count,
            "providers": self.providers,
            "source_tags": self.source_tags,
            "role_mix": self.role_mix,
            "conversations": self.conversations,
            "base_confidence": self.base_confidence,
            "boosted_confidence": self.boosted_confidence,
            "snr_score": self.snr_score,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


@dataclass
class SignalEdge:
    """Weighted co-occurrence edge between two signal nodes."""

    source_node_id: str
    target_node_id: str
    co_occurrence_count: int
    shared_conversations: list[str]
    weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_node_id": self.source_node_id,
            "target_node_id": self.target_node_id,
            "co_occurrence_count": self.co_occurrence_count,
            "shared_conversations": self.shared_conversations,
            "weight": self.weight,
        }


@dataclass
class StereoscopicReport:
    """Serializable compilation output used by downstream GENESIS stages."""

    total_turns: int
    total_hints: int
    provider_coverage: list[str]
    cv: float
    snr_threshold: float
    elite_threshold: float
    provider_turn_counts: dict[str, int] = field(default_factory=dict)
    provider_hint_counts: dict[str, int] = field(default_factory=dict)
    provider_parse_failures: dict[str, int] = field(default_factory=dict)
    ingest_input_file_count: int = 0
    unknown_file_count: int = 0
    nodes: list[SignalNode] = field(default_factory=list)
    edges: list[SignalEdge] = field(default_factory=list)
    elite_nodes: list[SignalNode] = field(default_factory=list)
    giants_protocol: tuple[GiantPrinciple, ...] = GIANTS_PROTOCOL

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_turns": self.total_turns,
            "total_hints": self.total_hints,
            "provider_coverage": self.provider_coverage,
            "cv": self.cv,
            "snr_threshold": self.snr_threshold,
            "elite_threshold": self.elite_threshold,
            "provider_turn_counts": dict(sorted(self.provider_turn_counts.items())),
            "provider_hint_counts": dict(sorted(self.provider_hint_counts.items())),
            "provider_parse_failures": dict(sorted(self.provider_parse_failures.items())),
            "ingest_input_file_count": self.ingest_input_file_count,
            "unknown_file_count": self.unknown_file_count,
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "elite_count": len(self.elite_nodes),
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "elite_nodes": [node.to_dict() for node in self.elite_nodes],
            "giants_protocol": [
                {
                    "name": principle.name,
                    "principle": principle.principle,
                    "implementation": principle.implementation,
                }
                for principle in self.giants_protocol
            ],
        }


@dataclass
class _NodeAccumulator:
    """Mutable accumulator while building a final signal node."""

    kind: FragmentKind
    signal_display: str
    providers: set[str] = field(default_factory=set)
    sources: set[str] = field(default_factory=set)
    conversations: set[str] = field(default_factory=set)
    role_mix: dict[str, int] = field(default_factory=dict)
    evidence_count: int = 0
    confidence_sum: float = 0.0
    weighted_confidence_sum: float = 0.0
    first_seen: int = 0
    last_seen: int = 0

    def add(
        self,
        provider: str,
        source: str,
        conversation_id: str,
        role: str,
        confidence: float,
        timestamp: int,
    ) -> None:
        self.providers.add(provider)
        self.sources.add(source)
        self.conversations.add(conversation_id)
        self.role_mix[role] = self.role_mix.get(role, 0) + 1
        self.evidence_count += 1
        self.confidence_sum += confidence
        self.weighted_confidence_sum += confidence * _ROLE_WEIGHT.get(role, _ROLE_WEIGHT["unknown"])

        if timestamp > 0:
            if self.first_seen == 0 or timestamp < self.first_seen:
                self.first_seen = timestamp
            if timestamp > self.last_seen:
                self.last_seen = timestamp


@dataclass
class _EdgeAccumulator:
    """Mutable accumulator while building an edge."""

    co_occurrence_count: int = 0
    shared_conversations: set[str] = field(default_factory=set)

    def add(self, conversation_id: str) -> None:
        self.co_occurrence_count += 1
        self.shared_conversations.add(conversation_id)


def _canonical_signal(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _node_id(kind: FragmentKind, signal: str) -> str:
    return f"{kind.value}:{_canonical_signal(signal)}"


def _compute_cv(provider_coverage: set[str]) -> float:
    """CV computed against conversation platforms (identity-building dialogues)."""
    return round(len(provider_coverage & set(CONVERSATION_PLATFORMS)) / len(CONVERSATION_PLATFORMS), 4)


def _iter_json_files(paths: Iterable[str | Path]) -> Iterable[Path]:
    allowed_suffixes = {".json", ".jsonl"}
    for raw in paths:
        root = Path(raw).expanduser().resolve()
        if not root.exists():
            continue
        if root.is_file() and root.suffix.lower() in allowed_suffixes:
            yield root
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in allowed_suffixes:
                yield path


def _load_payload(path: Path) -> Any | None:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() == ".jsonl":
            rows: list[dict[str, Any]] = []
            for line in raw.splitlines():
                text = line.strip()
                if not text:
                    continue
                try:
                    item = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if isinstance(item, dict):
                    rows.append(item)
            return rows
        return json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return None


class AutonomousSNRGoTEngine:
    """Compile conversation turns into high-SNR stereoscopic signal graphs."""

    def __init__(self, snr_threshold: float = 0.85, elite_threshold: float = 0.95) -> None:
        self.snr_threshold = max(0.0, min(1.0, snr_threshold))
        self.elite_threshold = max(self.snr_threshold, min(1.0, elite_threshold))

    def compile(
        self,
        turns: Iterable[ConversationTurn],
        provider_coverage: Iterable[str] | None = None,
        provider_parse_failures: dict[str, int] | None = None,
        ingest_input_file_count: int = 0,
        unknown_file_count: int = 0,
    ) -> StereoscopicReport:
        coverage = set(provider_coverage or [])
        node_map: dict[str, _NodeAccumulator] = {}
        edge_map: dict[tuple[str, str], _EdgeAccumulator] = {}
        provider_turn_counts: dict[str, int] = {}
        provider_hint_counts: dict[str, int] = {}

        total_turns = 0
        total_hints = 0

        for turn in turns:
            total_turns += 1
            coverage.add(turn.provider)
            provider_turn_counts[turn.provider] = provider_turn_counts.get(turn.provider, 0) + 1

            turn_node_ids: set[str] = set()
            for hint in turn.fragment_hints:
                total_hints += 1
                provider_hint_counts[turn.provider] = provider_hint_counts.get(turn.provider, 0) + 1
                nid = _node_id(hint.kind, hint.signal)
                acc = node_map.get(nid)
                if acc is None:
                    acc = _NodeAccumulator(kind=hint.kind, signal_display=hint.signal.strip())
                    node_map[nid] = acc
                acc.add(
                    provider=turn.provider,
                    source=hint.source,
                    conversation_id=turn.conversation_id,
                    role=turn.role,
                    confidence=hint.confidence,
                    timestamp=turn.timestamp,
                )
                turn_node_ids.add(nid)

            for left, right in combinations(sorted(turn_node_ids), 2):
                edge_key = (left, right)
                edge_acc = edge_map.get(edge_key)
                if edge_acc is None:
                    edge_acc = _EdgeAccumulator()
                    edge_map[edge_key] = edge_acc
                edge_acc.add(turn.conversation_id)

        nodes = self._finalize_nodes(node_map)
        edges = self._finalize_edges(edge_map)
        elite_nodes = [node for node in nodes if node.snr_score >= self.elite_threshold]

        return StereoscopicReport(
            total_turns=total_turns,
            total_hints=total_hints,
            provider_coverage=sorted(coverage),
            cv=_compute_cv(coverage),
            snr_threshold=self.snr_threshold,
            elite_threshold=self.elite_threshold,
            provider_turn_counts=provider_turn_counts,
            provider_hint_counts=provider_hint_counts,
            provider_parse_failures=dict(sorted((provider_parse_failures or {}).items())),
            ingest_input_file_count=ingest_input_file_count,
            unknown_file_count=unknown_file_count,
            nodes=[node for node in nodes if node.snr_score >= self.snr_threshold],
            edges=edges,
            elite_nodes=elite_nodes,
        )

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # Self-Compilation — Conversation Genesis Feedback Loop
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Map from Rust AtomKind names to Python FragmentKind values.
    _ATOM_KIND_TO_FRAGMENT: dict[str, FragmentKind] = {
        "fact": FragmentKind.FACT,
        "preference": FragmentKind.PREFERENCE,
        "pattern": FragmentKind.PATTERN,
        "relationship": FragmentKind.RELATIONSHIP,
        "goal": FragmentKind.GOAL,
        "expertise": FragmentKind.EXPERTISE,
        "context": FragmentKind.EMOTION,
        "principle": FragmentKind.STYLE,
        "temporal": FragmentKind.TEMPORAL,
        "negation": FragmentKind.FACT,  # negation is a factual assertion
    }

    def compile_from_atoms(
        self,
        atoms: list[dict[str, Any]],
        session_id: str = "local",
    ) -> StereoscopicReport:
        """Compile identity signals from pre-extracted memory atoms.

        This accepts atoms already extracted by the Rust memory pipeline
        (via TEACH or auto-extraction) and runs them through the same
        stereoscopic compilation as imported conversations.

        Each atom dict should have:
        - kind: str (fact, preference, goal, expertise, pattern, etc.)
        - content: str
        - confidence: float (0.0-1.0)
        - timestamp: int (unix seconds, optional — defaults to 0)

        This closes the self-compilation loop: the user's own interactions
        with BIZRA generate atoms -> atoms feed the stereoscopic engine ->
        identity signals are compiled without needing external imports.

        Args:
            atoms: List of atom dictionaries from the Rust pipeline.
            session_id: Session identifier for conversation grouping.

        Returns:
            StereoscopicReport with compiled signal nodes and edges.
        """
        turns: list[ConversationTurn] = []
        conversation_id = f"session-{session_id}"

        for i, atom in enumerate(atoms):
            kind_str = str(atom.get("kind", "fact")).lower().strip()
            content = str(atom.get("content", "")).strip()
            confidence = float(atom.get("confidence", 0.0))
            timestamp = int(atom.get("timestamp", 0))

            if not content:
                continue
            confidence = max(0.0, min(1.0, confidence))

            # Map atom kind to Python FragmentKind
            fragment_kind = self._ATOM_KIND_TO_FRAGMENT.get(
                kind_str, FragmentKind.FACT
            )

            # Build a deterministic turn_id from kind + content
            turn_id = stable_turn_id(
                provider="bizra_self",
                raw_id=f"{kind_str}:{content[:64]}",
            )

            hint = FragmentHint(
                kind=fragment_kind,
                signal=content,
                confidence=confidence,
                source=f"bizra_self/{kind_str}",
            )

            turn = ConversationTurn(
                provider="bizra_self",
                conversation_id=conversation_id,
                turn_id=turn_id,
                role="user",
                content=content,
                timestamp=timestamp,
                model="sovereign-node",
                fragment_hints=[hint],
            )
            turns.append(turn)

        return self.compile(
            turns=turns,
            provider_coverage={"bizra_self"},
        )

    def compile_paths(self, paths: Iterable[str | Path]) -> StereoscopicReport:
        turns: list[ConversationTurn] = []
        coverage: set[str] = set()
        provider_parse_failures: dict[str, int] = {}
        ingest_input_file_count = 0
        unknown_file_count = 0

        for path in _iter_json_files(paths):
            ingest_input_file_count += 1
            parsed_turns = parse_file(path)
            if parsed_turns:
                turns.extend(parsed_turns)
                coverage.update(turn.provider for turn in parsed_turns)
                continue

            payload = _load_payload(path)
            if payload is None:
                continue
            detected = detect_provider(payload, source_path=str(path))
            if detected in CORE8:
                coverage.add(detected)
                provider_parse_failures[detected] = provider_parse_failures.get(detected, 0) + 1
            else:
                unknown_file_count += 1

        return self.compile(
            turns=turns,
            provider_coverage=coverage,
            provider_parse_failures=provider_parse_failures,
            ingest_input_file_count=ingest_input_file_count,
            unknown_file_count=unknown_file_count,
        )

    def _finalize_nodes(self, node_map: dict[str, _NodeAccumulator]) -> list[SignalNode]:
        nodes: list[SignalNode] = []
        for node_id, acc in node_map.items():
            if acc.evidence_count == 0:
                continue

            base_confidence = round(acc.confidence_sum / acc.evidence_count, 6)
            boosted_confidence = apply_cross_platform_boost(
                base_confidence,
                acc.providers,
            )
            role_alignment = min(
                1.0,
                max(0.0, acc.weighted_confidence_sum / max(acc.confidence_sum, 1e-9)),
            )

            provider_diversity = min(1.0, len(acc.providers) / 3.0)
            evidence_density = min(1.0, math.log1p(acc.evidence_count) / math.log(6.0))
            provenance_depth = min(1.0, len(acc.sources) / 3.0)

            if acc.first_seen > 0 and acc.last_seen > 0 and acc.last_seen >= acc.first_seen:
                span = acc.last_seen - acc.first_seen
                temporal_stability = min(
                    1.0,
                    math.log1p(span) / math.log(30 * 24 * 3600),
                )
            else:
                temporal_stability = 0.5

            snr_score = round(
                min(
                    1.0,
                    max(
                        0.0,
                        (0.70 * boosted_confidence)
                        + (0.10 * provider_diversity)
                        + (0.07 * evidence_density)
                        + (0.04 * provenance_depth)
                        + (0.04 * role_alignment)
                        + (0.02 * temporal_stability)
                        + 0.10,
                    ),
                ),
                6,
            )

            nodes.append(
                SignalNode(
                    node_id=node_id,
                    kind=acc.kind,
                    signal=acc.signal_display,
                    evidence_count=acc.evidence_count,
                    provider_count=len(acc.providers),
                    providers=sorted(acc.providers),
                    source_tags=sorted(acc.sources),
                    role_mix=dict(sorted(acc.role_mix.items())),
                    conversations=sorted(acc.conversations),
                    base_confidence=base_confidence,
                    boosted_confidence=boosted_confidence,
                    snr_score=snr_score,
                    first_seen=acc.first_seen,
                    last_seen=acc.last_seen,
                )
            )

        return sorted(nodes, key=lambda n: (-n.snr_score, -n.evidence_count, n.node_id))

    def _finalize_edges(self, edge_map: dict[tuple[str, str], _EdgeAccumulator]) -> list[SignalEdge]:
        edges: list[SignalEdge] = []
        for (source_node_id, target_node_id), acc in edge_map.items():
            co_score = min(1.0, math.log1p(acc.co_occurrence_count) / math.log(8.0))
            conversation_score = min(1.0, len(acc.shared_conversations) / 4.0)
            weight = round((0.75 * co_score) + (0.25 * conversation_score), 6)
            edges.append(
                SignalEdge(
                    source_node_id=source_node_id,
                    target_node_id=target_node_id,
                    co_occurrence_count=acc.co_occurrence_count,
                    shared_conversations=sorted(acc.shared_conversations),
                    weight=weight,
                )
            )
        return sorted(edges, key=lambda e: (-e.weight, -e.co_occurrence_count, e.source_node_id, e.target_node_id))
