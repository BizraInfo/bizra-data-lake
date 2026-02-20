"""
Proactive Information Retriever — Anticipatory Knowledge System

Implements proactive information engineering:
- Predicts what information will be needed
- Pre-fetches relevant knowledge
- Suggests related memories
- Identifies knowledge gaps

Standing on Giants: Shannon (Information Theory) + Attention Mechanisms
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from core.living_memory.core import (
    LivingMemoryCore,
    MemoryEntry,
    MemoryType,
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionContext:
    """Context for proactive prediction."""

    current_query: Optional[str] = None
    recent_queries: List[str] = field(default_factory=list)
    active_topics: Set[str] = field(default_factory=set)
    user_intent: Optional[str] = None
    session_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ProactiveSuggestion:
    """A proactive knowledge suggestion."""

    memory: MemoryEntry
    reason: str
    confidence: float
    urgency: float  # How time-sensitive is this suggestion


@dataclass
class KnowledgeGap:
    """Identified gap in knowledge."""

    topic: str
    description: str
    suggested_sources: List[str]
    priority: float


class ProactiveRetriever:
    """
    Anticipatory knowledge retrieval system.

    Proactively identifies and surfaces relevant information
    before it's explicitly requested.

    Phase 46: Optional HMM engine for cognitive state prediction.
    Gated by BIZRA_PHASE46_HMM_ENABLED env var.
    """

    def __init__(
        self,
        memory: LivingMemoryCore,
        llm_fn: Optional[Callable[[str], str]] = None,
        max_suggestions: int = 5,
        hmm_engine: Optional[Any] = None,
    ):
        self.memory = memory
        self.llm_fn = llm_fn
        self.max_suggestions = max_suggestions

        # Prediction state
        self._context = PredictionContext()
        self._topic_history: List[str] = []
        self._suggestion_cache: List[ProactiveSuggestion] = []

        # Topic tracking
        self._topic_transitions: Dict[str, Dict[str, int]] = (
            {}
        )  # topic -> next_topic -> count

        # Phase 46: HMM cognitive state prediction
        self._hmm_engine: Optional[Any] = hmm_engine
        self._hmm_enabled = self._init_hmm()

        # Phase 47.1: Canary routing for HMM observations
        from core.rollout.canary import CanaryRouter
        self._canary = CanaryRouter()

        # Phase 49.8: HMM caller-gate isolation (single-caller mode by default)
        self._hmm_gate: Optional[Any] = None
        try:
            from core.rollout.hmm_gate import HMMCallerGate
            self._hmm_gate = HMMCallerGate()
        except Exception:
            pass  # gate unavailable — direct observe only

        # Phase 49.8: Metrics for proactive HMM observations
        from core.rollout.metrics import get_shared_metrics
        self._p46_metrics = get_shared_metrics()

    def _init_hmm(self) -> bool:
        """Lazily initialise the HMM engine if Phase 46 is enabled."""
        import os

        if os.getenv("BIZRA_PHASE46_HMM_ENABLED", "0").lower() not in {
            "1",
            "true",
            "yes",
        }:
            return False
        if self._hmm_engine is not None:
            return True
        try:
            from core.prediction import HMMEngine

            self._hmm_engine = HMMEngine()
            logger.info("ProactiveRetriever: HMM engine initialised (Phase 46)")
            return True
        except Exception as exc:
            logger.warning("ProactiveRetriever: HMM init failed: %s", exc)
            return False

    def predict_state(self) -> Optional[Any]:
        """Return current HMM prediction, or None if HMM unavailable."""
        if self._hmm_engine is None:
            return None
        try:
            return self._hmm_engine.predict_next()
        except Exception as exc:
            logger.warning("ProactiveRetriever: HMM predict failed: %s", exc)
            return None

    # --- HMM observation mapping ---
    _TOPIC_TO_SYMBOL: Dict[str, str] = {
        "search": "search",
        "find": "search",
        "query": "search",
        "edit": "edit",
        "modify": "edit",
        "change": "edit",
        "fix": "edit",
        "review": "review",
        "check": "review",
        "test": "test",
        "deploy": "deploy",
        "organize": "organize",
        "sort": "organize",
        "chat": "chat",
        "open": "file_open",
        "read": "file_open",
        "save": "file_save",
        "write": "file_save",
        "compile": "compile",
        "build": "compile",
        "run": "compile",
    }

    def _observe_hmm(self, topics: Set[str]) -> None:
        """Feed extracted topics to the HMM as observation symbols.

        Phase 47.1: Each observation is gated through CanaryRouter using the
        symbol as request_key for deterministic percent-based routing.
        Phase 49.8: Observations pass through HMMCallerGate for caller isolation.
        """
        if self._hmm_engine is None:
            return
        for topic in topics:
            symbol = self._TOPIC_TO_SYMBOL.get(topic)
            if symbol and self._canary.should_route("hmm", symbol):
                try:
                    # Gate through HMMCallerGate if available (single-caller isolation)
                    if self._hmm_gate is not None:
                        result = self._hmm_gate.observe(symbol, "proactive")
                        if result is None:
                            logger.debug("HMM observation rejected by caller gate: %s", symbol)
                            continue
                    self._hmm_engine.observe(symbol)
                    self._p46_metrics.inc("hmm_proactive_observations")
                except Exception as exc:
                    self._p46_metrics.inc("hmm_proactive_errors")
                    logger.debug("HMM observe failed for symbol %s: %s", symbol, exc)

    def update_context(
        self,
        query: Optional[str] = None,
        topics: Optional[Set[str]] = None,
        intent: Optional[str] = None,
    ) -> None:
        """Update prediction context with new information."""
        if query:
            self._context.current_query = query
            self._context.recent_queries.append(query)
            if len(self._context.recent_queries) > 20:
                self._context.recent_queries.pop(0)

            # Extract topics from query
            extracted = self._extract_topics(query)
            self._context.active_topics.update(extracted)

            # Track topic transitions for prediction
            for topic in extracted:
                self._update_topic_transitions(topic)

            # Phase 46: Feed topics to HMM
            if self._hmm_enabled:
                self._observe_hmm(extracted)

        if topics:
            self._context.active_topics.update(topics)

        if intent:
            self._context.user_intent = intent

    def _extract_topics(self, text: str) -> Set[str]:
        """Extract topic keywords from text."""
        # Simple keyword extraction (could be enhanced with NER)
        words = text.lower().split()
        stopwords = {
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
        }

        topics = {w for w in words if len(w) > 3 and w not in stopwords}
        return topics

    def _update_topic_transitions(self, current_topic: str) -> None:
        """Track topic transitions for prediction."""
        if self._topic_history:
            prev_topic = self._topic_history[-1]
            if prev_topic not in self._topic_transitions:
                self._topic_transitions[prev_topic] = {}
            self._topic_transitions[prev_topic][current_topic] = (
                self._topic_transitions[prev_topic].get(current_topic, 0) + 1
            )

        self._topic_history.append(current_topic)
        if len(self._topic_history) > 100:
            self._topic_history.pop(0)

    def predict_next_topics(self, top_k: int = 5) -> List[Tuple[str, float]]:
        """Predict likely next topics based on history."""
        predictions = []

        for topic in self._context.active_topics:
            if topic in self._topic_transitions:
                next_topics = self._topic_transitions[topic]
                total = sum(next_topics.values())
                for next_topic, count in next_topics.items():
                    prob = count / total
                    predictions.append((next_topic, prob))

        # Aggregate and sort
        topic_scores: Dict[str, float] = {}
        for topic, prob in predictions:
            topic_scores[topic] = topic_scores.get(topic, 0) + prob

        sorted_topics = sorted(topic_scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_topics[:top_k]

    async def get_proactive_suggestions(self) -> List[ProactiveSuggestion]:
        """
        Generate proactive knowledge suggestions.

        Combines:
        - Predicted next topics
        - Related to recent queries
        - Time-sensitive knowledge
        """
        suggestions = []

        # 1. Retrieve based on predicted topics
        predicted_topics = self.predict_next_topics()
        for topic, prob in predicted_topics[:3]:
            memories = await self.memory.retrieve(
                query=topic,
                top_k=2,
                min_score=0.3,
            )
            for mem in memories:
                suggestions.append(
                    ProactiveSuggestion(
                        memory=mem,
                        reason=f"Related to predicted topic: {topic}",
                        confidence=prob * 0.8,
                        urgency=0.3,
                    )
                )

        # 2. Retrieve based on recent queries pattern
        if len(self._context.recent_queries) >= 3:
            pattern_query = " ".join(self._context.recent_queries[-3:])
            pattern_memories = await self.memory.retrieve(
                query=pattern_query,
                top_k=2,
                min_score=0.4,
            )
            for mem in pattern_memories:
                suggestions.append(
                    ProactiveSuggestion(
                        memory=mem,
                        reason="Related to recent conversation pattern",
                        confidence=0.7,
                        urgency=0.5,
                    )
                )

        # 3. Check for prospective memories (time-sensitive)
        prospective = await self.memory.retrieve(
            memory_type=MemoryType.PROSPECTIVE,
            top_k=3,
            min_score=0.1,
        )
        for mem in prospective:
            # Check if this prospective memory is becoming urgent
            age_hours = (
                datetime.now(timezone.utc) - mem.created_at
            ).total_seconds() / 3600
            urgency = min(age_hours / 24, 1.0)  # Increase urgency over time

            suggestions.append(
                ProactiveSuggestion(
                    memory=mem,
                    reason="Upcoming goal or plan",
                    confidence=mem.confidence,
                    urgency=urgency,
                )
            )

        # Deduplicate and sort
        seen_ids = set()
        unique_suggestions = []
        for s in suggestions:
            if s.memory.id not in seen_ids:
                seen_ids.add(s.memory.id)
                unique_suggestions.append(s)

        # Sort by urgency * confidence
        unique_suggestions.sort(key=lambda x: x.urgency * x.confidence, reverse=True)

        self._suggestion_cache = unique_suggestions[: self.max_suggestions]
        return self._suggestion_cache

    async def identify_knowledge_gaps(self) -> List[KnowledgeGap]:
        """
        Identify gaps in current knowledge.

        Analyzes:
        - Failed retrievals
        - Low-confidence topics
        - Missing connections
        """
        gaps = []

        # Check for topics with no strong memories
        for topic in self._context.active_topics:
            memories = await self.memory.retrieve(
                query=topic,
                top_k=3,
                min_score=0.5,
            )
            if len(memories) < 2:
                gaps.append(
                    KnowledgeGap(
                        topic=topic,
                        description=f"Limited knowledge about: {topic}",
                        suggested_sources=["web search", "documentation", "user input"],
                        priority=0.7,
                    )
                )

        # Check for low-confidence memories in active topics
        for topic in self._context.active_topics:
            memories = await self.memory.retrieve(
                query=topic,
                top_k=5,
                min_score=0.1,
            )
            low_confidence = [m for m in memories if m.confidence < 0.5]
            if low_confidence:
                gaps.append(
                    KnowledgeGap(
                        topic=topic,
                        description=f"Uncertain knowledge about: {topic}",
                        suggested_sources=["verification", "authoritative source"],
                        priority=0.5,
                    )
                )

        # Sort by priority
        gaps.sort(key=lambda x: x.priority, reverse=True)
        return gaps

    async def pre_fetch(self, predicted_queries: List[str]) -> int:
        """
        Pre-fetch knowledge for predicted queries.

        Warms up memory for anticipated needs.
        """
        fetched = 0

        for query in predicted_queries:
            # Retrieve to update access times (warming cache)
            memories = await self.memory.retrieve(
                query=query,
                top_k=5,
                min_score=0.2,
            )
            fetched += len(memories)

        return fetched

    def get_context_summary(self) -> Dict[str, Any]:
        """Get summary of current prediction context."""
        return {
            "current_query": self._context.current_query,
            "recent_queries_count": len(self._context.recent_queries),
            "active_topics": list(self._context.active_topics),
            "user_intent": self._context.user_intent,
            "session_duration_minutes": (
                datetime.now(timezone.utc) - self._context.session_start
            ).total_seconds()
            / 60,
            "predicted_topics": self.predict_next_topics(3),
            "cached_suggestions": len(self._suggestion_cache),
        }
