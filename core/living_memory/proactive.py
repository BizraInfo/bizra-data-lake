"""
Proactive Information Retriever — Anticipatory Knowledge System

Implements proactive information engineering:
- Predicts what information will be needed
- Pre-fetches relevant knowledge
- Suggests related memories
- Identifies knowledge gaps

Standing on Giants: Shannon (Information Theory) + Attention Mechanisms
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import numpy as np

from core.living_memory.core import (
    LivingMemoryCore,
    MemoryEntry,
    MemoryType,
    MemoryState,
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
    """

    def __init__(
        self,
        memory: LivingMemoryCore,
        llm_fn: Optional[Callable[[str], str]] = None,
        max_suggestions: int = 5,
    ):
        self.memory = memory
        self.llm_fn = llm_fn
        self.max_suggestions = max_suggestions

        # Prediction state
        self._context = PredictionContext()
        self._topic_history: List[str] = []
        self._suggestion_cache: List[ProactiveSuggestion] = []

        # Topic tracking
        self._topic_transitions: Dict[str, Dict[str, int]] = {}  # topic -> next_topic -> count

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

        if topics:
            self._context.active_topics.update(topics)

        if intent:
            self._context.user_intent = intent

    def _extract_topics(self, text: str) -> Set[str]:
        """Extract topic keywords from text."""
        # Simple keyword extraction (could be enhanced with NER)
        words = text.lower().split()
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'can',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after'}

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
                suggestions.append(ProactiveSuggestion(
                    memory=mem,
                    reason=f"Related to predicted topic: {topic}",
                    confidence=prob * 0.8,
                    urgency=0.3,
                ))

        # 2. Retrieve based on recent queries pattern
        if len(self._context.recent_queries) >= 3:
            pattern_query = " ".join(self._context.recent_queries[-3:])
            pattern_memories = await self.memory.retrieve(
                query=pattern_query,
                top_k=2,
                min_score=0.4,
            )
            for mem in pattern_memories:
                suggestions.append(ProactiveSuggestion(
                    memory=mem,
                    reason="Related to recent conversation pattern",
                    confidence=0.7,
                    urgency=0.5,
                ))

        # 3. Check for prospective memories (time-sensitive)
        prospective = await self.memory.retrieve(
            memory_type=MemoryType.PROSPECTIVE,
            top_k=3,
            min_score=0.1,
        )
        for mem in prospective:
            # Check if this prospective memory is becoming urgent
            age_hours = (datetime.now(timezone.utc) - mem.created_at).total_seconds() / 3600
            urgency = min(age_hours / 24, 1.0)  # Increase urgency over time

            suggestions.append(ProactiveSuggestion(
                memory=mem,
                reason="Upcoming goal or plan",
                confidence=mem.confidence,
                urgency=urgency,
            ))

        # Deduplicate and sort
        seen_ids = set()
        unique_suggestions = []
        for s in suggestions:
            if s.memory.id not in seen_ids:
                seen_ids.add(s.memory.id)
                unique_suggestions.append(s)

        # Sort by urgency * confidence
        unique_suggestions.sort(key=lambda x: x.urgency * x.confidence, reverse=True)

        self._suggestion_cache = unique_suggestions[:self.max_suggestions]
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
                gaps.append(KnowledgeGap(
                    topic=topic,
                    description=f"Limited knowledge about: {topic}",
                    suggested_sources=["web search", "documentation", "user input"],
                    priority=0.7,
                ))

        # Check for low-confidence memories in active topics
        for topic in self._context.active_topics:
            memories = await self.memory.retrieve(
                query=topic,
                top_k=5,
                min_score=0.1,
            )
            low_confidence = [m for m in memories if m.confidence < 0.5]
            if low_confidence:
                gaps.append(KnowledgeGap(
                    topic=topic,
                    description=f"Uncertain knowledge about: {topic}",
                    suggested_sources=["verification", "authoritative source"],
                    priority=0.5,
                ))

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
            ).total_seconds() / 60,
            "predicted_topics": self.predict_next_topics(3),
            "cached_suggestions": len(self._suggestion_cache),
        }
