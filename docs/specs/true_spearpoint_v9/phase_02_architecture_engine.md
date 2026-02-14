# Phase 02: Architecture Engine Upgrades

## Status: SPEC
## Depends On: Existing `core/benchmark/moe_router.py`, `core/inference/gateway.py`
## Produces: FederatedMoEUpgrade, MIRASMemory, SequentialAttention integration

---

## 1. Context

The existing `MoERouter` in `core/benchmark/moe_router.py` provides:
- `ExpertTier` enum (NANO → FRONTIER) with cost/speed data
- `ComplexityClassifier` (keyword-based)
- `SequentialAttention` (OMP-equivalent subset selection)
- `federated_dispatch()` for parallel tier execution
- `ExpertStats` with Expertise-per-Token (EPT) metric

The existing `InferenceGateway` in `core/inference/gateway.py` provides:
- Tiered backend selection (LM Studio → Ollama → Cloud)
- Health checks and failover

### Gaps to Fill

1. **Z-Scorer routing** — ML-based complexity scoring (replaces keyword heuristic)
2. **MIRAS Memory** — Multi-tier memory preventing attention sink in long tasks
3. **SequentialAttention ↔ Gateway wiring** — Use SA for context compression
4. **Zero-Failure Training Protocol** — Init selection + gradient monitoring

---

## 2. Z-Scorer: ML-Based Complexity Classifier

### Purpose
Replace the keyword-based `ComplexityClassifier` with a trainable scorer that
improves routing accuracy from feedback. Lightweight — no external ML deps.

### Pseudocode

```
MODULE core.benchmark.z_scorer

IMPORT ComplexityClassifier, QueryComplexity FROM core.benchmark.moe_router
IMPORT ExpertTier FROM core.benchmark.moe_router

DATACLASS ZScoreFeatures:
    token_count: int
    question_depth: int        # Nested clause count
    domain_specificity: float  # 0.0 generic → 1.0 highly specialized
    reasoning_steps: int       # Estimated multi-step count
    tool_requirements: int     # Number of distinct tools needed
    context_length: int        # Total context tokens

DATACLASS ZScoreResult:
    complexity: QueryComplexity
    confidence: float          # 0.0 → 1.0
    recommended_tier: ExpertTier
    features: ZScoreFeatures
    reasoning: str

CLASS ZScorer:
    """
    Trainable complexity classifier. Starts with heuristic rules,
    learns from routing outcomes via exponential moving average.

    No ML framework dependency — uses simple weight vectors.
    """

    INIT():
        # Feature weights (learned from feedback)
        self._weights = {
            'token_count': 0.15,
            'question_depth': 0.25,
            'domain_specificity': 0.20,
            'reasoning_steps': 0.25,
            'tool_requirements': 0.10,
            'context_length': 0.05,
        }
        # Complexity thresholds
        self._thresholds = [0.2, 0.4, 0.6, 0.8]  # TRIVIAL→FRONTIER boundaries
        # Outcome history for learning
        self._history: list[tuple[ZScoreFeatures, ExpertTier, float]] = []
        self._learning_rate = 0.01
        self._fallback = ComplexityClassifier()

    score(query: str, context: dict = None) -> ZScoreResult:
        """Score query complexity and recommend tier."""
        features = self._extract_features(query, context)
        raw_score = self._compute_score(features)
        complexity = self._score_to_complexity(raw_score)
        tier = self._complexity_to_tier(complexity)

        RETURN ZScoreResult(
            complexity=complexity,
            confidence=self._compute_confidence(features),
            recommended_tier=tier,
            features=features,
            reasoning=f"score={raw_score:.3f}, tier={tier.name}",
        )

    _extract_features(query: str, context: dict) -> ZScoreFeatures:
        """Extract scoring features from query text."""
        tokens = query.split()
        # Question depth: count subordinate clauses
        depth_markers = ['because', 'although', 'if', 'when', 'while',
                         'considering', 'given that', 'assuming']
        depth = SUM(1 FOR m IN depth_markers IF m IN query.lower())

        # Reasoning steps: count sequential indicators
        step_markers = ['first', 'then', 'next', 'finally', 'step',
                        'compare', 'analyze', 'evaluate', 'synthesize']
        steps = SUM(1 FOR m IN step_markers IF m IN query.lower())

        # Domain specificity: ratio of uncommon words (> 8 chars)
        long_words = [w FOR w IN tokens IF len(w) > 8]
        specificity = len(long_words) / max(len(tokens), 1)

        # Tool requirements: detect tool-related keywords
        tool_markers = ['search', 'calculate', 'browse', 'execute',
                        'code', 'file', 'database', 'api']
        tools = SUM(1 FOR m IN tool_markers IF m IN query.lower())

        RETURN ZScoreFeatures(
            token_count=len(tokens),
            question_depth=depth,
            domain_specificity=min(specificity, 1.0),
            reasoning_steps=steps,
            tool_requirements=tools,
            context_length=len(context.get('history', '')) IF context ELSE 0,
        )

    _compute_score(features: ZScoreFeatures) -> float:
        """Weighted sum of normalized features."""
        normalized = {
            'token_count': min(features.token_count / 500, 1.0),
            'question_depth': min(features.question_depth / 5, 1.0),
            'domain_specificity': features.domain_specificity,
            'reasoning_steps': min(features.reasoning_steps / 5, 1.0),
            'tool_requirements': min(features.tool_requirements / 4, 1.0),
            'context_length': min(features.context_length / 10000, 1.0),
        }
        RETURN SUM(self._weights[k] * normalized[k] FOR k IN self._weights)

    update(features: ZScoreFeatures, actual_tier: ExpertTier, quality: float):
        """Update weights from routing outcome feedback."""
        self._history.append((features, actual_tier, quality))

        # Only update after sufficient history
        IF len(self._history) < 10:
            RETURN

        # EMA update: shift weights toward features that predicted success
        # (quality > 0.9) and away from features that predicted failure
        recent = self._history[-20:]
        FOR feature_name IN self._weights:
            successes = [getattr(h[0], feature_name)
                         FOR h IN recent IF h[2] > 0.9]
            failures = [getattr(h[0], feature_name)
                        FOR h IN recent IF h[2] < 0.5]

            IF successes AND failures:
                signal = MEAN(successes) - MEAN(failures)
                self._weights[feature_name] += self._learning_rate * signal

        # Normalize weights to sum to 1.0
        total = SUM(self._weights.values())
        IF total > 0:
            self._weights = {k: v/total FOR k, v IN self._weights.items()}
```

---

## 3. MIRAS Memory — Multi-Tier Retrieval

### Purpose
Prevent catastrophic forgetting in long-horizon tasks (SWE-bench, multi-step
reasoning). Three tiers: short-term (LRU), long-term (compressed embeddings),
episodic (structured traces).

### Design Constraint
**Zero new dependencies.** Uses stdlib `collections.OrderedDict` for LRU,
existing `core/living_memory/core.py` for persistence, and the existing
FAISS/vector infrastructure only if available.

### Pseudocode

```
MODULE core.benchmark.miras_memory

IMPORT OrderedDict FROM collections
IMPORT time, hashlib, json

DATACLASS MemoryEntry:
    key: str
    content: str
    timestamp: float
    relevance: float       # Last retrieval relevance score
    access_count: int
    tier: str              # "short_term" | "long_term" | "episodic"
    metadata: dict

DATACLASS RetrievalResult:
    entries: list[MemoryEntry]
    sources: dict[str, int]  # tier -> count
    total_retrieved: int
    dedup_removed: int

CLASS MIRASMemory:
    """
    Memory Integration for Reduced Attention Sink.

    Three tiers prevent context overflow in long-running agents:
    1. Short-term: LRU cache of recent interactions (fast, bounded)
    2. Long-term: Compressed summaries (larger, slower)
    3. Episodic: Structured action-result traces (graph traversal)

    No new dependencies — uses stdlib + existing living_memory if available.
    """

    INIT(short_term_capacity: int = 100,
         long_term_capacity: int = 10000,
         compression_threshold: int = 50):
        # Short-term: LRU with bounded size
        self._short_term = OrderedDict()  # key -> MemoryEntry
        self._st_capacity = short_term_capacity

        # Long-term: dict with eviction by relevance
        self._long_term: dict[str, MemoryEntry] = {}
        self._lt_capacity = long_term_capacity

        # Episodic: list of structured traces
        self._episodic: list[MemoryEntry] = []

        # Compression threshold: move to long-term after N accesses
        self._compression_threshold = compression_threshold

    store(content: str, metadata: dict = None) -> str:
        """Store new information in short-term memory."""
        key = hashlib.sha256(content.encode()).hexdigest()[:16]

        entry = MemoryEntry(
            key=key,
            content=content,
            timestamp=time.time(),
            relevance=1.0,
            access_count=0,
            tier="short_term",
            metadata=metadata OR {},
        )

        # Add to short-term
        self._short_term[key] = entry
        self._short_term.move_to_end(key)

        # Evict oldest if over capacity
        WHILE len(self._short_term) > self._st_capacity:
            evicted_key, evicted = self._short_term.popitem(last=False)
            # Promote to long-term if accessed enough
            IF evicted.access_count >= self._compression_threshold:
                self._promote_to_long_term(evicted)

        RETURN key

    store_episodic(action: str, result: str, context: dict = None):
        """Store a structured action-result trace."""
        key = hashlib.sha256(f"{action}:{result}".encode()).hexdigest()[:16]
        entry = MemoryEntry(
            key=key,
            content=json.dumps({"action": action, "result": result}),
            timestamp=time.time(),
            relevance=1.0,
            access_count=0,
            tier="episodic",
            metadata=context OR {},
        )
        self._episodic.append(entry)

    retrieve(query: str, k: int = 10) -> RetrievalResult:
        """Retrieve relevant memories across all tiers."""
        query_lower = query.lower()
        candidates: list[tuple[float, MemoryEntry]] = []

        # 1. Short-term: recency-weighted keyword match
        FOR entry IN self._short_term.values():
            score = self._keyword_relevance(query_lower, entry.content.lower())
            recency = 1.0 / (1.0 + time.time() - entry.timestamp)
            combined = 0.6 * score + 0.4 * recency
            IF combined > 0.1:
                candidates.append((combined, entry))
                entry.access_count += 1

        # 2. Long-term: keyword match (no recency bonus)
        FOR entry IN self._long_term.values():
            score = self._keyword_relevance(query_lower, entry.content.lower())
            IF score > 0.1:
                candidates.append((score * 0.9, entry))  # Slight discount
                entry.access_count += 1

        # 3. Episodic: structured match on action/result
        FOR entry IN self._episodic:
            score = self._keyword_relevance(query_lower, entry.content.lower())
            IF score > 0.1:
                candidates.append((score * 0.85, entry))

        # Sort by score, deduplicate, take top k
        candidates.sort(key=LAMBDA x: x[0], reverse=True)
        seen_keys = set()
        results = []
        dedup_removed = 0

        FOR score, entry IN candidates:
            IF entry.key IN seen_keys:
                dedup_removed += 1
                CONTINUE
            seen_keys.add(entry.key)
            results.append(entry)
            IF len(results) >= k:
                BREAK

        sources = {"short_term": 0, "long_term": 0, "episodic": 0}
        FOR entry IN results:
            sources[entry.tier] += 1

        RETURN RetrievalResult(
            entries=results,
            sources=sources,
            total_retrieved=len(results),
            dedup_removed=dedup_removed,
        )

    _keyword_relevance(query: str, content: str) -> float:
        """Simple keyword overlap relevance (no ML deps)."""
        query_words = set(query.split())
        content_words = set(content.split())
        IF NOT query_words:
            RETURN 0.0
        overlap = len(query_words & content_words)
        RETURN overlap / len(query_words)

    _promote_to_long_term(entry: MemoryEntry):
        """Move entry from short-term to long-term."""
        entry.tier = "long_term"
        self._long_term[entry.key] = entry

        # Evict lowest-relevance if over capacity
        IF len(self._long_term) > self._lt_capacity:
            worst_key = MIN(
                self._long_term,
                key=LAMBDA k: self._long_term[k].relevance
            )
            DEL self._long_term[worst_key]

    consolidate():
        """Merge related memories in long-term to reduce redundancy."""
        # Group by metadata similarity, merge content
        # This is a periodic maintenance operation
        PASS  # V9.1 — not critical for initial release

    get_stats() -> dict:
        RETURN {
            "short_term_count": len(self._short_term),
            "long_term_count": len(self._long_term),
            "episodic_count": len(self._episodic),
            "total": (len(self._short_term) +
                      len(self._long_term) +
                      len(self._episodic)),
        }
```

---

## 4. Zero-Failure Training Protocol

### Purpose
Guarantee training stability for rapid iteration in the dominance loop.
Select optimal initialization strategy via quick trials.

### Pseudocode

```
MODULE core.benchmark.zero_failure_trainer

DATACLASS InitStrategy:
    name: str               # "xavier_uniform", "kaiming_normal", etc.
    steps_to_threshold: int # Steps to reach 10% loss reduction
    final_loss: float       # Loss after trial

DATACLASS TrainingGuardrails:
    gradient_clip: float = 1.0
    warmup_steps: int = 100
    residual_connections: bool = True
    gradient_variance_max: float = 10.0  # Abort if exceeded

CLASS ZeroFailureTrainer:
    """
    Pre-flight init selection + adaptive warmup.
    Ensures no wasted compute from divergent training runs.
    """

    INIT(trial_steps: int = 100,
         loss_reduction_target: float = 0.10):
        self._trial_steps = trial_steps
        self._loss_target = loss_reduction_target

    select_initialization(
        model_fn: callable,
        data_sample: list,
        candidates: list[str] = None,
    ) -> InitStrategy:
        """
        Run quick trials with different init strategies.
        Select the one that reaches target loss reduction fastest.
        """
        candidates = candidates OR [
            "xavier_uniform", "xavier_normal",
            "kaiming_uniform", "kaiming_normal",
        ]

        results: list[InitStrategy] = []

        FOR init_name IN candidates:
            # Quick trial (simulation — actual training is backend-specific)
            initial_loss = self._simulate_trial(init_name, data_sample)
            results.append(InitStrategy(
                name=init_name,
                steps_to_threshold=initial_loss['steps'],
                final_loss=initial_loss['final'],
            ))

        # Select fastest convergence
        best = MIN(results, key=LAMBDA r: r.steps_to_threshold)
        RETURN best

    compute_warmup(gradient_variance: float) -> int:
        """
        Adaptive warmup: higher variance = more warmup steps.
        Linear scaling from 50 (low var) to 500 (high var).
        """
        IF gradient_variance < 1.0:
            RETURN 50
        IF gradient_variance > 10.0:
            RETURN 500
        # Linear interpolation
        RETURN int(50 + (gradient_variance - 1.0) / 9.0 * 450)

    _simulate_trial(init_name: str, data: list) -> dict:
        """
        Simulated trial for init selection.
        In production, this calls actual model training for N steps.
        """
        # Simulation: different inits have different convergence profiles
        profiles = {
            "xavier_uniform": {"steps": 80, "final": 0.35},
            "xavier_normal": {"steps": 90, "final": 0.33},
            "kaiming_uniform": {"steps": 70, "final": 0.38},
            "kaiming_normal": {"steps": 75, "final": 0.36},
        }
        RETURN profiles.get(init_name, {"steps": 100, "final": 0.40})
```

---

## 5. TDD Anchors

```
TEST test_z_scorer_basic_classification:
    scorer = ZScorer()
    result = scorer.score("What is 2+2?")
    ASSERT result.complexity == QueryComplexity.TRIVIAL
    ASSERT result.confidence > 0.5

TEST test_z_scorer_complex_query:
    scorer = ZScorer()
    query = ("Analyze the implications of quantum entanglement on "
             "cryptographic protocols, considering both theoretical "
             "and practical aspects, then evaluate against existing "
             "standards and synthesize recommendations")
    result = scorer.score(query)
    ASSERT result.complexity IN [QueryComplexity.COMPLEX, QueryComplexity.FRONTIER]

TEST test_z_scorer_learns_from_feedback:
    scorer = ZScorer()
    initial_weights = dict(scorer._weights)
    FOR i IN range(20):
        features = ZScoreFeatures(token_count=100, ...)
        scorer.update(features, ExpertTier.FRONTIER, quality=0.95)
    ASSERT scorer._weights != initial_weights  # Weights shifted

TEST test_miras_store_and_retrieve:
    mem = MIRASMemory(short_term_capacity=10)
    mem.store("Python is a programming language")
    mem.store("Rust is a systems language")
    result = mem.retrieve("programming language", k=5)
    ASSERT result.total_retrieved >= 1
    ASSERT "Python" IN result.entries[0].content

TEST test_miras_lru_eviction:
    mem = MIRASMemory(short_term_capacity=3)
    mem.store("entry 1")
    mem.store("entry 2")
    mem.store("entry 3")
    mem.store("entry 4")  # Should evict entry 1
    ASSERT mem.get_stats()["short_term_count"] == 3

TEST test_miras_episodic:
    mem = MIRASMemory()
    mem.store_episodic("run tests", "all 37 passed")
    result = mem.retrieve("test results", k=5)
    ASSERT result.sources["episodic"] >= 1

TEST test_miras_deduplication:
    mem = MIRASMemory()
    mem.store("duplicate content")
    mem.store("duplicate content")  # Same hash = same key
    result = mem.retrieve("duplicate", k=10)
    ASSERT result.dedup_removed == 0  # Same key overwrites in dict

TEST test_zero_failure_selects_best_init:
    trainer = ZeroFailureTrainer()
    result = trainer.select_initialization(None, [])
    ASSERT result.name IN ["xavier_uniform", "xavier_normal",
                           "kaiming_uniform", "kaiming_normal"]
    ASSERT result.steps_to_threshold > 0

TEST test_warmup_scales_with_variance:
    trainer = ZeroFailureTrainer()
    ASSERT trainer.compute_warmup(0.5) == 50   # Low variance
    ASSERT trainer.compute_warmup(10.0) == 500  # High variance
    ASSERT 50 < trainer.compute_warmup(5.0) < 500  # Mid variance
```

---

## 6. File Deliverables

| File | Lines | Purpose |
|------|-------|---------|
| `core/benchmark/z_scorer.py` | ~180 | ML-based complexity scoring |
| `core/benchmark/miras_memory.py` | ~200 | Multi-tier memory system |
| `core/benchmark/zero_failure_trainer.py` | ~100 | Init selection + warmup |
| `tests/core/benchmark/test_z_scorer.py` | ~120 | Z-Scorer tests |
| `tests/core/benchmark/test_miras_memory.py` | ~120 | MIRAS tests |
| `tests/core/benchmark/test_zero_failure_trainer.py` | ~60 | Trainer tests |
