# BIZRA Unified Orchestrator v3.0
# The Crown Jewel: Integrates Hypergraph RAG + ARTE + PAT + KEP + Multi-Modal
# Standing on Giants Protocol: Combining best of symbolic AI, neural retrieval, multi-agent systems,
#                              Knowledge Explosion Point discovery, and multi-modal processing
# Achieves: SNR > 0.99, Graph-of-Thoughts reasoning, Ihsan excellence, Cross-domain synergy, Vision+Audio

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple, Union
import numpy as np

from bizra_config import (
    CHUNKS_TABLE_PATH, CORPUS_TABLE_PATH, GRAPH_PATH, GOLD_PATH,
    SNR_THRESHOLD, IHSAN_CONSTRAINT, INDEXED_PATH,
    VISION_ENABLED, AUDIO_ENABLED, IMAGE_EMBEDDINGS_PATH
)

# Import BIZRA engines
from hypergraph_engine import HypergraphRAGEngine, RetrievalMode, QueryContext
from arte_engine import ARTEEngine, ThoughtType, TensionType
from pat_engine import PATOrchestrator, OllamaBackend, LMStudioBackend, ThinkingMode

# Import KEP Bridge
try:
    from kep_bridge import (
        KEPBridge, KEPResult, SynergyCandidate, CompoundProposal, IhsanCheck
    )
    KEP_AVAILABLE = True
except ImportError:
    KEP_AVAILABLE = False

# Import Discipline Synthesis Engine (47-Discipline Cognitive Topology)
try:
    from discipline_synthesis import DisciplineSynthesisEngine, Generator
    DISCIPLINE_ENGINE_AVAILABLE = True
except ImportError:
    DISCIPLINE_ENGINE_AVAILABLE = False
    DisciplineSynthesisEngine = None

# Import Multi-Modal Engine
try:
    from multimodal_engine import MultiModalEngine, ModalityType, ImageContent, AudioContent
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False

# Import Dual Agentic Bridge
try:
    from dual_agentic_bridge import DualAgenticBridge, ModelCapability, KnowledgeEnhancedRouter
    DUAL_AGENTIC_AVAILABLE = True
except ImportError:
    DUAL_AGENTIC_AVAILABLE = False

# Import SNR Optimizer for Ihsān Achievement
try:
    from snr_optimizer import SNROptimizer, OptimizationResult, OptimizationStrategy
    SNR_OPTIMIZER_AVAILABLE = True
except ImportError:
    SNR_OPTIMIZER_AVAILABLE = False

# Import Sovereign Bridge (High-Performance Caching & Event Bus)
try:
    from sovereign_bridge import (
        SovereignBridge, get_bridge, initialize_bridge,
        BridgeEventType, QueryResultCache, EmbeddingCache, ContextCache
    )
    SOVEREIGN_BRIDGE_AVAILABLE = True
except ImportError:
    SOVEREIGN_BRIDGE_AVAILABLE = False
    SovereignBridge = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | BIZRA | %(message)s',
    handlers=[
        logging.FileHandler(INDEXED_PATH / "bizra_orchestrator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("BIZRA")


class QueryComplexity(Enum):
    """Query complexity levels for adaptive processing."""
    SIMPLE = "simple"         # Direct lookup, single-hop
    MODERATE = "moderate"     # Multi-hop retrieval
    COMPLEX = "complex"       # Full reasoning pipeline
    RESEARCH = "research"     # Deep multi-agent analysis


@dataclass
class BIZRAQuery:
    """Structured query with multi-modal support."""
    text: str
    complexity: QueryComplexity = QueryComplexity.MODERATE
    require_sources: bool = True
    max_tokens: int = 4000
    snr_threshold: float = SNR_THRESHOLD
    enable_kep: bool = True  # Enable KEP synergy detection
    min_synergy_strength: float = 0.6  # Minimum synergy strength
    # Multi-modal fields
    image_path: Optional[str] = None  # Path to image for vision queries
    audio_path: Optional[str] = None  # Path to audio for voice queries
    enable_vision: bool = True  # Enable vision processing
    enable_audio: bool = True  # Enable audio processing
    cross_modal_search: bool = True  # Enable cross-modal retrieval
    metadata: Dict = field(default_factory=dict)


@dataclass
class BIZRAResponse:
    """Complete response with full provenance and multi-modal support."""
    query: str
    answer: str
    snr_score: float
    ihsan_achieved: bool
    sources: List[Dict]
    reasoning_trace: List[str]
    tension_analysis: Dict
    execution_time: float
    # KEP fields
    synergies: List[Dict] = field(default_factory=list)
    compounds: List[Dict] = field(default_factory=list)
    learning_boost: float = 1.0
    # 47-Discipline fields
    discipline_coverage: Optional[float] = None
    generator_strengths: Optional[Dict[str, float]] = None
    # Multi-modal fields
    image_analysis: Optional[str] = None  # Vision model analysis
    audio_transcript: Optional[str] = None  # Audio transcription
    similar_images: List[Dict] = field(default_factory=list)  # Cross-modal image matches
    modality_used: List[str] = field(default_factory=list)  # Which modalities were processed
    metadata: Dict = field(default_factory=dict)


class BIZRAOrchestrator:
    """
    BIZRA Unified Orchestrator v3.0

    The integration layer that combines:
    - Hypergraph RAG: HNSW vector search + graph traversal
    - ARTE Engine: Symbolic-neural bridging + SNR validation
    - PAT Engine: Multi-agent reasoning + LLM generation
    - KEP Bridge: Cross-domain synergy detection + compound discovery
    - Multi-Modal Engine: Vision (CLIP) + Audio (Whisper) processing
    - Dual Agentic Bridge: Connection to multi-model router

    Pipeline:
    1. Query Analysis → Determine complexity + modality
    2. Multi-Modal Processing → Image/audio analysis if present
    3. Context Retrieval → Hypergraph RAG + cross-modal search
    4. Tension Resolution → ARTE symbolic-neural bridge
    5. KEP Processing → Synergy detection + compound discovery
    6. Response Generation → PAT agents (if complex) + vision context
    7. Quality Validation → SNR + Ihsan constraints
    8. Response Assembly → Final output with multi-modal insights
    """

    def __init__(
        self,
        enable_pat: bool = True,
        enable_kep: bool = True,
        enable_multimodal: bool = True,
        enable_discipline: bool = True,
        ollama_model: str = "liquid/lfm2.5-1.2b"
    ):
        logger.info("Initializing BIZRA Unified Orchestrator v3.0")

        # Initialize engines
        self.hypergraph = HypergraphRAGEngine()
        self.arte = ARTEEngine()

        self.pat_enabled = enable_pat
        self.pat: Optional[PATOrchestrator] = None

        if enable_pat:
            try:
                # Try LM Studio first, fall back to Ollama
                lm_backend = LMStudioBackend()
                # Check synchronously by creating event loop if needed
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Use LM Studio backend (it will be checked during first call)
                self.pat = PATOrchestrator(lm_backend, model=ollama_model)
                logger.info(f"PAT Engine initialized with LM Studio (model: {ollama_model})")
            except Exception as e:
                # Fallback to Ollama
                try:
                    backend = OllamaBackend()
                    self.pat = PATOrchestrator(backend, model=ollama_model)
                    logger.info("PAT Engine initialized with Ollama (fallback)")
                except Exception as e2:
                    logger.warning(f"PAT Engine unavailable: {e2}")
                    self.pat_enabled = False

        # Initialize KEP Bridge
        self.kep_enabled = enable_kep and KEP_AVAILABLE
        self.kep: Optional[KEPBridge] = None

        if enable_kep and KEP_AVAILABLE:
            try:
                self.kep = KEPBridge()
                logger.info("KEP Bridge initialized")
            except Exception as e:
                logger.warning(f"KEP Bridge unavailable: {e}")
                self.kep_enabled = False
        elif enable_kep and not KEP_AVAILABLE:
            logger.warning("KEP Bridge module not available")

        # Initialize 47-Discipline Synthesis Engine
        self.discipline_enabled = enable_discipline and DISCIPLINE_ENGINE_AVAILABLE
        self.discipline_engine: Optional[DisciplineSynthesisEngine] = None

        if enable_discipline and DISCIPLINE_ENGINE_AVAILABLE:
            try:
                self.discipline_engine = DisciplineSynthesisEngine()
                logger.info("47-Discipline Synthesis Engine initialized (4-Generator Theory)")
            except Exception as e:
                logger.warning(f"Discipline Engine unavailable: {e}")
                self.discipline_enabled = False
        elif enable_discipline and not DISCIPLINE_ENGINE_AVAILABLE:
            logger.warning("Discipline Synthesis Engine module not available")

        # Initialize Multi-Modal Engine
        self.multimodal_enabled = enable_multimodal and MULTIMODAL_AVAILABLE
        self.multimodal: Optional[MultiModalEngine] = None

        if enable_multimodal and MULTIMODAL_AVAILABLE:
            try:
                self.multimodal = MultiModalEngine()
                logger.info("Multi-Modal Engine created (lazy initialization)")
            except Exception as e:
                logger.warning(f"Multi-Modal Engine unavailable: {e}")
                self.multimodal_enabled = False
        elif enable_multimodal and not MULTIMODAL_AVAILABLE:
            logger.warning("Multi-Modal Engine module not available")

        # Initialize Dual Agentic Bridge
        self.dual_agentic_enabled = DUAL_AGENTIC_AVAILABLE
        self.dual_agentic: Optional[DualAgenticBridge] = None

        if DUAL_AGENTIC_AVAILABLE:
            try:
                self.dual_agentic = DualAgenticBridge()
                logger.info("Dual Agentic Bridge created")
            except Exception as e:
                logger.warning(f"Dual Agentic Bridge unavailable: {e}")
                self.dual_agentic_enabled = False

        # Initialize Sovereign Bridge (High-Performance Caching Layer)
        self.sovereign_bridge_enabled = SOVEREIGN_BRIDGE_AVAILABLE
        self.sovereign_bridge: Optional[SovereignBridge] = None

        if SOVEREIGN_BRIDGE_AVAILABLE:
            try:
                self.sovereign_bridge = get_bridge()
                logger.info("✨ Sovereign Bridge connected (B+ Tree + Bloom Filter + LRU Cache)")
            except Exception as e:
                logger.warning(f"Sovereign Bridge unavailable: {e}")
                self.sovereign_bridge_enabled = False
        else:
            logger.info("Sovereign Bridge module not available")

        # Initialize hypergraph index
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize all engines including multi-modal and Sovereign Bridge."""
        if self._initialized:
            return True

        logger.info("Initializing BIZRA engines...")

        # Initialize Sovereign Bridge first (caching layer)
        if self.sovereign_bridge_enabled and self.sovereign_bridge:
            try:
                await self.sovereign_bridge.initialize()
                logger.info("✨ Sovereign Bridge initialized (high-performance caching active)")
            except Exception as e:
                logger.warning(f"Sovereign Bridge initialization failed: {e}")

        # Initialize Hypergraph RAG
        if not self.hypergraph.initialize():
            logger.error("Failed to initialize Hypergraph RAG")
            return False

        # Check ARTE integrity
        health = self.arte.check_system_integrity()
        if health["integration"]["tension_level"] == "critical":
            logger.warning("ARTE shows critical tension level")

        # Initialize Multi-Modal Engine (lazy load models)
        if self.multimodal_enabled and self.multimodal:
            try:
                self.multimodal.initialize()
                logger.info("Multi-Modal Engine initialized")
            except Exception as e:
                logger.warning(f"Multi-Modal initialization failed: {e}")

        # Check Dual Agentic availability
        if self.dual_agentic_enabled and self.dual_agentic:
            available = await self.dual_agentic.check_availability()
            if available:
                logger.info("Dual Agentic System connected")
            else:
                logger.info("Dual Agentic System not available (will use fallbacks)")

        self._initialized = True
        logger.info("BIZRA Orchestrator v3.0 initialized successfully")

        return True

    async def query(self, query: BIZRAQuery) -> BIZRAResponse:
        """
        Process a query through the full BIZRA pipeline with multi-modal support.

        Steps:
        1. Initialize if needed
        2. Multi-modal preprocessing (image/audio analysis)
        3. Retrieve context via Hypergraph RAG + cross-modal search
        4. Apply ARTE symbolic-neural bridging
        5. KEP Processing - Synergy detection + compound discovery
        6. Generate response (direct or via PAT agents) with vision context
        7. Validate quality with KEP Ihsan check
        8. Assemble response with multi-modal insights
        """
        start_time = time.time()
        reasoning_trace = []
        kep_result: Optional[KEPResult] = None
        modalities_used = ["text"]
        image_analysis = None
        audio_transcript = None
        similar_images = []

        # Step 1: Initialize
        if not self._initialized:
            await self.initialize()

        reasoning_trace.append(f"Query: {query.text}")
        reasoning_trace.append(f"Complexity: {query.complexity.value}")

        # Step 2: Multi-modal preprocessing
        if query.image_path and query.enable_vision and self.multimodal_enabled:
            image_analysis, similar_images = await self._process_image_query(
                query.image_path, query.text
            )
            if image_analysis:
                modalities_used.append("vision")
                reasoning_trace.append(f"Vision: Processed image with CLIP + analysis")

        if query.audio_path and query.enable_audio and self.multimodal_enabled:
            audio_transcript = await self._process_audio_query(query.audio_path)
            if audio_transcript:
                modalities_used.append("audio")
                reasoning_trace.append(f"Audio: Transcribed with Whisper")

        # Step 2: Context Retrieval via Hypergraph RAG
        retrieval_mode = self._select_retrieval_mode(query.complexity)
        reasoning_trace.append(f"Retrieval mode: {retrieval_mode.value}")

        context = self.hypergraph.retrieve(
            query=query.text,
            mode=retrieval_mode,
            k=10 if query.complexity in [QueryComplexity.COMPLEX, QueryComplexity.RESEARCH] else 5,
            max_hops=3 if query.complexity == QueryComplexity.RESEARCH else 2,
            snr_threshold=query.snr_threshold,
            max_tokens=query.max_tokens
        )

        reasoning_trace.extend(context.reasoning_trace)

        # Step 3: ARTE Symbolic-Neural Resolution
        symbolic_facts = self._extract_symbolic_facts(context)
        neural_results = self._extract_neural_results(context)

        arte_result = self.arte.resolve_tension(
            query=query.text,
            symbolic_facts=symbolic_facts,
            neural_results=neural_results,
            query_embedding=context.query_embedding,
            context_embeddings=np.array([
                self.hypergraph.index.get_chunk_embedding(r.chunk_id)
                for r in context.results
                if self.hypergraph.index.get_chunk_embedding(r.chunk_id) is not None
            ]) if context.results else np.zeros((1, 384))
        )

        reasoning_trace.append(f"ARTE SNR: {arte_result['snr_score']}")
        reasoning_trace.append(f"Tension: {arte_result['tension_analysis']['type']}")

        # Step 4: KEP Processing (synergy detection + compound discovery)
        if query.enable_kep and self.kep_enabled and self.kep:
            # Prepare retrieval results for KEP
            kep_retrieval_results = [
                {
                    "doc_id": r.doc_id,
                    "chunk_id": r.chunk_id,
                    "text": r.text,
                    "score": r.score,
                    "embedding": self.hypergraph.index.get_chunk_embedding(r.chunk_id).tolist()
                    if self.hypergraph.index.get_chunk_embedding(r.chunk_id) is not None
                    else None
                }
                for r in context.results
            ]

            kep_result = await self.kep.process(
                query=query.text,
                retrieval_results=kep_retrieval_results,
                query_embedding=context.query_embedding,
                snr_score=arte_result['snr_score'],
                min_synergy_strength=query.min_synergy_strength,
                max_compounds=3 if query.complexity in [QueryComplexity.COMPLEX, QueryComplexity.RESEARCH] else 1
            )

            reasoning_trace.append(f"KEP synergies detected: {len(kep_result.synergies)}")
            reasoning_trace.append(f"KEP compounds discovered: {len(kep_result.compounds)}")
            reasoning_trace.append(f"KEP Ihsan check: {'PASSED' if kep_result.ihsan_check.passed else 'FAILED'}")
            reasoning_trace.append(f"KEP learning boost: {kep_result.learning_boost:.2f}x")

        # Step 5: Response Generation
        if query.complexity in [QueryComplexity.COMPLEX, QueryComplexity.RESEARCH] and self.pat_enabled:
            # Use PAT agents for complex queries (with KEP context if available)
            answer = await self._generate_with_pat(query, context, arte_result, kep_result)
            reasoning_trace.append("Response generated via PAT agents")
        else:
            # Direct answer assembly for simpler queries
            answer = self._assemble_direct_answer(query, context, arte_result, kep_result)
            reasoning_trace.append("Direct response assembled")

        # Step 6: Final SNR validation (with KEP Ihsan if available)
        final_snr = self._calculate_final_snr(
            arte_result['snr_score'],
            context.snr_score,
            len(context.results)
        )

        # Use KEP Ihsan check if available, otherwise fall back to SNR threshold
        if kep_result and kep_result.ihsan_check:
            ihsan_achieved = kep_result.ihsan_check.passed
            final_snr = kep_result.ihsan_check.score  # Use KEP's comprehensive score
        else:
            ihsan_achieved = final_snr >= IHSAN_CONSTRAINT

        if not ihsan_achieved and query.snr_threshold >= IHSAN_CONSTRAINT:
            reasoning_trace.append(f"Warning: SNR {final_snr:.3f} below Ihsan threshold {IHSAN_CONSTRAINT}")

        # Step 7: Build response
        execution_time = time.time() - start_time

        sources = [
            {
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "score": r.score,
                "text_preview": r.text[:200] + "..." if len(r.text) > 200 else r.text
            }
            for r in context.results[:5]
        ]

        # Extract KEP synergies and compounds if available
        synergies_data = []
        compounds_data = []
        learning_boost = 1.0

        if kep_result:
            synergies_data = [
                {
                    "source_domain": s.source_domain,
                    "target_domain": s.target_domain,
                    "synergy_type": s.synergy_type.value,
                    "strength": s.strength,
                    "bridging_concepts": s.bridging_concepts
                }
                for s in kep_result.synergies
            ]
            compounds_data = [
                {
                    "compound_type": c.compound_type.value,
                    "hypothesis": c.hypothesis,
                    "confidence": c.confidence,
                    "implications": c.implications
                }
                for c in kep_result.compounds
            ]
            learning_boost = kep_result.learning_boost

        # Get discipline coverage if engine available
        discipline_coverage = None
        generator_strengths = None
        if self.discipline_enabled and self.discipline_engine:
            try:
                report = self.discipline_engine.generate_report()
                discipline_coverage = report.overall_coverage
                generator_strengths = report.generator_strength
            except Exception as e:
                logger.warning(f"Discipline coverage unavailable: {e}")

        return BIZRAResponse(
            query=query.text,
            answer=answer,
            snr_score=final_snr,
            ihsan_achieved=ihsan_achieved,
            sources=sources if query.require_sources else [],
            reasoning_trace=reasoning_trace,
            tension_analysis=arte_result['tension_analysis'],
            execution_time=round(execution_time, 3),
            synergies=synergies_data,
            compounds=compounds_data,
            learning_boost=learning_boost,
            discipline_coverage=discipline_coverage,
            generator_strengths=generator_strengths,
            image_analysis=image_analysis,
            audio_transcript=audio_transcript,
            similar_images=similar_images,
            modality_used=modalities_used,
            metadata={
                "complexity": query.complexity.value,
                "retrieval_mode": retrieval_mode.value,
                "chunks_retrieved": len(context.results),
                "tokens_estimated": context.total_tokens_est,
                "kep_enabled": kep_result is not None,
                "discipline_enabled": self.discipline_enabled,
                "multimodal_enabled": self.multimodal_enabled,
                "vision_used": "vision" in modalities_used,
                "audio_used": "audio" in modalities_used
            }
        )

    def _select_retrieval_mode(self, complexity: QueryComplexity) -> RetrievalMode:
        """Select retrieval mode based on query complexity."""
        mapping = {
            QueryComplexity.SIMPLE: RetrievalMode.SEMANTIC,
            QueryComplexity.MODERATE: RetrievalMode.HYBRID,
            QueryComplexity.COMPLEX: RetrievalMode.MULTI_HOP,
            QueryComplexity.RESEARCH: RetrievalMode.MULTI_HOP
        }
        return mapping.get(complexity, RetrievalMode.HYBRID)

    def _extract_symbolic_facts(self, context: QueryContext) -> List[Dict]:
        """Extract symbolic facts from retrieval context."""
        facts = []
        seen_docs = set()

        for result in context.results:
            if result.doc_id not in seen_docs:
                seen_docs.add(result.doc_id)
                facts.append({
                    "doc_id": result.doc_id,
                    "text": result.text[:500],
                    "graph_distance": result.graph_distance
                })

        return facts

    def _extract_neural_results(self, context: QueryContext) -> List[Dict]:
        """Extract neural search results."""
        return [
            {
                "chunk_id": r.chunk_id,
                "doc_id": r.doc_id,
                "score": r.score,
                "text": r.text
            }
            for r in context.results
        ]

    async def _generate_with_pat(
        self,
        query: BIZRAQuery,
        context: QueryContext,
        arte_result: Dict,
        kep_result: Optional[KEPResult] = None
    ) -> str:
        """Generate response using PAT multi-agent system with KEP context."""
        if not self.pat:
            return self._assemble_direct_answer(query, context, arte_result, kep_result)

        # Build context for PAT
        retrieved_context = "\n\n".join([
            f"[Source {i+1}] (score: {r.score:.2f})\n{r.text}"
            for i, r in enumerate(context.results[:5])
        ])

        pat_context = {
            "retrieved_context": retrieved_context,
            "snr_score": arte_result['snr_score'],
            "tension_type": arte_result['tension_analysis']['type'],
            "sources": [r.doc_id for r in context.results[:5]]
        }

        # Add KEP synergies and compounds to PAT context
        if kep_result:
            pat_context["synergies"] = [
                {
                    "domains": f"{s.source_domain} <-> {s.target_domain}",
                    "type": s.synergy_type.value,
                    "strength": s.strength,
                    "bridges": s.bridging_concepts
                }
                for s in kep_result.synergies
            ]
            pat_context["compounds"] = [
                {
                    "type": c.compound_type.value,
                    "hypothesis": c.hypothesis,
                    "confidence": c.confidence
                }
                for c in kep_result.compounds
            ]
            pat_context["learning_boost"] = kep_result.learning_boost

        try:
            pat_result = await self.pat.process_task(
                task=query.text,
                context=pat_context,
                agents_to_use=["researcher", "analyst"] if query.complexity == QueryComplexity.COMPLEX else ["researcher", "analyst", "creator"],
                require_guardian_approval=True
            )

            return pat_result.get("synthesis", self._assemble_direct_answer(query, context, arte_result, kep_result))

        except Exception as e:
            logger.error(f"PAT generation failed: {e}")
            return self._assemble_direct_answer(query, context, arte_result, kep_result)

    def _assemble_direct_answer(
        self,
        query: BIZRAQuery,
        context: QueryContext,
        arte_result: Dict,
        kep_result: Optional[KEPResult] = None
    ) -> str:
        """Assemble direct answer from retrieved context with KEP insights."""
        if not context.results:
            return "No relevant information found in the knowledge base."

        # Build answer from top results
        parts = []

        # Introduction
        parts.append(f"Based on {len(context.results)} retrieved sources (SNR: {arte_result['snr_score']:.2f}):\n")

        # Key information from top results
        for i, result in enumerate(context.results[:3]):
            parts.append(f"\n**Source {i+1}** (relevance: {result.score:.2f}):")
            # Extract key sentences
            text = result.text.strip()
            if len(text) > 300:
                text = text[:300] + "..."
            parts.append(text)

        # Add ARTE insights
        tension = arte_result['tension_analysis']
        if tension['type'] != 'coherent':
            parts.append(f"\n**Note:** {tension['type'].replace('_', ' ').title()} detected. Recommendations: {', '.join(tension.get('recommendations', ['Review sources']))}")

        # Add KEP synergies and compounds
        if kep_result and kep_result.synergies:
            parts.append("\n\n**Cross-Domain Synergies Detected:**")
            for syn in kep_result.synergies[:3]:
                parts.append(f"  - {syn.source_domain} <-> {syn.target_domain} ({syn.synergy_type.value}, strength: {syn.strength:.2f})")
                if syn.bridging_concepts:
                    parts.append(f"    Bridging concepts: {', '.join(syn.bridging_concepts[:3])}")

        if kep_result and kep_result.compounds:
            parts.append("\n**Compound Discoveries:**")
            for comp in kep_result.compounds[:2]:
                parts.append(f"  - [{comp.compound_type.value.upper()}] {comp.hypothesis[:150]}...")
                parts.append(f"    Confidence: {comp.confidence:.2f}")

        if kep_result:
            parts.append(f"\n*Learning boost: {kep_result.learning_boost:.2f}x*")

        return "\n".join(parts)

    def _calculate_final_snr(
        self,
        arte_snr: float,
        retrieval_snr: float,
        result_count: int
    ) -> float:
        """Calculate final SNR combining all factors."""
        if result_count == 0:
            return 0.0

        # Weighted combination
        base_snr = (arte_snr * 0.6 + retrieval_snr * 0.4)

        # Penalty for very few results
        if result_count < 3:
            base_snr *= 0.8

        # Bonus for high result count with good scores
        elif result_count >= 5 and retrieval_snr > 0.7:
            base_snr = min(base_snr * 1.1, 0.99)

        return round(base_snr, 4)

    # =========================================================================
    # MULTI-MODAL PROCESSING METHODS
    # =========================================================================

    async def _process_image_query(
        self,
        image_path: str,
        query_text: str
    ) -> Tuple[Optional[str], List[Dict]]:
        """
        Process an image query using vision models.

        Args:
            image_path: Path to image file
            query_text: Query text for context

        Returns:
            Tuple of (image_analysis, similar_images)
        """
        image_analysis = None
        similar_images = []

        if not self.multimodal or not self.multimodal.image_processor:
            return None, []

        try:
            # Process image
            image_content = self.multimodal.image_processor.process_image(
                image_path, extract_ocr=True, describe=False
            )

            # Get image analysis via Dual Agentic or LLaVA
            if self.dual_agentic_enabled and self.dual_agentic:
                image_analysis = await self.dual_agentic.analyze_image(
                    image_path,
                    f"Analyze this image in the context of: {query_text}"
                )
            else:
                # Use local LLaVA
                image_analysis = await self.multimodal.image_processor.describe_image_async(
                    image_path, use_local=True
                )

            # Find similar images in the knowledge base
            if image_content.embedding is not None:
                similar_images = await self._find_similar_images(image_content.embedding)

        except Exception as e:
            logger.warning(f"Image processing failed: {e}")

        return image_analysis, similar_images

    async def _process_audio_query(self, audio_path: str) -> Optional[str]:
        """
        Process an audio query using Whisper transcription.

        Args:
            audio_path: Path to audio file

        Returns:
            Transcribed text
        """
        if not self.multimodal or not self.multimodal.audio_processor:
            return None

        try:
            # Use Dual Agentic if available
            if self.dual_agentic_enabled and self.dual_agentic:
                transcript = await self.dual_agentic.transcribe_audio(audio_path)
                return transcript

            # Otherwise use local Whisper
            audio_content = self.multimodal.audio_processor.process_audio(audio_path)
            return audio_content.transcript if audio_content else None

        except Exception as e:
            logger.warning(f"Audio processing failed: {e}")
            return None

    async def _find_similar_images(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict]:
        """
        Find similar images in the knowledge base using CLIP embeddings.

        Args:
            query_embedding: CLIP embedding to search with
            k: Number of results to return

        Returns:
            List of similar images with scores
        """
        similar_images = []

        try:
            import pandas as pd

            # Load image embeddings if they exist
            image_chunks_path = IMAGE_EMBEDDINGS_PATH / "image_chunks.parquet"
            if not image_chunks_path.exists():
                return []

            df_images = pd.read_parquet(image_chunks_path)

            if len(df_images) == 0:
                return []

            # Calculate cosine similarity
            for _, row in df_images.iterrows():
                embedding = np.array(row['embedding'])
                similarity = np.dot(query_embedding, embedding)

                similar_images.append({
                    'chunk_id': row['chunk_id'],
                    'file_path': row.get('file_path', ''),
                    'text': row.get('chunk_text', ''),
                    'score': float(similarity)
                })

            # Sort by similarity and return top k
            similar_images.sort(key=lambda x: x['score'], reverse=True)
            return similar_images[:k]

        except Exception as e:
            logger.warning(f"Similar image search failed: {e}")
            return []

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        status = {
            "initialized": self._initialized,
            "version": "3.0",
            "engines": {
                "hypergraph_rag": "ready" if self._initialized else "not_initialized",
                "arte": "ready",
                "pat": "ready" if self.pat_enabled else "disabled",
                "kep": "ready" if self.kep_enabled else "disabled",
                "multimodal": "ready" if self.multimodal_enabled else "disabled",
                "dual_agentic": "ready" if self.dual_agentic_enabled else "disabled"
            }
        }

        # Add ARTE health check
        if self._initialized:
            arte_health = self.arte.check_system_integrity()
            status["arte_health"] = {
                "symbolic_nodes": arte_health["symbolic"]["nodes"],
                "neural_chunks": arte_health["neural"]["chunks"],
                "integration_snr": arte_health["integration"]["snr_score"],
                "tension_level": arte_health["integration"]["tension_level"]
            }

        # Add KEP status
        if self.kep_enabled and self.kep:
            kep_status = self.kep.get_status()
            status["kep_status"] = kep_status

        # Add Multi-Modal status
        if self.multimodal_enabled and self.multimodal:
            status["multimodal_status"] = self.multimodal.get_status()

        # Add Dual Agentic status
        if self.dual_agentic_enabled and self.dual_agentic:
            status["dual_agentic_status"] = self.dual_agentic.get_status()

        return status


async def main():
    """Demonstration of BIZRA Unified Orchestrator v3.0 with multi-modal support."""
    print("=" * 70)
    print("BIZRA UNIFIED ORCHESTRATOR v3.0")
    print("Hypergraph RAG + ARTE + PAT + KEP + Multi-Modal Integration")
    print("Standing on Giants Protocol: Vision + Voice + Cross-Domain Synergy")
    print("=" * 70)

    # Initialize orchestrator with all capabilities
    orchestrator = BIZRAOrchestrator(
        enable_pat=True,
        enable_kep=True,
        enable_multimodal=True
    )

    # Initialize
    if not await orchestrator.initialize():
        print("Failed to initialize orchestrator")
        return

    # Get system status
    status = orchestrator.get_system_status()
    print(f"\nSystem Status (v{status.get('version', '3.0')}):")
    print(f"  Hypergraph RAG: {status['engines']['hypergraph_rag']}")
    print(f"  ARTE Engine: {status['engines']['arte']}")
    print(f"  PAT Engine: {status['engines']['pat']}")
    print(f"  KEP Bridge: {status['engines']['kep']}")
    print(f"  Multi-Modal: {status['engines'].get('multimodal', 'unknown')}")
    print(f"  Dual Agentic: {status['engines'].get('dual_agentic', 'unknown')}")

    if "arte_health" in status:
        health = status["arte_health"]
        print(f"\nARTE Health:")
        print(f"  Symbolic Nodes: {health['symbolic_nodes']}")
        print(f"  Neural Chunks: {health['neural_chunks']}")
        print(f"  Integration SNR: {health['integration_snr']}")
        print(f"  Tension Level: {health['tension_level']}")

    if "kep_status" in status:
        kep = status["kep_status"]
        print(f"\nKEP Bridge Status:")
        print(f"  Initialized: {kep.get('initialized', False)}")
        print(f"  Synergy Weights: {kep.get('synergy_weights', 0)}")
        print(f"  Learning History: {kep.get('learning_history', 0)}")

    # Test queries
    test_queries = [
        BIZRAQuery(
            text="How does the BIZRA data lake process incoming files?",
            complexity=QueryComplexity.MODERATE
        ),
        BIZRAQuery(
            text="Explain the architecture of the embedding generation pipeline",
            complexity=QueryComplexity.COMPLEX
        ),
        BIZRAQuery(
            text="What is the relationship between ARTE and the hypergraph?",
            complexity=QueryComplexity.RESEARCH
        )
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query.text}")
        print(f"Complexity: {query.complexity.value}")
        print("-" * 70)

        response = await orchestrator.query(query)

        print(f"\nSNR Score: {response.snr_score}")
        print(f"Ihsan Achieved: {response.ihsan_achieved}")
        print(f"Execution Time: {response.execution_time}s")
        print(f"Sources Retrieved: {len(response.sources)}")
        print(f"Learning Boost: {response.learning_boost:.2f}x")

        print(f"\nReasoning Trace:")
        for step in response.reasoning_trace[:8]:
            print(f"  - {step}")

        # Display KEP synergies
        if response.synergies:
            print(f"\nKEP Synergies ({len(response.synergies)} detected):")
            for syn in response.synergies[:3]:
                print(f"  - {syn['source_domain']} <-> {syn['target_domain']}")
                print(f"    Type: {syn['synergy_type']}, Strength: {syn['strength']:.2f}")
                if syn.get('bridging_concepts'):
                    print(f"    Bridges: {', '.join(syn['bridging_concepts'][:3])}")

        # Display KEP compounds
        if response.compounds:
            print(f"\nKEP Compound Discoveries ({len(response.compounds)}):")
            for comp in response.compounds[:2]:
                print(f"  - [{comp['compound_type'].upper()}]")
                print(f"    Hypothesis: {comp['hypothesis'][:100]}...")
                print(f"    Confidence: {comp['confidence']:.2f}")

        print(f"\nAnswer Preview:")
        answer_preview = response.answer[:500] + "..." if len(response.answer) > 500 else response.answer
        print(answer_preview)

        if response.sources:
            print(f"\nTop Sources:")
            for src in response.sources[:3]:
                print(f"  - {src['doc_id']} (score: {src['score']:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
