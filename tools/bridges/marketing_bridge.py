# BIZRA Marketing Bridge v1.0
# Connects Data Lake Intelligence to Marketing Swarms
# Enables: Knowledge retrieval, synergy-informed campaigns, brand-safe content

"""
Marketing Bridge Architecture:

┌─────────────────────────────────────────────────────────────────────────┐
│                      BIZRA DATA LAKE                                    │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────────────┐│
│  │ Hypergraph RAG │  │  KEP Bridge    │  │      ARTE Engine           ││
│  │ (Knowledge)    │  │  (Synergies)   │  │      (SNR/Quality)         ││
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────────────────┘│
└──────────┼───────────────────┼───────────────────┼──────────────────────┘
           │                   │                   │
           ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     MARKETING BRIDGE                                    │
│  ┌────────────────────────────────────────────────────────────────────┐│
│  │  MarketingKnowledge │ CampaignIntelligence │ BrandSafetyValidator  ││
│  └────────────────────────────────────────────────────────────────────┘│
│                                HTTP API                                 │
└─────────────────────────────────────────────────────────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    MARKETING SWARMS (TypeScript)                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐   │
│  │ Historical  │ │ Simulation  │ │  Creative   │ │  Quality        │   │
│  │ Memory Agent│ │   Agent     │ │ Genome Agent│ │  Guardian Agent │   │
│  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import List, Dict, Optional, Any
import threading
import ssl

from bizra_config import INDEXED_PATH, SNR_THRESHOLD, IHSAN_CONSTRAINT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | MARKETING-BRIDGE | %(message)s',
    handlers=[
        logging.FileHandler(INDEXED_PATH / "marketing_bridge.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("MARKETING-BRIDGE")


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class MarketingContext(Enum):
    """Marketing context types for knowledge retrieval."""
    CAMPAIGN_STRATEGY = "campaign_strategy"
    AUDIENCE_INSIGHT = "audience_insight"
    CREATIVE_BRIEF = "creative_brief"
    COMPETITOR_ANALYSIS = "competitor_analysis"
    BRAND_GUIDELINES = "brand_guidelines"
    PERFORMANCE_HISTORY = "performance_history"
    MARKET_TRENDS = "market_trends"


@dataclass
class MarketingQuery:
    """Query from marketing swarm to data lake."""
    query_text: str
    context_type: MarketingContext
    campaign_id: Optional[str] = None
    platform: Optional[str] = None  # google_ads, meta, tiktok, etc.
    max_results: int = 5
    require_synergies: bool = True
    brand_safety_check: bool = True
    metadata: Dict = field(default_factory=dict)


@dataclass
class MarketingInsight:
    """Insight returned to marketing swarm."""
    query: str
    insights: List[Dict]
    synergies: List[Dict]
    brand_safety_score: float
    confidence: float
    recommendations: List[str]
    execution_time: float
    metadata: Dict = field(default_factory=dict)


# ============================================================================
# MARKETING KNOWLEDGE RETRIEVER
# ============================================================================

class MarketingKnowledgeRetriever:
    """
    Retrieves marketing-relevant knowledge from the Data Lake.

    Maps marketing contexts to appropriate retrieval strategies:
    - Campaign Strategy → Multi-hop reasoning through related concepts
    - Audience Insight → Semantic search for behavioral patterns
    - Creative Brief → Graph traversal for brand elements
    - Competitor Analysis → Structural similarity search
    """

    def __init__(self):
        self.orchestrator = None
        self._initialized = False

    async def initialize(self):
        """Initialize connection to BIZRA Orchestrator."""
        if self._initialized:
            return True

        try:
            from bizra_orchestrator import BIZRAOrchestrator
            self.orchestrator = BIZRAOrchestrator(enable_pat=True, enable_kep=True)
            await self.orchestrator.initialize()
            self._initialized = True
            logger.info("Marketing Knowledge Retriever initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            return False

    async def retrieve(self, query: MarketingQuery) -> MarketingInsight:
        """
        Retrieve marketing insights from the knowledge base.
        """
        import time
        start_time = time.time()

        if not self._initialized:
            await self.initialize()

        # Map marketing context to query complexity
        from bizra_orchestrator import BIZRAQuery, QueryComplexity

        complexity_mapping = {
            MarketingContext.CAMPAIGN_STRATEGY: QueryComplexity.COMPLEX,
            MarketingContext.AUDIENCE_INSIGHT: QueryComplexity.MODERATE,
            MarketingContext.CREATIVE_BRIEF: QueryComplexity.MODERATE,
            MarketingContext.COMPETITOR_ANALYSIS: QueryComplexity.RESEARCH,
            MarketingContext.BRAND_GUIDELINES: QueryComplexity.SIMPLE,
            MarketingContext.PERFORMANCE_HISTORY: QueryComplexity.MODERATE,
            MarketingContext.MARKET_TRENDS: QueryComplexity.RESEARCH,
        }

        # Build enriched query
        enriched_query = self._enrich_query(query)

        # Execute through orchestrator
        bizra_query = BIZRAQuery(
            text=enriched_query,
            complexity=complexity_mapping.get(query.context_type, QueryComplexity.MODERATE),
            enable_kep=query.require_synergies,
            min_synergy_strength=0.6,
            max_tokens=2000
        )

        response = await self.orchestrator.query(bizra_query)

        # Extract marketing-relevant insights
        insights = self._extract_marketing_insights(response, query)

        # Format synergies for marketing use
        synergies = self._format_synergies_for_marketing(response.synergies, query)

        # Calculate brand safety score
        brand_safety_score = self._calculate_brand_safety(response, query)

        # Generate recommendations
        recommendations = self._generate_recommendations(response, query, synergies)

        execution_time = time.time() - start_time

        return MarketingInsight(
            query=query.query_text,
            insights=insights,
            synergies=synergies,
            brand_safety_score=brand_safety_score,
            confidence=response.snr_score,
            recommendations=recommendations,
            execution_time=round(execution_time, 3),
            metadata={
                "context_type": query.context_type.value,
                "platform": query.platform,
                "campaign_id": query.campaign_id,
                "ihsan_achieved": response.ihsan_achieved,
                "learning_boost": response.learning_boost
            }
        )

    def _enrich_query(self, query: MarketingQuery) -> str:
        """Enrich query with marketing context."""
        context_prefixes = {
            MarketingContext.CAMPAIGN_STRATEGY: "For marketing campaign strategy: ",
            MarketingContext.AUDIENCE_INSIGHT: "Regarding target audience behavior and preferences: ",
            MarketingContext.CREATIVE_BRIEF: "For creative content development: ",
            MarketingContext.COMPETITOR_ANALYSIS: "Analyzing competitive landscape: ",
            MarketingContext.BRAND_GUIDELINES: "According to brand guidelines and identity: ",
            MarketingContext.PERFORMANCE_HISTORY: "Based on historical campaign performance: ",
            MarketingContext.MARKET_TRENDS: "Current market trends and opportunities: ",
        }

        prefix = context_prefixes.get(query.context_type, "")
        platform_suffix = f" (Platform: {query.platform})" if query.platform else ""

        return f"{prefix}{query.query_text}{platform_suffix}"

    def _extract_marketing_insights(self, response, query: MarketingQuery) -> List[Dict]:
        """Extract marketing-relevant insights from response."""
        insights = []

        for source in response.sources[:query.max_results]:
            insight = {
                "source_id": source.get("doc_id", "unknown"),
                "relevance_score": source.get("score", 0.0),
                "content_preview": source.get("text_preview", "")[:300],
                "marketing_applicability": self._assess_marketing_applicability(
                    source, query.context_type
                )
            }
            insights.append(insight)

        return insights

    def _assess_marketing_applicability(self, source: Dict, context: MarketingContext) -> str:
        """Assess how applicable a source is to marketing context."""
        text = source.get("text_preview", "").lower()

        marketing_keywords = {
            MarketingContext.CAMPAIGN_STRATEGY: ["strategy", "campaign", "launch", "objective", "goal"],
            MarketingContext.AUDIENCE_INSIGHT: ["audience", "customer", "user", "behavior", "preference"],
            MarketingContext.CREATIVE_BRIEF: ["creative", "content", "design", "visual", "message"],
            MarketingContext.COMPETITOR_ANALYSIS: ["competitor", "market", "industry", "benchmark"],
            MarketingContext.BRAND_GUIDELINES: ["brand", "identity", "tone", "voice", "style"],
            MarketingContext.PERFORMANCE_HISTORY: ["performance", "metric", "roi", "conversion", "result"],
            MarketingContext.MARKET_TRENDS: ["trend", "growth", "opportunity", "emerging", "shift"],
        }

        keywords = marketing_keywords.get(context, [])
        matches = sum(1 for kw in keywords if kw in text)

        if matches >= 3:
            return "highly_applicable"
        elif matches >= 1:
            return "moderately_applicable"
        else:
            return "tangentially_related"

    def _format_synergies_for_marketing(self, synergies: List[Dict], query: MarketingQuery) -> List[Dict]:
        """Format synergies in marketing-friendly terms."""
        marketing_synergies = []

        for syn in synergies:
            marketing_syn = {
                "cross_domain_opportunity": f"{syn.get('source_domain', 'unknown')} insights applicable to {syn.get('target_domain', 'unknown')}",
                "strength": syn.get("strength", 0.0),
                "bridging_concepts": syn.get("bridging_concepts", []),
                "marketing_implication": self._derive_marketing_implication(syn, query)
            }
            marketing_synergies.append(marketing_syn)

        return marketing_synergies

    def _derive_marketing_implication(self, synergy: Dict, query: MarketingQuery) -> str:
        """Derive marketing implication from synergy."""
        synergy_type = synergy.get("synergy_type", "unknown")

        implications = {
            "conceptual": "Shared messaging opportunity across channels",
            "methodological": "Transferable campaign tactics",
            "structural": "Consistent creative structure pattern",
            "causal": "Cause-effect relationship for attribution",
            "analogical": "Cross-category positioning opportunity",
            "emergent": "Novel creative combination potential"
        }

        return implications.get(synergy_type, "Potential cross-pollination opportunity")

    def _calculate_brand_safety(self, response, query: MarketingQuery) -> float:
        """Calculate brand safety score."""
        if not query.brand_safety_check:
            return 1.0

        # Base on SNR (high SNR = high quality = safer)
        base_score = response.snr_score

        # Check for problematic patterns in content
        all_text = " ".join(
            s.get("text_preview", "").lower()
            for s in response.sources
        )

        # Negative indicators
        risk_patterns = [
            "controversial", "political", "sensitive", "explicit",
            "violence", "discriminat", "illegal", "offensive"
        ]

        risk_count = sum(1 for p in risk_patterns if p in all_text)
        risk_penalty = risk_count * 0.1

        brand_safety = max(0.0, min(1.0, base_score - risk_penalty))

        return round(brand_safety, 3)

    def _generate_recommendations(
        self,
        response,
        query: MarketingQuery,
        synergies: List[Dict]
    ) -> List[str]:
        """Generate actionable marketing recommendations."""
        recommendations = []

        # Based on context type
        context_recs = {
            MarketingContext.CAMPAIGN_STRATEGY: [
                "Consider multi-channel approach based on knowledge graph connections",
                f"Leverage {len(synergies)} detected cross-domain synergies"
            ],
            MarketingContext.AUDIENCE_INSIGHT: [
                "Segment audiences based on behavioral patterns identified",
                "Personalize messaging using discovered preference clusters"
            ],
            MarketingContext.CREATIVE_BRIEF: [
                "Use bridging concepts for headline ideation",
                "Apply structural patterns to creative templates"
            ],
            MarketingContext.COMPETITOR_ANALYSIS: [
                "Differentiate using unique synergy combinations",
                "Monitor competitor gaps revealed in analysis"
            ],
        }

        recommendations.extend(
            context_recs.get(query.context_type, ["Review insights for campaign optimization"])
        )

        # Based on SNR
        if response.snr_score < 0.7:
            recommendations.append("Low confidence - consider additional research")
        elif response.snr_score > 0.9:
            recommendations.append("High confidence insights - ready for execution")

        # Based on synergies
        if synergies:
            top_synergy = max(synergies, key=lambda s: s.get("strength", 0))
            if top_synergy.get("strength", 0) > 0.8:
                recommendations.append(
                    f"Strong synergy detected: {top_synergy.get('cross_domain_opportunity')}"
                )

        return recommendations[:5]  # Limit to top 5


# ============================================================================
# HTTP API SERVER
# ============================================================================

class MarketingBridgeHandler(BaseHTTPRequestHandler):
    """HTTP handler for Marketing Bridge API."""

    retriever: Optional[MarketingKnowledgeRetriever] = None
    loop: Optional[asyncio.AbstractEventLoop] = None

    def _set_headers(self, status: int = 200, content_type: str = 'application/json'):
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')  # CORS for TypeScript client
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self._set_headers(200)

    def do_GET(self):
        """Status page."""
        self._set_headers(200, 'text/html')
        html = """<!DOCTYPE html>
<html>
<head>
    <title>BIZRA Marketing Bridge</title>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background: #0d1117; color: #c9d1d9; padding: 40px; }
        .container { max-width: 900px; margin: 0 auto; }
        h1 { color: #58a6ff; }
        .endpoint { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 15px; margin: 15px 0; }
        .method { color: #3fb950; font-weight: bold; }
        code { background: #21262d; padding: 2px 6px; border-radius: 4px; }
        pre { background: #21262d; padding: 15px; border-radius: 6px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="container">
        <h1>BIZRA Marketing Bridge API</h1>
        <p>Connects Marketing Swarms to Data Lake Intelligence</p>

        <div class="endpoint">
            <p><span class="method">POST</span> <code>/query</code></p>
            <p>Query the knowledge base for marketing insights</p>
            <pre>{
  "query_text": "What are effective strategies for B2B SaaS?",
  "context_type": "campaign_strategy",
  "platform": "google_ads",
  "max_results": 5,
  "require_synergies": true,
  "brand_safety_check": true
}</pre>
        </div>

        <div class="endpoint">
            <p><span class="method">POST</span> <code>/campaign_intelligence</code></p>
            <p>Get AI-powered campaign recommendations</p>
            <pre>{
  "campaign_id": "camp_123",
  "objective": "lead_generation",
  "target_audience": "enterprise_decision_makers",
  "budget_range": "medium"
}</pre>
        </div>

        <div class="endpoint">
            <p><span class="method">GET</span> <code>/health</code></p>
            <p>Health check for the bridge service</p>
        </div>

        <h2>Context Types</h2>
        <ul>
            <li><code>campaign_strategy</code> - Strategic campaign planning</li>
            <li><code>audience_insight</code> - Target audience analysis</li>
            <li><code>creative_brief</code> - Creative development guidance</li>
            <li><code>competitor_analysis</code> - Competitive intelligence</li>
            <li><code>brand_guidelines</code> - Brand consistency checks</li>
            <li><code>performance_history</code> - Historical performance data</li>
            <li><code>market_trends</code> - Market trend analysis</li>
        </ul>
    </div>
</body>
</html>"""
        self.wfile.write(html.encode())

    def do_POST(self):
        """Handle API requests."""
        # Validate Content-Length
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length <= 0:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "Missing Content-Length"}).encode())
            return

        if content_length > 1024 * 1024:  # 1MB limit
            self._set_headers(413)
            self.wfile.write(json.dumps({"error": "Request too large"}).encode())
            return

        # Parse request
        try:
            post_data = self.rfile.read(content_length)
            request = json.loads(post_data)
        except json.JSONDecodeError:
            self._set_headers(400)
            self.wfile.write(json.dumps({"error": "Invalid JSON"}).encode())
            return

        # Route request
        if self.path == '/query':
            self._handle_query(request)
        elif self.path == '/campaign_intelligence':
            self._handle_campaign_intelligence(request)
        elif self.path == '/health':
            self._handle_health()
        else:
            self._set_headers(404)
            self.wfile.write(json.dumps({"error": "Not found"}).encode())

    def _handle_query(self, request: Dict):
        """Handle knowledge query."""
        try:
            # Build query
            context_type = MarketingContext(request.get("context_type", "campaign_strategy"))

            query = MarketingQuery(
                query_text=request.get("query_text", ""),
                context_type=context_type,
                campaign_id=request.get("campaign_id"),
                platform=request.get("platform"),
                max_results=request.get("max_results", 5),
                require_synergies=request.get("require_synergies", True),
                brand_safety_check=request.get("brand_safety_check", True)
            )

            # Execute query in async loop
            if self.loop and self.retriever:
                future = asyncio.run_coroutine_threadsafe(
                    self.retriever.retrieve(query),
                    self.loop
                )
                result = future.result(timeout=30)

                self._set_headers(200)
                self.wfile.write(json.dumps(asdict(result)).encode())
            else:
                self._set_headers(503)
                self.wfile.write(json.dumps({"error": "Service not initialized"}).encode())

        except Exception as e:
            logger.error(f"Query error: {e}")
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _handle_campaign_intelligence(self, request: Dict):
        """Handle campaign intelligence request."""
        # Build comprehensive query from campaign parameters
        campaign_id = request.get("campaign_id", "unknown")
        objective = request.get("objective", "awareness")
        audience = request.get("target_audience", "general")
        budget = request.get("budget_range", "medium")

        query_text = (
            f"Campaign optimization for {objective} targeting {audience} "
            f"with {budget} budget. What strategies and creative approaches "
            f"have proven effective?"
        )

        query = MarketingQuery(
            query_text=query_text,
            context_type=MarketingContext.CAMPAIGN_STRATEGY,
            campaign_id=campaign_id,
            require_synergies=True,
            brand_safety_check=True
        )

        try:
            if self.loop and self.retriever:
                future = asyncio.run_coroutine_threadsafe(
                    self.retriever.retrieve(query),
                    self.loop
                )
                result = future.result(timeout=30)

                # Enhance with campaign-specific structure
                response = {
                    "campaign_id": campaign_id,
                    "objective": objective,
                    "intelligence": asdict(result),
                    "recommended_actions": result.recommendations,
                    "synergy_opportunities": result.synergies,
                    "brand_safety_clearance": result.brand_safety_score >= 0.8
                }

                self._set_headers(200)
                self.wfile.write(json.dumps(response).encode())
            else:
                self._set_headers(503)
                self.wfile.write(json.dumps({"error": "Service not initialized"}).encode())

        except Exception as e:
            logger.error(f"Campaign intelligence error: {e}")
            self._set_headers(500)
            self.wfile.write(json.dumps({"error": str(e)}).encode())

    def _handle_health(self):
        """Health check endpoint."""
        self._set_headers(200)
        health = {
            "status": "healthy" if self.retriever and self.retriever._initialized else "initializing",
            "service": "marketing_bridge",
            "version": "1.0.0"
        }
        self.wfile.write(json.dumps(health).encode())


# ============================================================================
# SERVER STARTUP
# ============================================================================

async def start_marketing_bridge(port: int = 8444):
    """Start the Marketing Bridge server."""
    print("=" * 70)
    print("BIZRA MARKETING BRIDGE v1.0")
    print("Data Lake Intelligence for Marketing Swarms")
    print("=" * 70)

    # Initialize retriever
    retriever = MarketingKnowledgeRetriever()
    await retriever.initialize()

    # Set up handler class variables
    MarketingBridgeHandler.retriever = retriever
    MarketingBridgeHandler.loop = asyncio.get_event_loop()

    # Create server (localhost only for security)
    server_address = ('127.0.0.1', port)
    httpd = HTTPServer(server_address, MarketingBridgeHandler)

    print(f"\nMarketing Bridge running on http://localhost:{port}")
    print(f"Documentation: http://localhost:{port}/")
    print(f"\nEndpoints:")
    print(f"  POST /query - Knowledge queries")
    print(f"  POST /campaign_intelligence - Campaign recommendations")
    print(f"  GET /health - Health check")
    print("\nPress Ctrl+C to stop")

    # Run server in thread
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    # Keep main loop running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down Marketing Bridge...")
        httpd.shutdown()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BIZRA Marketing Bridge")
    parser.add_argument("--port", type=int, default=8444, help="Port (default: 8444)")
    args = parser.parse_args()

    asyncio.run(start_marketing_bridge(port=args.port))
