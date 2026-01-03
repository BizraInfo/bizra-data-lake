import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from core.meta_prompt.models import MetaPromptRequest, MetaPromptResponse, MetaPromptResult
from core.meta_prompt.agents import (
    OntologyArchitect, LanguageMaster, LearningCore, PromptEngineer,
    KnowledgeHarvester, ReasoningEngine, MetaLearner, OutputSynthesizer, EthicsGuardian
)

class MetaPromptEngine:
    """Orchestrates the multi-agent meta-prompt workflow.

    Defaults:
    - Per-agent capabilities come from DEFAULT_CAPABILITIES below.
    - Deployments may override capabilities by passing a mapping into the constructor.
    """

    DEFAULT_CAPABILITIES: Dict[str, List[str]] = {
        "OntologyArchitect": [
            "Hierarchical concept mapping",
            "Dynamic taxonomy updates",
            "Cross-domain relationship identification",
        ],
        "LanguageMaster": [
            "Context-aware interpretation",
            "Sentiment and intent analysis",
            "Multi-lingual understanding and generation",
        ],
        "LearningCore": [
            "Transfer learning",
            "Few-shot learning",
            "Continual learning",
        ],
        "PromptEngineer": [
            "Template-based construction",
            "Context-aware refinement",
            "Creativity algorithms",
        ],
        "KnowledgeHarvester": [
            "Multi-source data ingestion",
            "Information synthesis",
            "Reliability assessment",
        ],
        "ReasoningEngine": [
            "Deductive reasoning",
            "Inductive reasoning",
            "Abductive reasoning",
            "Bayesian inference",
        ],
        "MetaLearner": [
            "Performance analysis",
            "Strategy optimization",
            "User preference learning",
        ],
        "OutputSynthesizer": [
            "Natural language generator",
            "Data visualizer",
            "Multi-modal formatter",
        ],
        "EthicsGuardian": [
            "Bias detection",
            "Privacy protection",
            "Ethical decision making",
        ],
    }

    def __init__(self, *, capabilities: Optional[Dict[str, List[str]]] = None, wisdom: Optional[Any] = None):
        caps = dict(self.DEFAULT_CAPABILITIES)
        if capabilities:
            for k, v in capabilities.items():
                if isinstance(k, str) and isinstance(v, list):
                    caps[k] = [str(x) for x in v if str(x).strip()]

        self.wisdom = wisdom
        self.agents = {
            "OntologyArchitect": OntologyArchitect(name="OntologyArchitect", role="Structure", capabilities=caps.get("OntologyArchitect", [])),
            "LanguageMaster": LanguageMaster(name="LanguageMaster", role="NLP", capabilities=caps.get("LanguageMaster", [])),
            "LearningCore": LearningCore(name="LearningCore", role="Learning", capabilities=caps.get("LearningCore", [])),
            "PromptEngineer": PromptEngineer(name="PromptEngineer", role="Prompts", capabilities=caps.get("PromptEngineer", [])),
            "KnowledgeHarvester": KnowledgeHarvester(name="KnowledgeHarvester", role="Info", capabilities=caps.get("KnowledgeHarvester", [])),
            "ReasoningEngine": ReasoningEngine(name="ReasoningEngine", role="Logic", capabilities=caps.get("ReasoningEngine", [])),
            "MetaLearner": MetaLearner(name="MetaLearner", role="Optimization", capabilities=caps.get("MetaLearner", [])),
            "OutputSynthesizer": OutputSynthesizer(name="OutputSynthesizer", role="Format", capabilities=caps.get("OutputSynthesizer", [])),
            "EthicsGuardian": EthicsGuardian(name="EthicsGuardian", role="Ethics", capabilities=caps.get("EthicsGuardian", [])),
        }

    async def run_knowledge_expansion(self, request: MetaPromptRequest) -> MetaPromptResponse:
        workflow_id = uuid.uuid4()
        context = dict(request.context)
        context["query"] = request.query
        context.setdefault("preferences", dict(request.preferences))
        if self.wisdom is not None:
            context.setdefault("wisdom", self.wisdom)
        results: List[MetaPromptResult] = []

        # Policy: on non-critical errors, continue; on EthicsGuardian failure, stop.
        steps = [
            "LanguageMaster",
            "KnowledgeHarvester",
            "OntologyArchitect",
            "ReasoningEngine",
            "PromptEngineer",
            "LearningCore",
            "MetaLearner",
            "OutputSynthesizer",
            "EthicsGuardian",
        ]

        critical_fail = False
        for agent_name in steps:
            try:
                res = await self.agents[agent_name].execute(context)
                results.append(self._result(agent_name, res, success=True, error=None))
                if isinstance(res, dict):
                    context.update(res)
            except Exception as e:
                results.append(self._result(agent_name, {"error": str(e)}, success=False, error=str(e)))
                if agent_name == "EthicsGuardian":
                    critical_fail = True
                    break

        success_count = sum(1 for r in results if getattr(r, "success", False))
        total = max(1, len(results))
        confidence = success_count / total
        if critical_fail:
            confidence = min(confidence, 0.49)

        explanation = "Workflow completed" if not critical_fail else "Workflow completed with critical failure"

        return MetaPromptResponse(
            results=results,
            explanation=explanation,
            confidence=float(confidence),
            workflow_id=workflow_id,
        )

    def _result(self, agent: str, content: Any, *, success: bool = True, error: Optional[str] = None) -> MetaPromptResult:
        return MetaPromptResult(
            content=content,
            source_agent=agent,
            timestamp=datetime.now(timezone.utc),
            success=success,
            error=error,
        )
