from typing import List, Dict, Any
from pydantic import BaseModel

def _evidence_limit(context: Dict[str, Any], default: int = 5) -> int:
    raw = None
    preferences = context.get("preferences")
    if isinstance(preferences, dict):
        raw = preferences.get("evidence_limit")
    if raw is None:
        raw = context.get("evidence_limit")
    try:
        limit = int(raw)
    except (TypeError, ValueError):
        limit = default
    return max(1, min(limit, 25))


def _normalize_evidence(rows: Any) -> List[Dict[str, Any]]:
    evidence: List[Dict[str, Any]] = []
    if not isinstance(rows, list):
        return evidence
    for row in rows:
        if not isinstance(row, dict):
            continue
        entry: Dict[str, Any] = {}
        hash_value = row.get("hash")
        path_value = row.get("path")
        if hash_value:
            entry["hash"] = hash_value
        if path_value:
            entry["path"] = path_value
        if row.get("filename"):
            entry["filename"] = row.get("filename")
        if row.get("hash_kind"):
            entry["hash_kind"] = row.get("hash_kind")
        if entry:
            evidence.append(entry)
    return evidence


class MetaAgent(BaseModel):
    name: str
    role: str
    capabilities: List[str]
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent's task."""
        raise NotImplementedError("Subclasses must implement execute")

class OntologyArchitect(MetaAgent):
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        topic = str(context.get("ontology_topic") or context.get("query") or "").strip()
        if not topic:
            return {"action": "organize_knowledge", "status": "no_topic", "evidence": []}

        wisdom = context.get("wisdom")
        if wisdom is None or not hasattr(wisdom, "query_knowledge"):
            return {
                "action": "organize_knowledge",
                "status": "offline",
                "topic": topic,
                "evidence": [],
                "error": "wisdom_unavailable",
            }

        limit = _evidence_limit(context)
        try:
            rows = wisdom.query_knowledge(topic=topic, limit=limit)
        except Exception as exc:
            return {
                "action": "organize_knowledge",
                "status": "offline",
                "topic": topic,
                "evidence": [],
                "error": str(exc),
            }

        evidence = _normalize_evidence(rows)
        status = "enriched" if evidence else "empty"
        return {
            "action": "organize_knowledge",
            "status": status,
            "topic": topic,
            "evidence": evidence,
            "evidence_count": len(evidence),
        }

class LanguageMaster(MetaAgent):
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        # Placeholder for LLM integration
        return {"action": "interpret_query", "intent": "simulated"}

class LearningCore(MetaAgent):
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "process_prompts", "status": "simulated"}

class PromptEngineer(MetaAgent):
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "generate_prompts", "prompts": ["simulated_prompt"]}

class KnowledgeHarvester(MetaAgent):
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "gather_information", "facts": []}

class ReasoningEngine(MetaAgent):
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "analyze_information", "logic": "simulated"}

class MetaLearner(MetaAgent):
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "evaluate_results", "score": 0.95}

class OutputSynthesizer(MetaAgent):
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "present_findings", "output": "Simulated Output"}

class EthicsGuardian(MetaAgent):
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "ensure_ethics", "verdict": "APPROVED"}
