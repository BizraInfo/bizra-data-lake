import pytest
from core.meta_prompt.engine import MetaPromptEngine
from core.meta_prompt.models import MetaPromptRequest

@pytest.mark.asyncio
async def test_meta_prompt_engine_workflow():
    engine = MetaPromptEngine()
    request = MetaPromptRequest(
        query="Test Query",
        context={"test": "context"},
        preferences={}
    )
    
    response = await engine.run_knowledge_expansion(request)

    expected = [
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

    assert [r.source_agent for r in response.results] == expected
    assert 0.0 <= response.confidence <= 1.0
    assert response.explanation
