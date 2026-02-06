"""Re-export from canonical location: core.bridges.knowledge_integrator"""
# Backwards compatibility — canonical implementation is in core/bridges/
from core.bridges.knowledge_integrator import *  # noqa: F401,F403
from core.bridges.knowledge_integrator import (
    KnowledgeIntegrator,
    KnowledgeQuery,
    KnowledgeResult,
    KnowledgeSource,
    create_knowledge_integrator,
)
