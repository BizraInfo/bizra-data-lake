"""Re-export from canonical location: core.bridges.swarm_knowledge_bridge"""

# Backwards compatibility — canonical implementation is in core/bridges/
from core.bridges.swarm_knowledge_bridge import *  # noqa: F401,F403
from core.bridges.swarm_knowledge_bridge import (
    ROLE_KNOWLEDGE_ACCESS,
    AgentKnowledgeContext,
    KnowledgeInjection,
    SwarmKnowledgeBridge,
    create_swarm_knowledge_bridge,
)
