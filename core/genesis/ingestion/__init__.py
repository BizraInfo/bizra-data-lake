"""Multi-platform conversation ingestion pipeline for User Zero Bootstrap.

Standing on Giants: Shannon (information theory) - Lamport (distributed systems)
"""

from core.genesis.ingestion.dedup import deduplicate
from core.genesis.ingestion.pipeline import IngestPipeline
from core.genesis.ingestion.schema import ConversationTurn, Platform, Role

__all__ = ["ConversationTurn", "Platform", "Role", "IngestPipeline", "deduplicate"]
