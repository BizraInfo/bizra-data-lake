"""
kg/__init__.py — Knowledge Substrate module

P1.11 Implementation: Receipt-native, fail-closed, audit-grade knowledge service.
"""

from kg.embeddings import get_embedder, Embedder, EmbeddingResult, NullEmbedder
from kg.retrieve import retrieve, search_by_entity, RetrievalBundle, RetrievalResult
from kg.receipts import (
    emit_receipt,
    emit_query_receipt,
    emit_ingest_receipt,
    get_receipt,
    list_recent_receipts,
    ReceiptKind,
    Decision,
    RejectionReason,
    IhsanScore,
    SapeVector,
    SnrMetrics
)

__all__ = [
    # Embeddings
    "get_embedder",
    "Embedder",
    "EmbeddingResult",
    "NullEmbedder",
    # Retrieval
    "retrieve",
    "search_by_entity",
    "RetrievalBundle",
    "RetrievalResult",
    # Receipts
    "emit_receipt",
    "emit_query_receipt",
    "emit_ingest_receipt",
    "get_receipt",
    "list_recent_receipts",
    "ReceiptKind",
    "Decision",
    "RejectionReason",
    "IhsanScore",
    "SapeVector",
    "SnrMetrics",
]
