"""MVDA v0.2 — Local corpus evidence retrieval from 04_GOLD."""

import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd

GOLD_DIR = Path("/data/bizra/04_GOLD")


def search_corpus(query: str, top_k: int = 5) -> List[Tuple[str, str, str]]:
    """Search chunks.parquet by keyword match. Returns (chunk_id, doc_id, text)."""
    chunks_path = GOLD_DIR / "chunks.parquet"
    if not chunks_path.exists():
        return []

    df = pd.read_parquet(chunks_path, columns=["chunk_id", "doc_id", "chunk_text"])
    query_lower = query.lower()
    keywords = [w for w in query_lower.split() if len(w) > 3]

    if not keywords:
        return []

    mask = df["chunk_text"].str.lower().str.contains(keywords[0], na=False)
    for kw in keywords[1:]:
        mask = mask | df["chunk_text"].str.lower().str.contains(kw, na=False)

    hits = df[mask].head(top_k)
    return [
        (row["chunk_id"], row["doc_id"], row["chunk_text"][:500])
        for _, row in hits.iterrows()
    ]


def get_doc_title(doc_id: str) -> str:
    """Look up document title from documents.parquet."""
    docs_path = GOLD_DIR / "documents.parquet"
    if not docs_path.exists():
        return doc_id

    df = pd.read_parquet(docs_path, columns=["doc_id", "title"])
    match = df[df["doc_id"] == doc_id]
    if not match.empty:
        return match.iloc[0]["title"]
    return doc_id
