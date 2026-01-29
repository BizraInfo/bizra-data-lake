"""
BIZRA QURANIC SEMANTIC SEARCH
"Interdisciplinary Query Engine"

Demonstrates the 'State of Art' capability to search the newly ingested
Quranic Corpus using vector semantics rather than just keyword matching.
"""

import pandas as pd
import numpy as np
from vector_engine import VectorEngine
from bizra_config import INDEXED_PATH

DATA_DIR = INDEXED_PATH / "knowledge/quran"
PARQUET_PATH = DATA_DIR / "quran_full.parquet"
VECTORS_PATH = DATA_DIR / "quran_vectors.npy"

def search(query, top_k=5):
    print(f"🧠 QUERY: '{query}'")
    
    # 1. Load Data
    print("   📂 Loading Corpus...")
    df = pd.read_parquet(PARQUET_PATH)
    vectors = np.load(VECTORS_PATH)
    
    # 2. Embed Query
    engine = VectorEngine()
    query_vec = engine.model.encode([query])[0]
    
    # 3. Compute Similarity (Cosine)
    # Cosine Similarity = (A . B) / (||A|| * ||B||)
    # Vectors are often normalized, but let's be safe
    print("   🧮 Computing Semantic Distance...")
    norm_vectors = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
    norm_query = query_vec / np.linalg.norm(query_vec)
    
    scores = np.dot(norm_vectors, norm_query)
    
    # 4. Rank
    top_indices = np.argsort(scores)[::-1][:top_k]
    
    print(f"\n✅ TOP {top_k} RESULTS FOR '{query}':")
    print("-" * 80)
    for idx in top_indices:
        row = df.iloc[idx]
        score = scores[idx]
        print(f"📖 {row['surah_name']} ({row['surah']}:{row['ayah']}) | Score: {score:.4f}")
        print(f"   AR: {row['arabic_text'][:80]}...")
        print(f"   EN: {row['english_text']}")
        print("-" * 80)

if __name__ == "__main__":
    # A query that embodies "Interdisciplinary Thinking" and "Intellect"
    search("using reason and intellect to reflect on signs", top_k=3)
