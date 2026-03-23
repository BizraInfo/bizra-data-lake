#!/usr/bin/env python3
"""
BIZRA House of Wisdom — Semantic Search CLI
════════════════════════════════════════════

Query your cognitive history across all eras using natural language.

Standing on Giants:
  Salton (1975) — vector space model for information retrieval
  Mikolov (2013) — distributed word representations
  Al-Khwarizmi (780-850) — original House of Wisdom scholar

Usage:
    python wisdom_search.py "consensus algorithms"
    python wisdom_search.py "what was I learning in August 2023"
    python wisdom_search.py "blockchain" --top 10
    python wisdom_search.py --timeline 2023-08
    python wisdom_search.py --stats
"""

import json
import math
import pickle
import re
import sys
from collections import Counter
from pathlib import Path

SOVEREIGN_ROOT = Path(r"B:\BIZRA-SOVEREIGN\05_DATA_LAKE")
EMBED_PATH = SOVEREIGN_ROOT / "03_INDEXED" / "embeddings" / "embeddings.pkl"
MANIFEST_DIR = SOVEREIGN_ROOT / "02_PROCESSED" / "manifests"
GRAPH_STATS = SOVEREIGN_ROOT / "03_INDEXED" / "graph" / "graph_stats.json"
TIMELINE_PATH = SOVEREIGN_ROOT / "03_INDEXED" / "timeline" / "thought_timeline.json"


def load_embeddings():
    """Load pre-computed embeddings from thought graph builder."""
    if not EMBED_PATH.exists():
        print("No embeddings found. Run thought_graph_builder.py first.")
        sys.exit(1)
    with open(EMBED_PATH, "rb") as f:
        data = pickle.load(f)
    return data["embeddings"], data["texts"]


def load_manifests():
    """Load manifest index for metadata lookup."""
    manifests = {}
    for mf in MANIFEST_DIR.glob("*.json"):
        try:
            with open(mf, "r") as f:
                m = json.load(f)
            manifests[m["id"]] = m
        except Exception:
            pass
    return manifests


def cosine_sim(a, b) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def semantic_search(
    query: str, embeddings: dict, texts: dict, manifests: dict, top_k: int = 5
):
    """Search embeddings by cosine similarity to query."""
    # Try sentence-transformers first
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        q_vec = model.encode(query, show_progress_bar=False)
    except ImportError:
        # TF-IDF fallback: match query words against stored texts
        print(
            "  (using keyword fallback - install sentence-transformers for semantic search)"
        )
        return keyword_search(query, texts, manifests, top_k)

    results = []
    for mid, vec in embeddings.items():
        sim = cosine_sim(q_vec, vec)
        results.append((mid, sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def keyword_search(query: str, texts: dict, manifests: dict, top_k: int = 5):
    """Keyword-based fallback search."""
    query_terms = set(re.findall(r"\b[a-z]{3,}\b", query.lower()))
    results = []
    for mid, text in texts.items():
        text_lower = text.lower()
        score = sum(1 for t in query_terms if t in text_lower)
        if score > 0:
            results.append((mid, score / len(query_terms) if query_terms else 0))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_k]


def display_results(results, manifests, texts):
    """Pretty-print search results."""
    if not results:
        print("\n  No results found.")
        return

    print(f"\n  {'='*60}")
    print(f"  TOP {len(results)} RESULTS")
    print(f"  {'='*60}")

    for rank, (mid, score) in enumerate(results, 1):
        m = manifests.get(mid, {})
        fname = m.get("original_filename", "unknown")
        ftype = m.get("file_type", "?")
        era = m.get("mindset_era", "?")
        date = m.get("modified", "?")[:10]
        preview = texts.get(mid, "")[:120].replace("\n", " ")

        print(f"\n  [{rank}] {fname}")
        print(f"      Score: {score:.3f} | Type: {ftype} | Era: {era} | Date: {date}")
        print(f"      {preview}...")


def show_timeline(month_filter: str = None):
    """Show thought timeline, optionally filtered by month."""
    if not TIMELINE_PATH.exists():
        print("No timeline found. Run thought_graph_builder.py first.")
        return

    with open(TIMELINE_PATH, "r") as f:
        timeline = json.load(f)

    print("\n  THOUGHT TIMELINE")
    print(f"  {'='*60}")

    for entry in timeline:
        date = entry["date"]
        if month_filter and not date.startswith(month_filter):
            continue
        count = entry["count"]
        types = Counter(f["type"] for f in entry["files"])
        type_str = ", ".join(f"{v} {k}" for k, v in types.most_common())
        bar = "#" * min(count, 40)
        print(f"  {date} | {count:3d} files | {bar} | {type_str}")

    if month_filter:
        filtered = [e for e in timeline if e["date"].startswith(month_filter)]
        total = sum(e["count"] for e in filtered)
        print(f"\n  Total for {month_filter}: {total} files")


def show_stats():
    """Show graph and knowledge base statistics."""
    if not GRAPH_STATS.exists():
        print("No graph stats found. Run thought_graph_builder.py first.")
        return

    with open(GRAPH_STATS, "r") as f:
        stats = json.load(f)

    print("\n  HOUSE OF WISDOM -- Knowledge Base Statistics")
    print(f"  {'='*60}")
    print(f"  Nodes (files):     {stats['nodes']}")
    print(f"  Edges (connections): {stats['edges']}")
    print(
        f"  Date range:        {stats['date_range']['earliest']} to {stats['date_range']['latest']}"
    )
    print("\n  Edge Types:")
    for etype, count in stats["edge_types"].items():
        pct = count / stats["edges"] * 100 if stats["edges"] > 0 else 0
        print(f"    {etype:20s}: {count:6d} ({pct:.1f}%)")
    print("\n  Eras:")
    for era, count in stats["eras"].items():
        print(f"    {era:30s}: {count:4d} files")
    print("\n  File Types:")
    for ftype, count in stats["file_types"].items():
        print(f"    {ftype:20s}: {count:4d} files")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="BIZRA House of Wisdom -- Semantic Search"
    )
    parser.add_argument(
        "query", nargs="?", default=None, help="Natural language search query"
    )
    parser.add_argument(
        "--top", type=int, default=5, help="Number of results (default: 5)"
    )
    parser.add_argument(
        "--timeline",
        type=str,
        default=None,
        help="Show timeline (optional: YYYY-MM filter)",
    )
    parser.add_argument(
        "--stats", action="store_true", help="Show knowledge base statistics"
    )
    args = parser.parse_args()

    if args.stats:
        show_stats()
        return

    if args.timeline is not None:
        month = args.timeline if args.timeline else None
        show_timeline(month)
        return

    if not args.query:
        parser.print_help()
        print("\nExamples:")
        print('  python wisdom_search.py "blockchain consensus"')
        print('  python wisdom_search.py "DeFi liquidity" --top 10')
        print("  python wisdom_search.py --timeline 2023-12")
        print("  python wisdom_search.py --stats")
        return

    print(f'  Searching: "{args.query}"')
    embeddings, texts = load_embeddings()
    manifests = load_manifests()
    results = semantic_search(args.query, embeddings, texts, manifests, top_k=args.top)
    display_results(results, manifests, texts)


if __name__ == "__main__":
    main()
