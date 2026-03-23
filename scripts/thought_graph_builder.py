#!/usr/bin/env python3
"""
BIZRA Thought Graph Builder — Stage 3 (INDEXED)
════════════════════════════════════════════════

Reads manifests from 02_PROCESSED, builds FAISS embeddings
and NetworkX thought graph with 5 connection types.

Standing on Giants:
  Mikolov (2013) — word/sentence embeddings
  Erdos (1959) — random graph theory
  Barabasi (1999) — scale-free networks
  Shannon (1948) — information content for edge weighting

Connection Types:
  1. TEMPORAL    — created same day/week
  2. TOPICAL     — cosine similarity >= threshold
  3. SEQUENTIAL  — filename/content patterns suggesting A→B
  4. EVOLUTIONARY — same topic, different maturity
  5. CROSS_MINDSET — different era, connected insight

Usage:
    python thought_graph_builder.py [--threshold 0.75] [--dry-run]
"""

import json
import pickle
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SOVEREIGN_ROOT = Path(r"B:\BIZRA-SOVEREIGN\05_DATA_LAKE")
MANIFEST_DIR = SOVEREIGN_ROOT / "02_PROCESSED" / "manifests"
INDEX_DIR = SOVEREIGN_ROOT / "03_INDEXED"
EMBED_DIR = INDEX_DIR / "embeddings"
GRAPH_DIR = INDEX_DIR / "graph"
TIMELINE_DIR = INDEX_DIR / "timeline"

# ═══════════════════════════════════════════════════════════════
# EMBEDDING ENGINE (Mikolov — distributed representations)
# ═══════════════════════════════════════════════════════════════


class EmbeddingEngine:
    """Generate and store sentence embeddings for semantic search."""

    def __init__(self):
        self.model = None
        self.embeddings = {}  # manifest_id → vector
        self.texts = {}  # manifest_id → text preview

    def load_model(self):
        """Load sentence-transformers model. Falls back to TF-IDF."""
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            print("  Embedding model: all-MiniLM-L6-v2 (384-dim)")
            return True
        except ImportError:
            print("  sentence-transformers not found, using TF-IDF fallback")
            return False

    def embed_text(self, text: str):
        """Generate embedding vector for text."""
        if self.model:
            return self.model.encode(text, show_progress_bar=False)
        return None

    def embed_manifests(self, manifests: list[dict]):
        """Embed all manifest text previews."""
        if not self.model:
            return self._tfidf_fallback(manifests)

        texts = []
        ids = []
        for m in manifests:
            text = m.get("extracted_text_preview", "")
            if text and not text.startswith("[binary:"):
                texts.append(text)
                ids.append(m["id"])
                self.texts[m["id"]] = text

        if not texts:
            return

        print(f"  Embedding {len(texts)} documents...")
        vectors = self.model.encode(texts, show_progress_bar=True, batch_size=64)
        for i, mid in enumerate(ids):
            self.embeddings[mid] = vectors[i]
        print(f"  Embedded: {len(self.embeddings)} vectors ({vectors[0].shape[0]}-dim)")

    def _tfidf_fallback(self, manifests: list[dict]):
        """TF-IDF fallback when sentence-transformers unavailable."""
        import math
        from collections import Counter

        texts = []
        ids = []
        for m in manifests:
            text = m.get("extracted_text_preview", "")
            if text and not text.startswith("[binary:"):
                texts.append(text.lower())
                ids.append(m["id"])
                self.texts[m["id"]] = text

        # Build vocabulary
        doc_freq = Counter()
        doc_terms = []
        for text in texts:
            terms = set(re.findall(r"\b[a-z]{3,}\b", text))
            doc_terms.append(terms)
            for t in terms:
                doc_freq[t] += 1

        n_docs = len(texts)
        vocab = [t for t, f in doc_freq.items() if 2 <= f <= n_docs * 0.8]
        vocab_idx = {t: i for i, t in enumerate(vocab)}

        # Build TF-IDF vectors

        for i, (mid, text) in enumerate(zip(ids, texts)):
            terms = re.findall(r"\b[a-z]{3,}\b", text)
            tf = Counter(terms)
            vec = [0.0] * len(vocab)
            for t, count in tf.items():
                if t in vocab_idx:
                    idf = math.log(n_docs / (1 + doc_freq[t]))
                    vec[vocab_idx[t]] = count * idf
            self.embeddings[mid] = vec

        print(f"  TF-IDF fallback: {len(self.embeddings)} vectors ({len(vocab)}-dim)")

    def cosine_similarity(self, vec_a, vec_b) -> float:
        """Compute cosine similarity between two vectors."""
        import math

        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def save(self, path: Path):
        """Save embeddings to disk."""
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "embeddings.pkl", "wb") as f:
            pickle.dump({"embeddings": self.embeddings, "texts": self.texts}, f)
        print(f"  Saved embeddings to {path}")


# ═══════════════════════════════════════════════════════════════
# THOUGHT GRAPH ENGINE (Erdos + Barabasi — network structure)
# ═══════════════════════════════════════════════════════════════


class ThoughtGraph:
    """NetworkX-compatible thought graph with 5 edge types."""

    def __init__(self):
        self.nodes = {}  # id → manifest dict
        self.edges = []  # list of (src, dst, type, weight)
        self.by_date = defaultdict(list)  # date_str → [ids]
        self.by_era = defaultdict(list)  # era → [ids]
        self.by_type = defaultdict(list)  # file_type → [ids]
        self.stats = defaultdict(int)

    def add_manifests(self, manifests: list[dict]):
        """Load all manifests as graph nodes."""
        for m in manifests:
            mid = m["id"]
            self.nodes[mid] = m
            # Index by date (YYYY-MM-DD)
            mod = m.get("modified", "")
            if mod:
                date_str = mod[:10]
                self.by_date[date_str].append(mid)
            # Index by era
            era = m.get("mindset_era", "unknown")
            self.by_era[era].append(mid)
            # Index by type
            ftype = m.get("file_type", "unknown")
            self.by_type[ftype].append(mid)
        print(f"  Nodes: {len(self.nodes)}")
        print(f"  Date buckets: {len(self.by_date)}")
        print(f"  Eras: {list(self.by_era.keys())}")

    def build_temporal_edges(self, window_days: int = 1):
        """TEMPORAL: connect files created within N days of each other."""
        dates = sorted(self.by_date.keys())
        count = 0
        for i, d1 in enumerate(dates):
            for j in range(i, min(i + window_days + 1, len(dates))):
                d2 = dates[j]
                if d1 == d2:
                    ids = self.by_date[d1]
                    for a_idx in range(len(ids)):
                        for b_idx in range(a_idx + 1, len(ids)):
                            self.edges.append((ids[a_idx], ids[b_idx], "TEMPORAL", 1.0))
                            count += 1
                else:
                    for a in self.by_date[d1]:
                        for b in self.by_date[d2]:
                            self.edges.append((a, b, "TEMPORAL", 0.5))
                            count += 1
        self.stats["temporal"] = count
        print(f"  TEMPORAL edges: {count}")

    def build_topical_edges(self, engine: EmbeddingEngine, threshold: float = 0.75):
        """TOPICAL: connect files with cosine similarity >= threshold."""
        ids = list(engine.embeddings.keys())
        count = 0
        n = len(ids)
        for i in range(n):
            for j in range(i + 1, n):
                sim = engine.cosine_similarity(
                    engine.embeddings[ids[i]], engine.embeddings[ids[j]]
                )
                if sim >= threshold:
                    self.edges.append((ids[i], ids[j], "TOPICAL", round(sim, 4)))
                    count += 1
        self.stats["topical"] = count
        print(f"  TOPICAL edges (>={threshold}): {count}")

    def build_sequential_edges(self):
        """SEQUENTIAL: detect A→B patterns from filenames."""
        # Pattern: numbered files, "part 1/2", versioned docs
        patterns = [
            (r"(.+?)[\s_-]*\((\d+)\)", "parens_num"),  # file (1), file (2)
            (r"(.+?)[\s_-]*(\d+)$", "trailing_num"),  # file1, file2
            (r"(.+?)[\s_-]*[Pp]art[\s_-]*(\d+)", "part"),  # Part 1, Part 2
            (r"(.+?)[\s_-]*[Vv](\d+)", "version"),  # v1, v2
        ]
        # Group by base name
        groups = defaultdict(list)
        for mid, m in self.nodes.items():
            fname = Path(m.get("original_filename", "")).stem
            for pat, kind in patterns:
                match = re.match(pat, fname)
                if match:
                    base = match.group(1).strip().lower()
                    num = int(match.group(2))
                    groups[(base, kind)].append((num, mid))
                    break

        count = 0
        for key, items in groups.items():
            items.sort()  # by number
            for i in range(len(items) - 1):
                self.edges.append((items[i][1], items[i + 1][1], "SEQUENTIAL", 0.9))
                count += 1
        self.stats["sequential"] = count
        print(f"  SEQUENTIAL edges: {count}")

    def build_evolutionary_edges(
        self, engine: EmbeddingEngine, threshold: float = 0.6, min_days_apart: int = 30
    ):
        """EVOLUTIONARY: same topic at different maturity (time + similarity)."""
        ids = list(engine.embeddings.keys())
        count = 0
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if a not in self.nodes or b not in self.nodes:
                    continue
                # Check time distance
                mod_a = self.nodes[a].get("modified", "")[:10]
                mod_b = self.nodes[b].get("modified", "")[:10]
                if not mod_a or not mod_b:
                    continue
                try:
                    da = datetime.fromisoformat(mod_a)
                    db = datetime.fromisoformat(mod_b)
                    days_apart = abs((da - db).days)
                except (ValueError, TypeError):
                    continue
                if days_apart < min_days_apart:
                    continue
                # Check topic similarity
                sim = engine.cosine_similarity(
                    engine.embeddings[a], engine.embeddings[b]
                )
                if sim >= threshold:
                    weight = sim * min(days_apart / 365, 1.0)
                    self.edges.append((a, b, "EVOLUTIONARY", round(weight, 4)))
                    count += 1
        self.stats["evolutionary"] = count
        print(f"  EVOLUTIONARY edges (>={threshold}, >={min_days_apart}d): {count}")

    def build_cross_mindset_edges(
        self, engine: EmbeddingEngine, threshold: float = 0.65
    ):
        """CROSS_MINDSET: different era, connected insight."""
        eras = list(self.by_era.keys())
        count = 0
        for i in range(len(eras)):
            for j in range(i + 1, len(eras)):
                era_a, era_b = eras[i], eras[j]
                if era_a == era_b or era_a == "unknown" or era_b == "unknown":
                    continue
                for a in self.by_era[era_a]:
                    if a not in engine.embeddings:
                        continue
                    for b in self.by_era[era_b]:
                        if b not in engine.embeddings:
                            continue
                        sim = engine.cosine_similarity(
                            engine.embeddings[a], engine.embeddings[b]
                        )
                        if sim >= threshold:
                            self.edges.append((a, b, "CROSS_MINDSET", round(sim, 4)))
                            count += 1
        self.stats["cross_mindset"] = count
        print(f"  CROSS_MINDSET edges (>={threshold}): {count}")

    def build_timeline(self):
        """Build chronological thought evolution."""
        timeline = []
        for date_str in sorted(self.by_date.keys()):
            entries = []
            for mid in self.by_date[date_str]:
                m = self.nodes[mid]
                entries.append(
                    {
                        "id": mid[:12],
                        "filename": m.get("original_filename", "unknown"),
                        "type": m.get("file_type", "unknown"),
                        "era": m.get("mindset_era", "unknown"),
                    }
                )
            timeline.append({"date": date_str, "count": len(entries), "files": entries})
        return timeline

    def save(self):
        """Save graph to disk."""
        GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        TIMELINE_DIR.mkdir(parents=True, exist_ok=True)

        # Save edges
        with open(GRAPH_DIR / "edges.jsonl", "w") as f:
            for src, dst, etype, weight in self.edges:
                f.write(
                    json.dumps(
                        {
                            "src": src[:16],
                            "dst": dst[:16],
                            "type": etype,
                            "weight": float(weight),
                        }
                    )
                    + "\n"
                )

        # Save node index
        node_index = {}
        for mid, m in self.nodes.items():
            node_index[mid[:16]] = {
                "filename": m.get("original_filename", ""),
                "type": m.get("file_type", ""),
                "era": m.get("mindset_era", ""),
                "date": m.get("modified", "")[:10],
            }
        with open(GRAPH_DIR / "nodes.json", "w") as f:
            json.dump(node_index, f, indent=2)

        # Save stats
        with open(GRAPH_DIR / "graph_stats.json", "w") as f:
            json.dump(
                {
                    "nodes": len(self.nodes),
                    "edges": len(self.edges),
                    "edge_types": dict(self.stats),
                    "eras": {k: len(v) for k, v in self.by_era.items()},
                    "file_types": {k: len(v) for k, v in self.by_type.items()},
                    "date_range": {
                        "earliest": min(self.by_date.keys()) if self.by_date else "",
                        "latest": max(self.by_date.keys()) if self.by_date else "",
                    },
                    "built": datetime.now(timezone.utc).isoformat(),
                },
                f,
                indent=2,
            )

        # Save timeline
        timeline = self.build_timeline()
        with open(TIMELINE_DIR / "thought_timeline.json", "w") as f:
            json.dump(timeline, f, indent=2)

        print("\n  Graph saved:")
        print(f"    Nodes: {GRAPH_DIR / 'nodes.json'}")
        print(f"    Edges: {GRAPH_DIR / 'edges.jsonl'}")
        print(f"    Stats: {GRAPH_DIR / 'graph_stats.json'}")
        print(f"    Timeline: {TIMELINE_DIR / 'thought_timeline.json'}")


# ═══════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════


def load_manifests() -> list[dict]:
    """Load all manifests from 02_PROCESSED."""
    manifests = []
    if not MANIFEST_DIR.exists():
        print("No manifests found. Run datalake_processor.py first.")
        sys.exit(1)
    for mf in MANIFEST_DIR.glob("*.json"):
        try:
            with open(mf, "r") as f:
                manifests.append(json.load(f))
        except Exception:
            pass
    return manifests


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="BIZRA Thought Graph Builder — Stage 3"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.75,
        help="Cosine similarity threshold for TOPICAL edges",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Build graph in memory without saving"
    )
    args = parser.parse_args()

    start = time.time()
    print("BIZRA Thought Graph Builder")
    print(f"{'='*60}")

    # Load manifests
    print("\n[1/6] Loading manifests...")
    manifests = load_manifests()
    print(f"  Loaded: {len(manifests)} manifests")

    # Build embeddings
    print("\n[2/6] Building embeddings...")
    engine = EmbeddingEngine()
    engine.load_model()
    engine.embed_manifests(manifests)

    # Build graph
    print("\n[3/6] Building thought graph...")
    graph = ThoughtGraph()
    graph.add_manifests(manifests)

    print("\n[4/6] Computing edges...")
    graph.build_temporal_edges(window_days=1)
    graph.build_topical_edges(engine, threshold=args.threshold)
    graph.build_sequential_edges()
    graph.build_evolutionary_edges(engine, threshold=0.6, min_days_apart=30)
    graph.build_cross_mindset_edges(engine, threshold=0.65)

    # Summary
    total_edges = len(graph.edges)
    elapsed = time.time() - start
    print("\n[5/6] Graph summary:")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {total_edges}")
    for etype, count in sorted(graph.stats.items()):
        pct = count / total_edges * 100 if total_edges > 0 else 0
        print(f"    {etype}: {count} ({pct:.1f}%)")
    print(f"  Duration: {elapsed:.1f}s")

    # Save
    if not args.dry_run:
        print("\n[6/6] Saving graph...")
        engine.save(EMBED_DIR)
        graph.save()
    else:
        print("\n[6/6] DRY RUN — graph not saved")

    print(f"\nDone in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
