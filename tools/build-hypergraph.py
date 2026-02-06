# BIZRA Hypergraph Knowledge Constructor v2.1
# Builds multi-dimensional knowledge graph with Assertion/Fact nodes
# Layer 3: Knowledge Graph + Hypergraph RAG logic

import os
import json
import networkx as nx
import hashlib
import pandas as pd
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from tqdm import tqdm
import re
from bizra_config import CORPUS_TABLE_PATH, CHUNKS_TABLE_PATH, GRAPH_PATH

# Output files
NODES_FILE = GRAPH_PATH / "nodes.jsonl"
EDGES_FILE = GRAPH_PATH / "edges.jsonl"
HYPERGRAPH_FILE = GRAPH_PATH / "hypergraph.json"
STATISTICS_FILE = GRAPH_PATH / "statistics.json"

class HypergraphConstructor:
    def __init__(self):
        print("🧬 Initializing BIZRA Hypergraph Constructor v2.1")
        GRAPH_PATH.mkdir(parents=True, exist_ok=True)
        self.assertions_file = CORPUS_TABLE_PATH.parent / "assertions.jsonl"
        
        # Initialize graph structures
        self.graph = nx.MultiDiGraph()
        self.stats = {
            "total_nodes": 0,
            "total_edges": 0,
            "assertions": 0,
            "start_time": datetime.now().isoformat()
        }
        
    def run(self):
        """Main entry point to build the hypergraph."""
        self.build_from_corpus()

    def build_from_corpus(self):
        """Builds graph using Documents and Chunks as the spine."""
        if not CORPUS_TABLE_PATH.exists():
            print("❌ No Corpus found. Run corpus_manager.py first.")
            return

        print(f"📖 Loading corpus from {CORPUS_TABLE_PATH}")
        df_docs = pd.read_parquet(CORPUS_TABLE_PATH)
        df_chunks = pd.read_parquet(CHUNKS_TABLE_PATH) if CHUNKS_TABLE_PATH.exists() else None

        print("🏗️ Building Graph Spine (Documents & Chunks)...")
        for _, doc in tqdm(df_docs.iterrows(), total=len(df_docs), desc="Mapping Documents"):
            self.graph.add_node(doc['doc_id'], label=doc['title'], kind="Document", source=doc['uri'])
            
            # Extract basic entities/concepts to create Assertion nodes later
            # (In a real system, this would call an LLM/NLP engine)
            concepts = re.findall(r'\b(BIZRA|SAPE|RTX|GPU|Organism|Node0)\b', doc['text'], re.I)
            for concept in set(concepts):
                concept_id = f"ent::{concept.lower()}"
                self.graph.add_node(concept_id, label=concept, kind="Entity")
                self.graph.add_edge(doc['doc_id'], concept_id, relation="MENTIONS")

        if df_chunks is not None:
            print("🏗️ Mapping Chunks and Hyperedge Assertions...")
            for _, chunk in tqdm(df_chunks.iterrows(), total=len(df_chunks), desc="Mapping Chunks"):
                self.graph.add_node(chunk['chunk_id'], kind="Chunk", doc_id=chunk['doc_id'])
                self.graph.add_edge(chunk['chunk_id'], chunk['doc_id'], relation="PART_OF")
                
                # Mock Hypergraph Assertion: Connect doc, chunk, and an entity
                # "In Doc X, Chunk Y discusses Concept Z"
                if "SAPE" in chunk['chunk_text']:
                    assertion_id = f"fact::{hashlib.blake2b(chunk['chunk_id'].encode(), digest_size=16).hexdigest()[:8]}"
                    self.graph.add_node(assertion_id, kind="Assertion", text="Discusses SAPE architecture")
                    self.graph.add_edge(assertion_id, chunk['chunk_id'], relation="ASSERTION_SUPPORTED_BY")
                    self.graph.add_edge(assertion_id, "ent::sape", relation="ASSERTION_INVOLVES")
                    self.stats["assertions"] += 1

        # Consume LangExtract assertions (Deterministic facts)
        if self.assertions_file.exists():
            print("🏗️ Integrating High-Precision Assertions from LangExtract...")
            with open(self.assertions_file, 'r') as f:
                for line in f:
                    try:
                        assertion_data = json.loads(line)
                        doc_id = assertion_data['doc_id']
                        if doc_id not in self.graph:
                            # If doc not in graph, skip or add placeholder
                            continue
                            
                        for i, ext in enumerate(assertion_data['extractions']):
                            fact_id = f"fact::{doc_id}_{i}"
                            self.graph.add_node(fact_id, kind="Fact", label=ext['text'], extraction_class=ext['class'], **ext.get('attributes', {}))
                            self.graph.add_edge(fact_id, doc_id, relation="FACT_FROM")
                            self.stats["assertions"] += 1
                    except Exception as e:
                        print(f"⚠️ Error parsing assertion: {e}")

        self.save_graph()

    def save_graph(self):
        """Saves as JSONL for scalability."""
        print(f"💾 Saving Graph to {GRAPH_PATH}...")
        
        with open(NODES_FILE, 'w', encoding='utf-8') as f:
            for node, data in self.graph.nodes(data=True):
                line = {"id": node}
                line.update(data)
                f.write(json.dumps(line) + "\n")
        
        with open(EDGES_FILE, 'w', encoding='utf-8') as f:
            for u, v, data in self.graph.edges(data=True):
                line = {"source": u, "target": v}
                line.update(data)
                f.write(json.dumps(line) + "\n")
                
        self.stats["total_nodes"] = self.graph.number_of_nodes()
        self.stats["total_edges"] = self.graph.number_of_edges()
        self.stats["end_time"] = datetime.now().isoformat()
        
        with open(STATISTICS_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, indent=2)
            
        print("\n" + "=" * 70)
        print("✅ HYPERGRAPH CONSTRUCTION COMPLETE")
        print("=" * 70)
        print(f"📊 Nodes: {self.stats['total_nodes']}")
        print(f"🔗 Edges: {self.stats['total_edges']}")
        print(f"🧬 Assertions: {self.stats['assertions']}")
        print(f"💾 Output: {GRAPH_PATH}")
        print("=" * 70)

if __name__ == "__main__":
    constructor = HypergraphConstructor()
    constructor.run()
