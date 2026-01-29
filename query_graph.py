# BIZRA Graph Query Engine v1.0
# Semantic & Graph-based retrieval over Layer 3

import json
import pandas as pd
import networkx as nx
from pathlib import Path
import argparse
from bizra_config import GRAPH_PATH, CORPUS_TABLE_PATH, CHUNKS_TABLE_PATH

NODES_FILE = GRAPH_PATH / "nodes.jsonl"
EDGES_FILE = GRAPH_PATH / "edges.jsonl"

def query_graph(query_text):
    print(f"🔍 Searching BIZRA Hypergraph for: '{query_text}'")
    
    if not NODES_FILE.exists():
        print("❌ Graph index not found. Run build-hypergraph.py first.")
        return

    # Load nodes
    nodes = []
    with open(NODES_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            nodes.append(json.loads(line))
    
    # Simple keyword search across nodes
    results = [n for n in nodes if query_text.lower() in str(n.get('label', '')).lower() 
               or query_text.lower() in str(n.get('text', '')).lower() 
               or query_text.lower() in str(n.get('kind', '')).lower()
               or query_text.lower() in str(n.get('id', '')).lower()]
    
    if not results:
        print("❓ No direct node matches found.")
        return

    print(f"✅ Found {len(results)} relevant nodes:\n")
    for res in results[:10]:
        kind = res.get('kind', 'Unknown')
        label = res.get('label', res.get('id', 'N/A'))
        print(f"[{kind}] {label}")
        if 'source' in res:
            print(f"   ∟ Source: {res['source']}")
        if 'extraction_class' in res:
            print(f"   ∟ Fact: {res['extraction_class']}")
    
    # Check for edges (connections)
    print("\n🔗 Exploring connections...")
    edges = []
    if EDGES_FILE.exists():
        with open(EDGES_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                edges.append(json.loads(line))
    
    relevant_ids = [r['id'] for r in results]
    connections = [e for e in edges if e['source'] in relevant_ids or e['target'] in relevant_ids]
    
    for conn in connections[:10]:
        print(f"   {conn['source']} --({conn['relation']})--> {conn['target']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()
    
    query_graph(args.query)
