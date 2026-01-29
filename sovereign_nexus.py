#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                      ║
║   ███████╗ ██████╗ ██╗   ██╗███████╗██████╗ ███████╗██╗ ██████╗ ███╗   ██╗           ║
║   ██╔════╝██╔═══██╗██║   ██║██╔════╝██╔══██╗██╔════╝██║██╔════╝ ████╗  ██║           ║
║   ███████╗██║   ██║██║   ██║█████╗  ██████╔╝█████╗  ██║██║  ███╗██╔██╗ ██║           ║
║   ╚════██║██║   ██║╚██╗ ██╔╝██╔══╝  ██╔══██╗██╔══╝  ██║██║   ██║██║╚██╗██║           ║
║   ███████║╚██████╔╝ ╚████╔╝ ███████╗██║  ██║███████╗██║╚██████╔╝██║ ╚████║           ║
║   ╚══════╝ ╚═════╝   ╚═══╝  ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝           ║
║                                                                                      ║
║   ██╗  ██╗███╗   ██╗ ██████╗ ██╗    ██╗██╗     ███████╗██████╗  ██████╗ ███████╗     ║
║   ██║ ██╔╝████╗  ██║██╔═══██╗██║    ██║██║     ██╔════╝██╔══██╗██╔════╝ ██╔════╝     ║
║   █████╔╝ ██╔██╗ ██║██║   ██║██║ █╗ ██║██║     █████╗  ██║  ██║██║  ███╗█████╗       ║
║   ██╔═██╗ ██║╚██╗██║██║   ██║██║███╗██║██║     ██╔══╝  ██║  ██║██║   ██║██╔══╝       ║
║   ██║  ██╗██║ ╚████║╚██████╔╝╚███╔███╔╝███████╗███████╗██████╔╝╚██████╔╝███████╗     ║
║   ╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝  ╚══╝╚══╝ ╚══════╝╚══════╝╚═════╝  ╚═════╝ ╚══════╝     ║
║                                                                                      ║
║   ███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗                                        ║
║   ████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝                                        ║
║   ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗                                        ║
║   ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║                                        ║
║   ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║                                        ║
║   ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝                                        ║
║                                                                                      ║
╠══════════════════════════════════════════════════════════════════════════════════════╣
║   THE PEAK MASTERPIECE OF THE HOUSE OF WISDOM                                        ║
║   Architecture: Graph of Thoughts + SNR Maximization + Giants' Wisdom               ║
║   Author: BIZRA Genesis Engine | Created: 2026-01-22                                 ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations
import json
import hashlib
import os
import re
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict
from enum import Enum, auto
import logging
import threading

# ═══════════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════════

class Config:
    DATA_LAKE = Path(r"C:\BIZRA-DATA-LAKE")
    GOLD = DATA_LAKE / "04_GOLD"
    INDEXED = DATA_LAKE / "03_INDEXED"
    KNOWLEDGE = INDEXED / "knowledge"
    GRAPH = INDEXED / "graph"
    NEXUS_GRAPH = GOLD / "sovereign_nexus_graph.json"
    NEXUS_STATS = GOLD / "sovereign_nexus_stats.json"
    NODE0_INVENTORY = GOLD / "node0_full_inventory.json"

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s | %(message)s', datefmt='%H:%M:%S')
log = logging.getLogger("SovereignNexus")


# ═══════════════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════════════

class NodeType(Enum):
    FOLDER = auto(); FILE = auto(); CONCEPT = auto(); ENTITY = auto()
    PATTERN = auto(); INSIGHT = auto(); SOFTWARE = auto(); HARDWARE = auto()
    PROJECT = auto(); RESEARCH = auto(); ASSET = auto()

class EdgeType(Enum):
    CONTAINS = auto(); REFERENCES = auto(); SIMILAR_TO = auto(); DEPENDS_ON = auto()
    DERIVED_FROM = auto(); RELATES_TO = auto(); INSTANCE_OF = auto(); PART_OF = auto()

@dataclass
class KnowledgeNode:
    id: str; name: str; type: NodeType; path: Optional[str] = None
    description: Optional[str] = None; metadata: Dict[str, Any] = field(default_factory=dict)
    snr_score: float = 0.0; created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self); d['type'] = self.type.name; return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'KnowledgeNode':
        d = d.copy(); d['type'] = NodeType[d['type']]; return cls(**d)

@dataclass
class KnowledgeEdge:
    source_id: str; target_id: str; type: EdgeType; weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self); d['type'] = self.type.name; return d
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'KnowledgeEdge':
        d = d.copy(); d['type'] = EdgeType[d['type']]; return cls(**d)

@dataclass
class ThoughtNode:
    id: str; thought: str; confidence: float; parent_id: Optional[str] = None
    children: List[str] = field(default_factory=list); evidence: List[str] = field(default_factory=list)

@dataclass
class QueryResult:
    nodes: List[KnowledgeNode]; edges: List[KnowledgeEdge]; paths: List[List[str]]
    insights: List[str]; snr_score: float; reasoning_chain: List[ThoughtNode]


# ═══════════════════════════════════════════════════════════════════════════════════════
# GRAPH OF THOUGHTS ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════

class GraphOfThoughts:
    """Multi-path reasoning via thought graph. Giants: Yao/Besta/Wei (ToT, GoT, CoT)."""
    
    def __init__(self):
        self.thoughts: Dict[str, ThoughtNode] = {}
        self.root_id: Optional[str] = None
        self._lock = threading.Lock()
    
    def _gen_id(self) -> str:
        return hashlib.blake2b(f"{datetime.now().isoformat()}{len(self.thoughts)}".encode(), digest_size=16).hexdigest()[:12]
    
    def create_root(self, thought: str) -> ThoughtNode:
        nid = self._gen_id()
        node = ThoughtNode(id=nid, thought=thought, confidence=1.0)
        with self._lock: self.thoughts[nid] = node; self.root_id = nid
        return node
    
    def branch(self, parent_id: str, thought: str, confidence: float, evidence: List[str] = None) -> ThoughtNode:
        nid = self._gen_id()
        node = ThoughtNode(id=nid, thought=thought, confidence=confidence, parent_id=parent_id, evidence=evidence or [])
        with self._lock:
            self.thoughts[nid] = node
            if parent_id in self.thoughts: self.thoughts[parent_id].children.append(nid)
        return node
    
    def merge(self, node_ids: List[str], merged_thought: str) -> ThoughtNode:
        nid = self._gen_id()
        confs = [self.thoughts[n].confidence for n in node_ids if n in self.thoughts]
        merged_conf = math.prod(confs) ** (1/len(confs)) if confs else 0.5
        all_ev = []; [all_ev.extend(self.thoughts[n].evidence) for n in node_ids if n in self.thoughts]
        node = ThoughtNode(id=nid, thought=merged_thought, confidence=merged_conf, evidence=list(set(all_ev)))
        with self._lock:
            self.thoughts[nid] = node
            for n in node_ids:
                if n in self.thoughts: self.thoughts[n].children.append(nid)
        return node
    
    def get_best_path(self) -> List[ThoughtNode]:
        if not self.root_id: return []
        path, cur = [], self.root_id
        while cur:
            node = self.thoughts.get(cur)
            if not node: break
            path.append(node)
            if node.children:
                child_confs = [(c, self.thoughts[c].confidence) for c in node.children if c in self.thoughts]
                cur = max(child_confs, key=lambda x: x[1])[0] if child_confs else None
            else: break
        return path
    
    def get_leaves(self) -> List[ThoughtNode]:
        return [n for n in self.thoughts.values() if not n.children]


# ═══════════════════════════════════════════════════════════════════════════════════════
# SNR MAXIMIZATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════════════

class SNREngine:
    """Signal-to-Noise Ratio optimization. Giants: Shannon, TF-IDF, PageRank."""
    
    @staticmethod
    def calculate_node_snr(node: KnowledgeNode, edge_count: int, total_nodes: int, avg_edges: float) -> float:
        connectivity = min(edge_count / max(avg_edges * 2, 1), 1.0)
        metadata_signal = min(len(node.metadata) / 5, 1.0)
        desc_signal = min(len(node.description or "") / 200, 1.0)
        return round(connectivity * 0.4 + metadata_signal * 0.3 + desc_signal * 0.3, 4)
    
    @staticmethod
    def calculate_query_snr(results: List[KnowledgeNode], terms: List[str], paths: int) -> float:
        if not results: return 0.0
        coverage = sum(1 for t in terms if any(t in (n.name + (n.description or "")).lower() for n in results))
        term_sig = coverage / max(len(terms), 1)
        avg_snr = sum(n.snr_score for n in results) / len(results)
        path_sig = min(paths / 3, 1.0)
        return round(term_sig * 0.4 + avg_snr * 0.4 + path_sig * 0.2, 4)


# ═══════════════════════════════════════════════════════════════════════════════════════
# SOVEREIGN KNOWLEDGE NEXUS — THE CORE
# ═══════════════════════════════════════════════════════════════════════════════════════

class SovereignKnowledgeNexus:
    """The Peak Masterpiece: Unified knowledge graph for Node0."""
    
    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, List[KnowledgeEdge]] = defaultdict(list)
        self.reverse_edges: Dict[str, List[KnowledgeEdge]] = defaultdict(list)
        self.type_index: Dict[NodeType, Set[str]] = defaultdict(set)
        self.name_index: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.RLock()
        self.stats = {"nodes": 0, "edges": 0, "types": {}, "created_at": None, "last_updated": None}
        log.info("🔮 Sovereign Knowledge Nexus initialized")
    
    def add_node(self, node: KnowledgeNode) -> str:
        with self._lock:
            self.nodes[node.id] = node
            self.type_index[node.type].add(node.id)
            for word in re.findall(r'\w+', node.name.lower()):
                if len(word) > 2: self.name_index[word].add(node.id)
            self.stats["nodes"] = len(self.nodes)
            self.stats["types"][node.type.name] = len(self.type_index[node.type])
        return node.id
    
    def add_edge(self, edge: KnowledgeEdge) -> None:
        with self._lock:
            self.edges[edge.source_id].append(edge)
            self.reverse_edges[edge.target_id].append(edge)
            self.stats["edges"] += 1
    
    def get_neighbors(self, node_id: str, direction: str = "both") -> List[KnowledgeNode]:
        neighbors = []
        if direction in ("out", "both"):
            for e in self.edges.get(node_id, []):
                if e.target_id in self.nodes: neighbors.append(self.nodes[e.target_id])
        if direction in ("in", "both"):
            for e in self.reverse_edges.get(node_id, []):
                if e.source_id in self.nodes: neighbors.append(self.nodes[e.source_id])
        return neighbors
    
    def index_node0_inventory(self) -> int:
        log.info("📊 Indexing Node0 inventory...")
        if not Config.NODE0_INVENTORY.exists(): return 0
        with open(Config.NODE0_INVENTORY, 'r', encoding='utf-8') as f: inv = json.load(f)
        n = 0
        
        # Root
        node0 = KnowledgeNode(id="node0_root", name="BIZRA Node0", type=NodeType.HARDWARE,
            description="Genesis Block. MSI Titan 18 HX. The complete machine.",
            metadata={"cpu": inv.get("hardware",{}).get("cpu",{}).get("name"),
                      "ram_gb": inv.get("hardware",{}).get("memory",{}).get("total_gb"),
                      "gpu": inv.get("hardware",{}).get("gpu",[{}])[0].get("name")})
        self.add_node(node0); n += 1
        
        # Hardware
        hw = inv.get("hardware", {})
        if hw.get("cpu"):
            cpu = KnowledgeNode(id="hw_cpu", name=hw["cpu"].get("name","CPU"), type=NodeType.HARDWARE, metadata=hw["cpu"])
            self.add_node(cpu); self.add_edge(KnowledgeEdge("node0_root","hw_cpu",EdgeType.CONTAINS)); n += 1
        
        for i, gpu in enumerate(hw.get("gpu", [])):
            g = KnowledgeNode(id=f"hw_gpu_{i}", name=gpu.get("name",f"GPU {i}"), type=NodeType.HARDWARE, metadata=gpu)
            self.add_node(g); self.add_edge(KnowledgeEdge("node0_root",f"hw_gpu_{i}",EdgeType.CONTAINS)); n += 1
        
        # Software
        for prog in inv.get("software",{}).get("programs",[]):
            pid = f"sw_{hashlib.blake2b(prog.encode(), digest_size=16).hexdigest()[:8]}"
            self.add_node(KnowledgeNode(id=pid, name=prog, type=NodeType.SOFTWARE))
            self.add_edge(KnowledgeEdge("node0_root", pid, EdgeType.CONTAINS)); n += 1
        
        for distro in inv.get("software",{}).get("wsl_distros",[]):
            did = f"wsl_{hashlib.blake2b(distro.encode(), digest_size=16).hexdigest()[:8]}"
            self.add_node(KnowledgeNode(id=did, name=f"WSL: {distro}", type=NodeType.SOFTWARE, metadata={"runtime":"WSL2"}))
            self.add_edge(KnowledgeEdge("node0_root", did, EdgeType.CONTAINS)); n += 1
        
        # Folders
        for folder in inv.get("data",{}).get("all_folders",[]):
            fid = f"folder_{hashlib.blake2b(folder['path'].encode(), digest_size=16).hexdigest()[:8]}"
            fn = KnowledgeNode(id=fid, name=folder["name"], type=NodeType.FOLDER, path=folder["path"],
                               metadata={"is_bizra": folder.get("is_bizra", False)})
            self.add_node(fn); self.add_edge(KnowledgeEdge("node0_root", fid, EdgeType.CONTAINS))
            if folder.get("is_bizra"): self.add_edge(KnowledgeEdge(fid, "node0_root", EdgeType.PART_OF, 2.0))
            n += 1
        
        # User folders
        user = inv.get("user_data",{}).get("folders",{})
        dl = user.get("downloads",{})
        if dl:
            dn = KnowledgeNode(id="folder_downloads", name="Downloads", type=NodeType.FOLDER, path=dl.get("path"),
                              metadata={"folders": dl.get("top_level_folders",0), "bizra": len(dl.get("bizra_related_folders",[]))})
            self.add_node(dn); self.add_edge(KnowledgeEdge("node0_root", "folder_downloads", EdgeType.CONTAINS)); n += 1
            
            for fname in dl.get("folder_names",[]):
                fid = f"dl_{hashlib.blake2b(fname.encode(), digest_size=16).hexdigest()[:8]}"
                is_biz = fname in dl.get("bizra_related_folders",[])
                self.add_node(KnowledgeNode(id=fid, name=fname, type=NodeType.FOLDER, path=f"{dl.get('path')}\\{fname}",
                                           metadata={"is_bizra": is_biz, "location": "downloads"}))
                self.add_edge(KnowledgeEdge("folder_downloads", fid, EdgeType.CONTAINS))
                if is_biz: self.add_edge(KnowledgeEdge(fid, "node0_root", EdgeType.PART_OF))
                n += 1
        
        log.info(f"  ✓ Indexed {n} nodes from inventory")
        return n
    
    def index_knowledge_files(self) -> int:
        log.info("📚 Indexing knowledge files...")
        n = 0
        for jf in Config.KNOWLEDGE.glob("*.jsonl"):
            with open(jf, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        e = json.loads(line)
                        et = e.get("type","concept").lower()
                        nt = {"folder":NodeType.FOLDER,"software":NodeType.SOFTWARE,"organization":NodeType.ENTITY,
                              "technology":NodeType.CONCEPT,"pattern":NodeType.PATTERN,"insight":NodeType.INSIGHT,
                              "reference":NodeType.RESEARCH}.get(et, NodeType.CONCEPT)
                        node = KnowledgeNode(id=e.get("id",hashlib.blake2b(line.encode(), digest_size=16).hexdigest()[:12]),
                                            name=e.get("name","?"), type=nt, path=e.get("path"),
                                            description=e.get("description"), metadata={k:v for k,v in e.items() 
                                                if k not in ("id","name","type","path","description")})
                        self.add_node(node); n += 1
                        if e.get("parent"): self.add_edge(KnowledgeEdge(e["parent"], node.id, EdgeType.CONTAINS))
                    except: continue
        log.info(f"  ✓ Indexed {n} nodes from knowledge files")
        return n
    
    def _calc_snr_all(self):
        log.info("📈 Calculating SNR scores...")
        total = sum(len(e) for e in self.edges.values())
        avg = total / max(len(self.nodes), 1)
        for nid, node in self.nodes.items():
            ec = len(self.edges.get(nid,[])) + len(self.reverse_edges.get(nid,[]))
            node.snr_score = SNREngine.calculate_node_snr(node, ec, len(self.nodes), avg)
    
    def build_full_index(self) -> Dict[str, int]:
        log.info("═" * 70)
        log.info("🏗️  BUILDING SOVEREIGN KNOWLEDGE NEXUS")
        log.info("═" * 70)
        results = {"inventory": self.index_node0_inventory(), "knowledge": self.index_knowledge_files()}
        self._calc_snr_all()
        self.stats["last_updated"] = datetime.now(timezone.utc).isoformat()
        if not self.stats["created_at"]: self.stats["created_at"] = self.stats["last_updated"]
        self.save()
        log.info("═" * 70)
        log.info("📊 INDEX COMPLETE")
        log.info(f"  Total Nodes: {self.stats['nodes']}")
        log.info(f"  Total Edges: {self.stats['edges']}")
        log.info("═" * 70)
        return results
    
    def query(self, query_text: str, max_results: int = 20) -> QueryResult:
        log.info(f"🔍 Query: '{query_text}'")
        got = GraphOfThoughts()
        terms = [t.lower() for t in re.findall(r'\w+', query_text) if len(t) > 2]
        got.create_root(f"Find: {query_text}")
        
        # Direct match
        matched: Set[str] = set()
        for term in terms:
            if term in self.name_index: matched.update(self.name_index[term])
        got.branch(got.root_id, f"Direct: {len(matched)}", 0.8, list(matched)[:5])
        
        # Expand
        expanded = set(matched)
        for nid in matched:
            for n in self.get_neighbors(nid)[:10]: expanded.add(n.id)
        got.branch(got.root_id, f"Expanded: {len(expanded)}", 0.7)
        
        # Rank
        ranked = sorted([self.nodes[n] for n in expanded if n in self.nodes], key=lambda x: x.snr_score, reverse=True)[:max_results]
        
        # Paths
        paths = []
        if len(matched) >= 2:
            ml = list(matched)[:4]
            for i in range(len(ml)):
                for j in range(i+1, len(ml)):
                    p = self._find_path(ml[i], ml[j])
                    if p: paths.append(p)
        
        # Insights
        insights = self._gen_insights(ranked, terms)
        got.merge([t.id for t in got.get_leaves()], f"Synthesized {len(ranked)} results")
        
        # Edges
        rids = {n.id for n in ranked}
        edges = [e for n in ranked for e in self.edges.get(n.id,[]) if e.target_id in rids]
        
        return QueryResult(nodes=ranked, edges=edges, paths=paths, insights=insights,
                          snr_score=SNREngine.calculate_query_snr(ranked, terms, len(paths)),
                          reasoning_chain=got.get_best_path())
    
    def _find_path(self, start: str, end: str, max_d: int = 4) -> Optional[List[str]]:
        if start == end: return [start]
        visited, queue = {start}, [(start, [start])]
        while queue:
            cur, path = queue.pop(0)
            if len(path) > max_d: continue
            for e in self.edges.get(cur, []):
                if e.target_id == end: return path + [end]
                if e.target_id not in visited:
                    visited.add(e.target_id); queue.append((e.target_id, path + [e.target_id]))
        return None
    
    def _gen_insights(self, nodes: List[KnowledgeNode], terms: List[str]) -> List[str]:
        ins = []
        tc = defaultdict(int); [tc.__setitem__(n.type.name, tc[n.type.name]+1) for n in nodes]
        if tc: dom = max(tc.items(), key=lambda x:x[1]); ins.append(f"Dominated by {dom[0]} ({dom[1]}/{len(nodes)})")
        biz = sum(1 for n in nodes if n.metadata.get("is_bizra") or "bizra" in n.name.lower())
        if biz: ins.append(f"{biz} BIZRA assets found")
        hi = [n for n in nodes if n.snr_score > 0.6]
        if hi: ins.append(f"{len(hi)} high-SNR nodes (>0.6)")
        return ins
    
    def save(self):
        Config.GOLD.mkdir(parents=True, exist_ok=True)
        gd = {"nodes": [n.to_dict() for n in self.nodes.values()], 
              "edges": [e.to_dict() for el in self.edges.values() for e in el]}
        with open(Config.NEXUS_GRAPH, 'w', encoding='utf-8') as f: json.dump(gd, f, indent=2, ensure_ascii=False)
        with open(Config.NEXUS_STATS, 'w', encoding='utf-8') as f: json.dump(self.stats, f, indent=2)
        log.info(f"💾 Saved to {Config.NEXUS_GRAPH}")
    
    def load(self) -> bool:
        if not Config.NEXUS_GRAPH.exists(): return False
        try:
            with open(Config.NEXUS_GRAPH, 'r', encoding='utf-8') as f: data = json.load(f)
            for nd in data.get("nodes",[]): self.add_node(KnowledgeNode.from_dict(nd))
            for ed in data.get("edges",[]): self.add_edge(KnowledgeEdge.from_dict(ed))
            if Config.NEXUS_STATS.exists():
                with open(Config.NEXUS_STATS, 'r', encoding='utf-8') as f: self.stats = json.load(f)
            log.info(f"📂 Loaded {len(self.nodes)} nodes")
            return True
        except Exception as e: log.error(f"Load failed: {e}"); return False


# ═══════════════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════════════

def main():
    import sys
    nexus = SovereignKnowledgeNexus()
    
    if len(sys.argv) < 2:
        print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║       SOVEREIGN KNOWLEDGE NEXUS — The Peak Masterpiece                        ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  python sovereign_nexus.py build      — Build full knowledge index            ║
║  python sovereign_nexus.py query "X"  — Query the knowledge graph             ║
║  python sovereign_nexus.py stats      — Show graph statistics                 ║
║  python sovereign_nexus.py bizra      — Find all BIZRA assets                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
        """)
        return
    
    cmd = sys.argv[1].lower()
    
    if cmd == "build":
        nexus.build_full_index()
    elif cmd == "query" and len(sys.argv) > 2:
        if nexus.load():
            q = " ".join(sys.argv[2:])
            r = nexus.query(q)
            print(f"\n🔍 RESULTS (SNR: {r.snr_score})\n" + "═"*60)
            for n in r.nodes[:15]:
                bar = "█" * int(n.snr_score * 10)
                print(f"  [{n.type.name:10}] {n.name[:40]:40} {bar}")
            if r.insights: print("\n💡 " + " | ".join(r.insights))
            print("═"*60)
        else: print("❌ Run 'build' first")
    elif cmd == "stats":
        if nexus.load():
            print(f"\n📊 STATS\n{'═'*50}")
            print(f"  Nodes: {nexus.stats['nodes']} | Edges: {nexus.stats['edges']}")
            print(f"  Types: {nexus.stats.get('types',{})}")
        else: print("❌ Run 'build' first")
    elif cmd == "bizra":
        if nexus.load():
            print(f"\n🌱 BIZRA ASSETS\n{'═'*50}")
            biz = [n for n in nexus.nodes.values() if n.metadata.get("is_bizra") or "bizra" in n.name.lower()]
            for n in sorted(biz, key=lambda x: x.snr_score, reverse=True)[:30]:
                print(f"  [{n.type.name:8}] {n.name}")
            print(f"\n  Total: {len(biz)}\n{'═'*50}")
        else: print("❌ Run 'build' first")
    else: print(f"Unknown: {cmd}")

if __name__ == "__main__": main()
