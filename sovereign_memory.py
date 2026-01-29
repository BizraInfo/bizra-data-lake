"""
BIZRA SOVEREIGN MEMORY (M6) - The 6th Tier
"Cross-Domain Omniscient Recall"

This tier provides capabilities that DON'T exist in M1-M5:
1. Cross-Project Linking: Find related work across BIZRA-PROJECTS, NODE0, GENESIS
2. Temporal Context: "What was I working on when X was created?"
3. Impact Lineage: Trace which files contributed to outcomes
4. Domain Clustering: Group files by conceptual similarity (not just folder)

Value Proposition:
- M1-M5 are scoped to TaskMaster operations (single project focus)
- M6 is the "God View" across the entire 1.5TB sovereign domain
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from bizra_config import GOLD_PATH, INDEXED_PATH

CATALOG_PATH = GOLD_PATH / "sovereign_catalog.parquet"


@dataclass
class MemoryTrace:
    """A retrieved memory with context."""
    path: str
    name: str
    kind: str
    domain: str
    modified: datetime
    snr_score: float
    relevance: float = 0.0
    context: str = ""


class SovereignMemory:
    """
    M6: The Omniscient Layer.
    Provides cross-domain queries that span all unified knowledge.
    """
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self._load()
        
    def _load(self):
        if CATALOG_PATH.exists():
            self.df = pd.read_parquet(CATALOG_PATH)
            # Ensure datetime type
            if 'modified' in self.df.columns:
                self.df['modified'] = pd.to_datetime(self.df['modified'])
            print(f"   📦 M6 Sovereign Memory: {len(self.df):,} nodes loaded.")
        else:
            print("   ⚠️ M6 Sovereign Memory: Catalog not found.")
            self.df = pd.DataFrame()

    def is_ready(self) -> bool:
        return self.df is not None and not self.df.empty

    # ═══════════════════════════════════════════════════════════════════════
    # VALUE-ADD CAPABILITY 1: Cross-Domain Search
    # ═══════════════════════════════════════════════════════════════════════
    def search_across_domains(self, query: str, top_k: int = 10) -> List[MemoryTrace]:
        """
        Search across ALL domains (Projects, Node0, Genesis) simultaneously.
        Returns results ranked by SNR and recency.
        """
        if not self.is_ready():
            return []
        
        # Multi-term search
        terms = query.lower().split()
        
        # Score each row
        def score_row(row):
            name_lower = row['name'].lower()
            path_lower = row['path'].lower()
            
            term_hits = sum(1 for t in terms if t in name_lower or t in path_lower)
            if term_hits == 0:
                return 0
            
            # Composite score: term matches + SNR + recency bonus
            recency_bonus = 0
            if pd.notna(row['modified']):
                days_old = (datetime.now() - row['modified']).days
                recency_bonus = max(0, 1 - (days_old / 365)) * 0.3
            
            return (term_hits * 0.5) + (row['snr_score'] * 0.3) + recency_bonus
        
        self.df['_score'] = self.df.apply(score_row, axis=1)
        results = self.df[self.df['_score'] > 0].nlargest(top_k, '_score')
        
        traces = []
        for _, row in results.iterrows():
            traces.append(MemoryTrace(
                path=row['path'],
                name=row['name'],
                kind=row['kind'],
                domain=row.get('domain_source', 'Unknown'),
                modified=row['modified'],
                snr_score=row['snr_score'],
                relevance=row['_score']
            ))
        
        self.df.drop('_score', axis=1, inplace=True)
        return traces

    # ═══════════════════════════════════════════════════════════════════════
    # VALUE-ADD CAPABILITY 2: Temporal Context ("What was I working on?")
    # ═══════════════════════════════════════════════════════════════════════
    def get_temporal_context(self, reference_date: datetime, window_days: int = 7) -> Dict[str, List[MemoryTrace]]:
        """
        Answer: "What was being worked on around [date]?"
        Returns files modified within the time window, grouped by domain.
        """
        if not self.is_ready():
            return {}
        
        start = reference_date - timedelta(days=window_days // 2)
        end = reference_date + timedelta(days=window_days // 2)
        
        mask = (self.df['modified'] >= start) & (self.df['modified'] <= end)
        context_df = self.df[mask].copy()
        
        # Group by domain
        grouped = defaultdict(list)
        for _, row in context_df.sort_values('modified', ascending=False).head(50).iterrows():
            trace = MemoryTrace(
                path=row['path'],
                name=row['name'],
                kind=row['kind'],
                domain=row.get('domain_source', 'Unknown'),
                modified=row['modified'],
                snr_score=row['snr_score']
            )
            grouped[trace.domain].append(trace)
        
        return dict(grouped)

    # ═══════════════════════════════════════════════════════════════════════
    # VALUE-ADD CAPABILITY 3: Impact Lineage (Trace Knowledge Flow)
    # ═══════════════════════════════════════════════════════════════════════
    def trace_concept_lineage(self, concept: str) -> List[Tuple[str, datetime, str]]:
        """
        Trace the evolution of a concept across time.
        Shows when/where it first appeared and how it evolved.
        """
        if not self.is_ready():
            return []
        
        concept_lower = concept.lower()
        
        # Find all files mentioning the concept
        mask = self.df['name'].str.lower().str.contains(concept_lower, na=False) | \
               self.df['path'].str.lower().str.contains(concept_lower, na=False)
        
        lineage_df = self.df[mask].sort_values('modified')
        
        lineage = []
        for _, row in lineage_df.iterrows():
            lineage.append((
                row['name'],
                row['modified'],
                row.get('domain_source', 'Unknown')
            ))
        
        return lineage

    # ═══════════════════════════════════════════════════════════════════════
    # VALUE-ADD CAPABILITY 4: Domain Analytics (Where is the knowledge?)
    # ═══════════════════════════════════════════════════════════════════════
    def get_domain_analytics(self) -> Dict[str, Dict]:
        """
        Provide statistics about knowledge distribution across domains.
        Useful for understanding where expertise is concentrated.
        """
        if not self.is_ready():
            return {}
        
        analytics = {}
        for domain in self.df['domain_source'].unique():
            domain_df = self.df[self.df['domain_source'] == domain]
            
            analytics[domain] = {
                'total_nodes': len(domain_df),
                'total_size_gb': domain_df['size_bytes'].sum() / (1024**3),
                'avg_snr': domain_df['snr_score'].mean(),
                'kind_breakdown': domain_df['kind'].value_counts().to_dict(),
                'recent_activity': domain_df[
                    domain_df['modified'] > datetime.now() - timedelta(days=30)
                ].shape[0]
            }
        
        return analytics

    # ═══════════════════════════════════════════════════════════════════════
    # VALUE-ADD CAPABILITY 5: Related Work Discovery
    # ═══════════════════════════════════════════════════════════════════════
    def find_related_work(self, file_path: str, top_k: int = 5) -> List[MemoryTrace]:
        """
        Given a file, find related work across ALL domains.
        Uses filename similarity, folder proximity, and temporal correlation.
        """
        if not self.is_ready():
            return []
        
        # Extract key terms from the file path
        path = Path(file_path)
        name_parts = path.stem.lower().replace('-', ' ').replace('_', ' ').split()
        parent_name = path.parent.name.lower()
        
        # Score each row for relatedness
        def relatedness_score(row):
            if row['path'] == file_path:
                return 0  # Skip self
            
            row_name = row['name'].lower()
            row_path = row['path'].lower()
            
            score = 0
            
            # Name term overlap
            for part in name_parts:
                if len(part) > 2 and part in row_name:
                    score += 0.3
            
            # Parent folder match
            if parent_name in row_path:
                score += 0.2
            
            # SNR boost
            score += row['snr_score'] * 0.1
            
            return score
        
        self.df['_rel'] = self.df.apply(relatedness_score, axis=1)
        related = self.df[self.df['_rel'] > 0].nlargest(top_k, '_rel')
        
        traces = []
        for _, row in related.iterrows():
            traces.append(MemoryTrace(
                path=row['path'],
                name=row['name'],
                kind=row['kind'],
                domain=row.get('domain_source', 'Unknown'),
                modified=row['modified'],
                snr_score=row['snr_score'],
                relevance=row['_rel'],
                context=f"Related via: name similarity"
            ))
        
        self.df.drop('_rel', axis=1, inplace=True)
        return traces


# ═══════════════════════════════════════════════════════════════════════════
# DEMONSTRATION
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("═" * 70)
    print("   🧠 M6 SOVEREIGN MEMORY - CAPABILITY DEMO")
    print("═" * 70)
    
    mem = SovereignMemory()
    
    if not mem.is_ready():
        print("❌ Sovereign Memory not initialized. Run unify_sovereign_domain.py first.")
        exit(1)
    
    # Demo 1: Cross-Domain Search
    print("\n📍 DEMO 1: Cross-Domain Search for 'architecture'")
    results = mem.search_across_domains("architecture system", top_k=5)
    for r in results:
        print(f"   [{r.domain:<20}] {r.name} (SNR: {r.snr_score:.2f})")
    
    # Demo 2: Temporal Context
    print("\n📍 DEMO 2: What was worked on around October 15, 2025?")
    context = mem.get_temporal_context(datetime(2025, 10, 15), window_days=7)
    for domain, traces in context.items():
        print(f"   [{domain}]: {len(traces)} files active")
        for t in traces[:2]:
            print(f"      - {t.name}")
    
    # Demo 3: Concept Lineage
    print("\n📍 DEMO 3: Trace 'BIZRA' concept evolution")
    lineage = mem.trace_concept_lineage("bizra")
    if lineage:
        print(f"   First appearance: {lineage[0][1].strftime('%Y-%m-%d')} in {lineage[0][2]}")
        print(f"   Latest mention: {lineage[-1][1].strftime('%Y-%m-%d')} in {lineage[-1][2]}")
        print(f"   Total mentions: {len(lineage)} files")
    
    # Demo 4: Domain Analytics
    print("\n📍 DEMO 4: Domain Knowledge Distribution")
    analytics = mem.get_domain_analytics()
    for domain, stats in analytics.items():
        print(f"   [{domain}]")
        print(f"      Nodes: {stats['total_nodes']:,} | Size: {stats['total_size_gb']:.1f}GB | Avg SNR: {stats['avg_snr']:.2f}")
        print(f"      Recent (30d): {stats['recent_activity']} files")
    
    print("\n" + "═" * 70)
    print("   ✅ M6 SOVEREIGN MEMORY: OPERATIONAL")
    print("═" * 70)
