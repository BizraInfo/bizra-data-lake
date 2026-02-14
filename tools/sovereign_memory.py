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

import json
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any, Iterable
from dataclasses import dataclass
from collections import defaultdict

from bizra_config import DATA_LAKE_ROOT, GOLD_PATH, INDEXED_PATH

CATALOG_PATH = GOLD_PATH / "sovereign_catalog.parquet"
CLAUDE_FLOW_DB = DATA_LAKE_ROOT / ".swarm/memory.db"
CLAUDE_FLOW_JSON_DIR = DATA_LAKE_ROOT / ".claude-flow/memory"

CLAUDE_FLOW_TYPE_SNR = {
    "working": 0.55,
    "episodic": 0.70,
    "semantic": 0.85,
    "procedural": 0.90,
    "pattern": 0.88,
}


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


# ═══════════════════════════════════════════════════════════════════════
# CLAUDE-FLOW MEMORY ADAPTER (MCP-ALIGNED)
# ═══════════════════════════════════════════════════════════════════════
class ClaudeFlowMemory:
    """Lightweight adapter for Claude-Flow memory sources."""

    def __init__(
        self,
        db_path: Path = CLAUDE_FLOW_DB,
        json_dir: Path = CLAUDE_FLOW_JSON_DIR,
        type_snr: Optional[Dict[str, float]] = None,
    ):
        self.db_path = Path(db_path)
        self.json_dir = Path(json_dir)
        self.type_snr = type_snr or dict(CLAUDE_FLOW_TYPE_SNR)

    def is_ready(self) -> bool:
        return self.db_path.exists() or self._has_json()

    def _has_json(self) -> bool:
        return self.json_dir.exists() and any(self.json_dir.glob("*.json"))

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        return conn

    def _ms_to_datetime(self, ms: Optional[int]) -> Optional[datetime]:
        if ms is None:
            return None
        try:
            return datetime.fromtimestamp(ms / 1000)
        except Exception:
            return None

    def _recency_bonus(self, modified: Optional[datetime]) -> float:
        if not modified:
            return 0.0
        days_old = max(0, (datetime.now() - modified).days)
        return max(0.0, 1.0 - (days_old / 365.0)) * 0.3

    def _score(self, haystack: str, terms: List[str], snr: float, modified: Optional[datetime], access_count: int) -> float:
        term_hits = sum(1 for t in terms if t in haystack)
        if term_hits == 0:
            return 0.0
        access_bonus = min(max(access_count, 0) / 10.0, 1.0) * 0.1
        return (term_hits * 0.5) + (snr * 0.3) + self._recency_bonus(modified) + access_bonus

    def _iter_json_entries(self) -> Iterable[Dict[str, Any]]:
        if not self._has_json():
            return []
        entries: List[Dict[str, Any]] = []
        for path in sorted(self.json_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                content = json.dumps(data, ensure_ascii=True)
                entries.append({
                    "id": path.stem,
                    "key": path.stem,
                    "namespace": "claude-flow",
                    "type": "session",
                    "content": content,
                    "tags": None,
                    "metadata": None,
                    "owner_id": None,
                    "updated_at": int(path.stat().st_mtime * 1000),
                    "created_at": int(path.stat().st_ctime * 1000),
                    "access_count": 0,
                    "status": "active",
                    "path": path,
                })
            except Exception:
                continue
        return entries

    def _query_db_candidates(self, terms: List[str], limit: int) -> List[sqlite3.Row]:
        if not self.db_path.exists():
            return []
        clauses = []
        params: List[Any] = []
        for term in terms:
            like = f"%{term}%"
            clauses.append(
                "(content LIKE ? COLLATE NOCASE OR key LIKE ? COLLATE NOCASE OR tags LIKE ? COLLATE NOCASE OR metadata LIKE ? COLLATE NOCASE)"
            )
            params.extend([like, like, like, like])
        where = " OR ".join(clauses) if clauses else "1=1"
        sql = (
            "SELECT id, key, namespace, type, content, tags, metadata, owner_id, "
            "created_at, updated_at, last_accessed_at, access_count, status "
            "FROM memory_entries "
            "WHERE status = 'active' AND (" + where + ") "
            "ORDER BY updated_at DESC "
            "LIMIT ?"
        )
        params.append(limit)
        try:
            conn = self._connect()
            try:
                return conn.execute(sql, params).fetchall()
            finally:
                conn.close()
        except sqlite3.Error:
            return []

    def search(self, query: str, top_k: int = 10) -> List[MemoryTrace]:
        terms = [t for t in query.lower().split() if t]
        if not terms:
            return []

        traces: List[MemoryTrace] = []
        candidate_rows = self._query_db_candidates(terms, limit=max(200, top_k * 20))

        for row in candidate_rows:
            key = row["key"] or ""
            content = row["content"] or ""
            tags = row["tags"] or ""
            metadata = row["metadata"] or ""
            haystack = f"{key} {content} {tags} {metadata}".lower()
            snr = self.type_snr.get(row["type"] or "semantic", 0.65)
            modified = self._ms_to_datetime(row["updated_at"] or row["created_at"])
            score = self._score(haystack, terms, snr, modified, row["access_count"] or 0)
            if score <= 0:
                continue

            traces.append(MemoryTrace(
                path=f"claude-flow://memory_entries/{row['id']}",
                name=key or row["id"],
                kind=f"Memory/{row['type']}",
                domain="claude-flow",
                modified=modified or datetime.now(),
                snr_score=snr,
                relevance=score,
                context=f"namespace={row['namespace']}",
            ))

        for entry in self._iter_json_entries():
            content = entry.get("content") or ""
            key = entry.get("key") or entry.get("id") or "session"
            haystack = f"{key} {content}".lower()
            snr = 0.7
            modified = self._ms_to_datetime(entry.get("updated_at"))
            score = self._score(haystack, terms, snr, modified, 0)
            if score <= 0:
                continue
            traces.append(MemoryTrace(
                path=f"claude-flow://memory_files/{key}",
                name=key,
                kind="Memory/session",
                domain="claude-flow",
                modified=modified or datetime.now(),
                snr_score=snr,
                relevance=score,
                context="source=.claude-flow/memory",
            ))

        traces.sort(key=lambda t: t.relevance, reverse=True)
        return traces[:top_k]

    def get_temporal_context(self, reference_date: datetime, window_days: int = 7) -> Dict[str, List[MemoryTrace]]:
        if not self.is_ready():
            return {}
        start = reference_date - timedelta(days=window_days // 2)
        end = reference_date + timedelta(days=window_days // 2)
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)

        traces: List[MemoryTrace] = []
        if self.db_path.exists():
            sql = (
                "SELECT id, key, namespace, type, content, created_at, updated_at, access_count "
                "FROM memory_entries "
                "WHERE status = 'active' AND updated_at BETWEEN ? AND ? "
                "ORDER BY updated_at DESC LIMIT 50"
            )
            try:
                conn = self._connect()
                try:
                    rows = conn.execute(sql, (start_ms, end_ms)).fetchall()
                finally:
                    conn.close()
            except sqlite3.Error:
                rows = []

            for row in rows:
                modified = self._ms_to_datetime(row["updated_at"] or row["created_at"])
                snr = self.type_snr.get(row["type"] or "semantic", 0.65)
                traces.append(MemoryTrace(
                    path=f"claude-flow://memory_entries/{row['id']}",
                    name=row["key"] or row["id"],
                    kind=f"Memory/{row['type']}",
                    domain="claude-flow",
                    modified=modified or datetime.now(),
                    snr_score=snr,
                    relevance=snr,
                    context=f"namespace={row['namespace']}",
                ))

        for entry in self._iter_json_entries():
            modified = self._ms_to_datetime(entry.get("updated_at"))
            if not modified or not (start <= modified <= end):
                continue
            traces.append(MemoryTrace(
                path=f"claude-flow://memory_files/{entry.get('key')}",
                name=entry.get("key") or "session",
                kind="Memory/session",
                domain="claude-flow",
                modified=modified,
                snr_score=0.7,
                relevance=0.7,
                context="source=.claude-flow/memory",
            ))

        return {"claude-flow": traces}

    def trace_concept_lineage(self, concept: str) -> List[Tuple[str, datetime, str]]:
        if not self.is_ready():
            return []
        concept_lower = concept.lower()
        lineage: List[Tuple[str, datetime, str]] = []

        if self.db_path.exists():
            like = f"%{concept_lower}%"
            sql = (
                "SELECT key, created_at, updated_at FROM memory_entries "
                "WHERE status = 'active' AND (key LIKE ? COLLATE NOCASE OR content LIKE ? COLLATE NOCASE) "
                "ORDER BY updated_at ASC LIMIT 200"
            )
            try:
                conn = self._connect()
                try:
                    rows = conn.execute(sql, (like, like)).fetchall()
                finally:
                    conn.close()
            except sqlite3.Error:
                rows = []
            for row in rows:
                modified = self._ms_to_datetime(row["updated_at"] or row["created_at"]) or datetime.now()
                name = row["key"] or row["id"]
                lineage.append((name, modified, "claude-flow"))

        for entry in self._iter_json_entries():
            content = entry.get("content") or ""
            key = entry.get("key") or entry.get("id") or "session"
            if concept_lower in content.lower() or concept_lower in key.lower():
                modified = self._ms_to_datetime(entry.get("updated_at")) or datetime.now()
                lineage.append((key, modified, "claude-flow"))

        lineage.sort(key=lambda item: item[1])
        return lineage

    def get_domain_analytics(self) -> Dict[str, Dict]:
        if not self.is_ready():
            return {}

        type_counts: Dict[str, int] = defaultdict(int)
        total_size_bytes = 0
        recent_activity = 0
        now_ms = int(datetime.now().timestamp() * 1000)
        cutoff_ms = now_ms - (30 * 24 * 60 * 60 * 1000)

        if self.db_path.exists():
            try:
                conn = self._connect()
                try:
                    for row in conn.execute(
                        "SELECT type, COUNT(*) AS count FROM memory_entries WHERE status = 'active' GROUP BY type"
                    ):
                        type_counts[row["type"] or "semantic"] += int(row["count"])
                    row = conn.execute(
                        "SELECT SUM(LENGTH(content)) AS total_size FROM memory_entries WHERE status = 'active'"
                    ).fetchone()
                    total_size_bytes += int(row["total_size"] or 0)
                    row = conn.execute(
                        "SELECT COUNT(*) AS count FROM memory_entries WHERE status = 'active' AND updated_at >= ?",
                        (cutoff_ms,),
                    ).fetchone()
                    recent_activity += int(row["count"] or 0)
                finally:
                    conn.close()
            except sqlite3.Error:
                pass

        for entry in self._iter_json_entries():
            content = entry.get("content") or ""
            total_size_bytes += len(content.encode("utf-8"))
            type_counts["session"] += 1
            updated_at = entry.get("updated_at") or 0
            if updated_at >= cutoff_ms:
                recent_activity += 1

        total_nodes = sum(type_counts.values())
        avg_snr = 0.0
        if total_nodes > 0:
            avg_snr = sum(self.type_snr.get(t, 0.65) * c for t, c in type_counts.items() if t != "session")
            avg_snr += type_counts.get("session", 0) * 0.7
            avg_snr /= total_nodes

        kind_breakdown = {f"Memory/{k}": v for k, v in type_counts.items()}

        return {
            "claude-flow": {
                "total_nodes": total_nodes,
                "total_size_gb": total_size_bytes / (1024 ** 3),
                "avg_snr": avg_snr,
                "kind_breakdown": kind_breakdown,
                "recent_activity": recent_activity,
            }
        }


# ═══════════════════════════════════════════════════════════════════════
# UNIFIED MEMORY (M6 + CLAUDE-FLOW)
# ═══════════════════════════════════════════════════════════════════════
class UnifiedMemory(SovereignMemory):
    """Unified memory interface combining Sovereign Catalog and Claude-Flow."""

    def __init__(self, include_claude_flow: bool = True):
        super().__init__()
        self._claude_flow = ClaudeFlowMemory() if include_claude_flow else None

    def is_ready(self) -> bool:
        claude_ready = self._claude_flow is not None and self._claude_flow.is_ready()
        return super().is_ready() or claude_ready

    def search_across_domains(self, query: str, top_k: int = 10) -> List[MemoryTrace]:
        results = super().search_across_domains(query, top_k=top_k)
        if self._claude_flow and self._claude_flow.is_ready():
            results.extend(self._claude_flow.search(query, top_k=top_k))
            results = sorted(results, key=lambda r: r.relevance, reverse=True)[:top_k]
        return results

    def get_temporal_context(self, reference_date: datetime, window_days: int = 7) -> Dict[str, List[MemoryTrace]]:
        context = super().get_temporal_context(reference_date, window_days)
        if self._claude_flow and self._claude_flow.is_ready():
            context.update(self._claude_flow.get_temporal_context(reference_date, window_days))
        return context

    def trace_concept_lineage(self, concept: str) -> List[Tuple[str, datetime, str]]:
        lineage = super().trace_concept_lineage(concept)
        if self._claude_flow and self._claude_flow.is_ready():
            lineage.extend(self._claude_flow.trace_concept_lineage(concept))
            lineage.sort(key=lambda item: item[1])
        return lineage

    def get_domain_analytics(self) -> Dict[str, Dict]:
        analytics = super().get_domain_analytics()
        if self._claude_flow and self._claude_flow.is_ready():
            analytics.update(self._claude_flow.get_domain_analytics())
        return analytics

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
