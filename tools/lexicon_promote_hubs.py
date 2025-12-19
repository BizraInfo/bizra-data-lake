#!/usr/bin/env python3
"""
lexicon_promote_hubs.py — Promote high-connectivity hub concepts to canonical lexicon

Analyzes the unified HyperGraph to identify high-frequency entities that are NOT yet
in the canonical lexicon, then generates a review queue for human approval.

Usage:
    python tools/lexicon_promote_hubs.py --graph evidence/bizra_hypergraph_unified --lexicon constitution/lexicon_v2.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml


@dataclass
class HubCandidate:
    """A concept discovered in the graph that may need canonicalization."""
    term: str
    frequency: int
    contexts: List[str]
    entity_type: Optional[str]
    sample_sources: List[str]
    recommendation: str  # "promote" | "alias" | "reject" | "review"


def load_lexicon(lexicon_path: Path) -> Dict[str, Any]:
    """Load lexicon YAML file."""
    with open(lexicon_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_canonical_terms(lexicon: Dict[str, Any]) -> Set[str]:
    """Extract all canonical terms and aliases from lexicon."""
    terms: Set[str] = set()
    
    # v1 style: terms dict
    if "terms" in lexicon:
        for term, details in lexicon["terms"].items():
            terms.add(term.lower())
            if isinstance(details, dict):
                for alias in details.get("aliases", []):
                    terms.add(alias.lower())
    
    # v2 style: hub_concepts dict
    if "hub_concepts" in lexicon:
        for term, details in lexicon["hub_concepts"].items():
            terms.add(term.lower())
            if isinstance(details, dict):
                for alias in details.get("aliases", []):
                    terms.add(alias.lower())
    
    return terms


def load_graph_entities(graph_dir: Path) -> List[Dict[str, Any]]:
    """Load entities from unified HyperGraph."""
    entities_file = graph_dir / "entities.json"
    if not entities_file.exists():
        print(f"❌ Entities file not found: {entities_file}")
        return []
    
    with open(entities_file, "r", encoding="utf-8") as f:
        return json.load(f)


def load_hyperedges(graph_dir: Path) -> List[Dict[str, Any]]:
    """Load hyperedges to compute term frequencies."""
    hyperedges_file = graph_dir / "hyperedges.json"
    if not hyperedges_file.exists():
        print(f"⚠️ Hyperedges file not found: {hyperedges_file}")
        return []
    
    with open(hyperedges_file, "r", encoding="utf-8") as f:
        return json.load(f)


def analyze_term_frequency(hyperedges: List[Dict[str, Any]]) -> Counter:
    """Count term occurrences across all hyperedges."""
    term_counter: Counter = Counter()
    
    for he in hyperedges:
        # Count entities mentioned in hyperedge
        for entity in he.get("entities", []):
            if isinstance(entity, str):
                term_counter[entity.lower()] += 1
            elif isinstance(entity, dict):
                term_counter[entity.get("name", "").lower()] += 1
        
        # Count concepts mentioned
        for concept in he.get("concepts", []):
            if isinstance(concept, str):
                term_counter[concept.lower()] += 1
    
    return term_counter


def extract_contexts(term: str, hyperedges: List[Dict[str, Any]], max_contexts: int = 3) -> List[str]:
    """Extract sample contexts where a term appears."""
    contexts: List[str] = []
    term_lower = term.lower()
    
    for he in hyperedges:
        if len(contexts) >= max_contexts:
            break
        
        segment = he.get("knowledge_segment", "")
        if term_lower in segment.lower():
            # Truncate to first 200 chars for context
            contexts.append(segment[:200] + "..." if len(segment) > 200 else segment)
    
    return contexts


def identify_hub_candidates(
    entities: List[Dict[str, Any]],
    hyperedges: List[Dict[str, Any]],
    canonical_terms: Set[str],
    min_frequency: int = 5
) -> List[HubCandidate]:
    """Identify high-frequency terms not yet in lexicon."""
    term_freq = analyze_term_frequency(hyperedges)
    candidates: List[HubCandidate] = []
    
    # Build entity type lookup
    entity_types: Dict[str, str] = {}
    entity_sources: Dict[str, List[str]] = {}
    for entity in entities:
        name = entity.get("name", "").lower()
        entity_types[name] = entity.get("type", "UNKNOWN")
        entity_sources[name] = entity.get("sources", [])[:3]
    
    for term, freq in term_freq.most_common(100):
        # Skip if already canonical
        if term in canonical_terms:
            continue
        
        # Skip low-frequency terms
        if freq < min_frequency:
            continue
        
        # Skip very short terms (likely noise)
        if len(term) < 3:
            continue
        
        # Skip purely numeric terms
        if term.replace(".", "").replace("-", "").isdigit():
            continue
        
        contexts = extract_contexts(term, hyperedges)
        
        # Determine recommendation based on frequency and type
        if freq >= 50:
            recommendation = "promote"
        elif freq >= 20:
            recommendation = "review"
        elif entity_types.get(term) in ("CONCEPT", "PROTOCOL", "MODULE"):
            recommendation = "review"
        else:
            recommendation = "alias"  # Likely an alias of existing term
        
        candidates.append(HubCandidate(
            term=term,
            frequency=freq,
            contexts=contexts,
            entity_type=entity_types.get(term),
            sample_sources=entity_sources.get(term, []),
            recommendation=recommendation
        ))
    
    return candidates


def generate_promotion_report(
    candidates: List[HubCandidate],
    output_path: Path
) -> None:
    """Generate YAML report of promotion candidates."""
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_candidates": len(candidates),
        "by_recommendation": {
            "promote": [c.term for c in candidates if c.recommendation == "promote"],
            "review": [c.term for c in candidates if c.recommendation == "review"],
            "alias": [c.term for c in candidates if c.recommendation == "alias"],
        },
        "candidates": []
    }
    
    for c in candidates:
        report["candidates"].append({
            "term": c.term,
            "frequency": c.frequency,
            "entity_type": c.entity_type,
            "recommendation": c.recommendation,
            "sample_contexts": c.contexts[:2],
            "sample_sources": c.sample_sources,
            "action": "pending_human_review"
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(report, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"✅ Promotion report written to: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Identify hub concepts for lexicon promotion"
    )
    parser.add_argument(
        "--graph",
        type=Path,
        default=Path("evidence/bizra_hypergraph_unified"),
        help="Path to unified HyperGraph directory"
    )
    parser.add_argument(
        "--lexicon",
        type=Path,
        default=Path("constitution/lexicon_v2.yaml"),
        help="Path to current lexicon file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/lexicon_promotion_queue.yaml"),
        help="Output path for promotion queue"
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=5,
        help="Minimum term frequency to consider"
    )
    
    args = parser.parse_args()
    
    print("🔍 Loading lexicon...")
    lexicon = load_lexicon(args.lexicon)
    canonical_terms = extract_canonical_terms(lexicon)
    print(f"   Found {len(canonical_terms)} canonical terms + aliases")
    
    print("📊 Loading HyperGraph...")
    entities = load_graph_entities(args.graph)
    hyperedges = load_hyperedges(args.graph)
    print(f"   Loaded {len(entities)} entities, {len(hyperedges)} hyperedges")
    
    print("🧠 Analyzing hub candidates...")
    candidates = identify_hub_candidates(
        entities, hyperedges, canonical_terms, args.min_frequency
    )
    print(f"   Found {len(candidates)} promotion candidates")
    
    if candidates:
        # Summary
        promote = [c for c in candidates if c.recommendation == "promote"]
        review = [c for c in candidates if c.recommendation == "review"]
        alias = [c for c in candidates if c.recommendation == "alias"]
        
        print(f"\n📋 RECOMMENDATION SUMMARY")
        print(f"   🟢 Promote (high confidence): {len(promote)}")
        for c in promote[:5]:
            print(f"      • {c.term} (freq={c.frequency}, type={c.entity_type})")
        
        print(f"   🟡 Review (needs human decision): {len(review)}")
        for c in review[:5]:
            print(f"      • {c.term} (freq={c.frequency})")
        
        print(f"   🔵 Alias candidates: {len(alias)}")
        
        generate_promotion_report(candidates, args.output)
    else:
        print("✨ No new hub candidates found — lexicon is up to date!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
