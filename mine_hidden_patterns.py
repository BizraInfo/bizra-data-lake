"""
BIZRA INDEPTH MINING ENGINE (v1.0)
"Excavating the Hidden Graph of Thoughts"

Mission:
1. Analyze the 'Sovereign Catalog' to detect "Thought Flow Patterns" (Temporal & Spatial clusters of High-SNR work).
2. Identify "Golden Gems": High-value narrative artifacts buried in the noise.
3. Extract the "Hidden Context": Read specific headers/summaries from these gems.

Embodies:
- Graph of Thoughts: Linking disparate files by time and topic.
- High SNR: Filtering for pure knowledge (Markdown/Text/JSON-Conversations).
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt # Optional, but we'll do ASCII viz
import re

# Configuration
from bizra_config import GOLD_PATH

CATALOG_PATH = GOLD_PATH / "sovereign_catalog.parquet"

def analyze_thought_flow(df):
    print("\n🌊 ANALYZING THOUGHT FLOW PATTERNS...")
    
    # Filter for Narrative/Knowledge
    knowledge_df = df[
        (df['kind'] == 'Knowledge/Narrative') | 
        (df['path'].str.contains('conversation|chat|logs|journal|diary|idea|plan', case=False))
    ].copy()
    
    if knowledge_df.empty:
        print("   ⚠️ No knowledge artifacts found for pattern analysis.")
        return

    # 1. Temporal Flow (When did the thoughts happen?)
    # Group by Month-Year
    knowledge_df['month_year'] = knowledge_df['modified'].dt.to_period('M')
    temporal_flow = knowledge_df.groupby('month_year').size().sort_index()
    
    print("\n   📅 CHRONOLOGICAL INTENSITY (The Pulse of Creation):")
    # ASCII Bar Chart
    max_val = temporal_flow.max() if not temporal_flow.empty else 1
    for period, count in temporal_flow.tail(12).items(): # Last 12 months
        bar_len = int((count / max_val) * 40)
        bar = '█' * bar_len
        print(f"      {period}: {bar} ({count})")

    # 2. Spatial Flow (Where do the thoughts live?)
    # Extract parent folders of knowledge files
    knowledge_df['parent_dir'] = knowledge_df['path'].apply(lambda x: Path(x).parent.name)
    spatial_flow = knowledge_df['parent_dir'].value_counts().head(10)
    
    print("\n   🧠 CORTICAL REGIONS (Highest Density of Knowledge):")
    for folder, count in spatial_flow.items():
        print(f"      📂 {folder:<30} : {count} nodes")

    return knowledge_df

def extract_golden_gems(knowledge_df):
    print("\n💎 PROSPECTING FOR HIDDEN GOLDEN GEMS...")
    
    # Heuristic for "Gem Quality":
    # 1. Keywords in name (High value concepts)
    # 2. File size (Not too small, not too huge - the "Goldilocks" zone of deep text)
    # 3. Recentness (Fresh thoughts are often sharper)
    
    gems = knowledge_df.copy()
    
    # Gem Score Calculation
    def calculate_gem_score(row):
        score = 0
        name = row['name'].lower()
        path = row['path'].lower()
        
        # High value keywords
        keywords = ['architecture', 'roadmap', 'manifesto', 'core', 'system', 'dream', 'god', 'master', 'final', 'secret', 'truth', 'bizra', 'sape', 'arte', 'ddagi']
        score += sum(2 for k in keywords if k in name)
        score += sum(1 for k in keywords if k in path)
        
        # Conversation logs usually hold raw thought streams
        if 'conversation' in path or 'chat' in path:
            score += 3
            
        return score

    gems['gem_score'] = gems.apply(calculate_gem_score, axis=1)
    
    # Sort by Score
    top_gems = gems.sort_values(by='gem_score', ascending=False).head(10)
    
    print("\n   🏆 TOP 10 DETECTED ARTIFACTS (Highest SNR):")
    print("-" * 100)
    print(f"   {'SCORE':<6} | {'DATE':<12} | {'NAME'}")
    print("-" * 100)
    
    for _, row in top_gems.iterrows():
        print(f"   {row['gem_score']:<6} | {row['modified'].strftime('%Y-%m-%d')}   | {row['name']}")
        # print(f"          path: {row['path']}")

    return top_gems

def read_gem_content(file_path):
    try:
        path = Path(file_path)
        if not path.exists(): return None
        
        # Read first 1kb to get the "Gist" or Intro
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(1500)
            return content.strip()
    except Exception:
        return None

def main():
    # Load
    try:
        df = pd.read_parquet(CATALOG_PATH)
    except Exception as e:
        print(f"❌ Could not load catalog: {e}")
        return

    # Analyze
    knowledge_df = analyze_thought_flow(df)
    
    if knowledge_df is not None:
        # Mine Gems
        top_gems = extract_golden_gems(knowledge_df)
        
        # Peek at the #1 Gem
        if not top_gems.empty:
            best_gem = top_gems.iloc[0]
            print(f"\n📜 EXPOSING TOP ARTIFACT: {best_gem['name']}")
            print(f"   📍 {best_gem['path']}")
            print("=" * 80)
            content = read_gem_content(best_gem['path'])
            if content:
                # Sanitize newlines for display
                print(content[:1000] + "...\n[...End of Preview...]")
            else:
                print("   [Content Unreadable]")
            print("=" * 80)

if __name__ == "__main__":
    main()
