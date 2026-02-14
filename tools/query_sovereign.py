"""
BIZRA PRECISION QUERY ENGINE
"Finding the Needle in the 700k Haystack"

Mission: Use the Sovereign Catalog (Parquet) to instantly locate high-value
assets (specifically the Quranic Corpus) without touching the slow filesystem.
"""

import pandas as pd
from bizra_config import GOLD_PATH

CATALOG_PATH = GOLD_PATH / "sovereign_catalog.parquet"

def omni_search(terms):
    print(f"🔍 LOADING SOVEREIGN INDEX...")
    try:
        df = pd.read_parquet(CATALOG_PATH)
    except Exception as e:
        print(f"❌ Error loading index: {e}")
        return

    print(f"📊 Index Loaded: {len(df):,} nodes.")
    print(f"🎯 Searching for terms: {terms}")

    # Case-insensitive "OR" search across Name and Path
    # Using specific terms relevant to the user's quest
    
    # 1. Search Logic
    # We want rows where the name contains ANY of the terms
    query_str = '|'.join(terms)
    
    results = df[
        df['name'].str.contains(query_str, case=False, na=False) | 
        df['path'].str.contains(query_str, case=False, na=False)
    ]
    
    # 2. Ranking (SNR Optimization)
    # Prefer exact matches or larger files (datasets)
    if not results.empty:
        results = results.sort_values(by='size_bytes', ascending=False)
        
        print(f"\n✅ FOUND {len(results)} CANDIDATES:")
        print("-" * 80)
        print(f"{'SIZE (MB)':<10} | {'KIND':<20} | {'PATH'}")
        print("-" * 80)
        
        for _, row in results.head(20).iterrows(): # Show top 20
            size_mb = row['size_bytes'] / (1024 * 1024)
            print(f"{size_mb:<10.2f} | {row['kind']:<20} | {row['path']}")
    else:
        print("❌ No matches found in the Sovereign Domain.")

if __name__ == "__main__":
    # The specific search terms for the lost artifact
    search_terms = ["quran", "corpus", "tanzil", "uthmani", "ayah", "surah"]
    omni_search(search_terms)
