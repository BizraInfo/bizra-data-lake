"""
BIZRA QURANIC LAKE INGESTION ENGINE (v1.0)
"Standing on the Shoulders of Giants"

Adapts the logic from 'quranic_indexer.py' (Bizra Taskmaster) to the 
Data Lake Architecture (Parquet + Local Vector Engine).

Mission:
1. Fetch the Complete Quranic Corpus (Arabic + English Translation).
2. Construct a high-precision DataFrame.
3. Generate Embeddings using the System Vector Engine.
4. Crystallize into the Data Lake for eternal retrieval.

References:
- http://api.alquran.cloud/v1 (Source)
- C:/BIZRA-DATA-LAKE/vector_engine.py (Capability)
"""

import json
import urllib.request
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import time

# Integration with existing systems
from vector_engine import VectorEngine
from bizra_config import INDEXED_PATH

OUTPUT_DIR = INDEXED_PATH / "knowledge/quran"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_PATH = OUTPUT_DIR / "quran_full.parquet"

API_BASE = "http://api.alquran.cloud/v1/quran"

def fetch_corpus():
    print("🌍 CONNECTING TO AL-QURAN CLOUD API...")
    
    # Fetch Arabic
    print("   ⬇️  Downloading Uthmani Script...")
    with urllib.request.urlopen(f"{API_BASE}/quran-uthmani") as url:
        arabic_data = json.loads(url.read().decode())
    
    # Fetch English
    print("   ⬇️  Downloading Sahih International Translation...")
    with urllib.request.urlopen(f"{API_BASE}/en.sahih") as url:
        english_data = json.loads(url.read().decode())
        
    return arabic_data['data'], english_data['data']

def process_data(arabic_quran, english_quran):
    print("⚙️  PROCESSING CORPUS...")
    records = []
    
    # The structure is Surahs -> Ayahs
    # We iterate both simultaneously as they are aligned
    for surah_idx, surah in enumerate(tqdm(arabic_quran['surahs'], desc="Aligning Surahs")):
        en_surah = english_quran['surahs'][surah_idx]
        
        surah_meta = {
            'surah_number': surah['number'],
            'surah_name_en': surah['englishName'],
            'surah_name_ar': surah['name'],
            'revelation_type': surah['revelationType']
        }
        
        for ayah_idx, ayah in enumerate(surah['ayahs']):
             en_ayah = en_surah['ayahs'][ayah_idx]
             
             # Validation
             if ayah['number'] != en_ayah['number']:
                 print(f"⚠️ Mismatch at Surah {surah['number']} Ayah {ayah['number']}")
                 
             text_combined = f"{surah['englishName']} ({surah['number']}:{ayah['numberInSurah']}): {en_ayah['text']}"
             
             records.append({
                 'id': f"quran_{surah['number']}_{ayah['numberInSurah']}",
                 'surah': surah['number'],
                 'ayah': ayah['numberInSurah'],
                 'surah_name': surah['englishName'],
                 'arabic_text': ayah['text'],
                 'english_text': en_ayah['text'],
                 'text_combined': text_combined, # For embedding
                 'juz': ayah['juz'],
                 'page': ayah['page']
             })
             
    return pd.DataFrame(records)

def generate_embeddings(df):
    print("🧠 ACTIVATING VECTOR ENGINE...")
    engine = VectorEngine() # Autoloads the model
    
    texts = df['text_combined'].tolist()
    
    print(f"   ⚡ Generating Embeddings for {len(texts)} verses...")
    # Batch process in chunks of 32 for efficiency
    embeddings = engine.model.encode(texts, batch_size=32, show_progress_bar=True)
    
    return embeddings

def main():
    try:
        if PARQUET_PATH.exists():
            print(f"ℹ️  Corpus already exists at {PARQUET_PATH}. Skipping fetch.")
            input("Press Enter to re-fetch/overwrite, or Ctrl+C to cancel...")
            
        print("🚀 STARTING QURANIC INGESTION SEQUENCE")
        
        # 1. Fetch
        ar, en = fetch_corpus()
        
        # 2. Process
        df = process_data(ar, en)
        print(f"   ✅ Processed {len(df)} Verses.")
        
        # 3. Embed
        embeddings = generate_embeddings(df)
        
        # 4. Store
        print("💾 CRYSTALLIZING DATA...")
        
        # Save Parquet (Data)
        df.to_parquet(PARQUET_PATH)
        
        # Save Embeddings (Vectors)
        # We store them as a separate numpy array for fast loading, 
        # or we could put them in the dataframe (but that makes parquet huge/slow to read just for text headers)
        # Let's keep them in the dataframe for "One File" portability, or separate?
        # A "Peak Masterpiece" uses the Data Lake standard.
        # Let's save a unified rich Parquet.
        
        #df['embedding'] = list(embeddings) # Warning: This can be heavy
        # Instead, let's save the numpy matrix specifically for the vector engine
        np.save(OUTPUT_DIR / "quran_vectors.npy", embeddings)
        
        # Save metadata linking rows to vectors
        meta_df = df[['id', 'text_combined']]
        meta_df.to_parquet(OUTPUT_DIR / "quran_meta.parquet")
        
        print("\n✅ MISSION COMPLETE")
        print(f"   📜 Corpus: {PARQUET_PATH}")
        print(f"   🧮 Vectors: {OUTPUT_DIR / 'quran_vectors.npy'}")
        print(f"   🔢 Verses: {len(df)}")
        print(f"   💎 Dimensions: {embeddings.shape}")

    except Exception as e:
        print(f"\n❌ FATAL ERROR: {e}")

if __name__ == "__main__":
    main()
