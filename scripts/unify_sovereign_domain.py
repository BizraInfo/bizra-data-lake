"""
BIZRA SOVEREIGN UNIFICATION ENGINE (v1.0 Peak)
"One Index to Rule Them All"

Embodiment:
- Interdisciplinary: Merges Code, Narrative, and Structural Data.
- Graph of Thoughts: Scans non-linearly across symbolic links.
- High SNR: Aggressively filters noise (git, binary artifacts) to retain only Signal.
- Giants Protocol: Leverages existing OS file structures without duplication.

Mission:
Catalog the newly linked 1.5TB Sovereign Domain into a lightweight,
queryable Parquet index ('sovereign_catalog.parquet') that feeds the Hypergraph.
"""

import os
import hashlib
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import mimetypes

# Load Configuration
from bizra_config import DATA_LAKE_ROOT, GOLD_PATH

# Target Roots (The Linked Giants)
EXTERNAL_ROOT = DATA_LAKE_ROOT / "01_RAW/external_links"
OUTPUT_CATALOG = GOLD_PATH / "sovereign_catalog.parquet"

# SNR Filters (Noise Reduction)
IGNORE_DIRS = {
    '.git', '.vscode', 'node_modules', '__pycache__', 'venv', '.venv',
    'bin', 'obj', 'build', 'dist', 'tmp', 'temp', '$Recycle.Bin'
}
IGNORE_EXTS = {
    '.exe', '.dll', '.so', '.dylib', '.class', '.pyc', '.pyd', 
    '.log', '.tmp', '.dat', '.cache', '.suo', '.user'
}

def calculate_snr_score(file_path):
    """
    Heuristic SNR Score.
    Text/Markdown = High Signal (1.0)
    Code = High Signal (0.9)
    Config/JSON = Medium Signal (0.7)
    Binary/Media = Low Signal (0.3)
    """
    ext = file_path.suffix.lower()
    if ext in ['.md', '.txt', '.org']: return 1.0
    if ext in ['.py', '.rs', '.js', '.ts', '.go', '.cpp', '.h', '.java', '.cs']: return 0.95
    if ext in ['.json', '.yaml', '.yml', '.xml', '.toml']: return 0.8
    if ext in ['.pdf', '.docx', '.pptx']: return 0.6 # High value but hard to parse quickly
    if ext in ['.png', '.jpg', '.mp4']: return 0.2
    return 0.1

def generate_sovereign_id(abs_path):
    """Deterministic ID based on the immutable absolute path."""
    return hashlib.sha256(str(abs_path).encode('utf-8')).hexdigest()[:16]

def unify_domain():
    print(f"🌟 INITIATING SOVEREIGN UNIFICATION PROTOCOL")
    print(f"🔭 Scoping: {EXTERNAL_ROOT}")
    
    if not EXTERNAL_ROOT.exists():
        print(f"❌ Critical: Link root {EXTERNAL_ROOT} not found.")
        return

    records = []
    start_time = datetime.now()
    
    # 1. Autonomous Crawl (The Engine)
    # Using os.walk for maximum speed over pathlib.rglob
    print("🕸️  Crawling the Graph of Directories...")
    
    file_count = 0
    skipped_count = 0
    
    for root, dirs, files in os.walk(EXTERNAL_ROOT):
        # In-place filtering of directories to prevent descending into Noise
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]
        
        path_root = Path(root)
        
        for file in files:
            file_path = path_root / file
            
            # SNR Filter: Extension check
            if file_path.suffix.lower() in IGNORE_EXTS:
                skipped_count += 1
                continue
                
            # SNR Filter: Calculate Signal Quality
            signal_score = calculate_snr_score(file_path)
            if signal_score < 0.1: # Absolute noise floor
                skipped_count += 1
                continue
                
            file_count += 1
            
            # Construct the Knowledge Node
            try:
                stats = file_path.stat()
                
                # Determine "Kind" (Interdisciplinary Tagging)
                kind = "Unknown"
                if signal_score >= 0.9: kind = "Knowledge/Narrative" if file_path.suffix in ['.md', '.txt'] else "Engineering/Code"
                elif signal_score >= 0.6: kind = "Data/Structure"
                else: kind = "Artifact/Media"

                records.append({
                    "id": generate_sovereign_id(file_path),
                    "path": str(file_path),
                    "name": file,
                    "kind": kind,
                    "size_bytes": stats.st_size,
                    "modified": datetime.fromtimestamp(stats.st_mtime),
                    "snr_score": signal_score,
                    "domain_source": path_root.parts[4] if len(path_root.parts) > 4 else "Unknown" # Extract which junction
                })
                
            except Exception as e:
                # Resilience: Log error but continue scanning
                pass
                
            if file_count % 10000 == 0:
                print(f"   ⚡ Scanned {file_count} elite nodes...", end='\r')

    print(f"\n✅ Scan Complete. {file_count} High-SNR Nodes identified. ({skipped_count} noise items filtered)")

    # 2. Crystallization (Parquet Dump)
    if records:
        print("💎 Crystallizing Catalog to Parquet format...")
        df = pd.DataFrame(records)
        df.to_parquet(OUTPUT_CATALOG, engine='pyarrow')
        
        # 3. Analytics (State of Art Reporting)
        total_size_gb = df['size_bytes'].sum() / (1024**3)
        print(f"\n📊 --- SOVEREIGN CATALOG STATISTICS ---")
        print(f"   Nodes: {len(df):,}")
        print(f"   Volume: {total_size_gb:.2f} GB")
        print(f"   Domains: {df['domain_source'].unique()}")
        print(f"   Breakdown: \n{df['kind'].value_counts()}")
        print(f"   Output: {OUTPUT_CATALOG}")
        print("---------------------------------------")
        print("👑 SYSTEM: UNIFIED.")
    else:
        print("⚠️ No valid nodes found. Check linking.")

if __name__ == "__main__":
    unify_domain()
