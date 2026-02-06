# BIZRA Knowledge Base Embedding Generator
# Uses RTX 4090 to generate embeddings for a massive knowledge base
# Optimized for performance and checkpointing

import os
import json
import torch
import hashlib
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import time
from datetime import datetime
from bizra_config import PROCESSED_PATH, VECTORS_PATH, CHECKPOINT_PATH, BATCH_SIZE, MAX_SEQ_LENGTH, INDEXED_PATH

# Configuration
ORGANIZED_PATH = PROCESSED_PATH
EMBEDDINGS_PATH = VECTORS_PATH.parent
KNOWLEDGE_BASE = INDEXED_PATH
# Model configuration
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# File processing limits
MAX_FILE_SIZE_MB = 10
MAX_LINES_TO_EMBED = 100

class EmbeddingGenerator:
    def __init__(self):
        print("🔥 Initializing BIZRA Embedding Generator")
        print(f"📊 Target: {ORGANIZED_PATH}")
        print(f"💾 Output: {EMBEDDINGS_PATH}")
        
        # Create output directories
        EMBEDDINGS_PATH.mkdir(parents=True, exist_ok=True)
        
        # Load model on GPU
        print(f"\n⚡ Loading model: {MODEL_NAME}")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🎮 Device: {self.device}")
        
        self.model = SentenceTransformer(MODEL_NAME, device=self.device)
        self.model.max_seq_length = MAX_SEQ_LENGTH
        
        if self.device == "cuda":
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Load checkpoint if exists
        self.processed_files = set()
        self.load_checkpoint()
        
        # Statistics
        self.stats = {
            "total_files": 0,
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "embeddings_generated": 0,
            "start_time": datetime.now().isoformat()
        }
    
    def load_checkpoint(self):
        """Load previously processed files"""
        if CHECKPOINT_PATH.exists():
            with open(CHECKPOINT_PATH) as f:
                checkpoint = json.load(f)
                self.processed_files = set(checkpoint.get("processed_files", []))
                print(f"📂 Loaded checkpoint: {len(self.processed_files)} files already processed")
    
    def save_checkpoint(self):
        """Save progress checkpoint"""
        checkpoint = {
            "processed_files": list(self.processed_files),
            "stats": self.stats,
            "last_updated": datetime.now().isoformat()
        }
        with open(CHECKPOINT_PATH, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def get_file_hash(self, file_path):
        """Generate unique hash for file"""
        return hashlib.sha256(str(file_path).encode()).hexdigest()[:16]
    
    def should_process_file(self, file_path):
        """Check if file should be processed"""
        # Skip if already processed
        if str(file_path) in self.processed_files:
            return False
        
        # Skip if too large
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                return False
        except:
            return False
        
        # Process only text-based files
        text_extensions = {
            '.md', '.txt', '.py', '.js', '.ts', '.rs', '.go', '.java',
            '.json', '.yaml', '.yml', '.toml', '.xml', '.html', '.css',
            '.c', '.cpp', '.h', '.hpp', '.sh', '.bash', '.sql', '.r'
        }
        
        return file_path.suffix.lower() in text_extensions
    
    def read_file_content(self, file_path):
        """Read file content with error handling"""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= MAX_LINES_TO_EMBED:
                        break
                    lines.append(line.strip())
                return '\n'.join(lines)
        except Exception as e:
            return None
    
    def generate_embedding(self, text):
        """Generate embedding for text"""
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            print(f"\n⚠️  Embedding error: {e}")
            return None
    
    def process_file(self, file_path):
        """Process a single file and generate embedding"""
        content = self.read_file_content(file_path)
        if not content:
            return None
        
        # Generate embedding
        embedding = self.generate_embedding(content)
        if embedding is None:
            return None
        
        # Create metadata
        metadata = {
            "file_path": str(file_path),
            "file_name": file_path.name,
            "file_hash": self.get_file_hash(file_path),
            "file_size": file_path.stat().st_size,
            "embedding_dim": len(embedding),
            "processed_at": datetime.now().isoformat(),
            "model": MODEL_NAME
        }
        
        # Save embedding
        embedding_file = EMBEDDINGS_PATH / f"{metadata['file_hash']}.json"
        with open(embedding_file, 'w') as f:
            json.dump({
                "metadata": metadata,
                "embedding": embedding
            }, f)
        
        return metadata
    
    def scan_directory(self):
        """Scan organized directory for all files"""
        print("\n📁 Scanning organized knowledge base...")
        all_files = []
        
        for file_path in ORGANIZED_PATH.rglob('*'):
            if file_path.is_file() and self.should_process_file(file_path):
                all_files.append(file_path)
        
        self.stats["total_files"] = len(all_files)
        print(f"✅ Found {len(all_files)} files to process")
        print(f"⏭️  Already processed: {len(self.processed_files)}")
        
        # Filter out already processed
        files_to_process = [f for f in all_files if str(f) not in self.processed_files]
        print(f"🎯 Remaining: {len(files_to_process)} files\n")
        
        return files_to_process
    
    def run(self):
        """Main execution loop"""
        print("=" * 70)
        print("🌊 BIZRA KNOWLEDGE BASE EMBEDDING GENERATION")
        print("=" * 70)
        
        files_to_process = self.scan_directory()
        
        if not files_to_process:
            print("✅ All files already processed!")
            return
        
        print(f"⏱️  Estimated time: {len(files_to_process) * 0.05 / 60:.1f} minutes")
        print(f"🔥 Starting generation...\n")
        
        start_time = time.time()
        checkpoint_interval = 100  # Save checkpoint every 100 files
        
        with tqdm(total=len(files_to_process), desc="Generating embeddings") as pbar:
            for i, file_path in enumerate(files_to_process):
                try:
                    metadata = self.process_file(file_path)
                    
                    if metadata:
                        self.processed_files.add(str(file_path))
                        self.stats["processed"] += 1
                        self.stats["embeddings_generated"] += 1
                    else:
                        self.stats["skipped"] += 1
                    
                except Exception as e:
                    self.stats["errors"] += 1
                    tqdm.write(f"❌ Error: {file_path.name}: {e}")
                
                pbar.update(1)
                
                # Save checkpoint periodically
                if (i + 1) % checkpoint_interval == 0:
                    self.save_checkpoint()
                    
                    # Show GPU stats
                    if self.device == "cuda":
                        mem_used = torch.cuda.memory_allocated() / 1e9
                        tqdm.write(f"\n💾 GPU Memory: {mem_used:.2f} GB")
        
        # Final save
        self.save_checkpoint()
        
        # Print final statistics
        elapsed = time.time() - start_time
        self.stats["elapsed_seconds"] = elapsed
        self.stats["end_time"] = datetime.now().isoformat()
        
        print("\n" + "=" * 70)
        print("✅ EMBEDDING GENERATION COMPLETE")
        print("=" * 70)
        print(f"📊 Total files: {self.stats['total_files']}")
        print(f"✅ Processed: {self.stats['processed']}")
        print(f"⏭️  Skipped: {self.stats['skipped']}")
        print(f"❌ Errors: {self.stats['errors']}")
        print(f"🎯 Embeddings generated: {self.stats['embeddings_generated']}")
        print(f"⏱️  Time: {elapsed / 60:.1f} minutes")
        print(f"💾 Output: {EMBEDDINGS_PATH}")
        print(f"📈 Avg speed: {self.stats['processed'] / elapsed:.1f} files/sec")
        print("=" * 70)
        
        # Save final statistics
        stats_file = KNOWLEDGE_BASE / "embeddings" / "generation_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)

if __name__ == "__main__":
    generator = EmbeddingGenerator()
    generator.run()
