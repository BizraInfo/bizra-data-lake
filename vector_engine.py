# BIZRA Vector Engine v2.0 (Layer 2: Semantic Indexing)
# Context-aware chunking + Multi-Modal Embeddings
# Engineering Excellence: Source-specific strategies + CLIP for vision

import os
import pandas as pd
import torch
import hashlib
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
from typing import Optional, List, Dict, Any, Union

from bizra_config import (
    CORPUS_TABLE_PATH, CHUNKS_TABLE_PATH, BATCH_SIZE, MAX_SEQ_LENGTH,
    VISION_ENABLED, CLIP_MODEL, CLIP_EMBEDDING_DIM, TEXT_EMBEDDING_DIM,
    IMAGE_EMBEDDINGS_PATH, IMAGE_EXTENSIONS, INDEXED_PATH, CHECKPOINT_PATH
)

# Optional CLIP imports
try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

class VectorEngine:
    """
    Multi-modal vector embedding engine.

    Capabilities:
    - Text embeddings via all-MiniLM-L6-v2 (384-dim)
    - Image embeddings via CLIP (512-dim)
    - Cross-modal text-to-image search via CLIP text encoder
    """

    def __init__(self, enable_vision: bool = VISION_ENABLED):
        print("🧠 Initializing BIZRA Vector Engine v2.0")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"   Device: {self.device}")

        # Text embedding model
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        self.text_model.max_seq_length = MAX_SEQ_LENGTH
        print(f"   Text Model: all-MiniLM-L6-v2 ({TEXT_EMBEDDING_DIM}-dim)")

        # Vision embedding model (CLIP)
        self.enable_vision = enable_vision and CLIP_AVAILABLE and PIL_AVAILABLE
        self.clip_model = None
        self.clip_processor = None

        if self.enable_vision:
            self._initialize_clip()
        else:
            print("   Vision: Disabled")

        # Ensure image embeddings directory exists
        IMAGE_EMBEDDINGS_PATH.mkdir(parents=True, exist_ok=True)

    def _initialize_clip(self):
        """Load CLIP model for vision embeddings."""
        try:
            print(f"   Loading CLIP model: {CLIP_MODEL}...")
            self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
            print(f"   Vision Model: CLIP ({CLIP_EMBEDDING_DIM}-dim)")
        except Exception as e:
            print(f"   ⚠️ Failed to load CLIP: {e}")
            self.enable_vision = False

    def chunk_document(self, doc):
        """Source-aware chunking logic."""
        text = doc['text']
        if not text: return []
        
        chunks = []
        source_type = doc['source_type']
        
        if source_type == "repo_file":
            # Simple code chunking by line count (could be class/function aware)
            lines = text.splitlines()
            for i in range(0, len(lines), 50):
                chunk_text = "\n".join(lines[i:i+50])
                chunks.append(self._create_chunk_dict(doc, chunk_text, i//50))
        
        elif source_type == "chat_turn":
            # Split by message or logical blocks
            blocks = text.split("\n\n")
            for i, block in enumerate(blocks):
                chunks.append(self._create_chunk_dict(doc, block, i))
        
        else:
            # Standard Fixed-size window for generic/notes
            words = text.split()
            for i in range(0, len(words), 200):
                chunk_text = " ".join(words[i:i+200])
                chunks.append(self._create_chunk_dict(doc, chunk_text, i//200))
                
        return chunks

    def _create_chunk_dict(self, doc, text, index):
        chunk_id = hashlib.sha256(f"{doc['doc_id']}_{index}".encode()).hexdigest()[:16]
        return {
            "chunk_id": chunk_id,
            "doc_id": doc['doc_id'],
            "chunk_index": index,
            "chunk_text": text,
            "token_est": len(text.split()) * 1.3,
            "created_at": doc['created_at'],
            "modality": "text",
            "chunk_metadata_json": json.dumps({"source_type": doc['source_type']})
        }

    # =========================================================================
    # IMAGE EMBEDDING METHODS
    # =========================================================================

    def encode_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Generate CLIP embedding for an image.

        Args:
            image_path: Path to image file

        Returns:
            512-dimensional normalized embedding or None
        """
        if not self.enable_vision:
            return None

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                embedding = image_features.cpu().numpy().flatten()
                # Normalize
                embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except Exception as e:
            print(f"⚠️ Failed to encode image {image_path}: {e}")
            return None

    def encode_text_for_image_search(self, text: str) -> Optional[np.ndarray]:
        """
        Generate CLIP text embedding for cross-modal image search.

        Args:
            text: Query text

        Returns:
            512-dimensional normalized embedding or None
        """
        if not self.enable_vision:
            return None

        try:
            inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                embedding = text_features.cpu().numpy().flatten()
                embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except Exception as e:
            print(f"⚠️ Failed to encode text for image search: {e}")
            return None

    def encode_images_batch(self, image_paths: List[str], batch_size: int = 32) -> List[tuple]:
        """
        Batch encode images for efficiency.

        Args:
            image_paths: List of image file paths
            batch_size: Images per batch

        Returns:
            List of (path, embedding) tuples
        """
        results = []

        if not self.enable_vision:
            return [(p, None) for p in image_paths]

        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch = image_paths[i:i + batch_size]

            for path in batch:
                embedding = self.encode_image(path)
                results.append((path, embedding))

        return results

    def is_image_file(self, file_path: str) -> bool:
        """Check if file is a supported image format."""
        return Path(file_path).suffix.lower() in IMAGE_EXTENSIONS

    def process_all(self, include_images: bool = True):
        """
        Process all documents and generate embeddings.

        Args:
            include_images: Whether to process image files with CLIP
        """
        if not CORPUS_TABLE_PATH.exists():
            print("❌ No Corpus Table found. Run corpus_manager.py first.")
            return

        df_docs = pd.read_parquet(CORPUS_TABLE_PATH)
        all_chunks = []
        image_docs = []

        print("✂️ Chunking documents...")
        for _, row in tqdm(df_docs.iterrows(), total=len(df_docs), desc="Contextual Chunking"):
            # Check if this is an image file
            file_path = row.get('file_path', '')
            if file_path and self.is_image_file(file_path):
                image_docs.append(row)
            else:
                all_chunks.extend(self.chunk_document(row))

        if not all_chunks and not image_docs:
            print("⚠️ No content to process.")
            return

        # Process text chunks
        if all_chunks:
            df_chunks = pd.DataFrame(all_chunks)

            # Chunked indexing + checkpointing to avoid OOM/SIGKILL
            save_every = int(os.getenv("BIZRA_TEXT_SAVE_EVERY", "0"))
            parts_dir = INDEXED_PATH / "embeddings" / "text_chunks"
            parts_dir.mkdir(parents=True, exist_ok=True)
            if save_every and save_every > 0:
                print(f"⚡ Generating text embeddings on {self.device} (chunked save every {save_every})...")
                part = 0
                for i in range(0, len(df_chunks), save_every):
                    batch_df = df_chunks.iloc[i:i+save_every].copy()
                    texts = batch_df['chunk_text'].tolist()
                    embeddings = self.text_model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)
                    batch_df['embedding'] = [e.tolist() for e in embeddings]
                    batch_df['embedding_model'] = "all-MiniLM-L6-v2"
                    batch_df['embedding_dim'] = TEXT_EMBEDDING_DIM
                    part_path = parts_dir / f"part-{part:05d}.parquet"
                    batch_df.to_parquet(part_path, index=False)
                    part += 1
                    CHECKPOINT_PATH.write_text(json.dumps({"last_index": i, "part": part}), encoding="utf-8")
                print(f"✅ Text Chunks saved (parts): {parts_dir}")
            else:
                print(f"⚡ Generating text embeddings on {self.device}...")
                texts = df_chunks['chunk_text'].tolist()
                embeddings = self.text_model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=True)

                # Store as list of floats for Parquet compatibility
                df_chunks['embedding'] = [e.tolist() for e in embeddings]
                df_chunks['embedding_model'] = "all-MiniLM-L6-v2"
                df_chunks['embedding_dim'] = TEXT_EMBEDDING_DIM

                df_chunks.to_parquet(CHUNKS_TABLE_PATH, index=False)
                print(f"✅ Text Chunks saved: {CHUNKS_TABLE_PATH} ({len(df_chunks)} chunks)")

        # Process images with CLIP
        if include_images and self.enable_vision and image_docs:
            self._process_images(image_docs)

    def _process_images(self, image_docs: List[Dict]):
        """Process image documents with CLIP embeddings."""
        print(f"\n🖼️ Processing {len(image_docs)} images with CLIP...")

        image_chunks = []

        for doc in tqdm(image_docs, desc="CLIP Embeddings"):
            file_path = doc.get('file_path', '')
            if not file_path or not Path(file_path).exists():
                continue

            embedding = self.encode_image(file_path)
            if embedding is None:
                continue

            chunk_id = hashlib.sha256(f"{doc['doc_id']}_img".encode()).hexdigest()[:16]

            # Get any OCR text if available
            text_content = doc.get('text', '') or f"[IMAGE: {Path(file_path).name}]"

            image_chunks.append({
                "chunk_id": chunk_id,
                "doc_id": doc['doc_id'],
                "chunk_index": 0,
                "chunk_text": text_content,
                "token_est": len(text_content.split()) * 1.3,
                "created_at": doc.get('created_at', ''),
                "modality": "image",
                "embedding": embedding.tolist(),
                "embedding_model": "clip-vit-base-patch32",
                "embedding_dim": CLIP_EMBEDDING_DIM,
                "file_path": file_path,
                "chunk_metadata_json": json.dumps({
                    "source_type": "image",
                    "original_path": file_path
                })
            })

        if image_chunks:
            df_images = pd.DataFrame(image_chunks)
            image_chunks_path = IMAGE_EMBEDDINGS_PATH / "image_chunks.parquet"
            df_images.to_parquet(image_chunks_path, index=False)
            print(f"✅ Image embeddings saved: {image_chunks_path} ({len(df_images)} images)")

    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            "device": self.device,
            "text_model": "all-MiniLM-L6-v2",
            "text_embedding_dim": TEXT_EMBEDDING_DIM,
            "vision_enabled": self.enable_vision,
            "vision_model": CLIP_MODEL if self.enable_vision else None,
            "vision_embedding_dim": CLIP_EMBEDDING_DIM if self.enable_vision else None,
            "corpus_exists": CORPUS_TABLE_PATH.exists(),
            "chunks_exist": CHUNKS_TABLE_PATH.exists()
        }


if __name__ == "__main__":
    engine = VectorEngine()

    print("\nEngine Status:")
    for k, v in engine.get_status().items():
        print(f"  {k}: {v}")

    print("\n" + "=" * 50)
    engine.process_all()
