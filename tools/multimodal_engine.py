# BIZRA Multi-Modal Engine v1.0
# Integrates CLIP (vision), Whisper (audio), and LLaVA (vision-language) for multi-modal RAG

import os
import json
import hashlib
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

from bizra_config import (
    VISION_ENABLED, AUDIO_ENABLED, CLIP_MODEL, VISION_LLM_LOCAL,
    WHISPER_LOCAL, IMAGE_BATCH_SIZE, AUDIO_CHUNK_DURATION,
    IMAGE_EXTENSIONS, AUDIO_EXTENSIONS, VIDEO_EXTENSIONS,
    CLIP_EMBEDDING_DIM, MULTIMODAL_CACHE, IMAGE_EMBEDDINGS_PATH,
    AUDIO_TRANSCRIPTS_PATH, GPU_ENABLED
)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

class ModalityType(Enum):
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"


@dataclass
class MultiModalContent:
    """Unified container for multi-modal content."""
    content_id: str
    modality: ModalityType
    source_path: str
    text_content: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class ImageContent(MultiModalContent):
    """Image-specific content with additional fields."""
    ocr_text: Optional[str] = None
    description: Optional[str] = None
    width: int = 0
    height: int = 0
    format: str = ""


@dataclass
class AudioContent(MultiModalContent):
    """Audio-specific content with transcription."""
    transcript: Optional[str] = None
    duration_seconds: float = 0.0
    language: str = "en"
    speaker_segments: List[Dict] = field(default_factory=list)


# ============================================================================
# IMAGE PROCESSOR
# ============================================================================

class ImageProcessor:
    """
    Process images for the BIZRA Data Lake using CLIP embeddings and OCR.

    Capabilities:
    - Extract CLIP embeddings (512-dim) for semantic image search
    - OCR text extraction from images
    - Image description via LLaVA (local) or Claude Vision (API)
    """

    def __init__(self, use_gpu: bool = GPU_ENABLED):
        self.use_gpu = use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.clip_model = None
        self.clip_processor = None
        self._initialized = False

        # Cache for embeddings
        self.cache_dir = MULTIMODAL_CACHE / "images"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self):
        """Load CLIP model (lazy initialization)."""
        if self._initialized:
            return

        if not CLIP_AVAILABLE:
            print("[MULTIMODAL] Warning: transformers not installed. CLIP disabled.")
            return

        if not PIL_AVAILABLE:
            print("[MULTIMODAL] Warning: PIL not installed. Image processing disabled.")
            return

        print(f"[MULTIMODAL] Loading CLIP model: {CLIP_MODEL} on {self.device}")
        try:
            self.clip_model = CLIPModel.from_pretrained(CLIP_MODEL).to(self.device)
            self.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL)
            self._initialized = True
            print(f"[MULTIMODAL] CLIP model loaded successfully")
        except Exception as e:
            print(f"[MULTIMODAL] Failed to load CLIP: {e}")

    def _get_cache_path(self, image_path: str) -> Path:
        """Get cache path for image embedding."""
        file_hash = hashlib.sha256(image_path.encode()).hexdigest()[:16]
        return self.cache_dir / f"{file_hash}.json"

    def _load_from_cache(self, image_path: str) -> Optional[np.ndarray]:
        """Load cached embedding if available."""
        cache_path = self._get_cache_path(image_path)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    if data.get('source_path') == image_path:
                        return np.array(data['embedding'])
            except Exception:
                pass
        return None

    def _save_to_cache(self, image_path: str, embedding: np.ndarray):
        """Save embedding to cache."""
        cache_path = self._get_cache_path(image_path)
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'source_path': image_path,
                    'embedding': embedding.tolist(),
                    'cached_at': datetime.now().isoformat()
                }, f)
        except Exception:
            pass

    def extract_embedding(self, image_path: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Extract CLIP embedding from image.

        Args:
            image_path: Path to image file
            use_cache: Whether to use cached embeddings

        Returns:
            512-dimensional CLIP embedding or None if failed
        """
        if use_cache:
            cached = self._load_from_cache(image_path)
            if cached is not None:
                return cached

        if not self._initialized:
            self.initialize()

        if self.clip_model is None:
            return None

        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                embedding = image_features.cpu().numpy().flatten()

                # Normalize
                embedding = embedding / np.linalg.norm(embedding)

            if use_cache:
                self._save_to_cache(image_path, embedding)

            return embedding

        except Exception as e:
            print(f"[MULTIMODAL] Failed to process image {image_path}: {e}")
            return None

    def extract_text_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Extract CLIP text embedding for cross-modal search.

        Args:
            text: Query text

        Returns:
            512-dimensional CLIP embedding
        """
        if not self._initialized:
            self.initialize()

        if self.clip_model is None:
            return None

        try:
            inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)

            with torch.no_grad():
                text_features = self.clip_model.get_text_features(**inputs)
                embedding = text_features.cpu().numpy().flatten()
                embedding = embedding / np.linalg.norm(embedding)

            return embedding

        except Exception as e:
            print(f"[MULTIMODAL] Failed to extract text embedding: {e}")
            return None

    def extract_ocr_text(self, image_path: str) -> Optional[str]:
        """
        Extract text from image using OCR (Tesseract).

        Args:
            image_path: Path to image file

        Returns:
            Extracted text or None
        """
        if not OCR_AVAILABLE:
            return None

        if not PIL_AVAILABLE:
            return None

        try:
            image = Image.open(image_path)
            text = pytesseract.image_to_string(image)
            return text.strip() if text.strip() else None
        except Exception as e:
            print(f"[MULTIMODAL] OCR failed for {image_path}: {e}")
            return None

    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """Get basic image metadata."""
        info = {
            'path': image_path,
            'width': 0,
            'height': 0,
            'format': '',
            'size_bytes': 0
        }

        try:
            info['size_bytes'] = os.path.getsize(image_path)

            if PIL_AVAILABLE:
                with Image.open(image_path) as img:
                    info['width'], info['height'] = img.size
                    info['format'] = img.format or ''
        except Exception:
            pass

        return info

    def batch_extract_embeddings(self, image_paths: List[str], batch_size: int = IMAGE_BATCH_SIZE) -> List[Tuple[str, Optional[np.ndarray]]]:
        """
        Extract embeddings for multiple images in batches.

        Args:
            image_paths: List of image file paths
            batch_size: Number of images per batch

        Returns:
            List of (path, embedding) tuples
        """
        if not self._initialized:
            self.initialize()

        results = []

        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            print(f"[MULTIMODAL] Processing image batch {i//batch_size + 1}/{(len(image_paths) + batch_size - 1)//batch_size}")

            for path in batch:
                embedding = self.extract_embedding(path)
                results.append((path, embedding))

        return results

    async def describe_image_async(self, image_path: str, use_local: bool = True) -> Optional[str]:
        """
        Generate image description using vision LLM.

        Args:
            image_path: Path to image
            use_local: Use local LLaVA model (True) or API (False)

        Returns:
            Text description of image
        """
        if use_local:
            return await self._describe_with_llava(image_path)
        else:
            return await self._describe_with_api(image_path)

    async def _describe_with_llava(self, image_path: str) -> Optional[str]:
        """Use local Ollama LLaVA model for image description."""
        try:
            import httpx

            # Read image as base64
            import base64
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode()

            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": VISION_LLM_LOCAL,
                        "prompt": "Describe this image in detail. Focus on the main subjects, text content, and any technical diagrams or charts.",
                        "images": [image_data],
                        "stream": False
                    }
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get('response', '')

        except Exception as e:
            print(f"[MULTIMODAL] LLaVA description failed: {e}")

        return None

    async def _describe_with_api(self, image_path: str) -> Optional[str]:
        """Use Claude Vision API for image description."""
        try:
            import anthropic
            import base64

            # Read image
            with open(image_path, 'rb') as f:
                image_data = base64.standard_b64encode(f.read()).decode()

            # Detect media type
            ext = Path(image_path).suffix.lower()
            media_types = {
                '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
                '.png': 'image/png', '.gif': 'image/gif',
                '.webp': 'image/webp'
            }
            media_type = media_types.get(ext, 'image/jpeg')

            client = anthropic.Anthropic()
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": "Describe this image in detail. Focus on the main subjects, any text content, and technical diagrams or charts if present."
                        }
                    ]
                }]
            )

            return message.content[0].text

        except Exception as e:
            print(f"[MULTIMODAL] Claude Vision API failed: {e}")

        return None

    def process_image(self, image_path: str, extract_ocr: bool = True, describe: bool = False) -> ImageContent:
        """
        Full image processing pipeline.

        Args:
            image_path: Path to image
            extract_ocr: Whether to extract OCR text
            describe: Whether to generate description (slower)

        Returns:
            ImageContent with all extracted data
        """
        content_id = hashlib.sha256(image_path.encode()).hexdigest()[:16]

        # Get basic info
        info = self.get_image_info(image_path)

        # Extract embedding
        embedding = self.extract_embedding(image_path)

        # Extract OCR
        ocr_text = self.extract_ocr_text(image_path) if extract_ocr else None

        # Combine text content
        text_content = ocr_text if ocr_text else ""

        return ImageContent(
            content_id=content_id,
            modality=ModalityType.IMAGE,
            source_path=image_path,
            text_content=text_content,
            embedding=embedding,
            ocr_text=ocr_text,
            description=None,  # Set async if needed
            width=info['width'],
            height=info['height'],
            format=info['format'],
            metadata={
                'size_bytes': info['size_bytes'],
                'has_ocr': ocr_text is not None
            }
        )


# ============================================================================
# AUDIO PROCESSOR
# ============================================================================

class AudioProcessor:
    """
    Process audio files for the BIZRA Data Lake using Whisper transcription.

    Capabilities:
    - Transcribe audio to text using OpenAI Whisper
    - Split long audio into chunks
    - Language detection
    """

    def __init__(self, model_size: str = WHISPER_LOCAL, use_gpu: bool = GPU_ENABLED):
        self.model_size = model_size
        self.use_gpu = use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.whisper_model = None
        self._initialized = False

        # Cache for transcripts
        self.cache_dir = MULTIMODAL_CACHE / "audio"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def initialize(self):
        """Load Whisper model (lazy initialization)."""
        if self._initialized:
            return

        if not WHISPER_AVAILABLE:
            print("[MULTIMODAL] Warning: whisper not installed. Audio processing disabled.")
            print("[MULTIMODAL] Install with: pip install openai-whisper")
            return

        print(f"[MULTIMODAL] Loading Whisper model: {self.model_size} on {self.device}")
        try:
            self.whisper_model = whisper.load_model(self.model_size, device=self.device)
            self._initialized = True
            print(f"[MULTIMODAL] Whisper model loaded successfully")
        except Exception as e:
            print(f"[MULTIMODAL] Failed to load Whisper: {e}")

    def _get_cache_path(self, audio_path: str) -> Path:
        """Get cache path for audio transcript."""
        file_hash = hashlib.sha256(audio_path.encode()).hexdigest()[:16]
        return self.cache_dir / f"{file_hash}.json"

    def _load_from_cache(self, audio_path: str) -> Optional[Dict]:
        """Load cached transcript if available."""
        cache_path = self._get_cache_path(audio_path)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    if data.get('source_path') == audio_path:
                        return data
            except Exception:
                pass
        return None

    def _save_to_cache(self, audio_path: str, transcript_data: Dict):
        """Save transcript to cache."""
        cache_path = self._get_cache_path(audio_path)
        try:
            with open(cache_path, 'w') as f:
                json.dump(transcript_data, f, indent=2)
        except Exception:
            pass

    def transcribe(self, audio_path: str, use_cache: bool = True) -> Optional[Dict]:
        """
        Transcribe audio file using Whisper.

        Args:
            audio_path: Path to audio file
            use_cache: Whether to use cached transcripts

        Returns:
            Dictionary with 'text', 'language', 'segments'
        """
        if use_cache:
            cached = self._load_from_cache(audio_path)
            if cached is not None:
                return cached

        if not self._initialized:
            self.initialize()

        if self.whisper_model is None:
            return None

        try:
            print(f"[MULTIMODAL] Transcribing: {audio_path}")
            result = self.whisper_model.transcribe(
                audio_path,
                language=None,  # Auto-detect
                task="transcribe"
            )

            transcript_data = {
                'source_path': audio_path,
                'text': result['text'],
                'language': result.get('language', 'unknown'),
                'segments': [
                    {
                        'start': seg['start'],
                        'end': seg['end'],
                        'text': seg['text']
                    }
                    for seg in result.get('segments', [])
                ],
                'transcribed_at': datetime.now().isoformat()
            }

            if use_cache:
                self._save_to_cache(audio_path, transcript_data)

            return transcript_data

        except Exception as e:
            print(f"[MULTIMODAL] Transcription failed for {audio_path}: {e}")
            return None

    def get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration in seconds."""
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
                 '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
                capture_output=True, text=True
            )
            return float(result.stdout.strip())
        except Exception:
            return 0.0

    def process_audio(self, audio_path: str) -> AudioContent:
        """
        Full audio processing pipeline.

        Args:
            audio_path: Path to audio file

        Returns:
            AudioContent with transcript and metadata
        """
        content_id = hashlib.sha256(audio_path.encode()).hexdigest()[:16]

        # Get duration
        duration = self.get_audio_duration(audio_path)

        # Transcribe
        transcript_data = self.transcribe(audio_path)

        text_content = ""
        language = "unknown"
        segments = []

        if transcript_data:
            text_content = transcript_data.get('text', '')
            language = transcript_data.get('language', 'unknown')
            segments = transcript_data.get('segments', [])

        return AudioContent(
            content_id=content_id,
            modality=ModalityType.AUDIO,
            source_path=audio_path,
            text_content=text_content,
            transcript=text_content,
            duration_seconds=duration,
            language=language,
            speaker_segments=segments,
            metadata={
                'file_size': os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
            }
        )


# ============================================================================
# MULTI-MODAL CHUNKER
# ============================================================================

class MultiModalChunker:
    """
    Create searchable chunks from multi-modal content.
    Bridges images and audio into the text-based RAG pipeline.
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_image_content(self, image_content: ImageContent) -> List[Dict[str, Any]]:
        """
        Create chunks from image content.

        Each image becomes one or more chunks containing:
        - OCR text (if present)
        - Description (if generated)
        - Reference to image embedding
        """
        chunks = []

        # Build text content
        text_parts = []
        if image_content.ocr_text:
            text_parts.append(f"[OCR TEXT]\n{image_content.ocr_text}")
        if image_content.description:
            text_parts.append(f"[IMAGE DESCRIPTION]\n{image_content.description}")

        combined_text = "\n\n".join(text_parts)

        if combined_text:
            # Split into chunks if needed
            if len(combined_text) <= self.chunk_size:
                chunks.append({
                    'content_id': image_content.content_id,
                    'text': combined_text,
                    'modality': 'image',
                    'source_path': image_content.source_path,
                    'has_embedding': image_content.embedding is not None,
                    'chunk_index': 0
                })
            else:
                # Split long text
                for i, start in enumerate(range(0, len(combined_text), self.chunk_size - self.overlap)):
                    chunk_text = combined_text[start:start + self.chunk_size]
                    chunks.append({
                        'content_id': f"{image_content.content_id}_{i}",
                        'text': chunk_text,
                        'modality': 'image',
                        'source_path': image_content.source_path,
                        'has_embedding': image_content.embedding is not None,
                        'chunk_index': i
                    })
        else:
            # No text content - create placeholder chunk for image-only search
            chunks.append({
                'content_id': image_content.content_id,
                'text': f"[IMAGE: {Path(image_content.source_path).name}]",
                'modality': 'image',
                'source_path': image_content.source_path,
                'has_embedding': image_content.embedding is not None,
                'chunk_index': 0,
                'image_only': True
            })

        return chunks

    def chunk_audio_content(self, audio_content: AudioContent, by_segments: bool = True) -> List[Dict[str, Any]]:
        """
        Create chunks from audio content.

        Can chunk by:
        - Time segments (from Whisper)
        - Fixed duration windows
        """
        chunks = []

        if by_segments and audio_content.speaker_segments:
            # Chunk by Whisper segments, grouping nearby segments
            current_text = []
            current_start = 0
            current_end = 0

            for i, seg in enumerate(audio_content.speaker_segments):
                current_text.append(seg['text'].strip())
                if i == 0:
                    current_start = seg['start']
                current_end = seg['end']

                # Create chunk if we've accumulated enough text or this is the last segment
                combined = ' '.join(current_text)
                if len(combined) >= self.chunk_size or i == len(audio_content.speaker_segments) - 1:
                    chunks.append({
                        'content_id': f"{audio_content.content_id}_{len(chunks)}",
                        'text': combined,
                        'modality': 'audio',
                        'source_path': audio_content.source_path,
                        'time_start': current_start,
                        'time_end': current_end,
                        'language': audio_content.language,
                        'chunk_index': len(chunks)
                    })
                    current_text = []
                    current_start = seg['end']
        else:
            # Chunk by text length
            full_text = audio_content.text_content or ""
            if full_text:
                for i, start in enumerate(range(0, len(full_text), self.chunk_size - self.overlap)):
                    chunk_text = full_text[start:start + self.chunk_size]
                    chunks.append({
                        'content_id': f"{audio_content.content_id}_{i}",
                        'text': chunk_text,
                        'modality': 'audio',
                        'source_path': audio_content.source_path,
                        'language': audio_content.language,
                        'chunk_index': i
                    })

        return chunks


# ============================================================================
# UNIFIED MULTI-MODAL ENGINE
# ============================================================================

class MultiModalEngine:
    """
    Unified engine for processing all multi-modal content.
    Integrates ImageProcessor, AudioProcessor, and MultiModalChunker.
    """

    def __init__(self, use_gpu: bool = GPU_ENABLED):
        self.image_processor = ImageProcessor(use_gpu=use_gpu) if VISION_ENABLED else None
        self.audio_processor = AudioProcessor(use_gpu=use_gpu) if AUDIO_ENABLED else None
        self.chunker = MultiModalChunker()
        self._initialized = False

    def initialize(self):
        """Initialize all processors."""
        if self._initialized:
            return

        print("[MULTIMODAL] Initializing Multi-Modal Engine...")

        if self.image_processor:
            self.image_processor.initialize()

        if self.audio_processor:
            self.audio_processor.initialize()

        self._initialized = True
        print("[MULTIMODAL] Multi-Modal Engine ready")

    def detect_modality(self, file_path: str) -> ModalityType:
        """Detect file modality from extension."""
        ext = Path(file_path).suffix.lower()

        if ext in IMAGE_EXTENSIONS:
            return ModalityType.IMAGE
        elif ext in AUDIO_EXTENSIONS:
            return ModalityType.AUDIO
        elif ext in VIDEO_EXTENSIONS:
            return ModalityType.VIDEO
        else:
            return ModalityType.TEXT

    def process_file(self, file_path: str, extract_ocr: bool = True, describe: bool = False) -> Optional[MultiModalContent]:
        """
        Process any supported file type.

        Args:
            file_path: Path to file
            extract_ocr: Whether to extract OCR from images
            describe: Whether to generate descriptions (slower)

        Returns:
            MultiModalContent or subclass
        """
        modality = self.detect_modality(file_path)

        if modality == ModalityType.IMAGE and self.image_processor:
            return self.image_processor.process_image(file_path, extract_ocr, describe)
        elif modality == ModalityType.AUDIO and self.audio_processor:
            return self.audio_processor.process_audio(file_path)
        elif modality == ModalityType.VIDEO:
            print(f"[MULTIMODAL] Video processing not yet implemented: {file_path}")
            return None
        else:
            return None

    def get_chunks(self, content: MultiModalContent) -> List[Dict[str, Any]]:
        """Get searchable chunks from multi-modal content."""
        if isinstance(content, ImageContent):
            return self.chunker.chunk_image_content(content)
        elif isinstance(content, AudioContent):
            return self.chunker.chunk_audio_content(content)
        else:
            return []

    def get_status(self) -> Dict[str, Any]:
        """Get engine status."""
        return {
            'initialized': self._initialized,
            'vision_enabled': VISION_ENABLED,
            'audio_enabled': AUDIO_ENABLED,
            'image_processor_ready': self.image_processor._initialized if self.image_processor else False,
            'audio_processor_ready': self.audio_processor._initialized if self.audio_processor else False,
            'device': 'cuda' if (TORCH_AVAILABLE and torch.cuda.is_available()) else 'cpu',
            'supported_image_formats': list(IMAGE_EXTENSIONS),
            'supported_audio_formats': list(AUDIO_EXTENSIONS)
        }


# ============================================================================
# MAIN / DEMO
# ============================================================================

def main():
    """Demonstrate multi-modal engine capabilities."""
    print("=" * 70)
    print("BIZRA Multi-Modal Engine v1.0")
    print("=" * 70)

    engine = MultiModalEngine()
    engine.initialize()

    status = engine.get_status()
    print("\nEngine Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")

    # Example usage
    print("\n" + "=" * 70)
    print("Example: Processing an image")
    print("=" * 70)
    print("""
    from multimodal_engine import MultiModalEngine

    engine = MultiModalEngine()
    engine.initialize()

    # Process an image
    content = engine.process_file("path/to/image.png")
    print(f"Embedding shape: {content.embedding.shape}")
    print(f"OCR text: {content.ocr_text[:100] if content.ocr_text else 'None'}")

    # Get chunks for RAG
    chunks = engine.get_chunks(content)
    print(f"Generated {len(chunks)} chunks")
    """)

    print("\n" + "=" * 70)
    print("Example: Processing an audio file")
    print("=" * 70)
    print("""
    # Process an audio file
    content = engine.process_file("path/to/audio.mp3")
    print(f"Transcript: {content.transcript[:200]}...")
    print(f"Duration: {content.duration_seconds}s")
    print(f"Language: {content.language}")

    # Get chunks for RAG
    chunks = engine.get_chunks(content)
    print(f"Generated {len(chunks)} chunks")
    """)


if __name__ == "__main__":
    main()
