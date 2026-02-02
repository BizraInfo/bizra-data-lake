# BIZRA Corpus Manager v2.0 (Layer 1: Canonical Storage)
# Converts Data Lake chaos into a structured 'Corpus Table' (Parquet)
# Implements Engineering Excellence: Provenance, Hashing, Multi-Modal Extraction
# Now supports: Documents, Images (OCR), Audio (Whisper)

import os
import hashlib
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import json
from typing import Optional, Dict, Any

from bizra_config import (
    PROCESSED_PATH, CORPUS_TABLE_PATH,
    VISION_ENABLED, AUDIO_ENABLED,
    IMAGE_EXTENSIONS, AUDIO_EXTENSIONS,
    LOG_DIR, INGEST_GATE_ENFORCE
)
from unstructured.partition.auto import partition

# Optional multi-modal imports
try:
    from multimodal_engine import MultiModalEngine, ModalityType
    MULTIMODAL_AVAILABLE = True
except ImportError:
    MULTIMODAL_AVAILABLE = False

# Optional ingestion gate
try:
    from core.iaas.ingest_gates import IngestGate
    INGEST_GATE_AVAILABLE = True
except Exception:
    INGEST_GATE_AVAILABLE = False

class CorpusManager:
    """
    BIZRA Corpus Manager v2.0

    Canonizes files from the Data Lake into a structured Parquet corpus.
    Now supports multi-modal content:
    - Documents: PDF, DOCX, PPTX, HTML, TXT, MD (via Unstructured-IO)
    - Code: Python, Rust, JS, TS, JSON, YAML
    - Images: JPG, PNG, GIF, etc. (OCR via Tesseract, CLIP for search)
    - Audio: MP3, WAV, FLAC, etc. (Whisper transcription)
    """

    def __init__(self, enable_multimodal: bool = True, enforce_ingest_gate: bool = False):
        print("🏛️ Initializing BIZRA Corpus Manager v2.0")
        self.processed_dir = PROCESSED_PATH
        self.output_path = CORPUS_TABLE_PATH
        self.index_data = []
        self.ingest_log_path = LOG_DIR / "ingest_gate.jsonl"

        # Multi-modal support
        self.multimodal_enabled = enable_multimodal and MULTIMODAL_AVAILABLE
        self.multimodal: Optional[MultiModalEngine] = None

        if self.multimodal_enabled:
            try:
                self.multimodal = MultiModalEngine()
                self.multimodal.initialize()
                print("   Multi-Modal Engine: Enabled (Vision + Audio)")
            except Exception as e:
                print(f"   Multi-Modal Engine: Disabled ({e})")
                self.multimodal_enabled = False
        else:
            print("   Multi-Modal Engine: Disabled")

        # Ingestion gate (optional)
        if INGEST_GATE_AVAILABLE:
            self.ingest_gate = IngestGate(enforce=(enforce_ingest_gate or INGEST_GATE_ENFORCE))
        else:
            self.ingest_gate = None
        if self.ingest_gate:
            mode = "ENFORCE" if self.ingest_gate.enforce else "SOFT"
            print(f"   Ingest Gate: Enabled ({mode})")

    def _is_image_file(self, path: Path) -> bool:
        """Check if file is a supported image format."""
        return path.suffix.lower() in IMAGE_EXTENSIONS

    def _is_audio_file(self, path: Path) -> bool:
        """Check if file is a supported audio format."""
        return path.suffix.lower() in AUDIO_EXTENSIONS

    def get_file_metadata(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Extracts stable ID and provenance with multi-modal support."""
        try:
            stat = file_path.stat()
            text_content = ""
            elements = []
            modality = "text"
            extra_metadata = {}

            # Route to appropriate processor based on file type
            if self._is_image_file(file_path):
                return self._process_image_file(file_path, stat)
            elif self._is_audio_file(file_path):
                return self._process_audio_file(file_path, stat)
            elif file_path.suffix.lower() in ['.pdf', '.docx', '.pptx', '.html', '.txt', '.md']:
                # Use Unstructured for documents
                print(f"📄 Partitioning: {file_path.name}")
                elements = partition(filename=str(file_path))
                text_content = "\n\n".join([str(el) for el in elements])
            elif file_path.suffix.lower() in ['.py', '.rs', '.js', '.ts', '.json', '.yaml', '.yml', '.toml', '.sh', '.bat', '.ps1']:
                # Code files
                text_content = file_path.read_text(errors='ignore')
                modality = "code"
            else:
                # Try to read as text, skip binary files
                try:
                    text_content = file_path.read_text(errors='ignore')
                except:
                    text_content = ""

            file_hash = hashlib.sha256(text_content.encode() if text_content else str(file_path).encode()).hexdigest()

            return {
                "doc_id": f"{file_hash[:16]}",
                "source_type": self._detect_source_type(file_path),
                "modality": modality,
                "uri": str(file_path.absolute()),
                "file_path": str(file_path),
                "title": file_path.name,
                "mime": self._get_mime(file_path),
                "lang": "en",
                "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "project": file_path.parts[-3] if len(file_path.parts) > 3 else "unknown",
                "hash_sha256": file_hash,
                "text": text_content,
                "text_quality": "high_fidelity" if elements else ("ok" if text_content else "no_text"),
                "metadata_json": json.dumps({
                    "size_bytes": stat.st_size,
                    "extension": file_path.suffix,
                    "element_count": len(elements),
                    **extra_metadata
                })
            }
        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            return None

    def _process_image_file(self, file_path: Path, stat) -> Optional[Dict[str, Any]]:
        """Process image file with OCR and CLIP embedding preparation."""
        text_content = ""
        ocr_text = None
        image_info = {}

        if self.multimodal_enabled and self.multimodal and self.multimodal.image_processor:
            try:
                print(f"🖼️ Processing image: {file_path.name}")
                image_content = self.multimodal.image_processor.process_image(
                    str(file_path), extract_ocr=True, describe=False
                )
                if image_content:
                    ocr_text = image_content.ocr_text
                    text_content = ocr_text if ocr_text else f"[IMAGE: {file_path.name}]"
                    image_info = {
                        "width": image_content.width,
                        "height": image_content.height,
                        "format": image_content.format,
                        "has_ocr": ocr_text is not None
                    }
            except Exception as e:
                print(f"   ⚠️ Image processing failed: {e}")
                text_content = f"[IMAGE: {file_path.name}]"
        else:
            text_content = f"[IMAGE: {file_path.name}]"

        file_hash = hashlib.sha256(str(file_path).encode()).hexdigest()

        return {
            "doc_id": f"{file_hash[:16]}",
            "source_type": "image",
            "modality": "image",
            "uri": str(file_path.absolute()),
            "file_path": str(file_path),
            "title": file_path.name,
            "mime": self._get_mime(file_path),
            "lang": "en",
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "project": file_path.parts[-3] if len(file_path.parts) > 3 else "unknown",
            "hash_sha256": file_hash,
            "text": text_content,
            "text_quality": "ocr" if ocr_text else "placeholder",
            "metadata_json": json.dumps({
                "size_bytes": stat.st_size,
                "extension": file_path.suffix,
                **image_info
            })
        }

    def _process_audio_file(self, file_path: Path, stat) -> Optional[Dict[str, Any]]:
        """Process audio file with Whisper transcription."""
        text_content = ""
        transcript = None
        audio_info = {}

        if self.multimodal_enabled and self.multimodal and self.multimodal.audio_processor:
            try:
                print(f"🎵 Transcribing: {file_path.name}")
                audio_content = self.multimodal.audio_processor.process_audio(str(file_path))
                if audio_content:
                    transcript = audio_content.transcript
                    text_content = transcript if transcript else f"[AUDIO: {file_path.name}]"
                    audio_info = {
                        "duration_seconds": audio_content.duration_seconds,
                        "language": audio_content.language,
                        "has_transcript": transcript is not None
                    }
            except Exception as e:
                print(f"   ⚠️ Audio processing failed: {e}")
                text_content = f"[AUDIO: {file_path.name}]"
        else:
            text_content = f"[AUDIO: {file_path.name}]"

        file_hash = hashlib.sha256(str(file_path).encode()).hexdigest()

        return {
            "doc_id": f"{file_hash[:16]}",
            "source_type": "audio",
            "modality": "audio",
            "uri": str(file_path.absolute()),
            "file_path": str(file_path),
            "title": file_path.name,
            "mime": self._get_mime(file_path),
            "lang": audio_info.get("language", "en"),
            "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "project": file_path.parts[-3] if len(file_path.parts) > 3 else "unknown",
            "hash_sha256": file_hash,
            "text": text_content,
            "text_quality": "transcript" if transcript else "placeholder",
            "metadata_json": json.dumps({
                "size_bytes": stat.st_size,
                "extension": file_path.suffix,
                **audio_info
            })
        }

    def _detect_source_type(self, path: Path):
        if "code" in path.parts: return "repo_file"
        if "chat_history" in str(path): return "chat_turn"
        if path.suffix == ".pdf": return "pdf"
        if path.suffix in [".md", ".txt"]: return "note"
        return "generic"

    def _get_mime(self, path: Path) -> str:
        """Detect MIME type from file extension."""
        ext = path.suffix.lower()
        mapping = {
            # Documents
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".html": "text/html",
            ".txt": "text/plain",
            ".md": "text/markdown",
            # Code
            ".py": "text/x-python",
            ".js": "text/javascript",
            ".ts": "text/typescript",
            ".json": "application/json",
            ".yaml": "application/x-yaml",
            ".yml": "application/x-yaml",
            # Images
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".bmp": "image/bmp",
            ".webp": "image/webp",
            ".svg": "image/svg+xml",
            # Audio
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".m4a": "audio/mp4",
            ".ogg": "audio/ogg",
            ".wma": "audio/x-ms-wma",
        }
        return mapping.get(ext, "application/octet-stream")

    def build_corpus(self):
        """Crawls PROCESSED_PATH and builds the documents table with multi-modal support."""
        print(f"🔍 Crawling {self.processed_dir} for canonization...")
        files = list(self.processed_dir.rglob("*"))
        files = [f for f in files if f.is_file()]

        # Track modality counts
        modality_counts = {"text": 0, "code": 0, "image": 0, "audio": 0}
        gate_rejects = 0

        for f in tqdm(files, desc="Canonizing Corpus"):
            meta = self.get_file_metadata(f)
            if meta:
                # Optional SNR + Ihsān gate
                if self.ingest_gate:
                    verdict = self.ingest_gate.evaluate(meta)
                    try:
                        md = json.loads(meta.get("metadata_json") or "{}")
                    except Exception:
                        md = {}
                    md["ingest_gate"] = verdict
                    meta["metadata_json"] = json.dumps(md)

                    # Log verdicts
                    try:
                        self.ingest_log_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(self.ingest_log_path, "a", encoding="utf-8") as logf:
                            logf.write(json.dumps({"file": meta.get("file_path"), **verdict}) + "\n")
                    except Exception:
                        pass

                    if self.ingest_gate.enforce and not verdict.get("passed", False):
                        gate_rejects += 1
                        continue

                self.index_data.append(meta)
                modality = meta.get("modality", "text")
                modality_counts[modality] = modality_counts.get(modality, 0) + 1

        df = pd.DataFrame(self.index_data)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output_path, index=False)

        print(f"\n✅ Corpus Table saved: {self.output_path}")
        print(f"   Total documents: {len(df)}")
        if self.ingest_gate and self.ingest_gate.enforce:
            print(f"   Gate rejects: {gate_rejects}")
        print(f"   By modality:")
        for modality, count in modality_counts.items():
            if count > 0:
                print(f"     - {modality}: {count}")

    def get_stats(self) -> Dict[str, Any]:
        """Get corpus statistics."""
        if not self.output_path.exists():
            return {"error": "Corpus not built yet"}

        df = pd.read_parquet(self.output_path)

        stats = {
            "total_documents": len(df),
            "by_modality": df.get("modality", pd.Series(["text"] * len(df))).value_counts().to_dict(),
            "by_source_type": df["source_type"].value_counts().to_dict(),
            "by_text_quality": df["text_quality"].value_counts().to_dict(),
            "total_text_chars": df["text"].str.len().sum(),
            "multimodal_enabled": self.multimodal_enabled
        }

        return stats


if __name__ == "__main__":
    manager = CorpusManager(enable_multimodal=True)
    manager.build_corpus()

    print("\n" + "=" * 50)
    print("Corpus Statistics:")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")


    def build_corpus_parallel(self, workers: int = 4):
        """Parallel corpus build (folder-level batching)."""
        print(f"🔍 Parallel crawling with {workers} workers...")
        files = [f for f in self.processed_dir.rglob('*') if f.is_file()]
        if not files:
            print("⚠️ No files to process.")
            return

        import concurrent.futures
        modality_counts = {"text": 0, "code": 0, "image": 0, "audio": 0}
        gate_rejects = 0

        def process_file(f):
            return self.get_file_metadata(f)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            for meta in ex.map(process_file, files):
                if not meta:
                    continue
                if self.ingest_gate:
                    verdict = self.ingest_gate.evaluate(meta)
                    try:
                        md = json.loads(meta.get("metadata_json") or "{}")
                    except Exception:
                        md = {}
                    md["ingest_gate"] = verdict
                    meta["metadata_json"] = json.dumps(md)
                    try:
                        self.ingest_log_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(self.ingest_log_path, "a", encoding="utf-8") as logf:
                            logf.write(json.dumps({"file": meta.get("file_path"), **verdict}) + "\n")
                    except Exception:
                        pass
                    if self.ingest_gate.enforce and not verdict.get("passed", False):
                        gate_rejects += 1
                        continue

                self.index_data.append(meta)
                modality = meta.get("modality", "text")
                modality_counts[modality] = modality_counts.get(modality, 0) + 1

        df = pd.DataFrame(self.index_data)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.output_path, index=False)

        print(f"\n✅ Corpus Table saved: {self.output_path}")
        print(f"   Total documents: {len(df)}")
        if self.ingest_gate and self.ingest_gate.enforce:
            print(f"   Gate rejects: {gate_rejects}")
        print(f"   By modality:")
        for modality, count in modality_counts.items():
            if count > 0:
                print(f"     - {modality}: {count}")
