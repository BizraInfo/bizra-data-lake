# BIZRA Sovereign Configuration Manager
# Centralizes all paths and hyperparameters to achieve Engineering Excellence
#
# PORTABILITY: All paths now configurable via environment variables
# Supports: Windows native, WSL, Linux, macOS

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load credentials
load_dotenv()

# ============================================================================
# PLATFORM DETECTION & PATH RESOLUTION
# ============================================================================

def _detect_platform() -> str:
    """Detect the current runtime platform."""
    if sys.platform.startswith("win"):
        return "windows"
    elif sys.platform.startswith("linux"):
        # Check for WSL
        if "microsoft" in os.uname().release.lower():
            return "wsl"
        return "linux"
    elif sys.platform == "darwin":
        return "macos"
    return "unknown"

PLATFORM = _detect_platform()

def _resolve_data_lake_root() -> Path:
    """
    Resolve DATA_LAKE_ROOT with fallback hierarchy:
    1. BIZRA_DATA_LAKE_ROOT env var (explicit override)
    2. Platform-specific defaults
    3. Current working directory fallback
    """
    # Explicit override always wins
    if env_root := os.getenv("BIZRA_DATA_LAKE_ROOT"):
        return Path(env_root)
    
    # Platform-specific defaults
    if PLATFORM == "windows":
        default = Path("C:/BIZRA-DATA-LAKE")
    elif PLATFORM == "wsl":
        # WSL: Try Windows path via /mnt/c, fallback to Linux home
        wsl_win_path = Path("/mnt/c/BIZRA-DATA-LAKE")
        if wsl_win_path.exists():
            return wsl_win_path
        default = Path.home() / "bizra-data-lake"
    elif PLATFORM == "linux":
        default = Path.home() / "bizra-data-lake"
    elif PLATFORM == "macos":
        default = Path.home() / "bizra-data-lake"
    else:
        default = Path.cwd() / "bizra-data-lake"
    
    return default


# --- CORE PATHS (Environment-Configurable) ---
DATA_LAKE_ROOT = _resolve_data_lake_root()
INTAKE_PATH = DATA_LAKE_ROOT / "00_INTAKE"
RAW_PATH = DATA_LAKE_ROOT / "01_RAW"
PROCESSED_PATH = DATA_LAKE_ROOT / "02_PROCESSED"
INDEXED_PATH = DATA_LAKE_ROOT / "03_INDEXED"
GOLD_PATH = DATA_LAKE_ROOT / "04_GOLD"

# External Sources & Sovereign Domains (environment-configurable)
def _resolve_downloads_path() -> Path:
    """Resolve downloads path based on platform."""
    if env_path := os.getenv("BIZRA_DOWNLOADS_PATH"):
        return Path(env_path)
    if PLATFORM == "windows":
        return Path("C:/Users/BIZRA-OS/Downloads")
    elif PLATFORM == "wsl":
        return Path("/mnt/c/Users/BIZRA-OS/Downloads")
    else:
        return Path.home() / "Downloads"

DOWNLOADS_PATH = _resolve_downloads_path()
NODE0_KNOWLEDGE = DATA_LAKE_ROOT / "01_RAW/external_links/knowledge"
BIZRA_PROJECTS = DATA_LAKE_ROOT / "01_RAW/external_links/BIZRA-PROJECTS"
GENESIS_NODE = DATA_LAKE_ROOT / "01_RAW/external_links/bizra-genesis-node-repaired"

# --- KNOWLEDGE KERNEL PATHS ---
GRAPH_PATH = INDEXED_PATH / "graph"
EMBEDDINGS_PATH = INDEXED_PATH / "embeddings"
VECTORS_PATH = EMBEDDINGS_PATH / "vectors"
CHECKPOINT_PATH = EMBEDDINGS_PATH / "checkpoint.json"

# --- CANONICAL STORAGE (PARQUET/DUCKDB) ---
CORPUS_TABLE_PATH = GOLD_PATH / "documents.parquet"
CHUNKS_TABLE_PATH = GOLD_PATH / "chunks.parquet"
DUCKDB_PATH = GOLD_PATH / "bizra_lake.duckdb"

# --- HARDWARE OPTIMIZATION ---
# Optimized for: RTX 4090 (16GB), 128GB RAM
BATCH_SIZE = 128
MAX_SEQ_LENGTH = 512
GPU_ENABLED = True

# --- REASONING PARAMETERS ---
SNR_THRESHOLD = 0.85     # Minimum acceptable signal quality
IHSAN_CONSTRAINT = 0.95  # Target for Excellence (Exceeding expectations)
ARTE_TENSION_LIMIT = 0.75

# --- EXTRACTION CONFIG (LANGEXTRACT) ---
EXTRACTION_MODEL = "gemini-1.5-flash"
EXTRACTION_PROMPT = """Extract core concepts, technical tools, and project decisions. 
Use verbatim text. Map everything to the underlying BIZRA architecture."""

# --- LOGGING & AUDIT ---
LOG_DIR = DATA_LAKE_ROOT / "02_PROCESSED" / "logs"

# ============================================================================
# MULTI-MODAL CONFIGURATION
# ============================================================================

# Vision Models
VISION_ENABLED = True
CLIP_MODEL = "openai/clip-vit-base-patch32"  # 512-dim embeddings
VISION_LLM_LOCAL = "llava:7b"  # Local via Ollama
VISION_LLM_API = "claude-3-5-sonnet-20241022"  # Cloud fallback

# Audio Models
AUDIO_ENABLED = True
WHISPER_MODEL = "openai/whisper-large-v3"  # Speech-to-text
WHISPER_LOCAL = "base"  # Local whisper model size: tiny, base, small, medium, large

# Multi-Modal Processing Parameters
IMAGE_BATCH_SIZE = 32  # Optimized for RTX 4090
AUDIO_CHUNK_DURATION = 30  # seconds per chunk
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.m4a', '.ogg', '.wma', '.aac'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.webm'}

# Multi-Modal Embedding Dimensions
CLIP_EMBEDDING_DIM = 512
TEXT_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2

# Cross-Modal Search Settings
CROSS_MODAL_WEIGHT = 0.7  # Weight for cross-modal similarity (vs text-only)
IMAGE_TEXT_THRESHOLD = 0.5  # Min similarity for image-text matching

# ============================================================================
# MULTI-BACKEND CONFIGURATION (LM Studio + Ollama)
# ============================================================================

# Dual Agentic System Bridge (LM Studio / Local Multi-Model Server)
DUAL_AGENTIC_URL = os.getenv("DUAL_AGENTIC_URL", "http://192.168.56.1:1234")
DUAL_AGENTIC_ENABLED = True

# LM Studio OpenAI-Compatible API Settings
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_URL", "http://192.168.56.1:1234/v1")
LM_STUDIO_ENABLED = True

# Ollama Local Backend (Fallback)
OLLAMA_BASE_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_ENABLED = True
OLLAMA_TIMEOUT = 120.0  # seconds

# Default Model Mappings
DEFAULT_TEXT_MODEL = os.getenv("DEFAULT_TEXT_MODEL", "liquid/lfm2.5-1.2b")
DEFAULT_REASONING_MODEL = os.getenv("DEFAULT_REASONING_MODEL", "agentflow-planner-7b-i1")
DEFAULT_CODE_MODEL = os.getenv("DEFAULT_CODE_MODEL", "qwen2.5-14b_uncensored_instruct")
DEFAULT_VISION_MODEL = os.getenv("DEFAULT_VISION_MODEL", "qwen/qwen3-vl-8b")

# Ollama Fallback Models
OLLAMA_TEXT_MODEL = os.getenv("OLLAMA_TEXT_MODEL", "llama3.2")
OLLAMA_CODE_MODEL = os.getenv("OLLAMA_CODE_MODEL", "codellama:7b")
OLLAMA_VISION_MODEL = os.getenv("OLLAMA_VISION_MODEL", "llava:7b")

# ============================================================================
# RESILIENCE CONFIGURATION
# ============================================================================

# Circuit Breaker Settings
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 3  # Failures before opening circuit
CIRCUIT_BREAKER_SUCCESS_THRESHOLD = 2  # Successes to close circuit
CIRCUIT_BREAKER_TIMEOUT = 30.0  # Seconds to wait before half-open

# Retry Settings
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # Seconds
RETRY_MAX_DELAY = 30.0  # Seconds
RETRY_EXPONENTIAL_BASE = 2.0

# Health Check Settings
HEALTH_CHECK_INTERVAL = 60.0  # Seconds between health checks
HEALTH_CHECK_TIMEOUT = 5.0  # Seconds to wait for health response

# Backend Priority (1 = highest)
BACKEND_PRIORITY = {
    "lm_studio": 1,  # Primary
    "ollama": 2,     # Fallback
    "openai": 3,     # Cloud fallback
}

# Multi-Modal Storage Paths
MULTIMODAL_CACHE = INDEXED_PATH / "multimodal_cache"
IMAGE_EMBEDDINGS_PATH = EMBEDDINGS_PATH / "image_vectors"
AUDIO_TRANSCRIPTS_PATH = GOLD_PATH / "audio_transcripts"

# ============================================================================
# XTR-WARP CONFIGURATION (Multi-Vector Contextualized Retrieval)
# ============================================================================

# WARP Index Paths
WARP_INDEX_ROOT = Path(os.getenv("INDEX_ROOT", str(INDEXED_PATH / "warp_indexes")))
WARP_EXPERIMENT_ROOT = Path(os.getenv("EXPERIMENT_ROOT", str(GOLD_PATH / "warp_experiments")))
BEIR_COLLECTION_PATH = Path(os.getenv("BEIR_COLLECTION_PATH", str(PROCESSED_PATH / "beir")))
LOTTE_COLLECTION_PATH = Path(os.getenv("LOTTE_COLLECTION_PATH", str(PROCESSED_PATH / "lotte")))

# WARP Model Settings
WARP_CHECKPOINT = os.getenv("WARP_CHECKPOINT", "answerdotai/answerai-colbert-small-v1")
WARP_NBITS = int(os.getenv("WARP_NBITS", "2"))
WARP_NPROBE = int(os.getenv("WARP_NPROBE", "16"))
WARP_T_PRIME = int(os.getenv("WARP_T_PRIME", "128"))
WARP_BOUND = int(os.getenv("WARP_BOUND", "128"))

# WARP Engine Settings
WARP_ENABLED = True
WARP_USE_GPU = GPU_ENABLED  # Prefer GPU for indexing
WARP_RUNTIME = "onnx"  # Options: "onnx", "openvino", "coreml" (macOS only)
WARP_FUSED_EXT = True  # Fused decompression + merge for multi-threaded

def initialize_directories():
    """Ensure all BIZRA infrastructure directories exist."""
    dirs = [INTAKE_PATH, RAW_PATH, PROCESSED_PATH, INDEXED_PATH, GOLD_PATH,
            GRAPH_PATH, EMBEDDINGS_PATH, VECTORS_PATH, LOG_DIR,
            MULTIMODAL_CACHE, IMAGE_EMBEDDINGS_PATH, AUDIO_TRANSCRIPTS_PATH,
            WARP_INDEX_ROOT, WARP_EXPERIMENT_ROOT]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    initialize_directories()
    print("✅ BIZRA Infrastructure Validated.")
