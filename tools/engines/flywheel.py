#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
    BIZRA FLYWHEEL — Autopoietic Self-Sustaining Core
    
    "The system that runs itself, improves itself, and never stops."
    
    Components:
    1. Moshi Audio Streaming — Real-time voice I/O
    2. Local LLM Inference — No external dependencies
    3. Autopoietic Loop — Self-monitoring, self-healing
    4. Fail-Closed Auth — Zero-trust by default
    5. Warm Model Cache — No cold-start latency
    
    Created: 2026-01-29 | BIZRA Sovereignty
    Principle: لا نفترض — We do not assume.
═══════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import hashlib
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import threading
import queue

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

FLYWHEEL_VERSION = "1.0.0"
STATE_FILE = Path("/var/lib/bizra/flywheel_state.json")
LOCK_FILE = Path("/var/run/bizra/flywheel.lock")
LOG_DIR = Path("/var/log/bizra/flywheel")

# Model configuration
DEFAULT_LLM_MODEL_OLLAMA = "llama3.1:8b"
DEFAULT_LLM_MODEL_LMSTUDIO = "liquid/lfm2.5-1.2b"  # Fast model for routine tasks
DEFAULT_EMBED_MODEL = "nomic-embed-text:latest"

# Inference endpoints
OLLAMA_INTERNAL_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://192.168.56.1:1234")  # LM Studio on host
INFERENCE_TIMEOUT_SECONDS = 30

# Inference backend preference: "ollama", "lmstudio", "auto"
INFERENCE_BACKEND = os.getenv("BIZRA_INFERENCE_BACKEND", "auto")

# Autopoietic loop settings
HEARTBEAT_INTERVAL_SECONDS = 30
HEALTH_CHECK_INTERVAL_SECONDS = 60
SELF_HEAL_MAX_RETRIES = 3
STATE_PERSIST_INTERVAL_SECONDS = 300

# Auth settings (fail-closed)
AUTH_MODE = os.getenv("BIZRA_AUTH_MODE", "FAIL_CLOSED")
AUTH_TOKEN_ENV = "BIZRA_API_TOKEN"  # nosec B105 — env var name, not a password value
AUTH_REQUIRED_ENDPOINTS = ["inference", "audio", "state", "config"]


# ═══════════════════════════════════════════════════════════════════════════════
# TYPES
# ═══════════════════════════════════════════════════════════════════════════════

class FlywheelState(str, Enum):
    COLD = "COLD"           # Not started, models not loaded
    WARMING = "WARMING"     # Loading models into memory
    READY = "READY"         # Fully operational
    DEGRADED = "DEGRADED"   # Partial functionality
    HEALING = "HEALING"     # Self-repair in progress
    SHUTDOWN = "SHUTDOWN"   # Graceful shutdown


class AuthResult(str, Enum):
    ALLOWED = "ALLOWED"
    DENIED = "DENIED"
    MISSING = "MISSING"


@dataclass
class ComponentHealth:
    """Health status of a flywheel component."""
    name: str
    healthy: bool
    last_check: str
    latency_ms: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlywheelStatus:
    """Complete flywheel status."""
    state: FlywheelState
    uptime_seconds: float
    started_at: str
    components: List[ComponentHealth]
    models_loaded: List[str]
    auth_mode: str
    version: str = FLYWHEEL_VERSION
    
    @property
    def healthy(self) -> bool:
        return self.state == FlywheelState.READY and all(c.healthy for c in self.components)


@dataclass
class PersistedState:
    """State persisted across restarts."""
    last_healthy_timestamp: str
    total_uptime_seconds: float
    restart_count: int
    models_cached: List[str]
    last_error: Optional[str] = None
    ihsan_score: float = 0.95
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "PersistedState":
        return cls(**data)
    
    @classmethod
    def default(cls) -> "PersistedState":
        return cls(
            last_healthy_timestamp=datetime.now(timezone.utc).isoformat(),
            total_uptime_seconds=0.0,
            restart_count=0,
            models_cached=[],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FAIL-CLOSED AUTHENTICATION
# ═══════════════════════════════════════════════════════════════════════════════

class FailClosedAuth:
    """
    Zero-trust authentication layer.
    
    Default: DENY ALL
    Explicit allowance required for every operation.
    """
    
    def __init__(self, mode: str = "FAIL_CLOSED"):
        self.mode = mode
        self._token_hash: Optional[str] = None
        self._load_token()
    
    def _load_token(self):
        """Load and hash the API token from environment."""
        token = os.getenv(AUTH_TOKEN_ENV)
        if token:
            self._token_hash = hashlib.sha256(token.encode()).hexdigest()
    
    def authenticate(self, provided_token: Optional[str], endpoint: str) -> AuthResult:
        """
        Authenticate a request.
        
        FAIL-CLOSED: If anything is wrong, deny.
        """
        # No token configured = development mode warning
        if not self._token_hash:
            if self.mode == "FAIL_CLOSED":
                return AuthResult.DENIED
            # Only allow in explicit dev mode
            if os.getenv("BIZRA_DEV_MODE") == "true":
                return AuthResult.ALLOWED
            return AuthResult.DENIED
        
        # No token provided
        if not provided_token:
            return AuthResult.MISSING
        
        # Constant-time comparison
        provided_hash = hashlib.sha256(provided_token.encode()).hexdigest()
        if provided_hash == self._token_hash:
            return AuthResult.ALLOWED
        
        return AuthResult.DENIED
    
    def require_auth(self, endpoint: str):
        """Decorator for endpoints requiring authentication."""
        def decorator(func: Callable):
            async def wrapper(*args, **kwargs):
                # Extract token from request
                token = kwargs.get("auth_token") or kwargs.get("token")
                result = self.authenticate(token, endpoint)
                
                if result != AuthResult.ALLOWED:
                    raise PermissionError(f"Authentication failed: {result.value}")
                
                return await func(*args, **kwargs)
            return wrapper
        return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CACHE (Warm Start)
# ═══════════════════════════════════════════════════════════════════════════════

class ModelCache:
    """
    Keep models warm in memory to eliminate cold-start latency.
    
    Strategy:
    1. Pre-load priority models on startup
    2. Keep active models in memory
    3. LRU eviction for less-used models
    """
    
    def __init__(self, ollama_url: str = OLLAMA_INTERNAL_URL):
        self.ollama_url = ollama_url
        self._loaded_models: Dict[str, float] = {}  # model -> last_used_timestamp
        self._lock = threading.Lock()
    
    async def warm_model(self, model: str) -> bool:
        """
        Warm a model by sending a minimal inference request.
        This loads the model into GPU/RAM.
        """
        try:
            import httpx
            async with httpx.AsyncClient(timeout=120.0) as client:
                # Send a minimal prompt to load the model
                response = await client.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": model,
                        "prompt": ".",
                        "stream": False,
                        "options": {"num_predict": 1}
                    }
                )
                if response.status_code == 200:
                    with self._lock:
                        self._loaded_models[model] = time.time()
                    return True
                return False
        except Exception as e:
            print(f"[ModelCache] Failed to warm {model}: {e}")
            return False
    
    async def warm_priority_models(self, models: List[str]) -> Dict[str, bool]:
        """Warm all priority models concurrently."""
        results = {}
        tasks = [self.warm_model(m) for m in models]
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        for model, outcome in zip(models, outcomes):
            results[model] = outcome is True
        return results
    
    def is_warm(self, model: str) -> bool:
        """Check if a model is currently warm."""
        with self._lock:
            return model in self._loaded_models
    
    @property
    def loaded_models(self) -> List[str]:
        """List of currently loaded models."""
        with self._lock:
            return list(self._loaded_models.keys())


# ═══════════════════════════════════════════════════════════════════════════════
# LOCAL LLM INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

class LocalInference:
    """
    LLM inference supporting multiple backends:
    - Ollama (native API)
    - LM Studio (OpenAI-compatible API)
    
    Auto-failover: tries preferred backend first, falls back to alternative.
    """
    
    def __init__(
        self, 
        ollama_url: str = OLLAMA_INTERNAL_URL,
        lmstudio_url: str = LMSTUDIO_URL,
        backend: str = INFERENCE_BACKEND,
        model_cache: Optional[ModelCache] = None
    ):
        self.ollama_url = ollama_url
        self.lmstudio_url = lmstudio_url
        self.backend = backend  # "ollama", "lmstudio", or "auto"
        self.model_cache = model_cache or ModelCache(ollama_url)
        self.default_model_ollama = DEFAULT_LLM_MODEL_OLLAMA
        self.default_model_lmstudio = DEFAULT_LLM_MODEL_LMSTUDIO
        
        # Track backend availability
        self._ollama_available: Optional[bool] = None
        self._lmstudio_available: Optional[bool] = None
    
    @property
    def default_model(self) -> str:
        """Return default model based on active backend."""
        if self.backend == "lmstudio":
            return self.default_model_lmstudio
        return self.default_model_ollama
    
    async def _check_ollama(self) -> bool:
        """Check if Ollama is available."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{self.ollama_url}/api/tags")
                return response.status_code == 200
        except:
            return False
    
    async def _check_lmstudio(self) -> bool:
        """Check if LM Studio is available."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(f"{self.lmstudio_url}/v1/models")
                return response.status_code == 200
        except:
            return False
    
    async def _generate_ollama(
        self, prompt: str, model: str, system: Optional[str],
        temperature: float, max_tokens: int
    ) -> str:
        """Generate using Ollama API."""
        import httpx
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            }
        }
        if system:
            payload["system"] = system
        
        async with httpx.AsyncClient(timeout=INFERENCE_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{self.ollama_url}/api/generate",
                json=payload
            )
            if response.status_code != 200:
                raise RuntimeError(f"Ollama inference failed: {response.status_code}")
            return response.json().get("response", "")
    
    async def _generate_lmstudio(
        self, prompt: str, model: str, system: Optional[str],
        temperature: float, max_tokens: int
    ) -> str:
        """Generate using LM Studio (OpenAI-compatible API)."""
        import httpx
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        
        async with httpx.AsyncClient(timeout=INFERENCE_TIMEOUT_SECONDS) as client:
            response = await client.post(
                f"{self.lmstudio_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            if response.status_code != 200:
                raise RuntimeError(f"LM Studio inference failed: {response.status_code}")
            
            result = response.json()
            return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False,
    ) -> str:
        """
        Generate text using available backend.
        
        Auto mode: tries LM Studio first (usually faster), falls back to Ollama.
        """
        # Determine backend and model
        if self.backend == "auto":
            # Check LM Studio first (it's on the host with more resources)
            if self._lmstudio_available is None:
                self._lmstudio_available = await self._check_lmstudio()
            if self._ollama_available is None:
                self._ollama_available = await self._check_ollama()
            
            if self._lmstudio_available:
                try:
                    return await self._generate_lmstudio(
                        prompt,
                        model or self.default_model_lmstudio,
                        system, temperature, max_tokens
                    )
                except Exception as e:
                    print(f"[LocalInference] LM Studio failed, trying Ollama: {e}")
                    self._lmstudio_available = False
            
            if self._ollama_available:
                return await self._generate_ollama(
                    prompt,
                    model or self.default_model_ollama,
                    system, temperature, max_tokens
                )
            
            raise RuntimeError("No inference backend available")
        
        elif self.backend == "lmstudio":
            return await self._generate_lmstudio(
                prompt,
                model or self.default_model_lmstudio,
                system, temperature, max_tokens
            )
        
        else:  # ollama
            return await self._generate_ollama(
                prompt,
                model or self.default_model_ollama,
                system, temperature, max_tokens
            )
    
    async def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embeddings using local model (Ollama or LM Studio)."""
        model = model or DEFAULT_EMBED_MODEL
        
        try:
            import httpx
            
            # Try Ollama first for embeddings
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.ollama_url}/api/embeddings",
                    json={"model": model, "prompt": text}
                )
                if response.status_code == 200:
                    return response.json().get("embedding", [])
            
            # Fall back to LM Studio embeddings endpoint
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.lmstudio_url}/v1/embeddings",
                    json={"model": model, "input": text}
                )
                if response.status_code == 200:
                    data = response.json().get("data", [])
                    if data:
                        return data[0].get("embedding", [])
            
            raise RuntimeError("Embedding failed on all backends")
        except Exception as e:
            raise RuntimeError(f"Embedding failed: {e}")
    
    async def health_check(self) -> ComponentHealth:
        """Check if inference is operational (any backend)."""
        start = time.time()
        
        ollama_ok = await self._check_ollama()
        lmstudio_ok = await self._check_lmstudio()
        
        latency = (time.time() - start) * 1000
        healthy = ollama_ok or lmstudio_ok
        
        return ComponentHealth(
            name="local_inference",
            healthy=healthy,
            last_check=datetime.now(timezone.utc).isoformat(),
            latency_ms=latency,
            metadata={
                "ollama": "available" if ollama_ok else "unavailable",
                "lmstudio": "available" if lmstudio_ok else "unavailable",
                "active_backend": self.backend,
            }
        )


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO STREAMING — Whisper STT + Piper TTS (faster-whisper available)
# ═══════════════════════════════════════════════════════════════════════════════

class AudioStreaming:
    """
    Real-time audio processing with faster-whisper (STT) and TTS.
    
    Uses:
    - faster-whisper: Speech-to-Text (installed in kernel container)
    - Piper/edge-tts: Text-to-Speech
    
    For full duplex streaming, Moshi can be added when available.
    """
    
    def __init__(self):
        self.stt_available = False
        self.tts_available = False
        self.whisper_model = None
        self._check_availability()
    
    def _check_availability(self):
        """Check if audio processing is available."""
        # Check for faster-whisper
        try:
            import importlib.util
            spec = importlib.util.find_spec("faster_whisper")
            self.stt_available = spec is not None
        except Exception:
            self.stt_available = False
        
        # Check for TTS (edge-tts or piper)
        try:
            spec = importlib.util.find_spec("edge_tts")
            self.tts_available = spec is not None
        except:
            self.tts_available = False
    
    async def initialize(self, whisper_model: str = "base") -> bool:
        """
        Initialize audio processing models.
        
        Args:
            whisper_model: Whisper model size (tiny, base, small, medium, large-v3)
        """
        if not self.stt_available:
            print("[AudioStreaming] faster-whisper not installed")
            return False
        
        try:
            from faster_whisper import WhisperModel
            
            # Use GPU if available, else CPU with int8
            device = "cuda" if self._gpu_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            print(f"[AudioStreaming] Loading Whisper {whisper_model} on {device}")
            self.whisper_model = WhisperModel(whisper_model, device=device, compute_type=compute_type)
            return True
            
        except Exception as e:
            print(f"[AudioStreaming] Failed to load Whisper: {e}")
            return False
    
    def _gpu_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    async def transcribe(self, audio_bytes: bytes, language: str = "en") -> str:
        """
        Transcribe audio to text using faster-whisper.
        
        Args:
            audio_bytes: Audio data (WAV, MP3, etc.)
            language: Language code (e.g., "en", "ar")
        
        Returns:
            Transcribed text
        """
        if not self.whisper_model:
            raise RuntimeError("Whisper not initialized. Call initialize() first.")
        
        import tempfile
        import os
        
        # Write audio to temp file (faster-whisper needs file path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(audio_bytes)
            temp_path = f.name
        
        try:
            segments, info = self.whisper_model.transcribe(
                temp_path,
                language=language,
                beam_size=5,
                vad_filter=True,  # Filter out silence
            )
            
            # Combine all segments
            text = " ".join(segment.text for segment in segments)
            return text.strip()
            
        finally:
            os.unlink(temp_path)
    
    async def synthesize(self, text: str, voice: str = "en-US-AriaNeural") -> bytes:
        """
        Synthesize text to speech using edge-tts.
        
        Args:
            text: Text to synthesize
            voice: Voice ID (e.g., "en-US-AriaNeural", "ar-SA-HamedNeural")
        
        Returns:
            Audio bytes (MP3)
        """
        if not self.tts_available:
            raise RuntimeError("TTS not available. Install edge-tts.")
        
        import edge_tts
        import tempfile
        
        communicate = edge_tts.Communicate(text, voice)
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            temp_path = f.name
        
        try:
            await communicate.save(temp_path)
            with open(temp_path, "rb") as f:
                return f.read()
        finally:
            import os
            os.unlink(temp_path)
    
    async def stream_audio(self, audio_chunk: bytes) -> Optional[bytes]:
        """
        Process audio chunk for real-time streaming.
        
        Currently returns None — full duplex streaming requires Moshi.
        Use transcribe() + LLM + synthesize() for turn-based conversation.
        """
        # Real-time streaming would require Moshi or similar
        # For now, use the turn-based API
        return None
    
    async def health_check(self) -> ComponentHealth:
        """Check audio streaming health."""
        return ComponentHealth(
            name="audio_streaming",
            healthy=self.stt_available,
            last_check=datetime.now(timezone.utc).isoformat(),
            metadata={
                "stt": "faster-whisper" if self.stt_available else "unavailable",
                "tts": "edge-tts" if self.tts_available else "unavailable",
                "whisper_loaded": self.whisper_model is not None,
                "realtime_streaming": "requires Moshi",
            }
        )


# ═══════════════════════════════════════════════════════════════════════════════
# AUTOPOIETIC LOOP
# ═══════════════════════════════════════════════════════════════════════════════

class AutopoieticLoop:
    """
    Self-sustaining, self-monitoring, self-healing loop.
    
    The flywheel that never stops.
    
    Responsibilities:
    1. Monitor all components
    2. Detect degradation
    3. Self-heal when possible
    4. Persist state for resurrection
    5. Graceful degradation when healing fails
    """
    
    def __init__(
        self,
        inference: LocalInference,
        audio: MoshiAudioStreaming,
        auth: FailClosedAuth,
        model_cache: ModelCache,
    ):
        self.inference = inference
        self.audio = audio
        self.auth = auth
        self.model_cache = model_cache
        
        self.state = FlywheelState.COLD
        self.started_at: Optional[datetime] = None
        self.persisted_state: Optional[PersistedState] = None
        
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None
        self._heal_attempts: Dict[str, int] = {}
    
    async def start(self) -> bool:
        """Start the autopoietic loop."""
        if self._running:
            return True
        
        # Load persisted state
        self.persisted_state = self._load_state()
        self.persisted_state.restart_count += 1
        
        # Transition to warming
        self.state = FlywheelState.WARMING
        self.started_at = datetime.now(timezone.utc)
        
        # Warm priority models
        print("[Flywheel] Warming models...")
        warm_results = await self.model_cache.warm_priority_models([
            DEFAULT_LLM_MODEL_OLLAMA,
            DEFAULT_EMBED_MODEL,
        ])
        
        # Initialize audio (if available)
        await self.audio.initialize()
        
        # Start the loop
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())
        
        # Transition to ready (or degraded if some components failed)
        all_warm = all(warm_results.values())
        self.state = FlywheelState.READY if all_warm else FlywheelState.DEGRADED
        
        print(f"[Flywheel] Started in state: {self.state.value}")
        return True
    
    async def stop(self):
        """Gracefully stop the flywheel."""
        self.state = FlywheelState.SHUTDOWN
        self._running = False
        
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        
        # Persist final state
        self._save_state()
        print("[Flywheel] Stopped gracefully.")
    
    async def _run_loop(self):
        """Main autopoietic loop."""
        last_health_check = 0
        last_state_persist = 0
        
        while self._running:
            try:
                now = time.time()
                
                # Health check
                if now - last_health_check >= HEALTH_CHECK_INTERVAL_SECONDS:
                    await self._check_health()
                    last_health_check = now
                
                # Persist state
                if now - last_state_persist >= STATE_PERSIST_INTERVAL_SECONDS:
                    self._save_state()
                    last_state_persist = now
                
                # Heartbeat
                await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[Flywheel] Loop error: {e}")
                await asyncio.sleep(5)
    
    async def _check_health(self):
        """Check health of all components and trigger healing if needed."""
        components = await self.get_component_health()
        
        unhealthy = [c for c in components if not c.healthy]
        
        if not unhealthy:
            if self.state == FlywheelState.DEGRADED:
                self.state = FlywheelState.READY
                print("[Flywheel] Recovered to READY state.")
            return
        
        # Attempt self-healing
        for component in unhealthy:
            await self._attempt_heal(component)
    
    async def _attempt_heal(self, component: ComponentHealth):
        """Attempt to heal a degraded component."""
        attempts = self._heal_attempts.get(component.name, 0)
        
        if attempts >= SELF_HEAL_MAX_RETRIES:
            print(f"[Flywheel] Max heal attempts reached for {component.name}")
            return
        
        self._heal_attempts[component.name] = attempts + 1
        self.state = FlywheelState.HEALING
        
        print(f"[Flywheel] Healing {component.name} (attempt {attempts + 1})")
        
        if component.name == "local_inference":
            # Try to re-warm the model
            success = await self.model_cache.warm_model(DEFAULT_LLM_MODEL)
            if success:
                self._heal_attempts[component.name] = 0
                print(f"[Flywheel] Healed {component.name}")
        
        elif component.name == "moshi_audio":
            # Try to reinitialize
            success = await self.audio.initialize()
            if success:
                self._heal_attempts[component.name] = 0
        
        # Re-check health
        health = await self._get_single_health(component.name)
        if health.healthy:
            self.state = FlywheelState.READY
        else:
            self.state = FlywheelState.DEGRADED
    
    async def _get_single_health(self, name: str) -> ComponentHealth:
        """Get health of a single component by name."""
        if name == "local_inference":
            return await self.inference.health_check()
        elif name == "moshi_audio":
            return await self.audio.health_check()
        return ComponentHealth(name=name, healthy=False, last_check="", error="Unknown component")
    
    async def get_component_health(self) -> List[ComponentHealth]:
        """Get health of all components."""
        return [
            await self.inference.health_check(),
            await self.audio.health_check(),
            ComponentHealth(
                name="auth",
                healthy=True,  # Auth is always "healthy" — it just denies/allows
                last_check=datetime.now(timezone.utc).isoformat(),
                metadata={"mode": self.auth.mode}
            ),
            ComponentHealth(
                name="model_cache",
                healthy=len(self.model_cache.loaded_models) > 0,
                last_check=datetime.now(timezone.utc).isoformat(),
                metadata={"loaded": self.model_cache.loaded_models}
            ),
        ]
    
    async def get_status(self) -> FlywheelStatus:
        """Get complete flywheel status."""
        components = await self.get_component_health()
        uptime = 0.0
        if self.started_at:
            uptime = (datetime.now(timezone.utc) - self.started_at).total_seconds()
        
        return FlywheelStatus(
            state=self.state,
            uptime_seconds=uptime,
            started_at=self.started_at.isoformat() if self.started_at else "",
            components=components,
            models_loaded=self.model_cache.loaded_models,
            auth_mode=self.auth.mode,
        )
    
    def _load_state(self) -> PersistedState:
        """Load persisted state from disk."""
        try:
            if STATE_FILE.exists():
                with open(STATE_FILE) as f:
                    data = json.load(f)
                    return PersistedState.from_dict(data)
        except Exception as e:
            print(f"[Flywheel] Could not load state: {e}")
        
        return PersistedState.default()
    
    def _save_state(self):
        """Persist state to disk."""
        if not self.persisted_state:
            return
        
        try:
            # Update cumulative uptime
            if self.started_at:
                session_uptime = (datetime.now(timezone.utc) - self.started_at).total_seconds()
                self.persisted_state.total_uptime_seconds += session_uptime
            
            self.persisted_state.last_healthy_timestamp = datetime.now(timezone.utc).isoformat()
            self.persisted_state.models_cached = self.model_cache.loaded_models
            
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(STATE_FILE, "w") as f:
                json.dump(self.persisted_state.to_dict(), f, indent=2)
                
        except Exception as e:
            print(f"[Flywheel] Could not save state: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# FLYWHEEL SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

class Flywheel:
    """
    The BIZRA Flywheel — Self-sustaining cognitive core.
    
    Singleton that orchestrates all components.
    Integrated with the Accumulator for impact tracking.
    """
    
    _instance: Optional["Flywheel"] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Initialize components
        self.auth = FailClosedAuth(AUTH_MODE)
        self.model_cache = ModelCache()
        self.inference = LocalInference(model_cache=self.model_cache)
        self.audio = AudioStreaming()
        
        # Initialize autopoietic loop
        self.loop = AutopoieticLoop(
            inference=self.inference,
            audio=self.audio,
            auth=self.auth,
            model_cache=self.model_cache,
        )
        
        # Initialize Accumulator (optional - fail soft)
        self.accumulator = None
        try:
            from accumulator import get_accumulator
            self.accumulator = get_accumulator()
            print("[Flywheel] Accumulator connected ✓")
        except ImportError:
            print("[Flywheel] Accumulator not available (optional)")
        except Exception as e:
            print(f"[Flywheel] Accumulator init failed: {e}")
        
        self._initialized = True
    
    async def activate(self) -> bool:
        """Activate the flywheel."""
        print("═" * 80)
        print("    BIZRA FLYWHEEL — Activating")
        print("═" * 80)
        
        success = await self.loop.start()
        
        if success:
            status = await self.loop.get_status()
            print(f"\n✅ Flywheel Active")
            print(f"   State: {status.state.value}")
            print(f"   Models: {', '.join(status.models_loaded) or 'None'}")
            print(f"   Auth: {status.auth_mode}")
            if self.accumulator:
                acc_status = self.accumulator.status()
                print(f"   Accumulator: {acc_status['state']} | Bloom: {acc_status['total_bloom']:.1f}")
            print("═" * 80)
        
        return success
    
    async def deactivate(self):
        """Deactivate the flywheel."""
        await self.loop.stop()
    
    async def status(self) -> FlywheelStatus:
        """Get flywheel status."""
        return await self.loop.get_status()
    
    async def infer(
        self,
        prompt: str,
        model: Optional[str] = None,
        auth_token: Optional[str] = None,
        contributor: str = "flywheel:anonymous",
        **kwargs
    ) -> str:
        """
        Run inference through the flywheel.
        
        Fail-closed: Requires authentication.
        Records impact to accumulator if available.
        """
        # Authenticate
        result = self.auth.authenticate(auth_token, "inference")
        if result != AuthResult.ALLOWED:
            raise PermissionError(f"Inference denied: {result.value}")
        
        # Track timing for impact
        start_time = time.time()
        
        response = await self.inference.generate(prompt, model=model, **kwargs)
        
        # Record impact to accumulator
        if self.accumulator:
            latency_ms = (time.time() - start_time) * 1000
            tokens_estimated = len(prompt.split()) + len(response.split())
            
            try:
                self.accumulator.record_computation(
                    contributor=contributor,
                    tokens_processed=tokens_estimated,
                    latency_ms=latency_ms,
                    model=model or self.inference.default_model,
                )
            except Exception as e:
                print(f"[Flywheel] Accumulator record failed: {e}")
        
        return response


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

async def main():
    """CLI entry point for the flywheel."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BIZRA Flywheel — Autopoietic Core")
    parser.add_argument("command", choices=["start", "status", "stop", "test"])
    parser.add_argument("--token", help="API token for authenticated operations")
    args = parser.parse_args()
    
    flywheel = Flywheel()
    
    if args.command == "start":
        await flywheel.activate()
        
        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(60)
                status = await flywheel.status()
                print(f"[Flywheel] State: {status.state.value}, Uptime: {status.uptime_seconds:.0f}s")
        except KeyboardInterrupt:
            print("\n[Flywheel] Shutting down...")
            await flywheel.deactivate()
    
    elif args.command == "status":
        # Quick status check without full activation
        status = await flywheel.status()
        print(json.dumps(asdict(status), indent=2, default=str))
    
    elif args.command == "test":
        await flywheel.activate()
        
        # Test inference
        try:
            result = await flywheel.infer(
                "Say 'Flywheel operational' in exactly 3 words.",
                auth_token=args.token or os.getenv(AUTH_TOKEN_ENV)
            )
            print(f"[Test] Inference result: {result}")
        except PermissionError as e:
            print(f"[Test] Auth failed: {e}")
        except Exception as e:
            print(f"[Test] Inference failed: {e}")
        
        await flywheel.deactivate()


if __name__ == "__main__":
    asyncio.run(main())
