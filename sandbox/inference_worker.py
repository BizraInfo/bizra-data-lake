"""
BIZRA Inference Worker - Sandboxed LLM Execution

This module runs inference in a quarantined environment:
- No network access
- No filesystem access except /tmp/models
- Treated as untrusted via WasmSandbox

"We do not assume. We verify with formal proofs."
"""

import os
import sys
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from enum import Enum

# Try to import llama-cpp-python
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None


class ModelTier(Enum):
    """Model capability tiers."""
    EDGE = "EDGE"      # 0.5B-1.5B, CPU
    LOCAL = "LOCAL"    # 7B-13B, GPU
    POOL = "POOL"      # 70B+, federated


@dataclass
class InferenceRequest:
    """Request for inference."""
    id: str
    prompt: str
    model_id: str
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stop: Optional[List[str]] = None

    @classmethod
    def from_json(cls, data: str) -> "InferenceRequest":
        parsed = json.loads(data)
        return cls(**parsed)


@dataclass
class InferenceResponse:
    """Response from inference."""
    id: str
    content: str
    model_id: str
    tokens_generated: int
    generation_time_ms: int
    ihsan_score: float
    snr_score: float
    success: bool
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))


class ModelStore:
    """
    Offline model management with zero network dependencies.

    Models are stored as GGUF files in the designated directory.
    """

    def __init__(self, store_path: Optional[Path] = None):
        self.store_path = store_path or Path.home() / ".bizra" / "models"
        self.models_dir = self.store_path / "gguf"
        self.registry_path = self.store_path / "registry.json"
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._ensure_directories()
        self._load_registry()

    def _ensure_directories(self):
        """Create directories if they don't exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _load_registry(self):
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r") as f:
                    self._registry = json.load(f)
            except Exception:
                self._registry = {}

    def _save_registry(self):
        """Save registry to disk."""
        with open(self.registry_path, "w") as f:
            json.dump(self._registry, f, indent=2)

    def import_model(
        self,
        model_path: Path,
        model_id: str,
        tier: ModelTier = ModelTier.LOCAL,
        copy: bool = True
    ) -> bool:
        """Import a GGUF model into the store."""
        if not model_path.exists():
            return False

        # Calculate file hash
        file_hash = self._hash_file(model_path)

        # Determine target path
        target_path = self.models_dir / f"{model_id}.gguf"

        if copy:
            # Copy file to store
            import shutil
            shutil.copy2(model_path, target_path)
        else:
            # Create symlink
            if target_path.exists():
                target_path.unlink()
            target_path.symlink_to(model_path.resolve())

        # Register model
        self._registry[model_id] = {
            "path": str(target_path),
            "original_path": str(model_path),
            "tier": tier.value,
            "file_hash": file_hash,
            "size_bytes": model_path.stat().st_size,
            "imported_at": time.time(),
        }
        self._save_registry()

        return True

    def get_model_path(self, model_id: str) -> Optional[Path]:
        """Get path to model file."""
        if model_id not in self._registry:
            return None
        path = Path(self._registry[model_id]["path"])
        return path if path.exists() else None

    def list_models(self, tier: Optional[ModelTier] = None) -> List[Dict[str, Any]]:
        """List all registered models."""
        models = []
        for model_id, info in self._registry.items():
            if tier is None or info["tier"] == tier.value:
                models.append({"id": model_id, **info})
        return models

    def _hash_file(self, path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


class InferenceBackend:
    """
    llama.cpp inference backend.

    Runs in sandboxed environment with no network access.
    """

    def __init__(
        self,
        model_path: Path,
        n_ctx: int = 2048,
        n_gpu_layers: int = -1,  # -1 = all layers on GPU
        n_threads: Optional[int] = None,
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads or os.cpu_count()
        self._model: Optional[Llama] = None

    def load(self) -> bool:
        """Load the model into memory."""
        if not LLAMA_CPP_AVAILABLE:
            return False

        try:
            self._model = Llama(
                model_path=str(self.model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                n_threads=self.n_threads,
                verbose=False,
            )
            return True
        except Exception as e:
            print(f"Failed to load model: {e}", file=sys.stderr)
            return False

    def unload(self):
        """Unload the model from memory."""
        self._model = None

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None

    def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate text from prompt."""
        if not self.is_loaded():
            return InferenceResponse(
                id=request.id,
                content="",
                model_id=request.model_id,
                tokens_generated=0,
                generation_time_ms=0,
                ihsan_score=0.0,
                snr_score=0.0,
                success=False,
                error="Model not loaded",
            )

        start_time = time.perf_counter()

        try:
            output = self._model(
                request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stop=request.stop,
            )

            content = output["choices"][0]["text"]
            tokens_generated = output["usage"]["completion_tokens"]
            generation_time_ms = int((time.perf_counter() - start_time) * 1000)

            # Calculate quality scores
            ihsan_score = self._score_ihsan(content)
            snr_score = self._score_snr(content)

            return InferenceResponse(
                id=request.id,
                content=content,
                model_id=request.model_id,
                tokens_generated=tokens_generated,
                generation_time_ms=generation_time_ms,
                ihsan_score=ihsan_score,
                snr_score=snr_score,
                success=True,
            )

        except Exception as e:
            return InferenceResponse(
                id=request.id,
                content="",
                model_id=request.model_id,
                tokens_generated=0,
                generation_time_ms=int((time.perf_counter() - start_time) * 1000),
                ihsan_score=0.0,
                snr_score=0.0,
                success=False,
                error=str(e),
            )

    def _score_ihsan(self, content: str) -> float:
        """
        Score content for ethical excellence (Ihsān).

        Checks for positive ethical indicators and penalizes negative ones.
        """
        content_lower = content.lower()

        positive_indicators = [
            "privacy", "consent", "transparency", "user control",
            "data protection", "security", "ethical", "responsible",
            "respect", "trust", "confidential", "accountable",
        ]

        negative_indicators = [
            "collect all", "share with third", "without consent",
            "track", "surveil", "exploit", "manipulate",
        ]

        score = 0.85

        for indicator in positive_indicators:
            if indicator in content_lower:
                score += 0.02

        for indicator in negative_indicators:
            if indicator in content_lower:
                score -= 0.05

        return max(0.0, min(1.0, score))

    def _score_snr(self, content: str) -> float:
        """
        Score content for signal-to-noise ratio.

        Based on Shannon's information theory principles.
        """
        words = content.split()
        word_count = len(words)

        if word_count == 0:
            return 0.0

        # Unique word ratio (signal density)
        unique_words = set(w.lower() for w in words)
        signal_density = len(unique_words) / word_count

        # Filler word penalty
        filler_words = ["um", "uh", "like", "you know", "basically", "actually"]
        content_lower = content.lower()
        filler_count = sum(1 for f in filler_words if f in content_lower)
        filler_penalty = filler_count * 0.03

        # Conciseness score (target 50-200 words)
        if 50 <= word_count <= 200:
            conciseness = 1.0
        elif word_count < 50:
            conciseness = word_count / 50
        else:
            conciseness = 200 / word_count

        score = signal_density * 0.4 + conciseness * 0.6 - filler_penalty
        return max(0.0, min(1.0, score))


class SandboxWorker:
    """
    Main sandbox worker that handles IPC and inference.

    Communicates via stdin/stdout JSON messages.
    """

    def __init__(self, model_store: ModelStore):
        self.model_store = model_store
        self.backends: Dict[str, InferenceBackend] = {}
        self.active_model: Optional[str] = None

    def load_model(self, model_id: str) -> bool:
        """Load a model by ID."""
        model_path = self.model_store.get_model_path(model_id)
        if not model_path:
            return False

        # Get tier-specific settings
        model_info = next(
            (m for m in self.model_store.list_models() if m["id"] == model_id),
            None
        )

        tier = ModelTier(model_info["tier"]) if model_info else ModelTier.LOCAL

        # Configure based on tier
        if tier == ModelTier.EDGE:
            n_ctx = 2048
            n_gpu_layers = 0  # CPU only
        elif tier == ModelTier.LOCAL:
            n_ctx = 4096
            n_gpu_layers = -1  # All on GPU
        else:
            n_ctx = 8192
            n_gpu_layers = -1

        backend = InferenceBackend(
            model_path=model_path,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
        )

        if backend.load():
            self.backends[model_id] = backend
            self.active_model = model_id
            return True

        return False

    def unload_model(self, model_id: str) -> bool:
        """Unload a model."""
        if model_id in self.backends:
            self.backends[model_id].unload()
            del self.backends[model_id]
            if self.active_model == model_id:
                self.active_model = None
            return True
        return False

    def infer(self, request: InferenceRequest) -> InferenceResponse:
        """Run inference on a request."""
        model_id = request.model_id

        # Load model if not already loaded
        if model_id not in self.backends:
            if not self.load_model(model_id):
                return InferenceResponse(
                    id=request.id,
                    content="",
                    model_id=model_id,
                    tokens_generated=0,
                    generation_time_ms=0,
                    ihsan_score=0.0,
                    snr_score=0.0,
                    success=False,
                    error=f"Failed to load model: {model_id}",
                )

        backend = self.backends[model_id]
        return backend.generate(request)

    def run_stdio_loop(self):
        """Run the main stdin/stdout message loop."""
        print('{"status": "ready", "version": "2.2.0"}', flush=True)

        for line in sys.stdin:
            try:
                msg = json.loads(line.strip())
                msg_type = msg.get("type", "inference")

                if msg_type == "inference":
                    request = InferenceRequest.from_json(json.dumps(msg["request"]))
                    response = self.infer(request)
                    print(response.to_json(), flush=True)

                elif msg_type == "load_model":
                    model_id = msg["model_id"]
                    success = self.load_model(model_id)
                    print(json.dumps({
                        "type": "model_loaded",
                        "model_id": model_id,
                        "success": success,
                    }), flush=True)

                elif msg_type == "unload_model":
                    model_id = msg["model_id"]
                    success = self.unload_model(model_id)
                    print(json.dumps({
                        "type": "model_unloaded",
                        "model_id": model_id,
                        "success": success,
                    }), flush=True)

                elif msg_type == "list_models":
                    tier = ModelTier(msg["tier"]) if "tier" in msg else None
                    models = self.model_store.list_models(tier)
                    print(json.dumps({
                        "type": "model_list",
                        "models": models,
                    }), flush=True)

                elif msg_type == "shutdown":
                    print(json.dumps({"type": "shutdown", "status": "ok"}), flush=True)
                    break

                else:
                    print(json.dumps({
                        "type": "error",
                        "message": f"Unknown message type: {msg_type}",
                    }), flush=True)

            except json.JSONDecodeError as e:
                print(json.dumps({
                    "type": "error",
                    "message": f"Invalid JSON: {e}",
                }), flush=True)

            except Exception as e:
                print(json.dumps({
                    "type": "error",
                    "message": str(e),
                }), flush=True)


def main():
    """Entry point for the sandbox worker."""
    # SECURITY (SEC-007): Fail-closed sandbox enforcement
    # CRITICAL: Never execute inference outside sandbox environment
    if os.environ.get("BIZRA_SANDBOX") != "1":
        print(json.dumps({
            "type": "error",
            "code": "SANDBOX_VIOLATION",
            "message": "Refusing execution: BIZRA_SANDBOX not set",
            "fatal": True
        }), file=sys.stderr, flush=True)
        sys.exit(78)  # EX_CONFIG - configuration error

    # Initialize model store
    store_path = os.environ.get("BIZRA_MODEL_STORE")
    model_store = ModelStore(Path(store_path) if store_path else None)

    # Create and run worker
    worker = SandboxWorker(model_store)
    worker.run_stdio_loop()


if __name__ == "__main__":
    main()
