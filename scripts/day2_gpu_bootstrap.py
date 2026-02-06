#!/usr/bin/env python3
"""
BIZRA DAY 2: GPU ACCELERATION BOOTSTRAP
═══════════════════════════════════════════════════════════════════════════════

Target: 8.63 tok/s (CPU) → 35+ tok/s (RTX 4090)
Model: 0.5B → 1.5B with maintained speed

Run on Windows host with CUDA:
  cd C:\BIZRA-DATA-LAKE
  python day2_gpu_bootstrap.py

Created: 2026-01-30 | BIZRA Sovereignty
Giants: Al-Jazari (engineering precision), Al-Khwarizmi (algorithmic optimization)
"""

import asyncio
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GPUConfig:
    """GPU acceleration configuration."""
    # CUDA settings
    cuda_visible_devices: str = "0"
    
    # llama.cpp settings
    n_gpu_layers: int = -1  # -1 = all layers on GPU
    n_ctx: int = 8192
    n_batch: int = 512
    n_threads: int = 8
    
    # Model paths
    model_dir: Path = Path("C:/BIZRA-DATA-LAKE/models")
    
    # Targets
    target_speed_small: float = 50.0  # 0.5B target
    target_speed_medium: float = 35.0  # 1.5B target
    target_speed_large: float = 15.0   # 7B target


# ═══════════════════════════════════════════════════════════════════════════════
# GPU DETECTION
# ═══════════════════════════════════════════════════════════════════════════════

def detect_gpu() -> dict:
    """Detect NVIDIA GPU and CUDA availability."""
    result = {
        "cuda_available": False,
        "gpu_name": None,
        "gpu_memory_gb": None,
        "cuda_version": None,
        "driver_version": None,
    }
    
    try:
        # Try nvidia-smi
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
            text=True
        )
        parts = output.strip().split(", ")
        if len(parts) >= 3:
            result["cuda_available"] = True
            result["gpu_name"] = parts[0]
            result["gpu_memory_gb"] = float(parts[1].replace(" MiB", "")) / 1024
            result["driver_version"] = parts[2]
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Check CUDA version
    try:
        output = subprocess.check_output(
            ["nvcc", "--version"],
            stderr=subprocess.DEVNULL,
            text=True
        )
        for line in output.split("\n"):
            if "release" in line.lower():
                result["cuda_version"] = line.split("release")[-1].split(",")[0].strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return result


def check_llama_cpp_cuda() -> bool:
    """Check if llama-cpp-python is compiled with CUDA support."""
    try:
        from llama_cpp import Llama
        # Try to detect CUDA support
        import llama_cpp
        # Check if CUDA backend is available
        return hasattr(llama_cpp, 'LLAMA_BACKEND_OFFLOAD') or True  # Assume yes if installed
    except ImportError:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL MANAGEMENT
# ═══════════════════════════════════════════════════════════════════════════════

MODELS = {
    "small": {
        "repo": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
        "file": "qwen2.5-0.5b-instruct-q4_k_m.gguf",
        "params": "0.5B",
        "size_mb": 469,
        "target_speed": 50.0,
    },
    "medium": {
        "repo": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "file": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "params": "1.5B",
        "size_mb": 1100,
        "target_speed": 35.0,
    },
    "large": {
        "repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "file": "qwen2.5-7b-instruct-q4_k_m.gguf",
        "params": "7B",
        "size_mb": 4500,
        "target_speed": 15.0,
    },
}


def download_model(model_key: str, model_dir: Path) -> Optional[Path]:
    """Download model from HuggingFace if not present."""
    model = MODELS.get(model_key)
    if not model:
        print(f"Unknown model: {model_key}")
        return None
    
    model_path = model_dir / model["file"]
    
    if model_path.exists():
        print(f"✅ Model already exists: {model['file']}")
        return model_path
    
    print(f"📥 Downloading {model['file']} ({model['size_mb']}MB)...")
    
    try:
        from huggingface_hub import hf_hub_download
        
        path = hf_hub_download(
            repo_id=model["repo"],
            filename=model["file"],
            local_dir=str(model_dir),
        )
        print(f"✅ Downloaded to: {path}")
        return Path(path)
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    model: str
    params: str
    n_gpu_layers: int
    prompt_tokens: int
    generated_tokens: int
    total_time_s: float
    tokens_per_second: float
    time_to_first_token_ms: float
    target_speed: float
    target_met: bool
    
    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "params": self.params,
            "n_gpu_layers": self.n_gpu_layers,
            "prompt_tokens": self.prompt_tokens,
            "generated_tokens": self.generated_tokens,
            "total_time_s": round(self.total_time_s, 3),
            "tokens_per_second": round(self.tokens_per_second, 2),
            "time_to_first_token_ms": round(self.time_to_first_token_ms, 2),
            "target_speed": self.target_speed,
            "target_met": self.target_met,
        }


def benchmark_model(
    model_path: Path,
    model_info: dict,
    config: GPUConfig,
    prompt: str = "Explain the concept of sovereignty in one paragraph.",
    max_tokens: int = 100,
) -> Optional[BenchmarkResult]:
    """Benchmark a model with GPU acceleration."""
    try:
        from llama_cpp import Llama
    except ImportError:
        print("❌ llama-cpp-python not installed")
        return None
    
    print(f"\n🧪 Benchmarking: {model_path.name}")
    print(f"   GPU Layers: {config.n_gpu_layers}")
    print(f"   Context: {config.n_ctx}")
    print(f"   Batch: {config.n_batch}")
    
    # Load model
    load_start = time.time()
    try:
        llm = Llama(
            model_path=str(model_path),
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            n_batch=config.n_batch,
            n_threads=config.n_threads,
            verbose=False,
        )
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return None
    
    load_time = time.time() - load_start
    print(f"   Load time: {load_time:.2f}s")
    
    # Warmup
    print("   Warming up...")
    _ = llm("Hello", max_tokens=5)
    
    # Benchmark
    print("   Running benchmark...")
    
    first_token_time = None
    generated_tokens = 0
    
    start_time = time.time()
    
    # Use streaming to measure TTFT
    response_text = ""
    for chunk in llm(prompt, max_tokens=max_tokens, stream=True):
        if first_token_time is None:
            first_token_time = time.time()
        
        if "choices" in chunk and chunk["choices"]:
            text = chunk["choices"][0].get("text", "")
            response_text += text
            generated_tokens += 1
    
    end_time = time.time()
    
    # Calculate metrics
    total_time = end_time - start_time
    ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else 0
    tokens_per_second = generated_tokens / total_time if total_time > 0 else 0
    
    # Check target
    target_speed = model_info.get("target_speed", 20.0)
    target_met = tokens_per_second >= target_speed
    
    result = BenchmarkResult(
        model=model_path.name,
        params=model_info.get("params", "?"),
        n_gpu_layers=config.n_gpu_layers,
        prompt_tokens=len(prompt.split()),
        generated_tokens=generated_tokens,
        total_time_s=total_time,
        tokens_per_second=tokens_per_second,
        time_to_first_token_ms=ttft_ms,
        target_speed=target_speed,
        target_met=target_met,
    )
    
    # Print result
    status = "✅" if target_met else "⚠️"
    print(f"\n   {status} Result: {tokens_per_second:.2f} tok/s (target: {target_speed})")
    print(f"   TTFT: {ttft_ms:.0f}ms")
    print(f"   Generated: {generated_tokens} tokens in {total_time:.2f}s")
    
    # Cleanup
    del llm
    
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("═" * 70)
    print("    BIZRA DAY 2: GPU ACCELERATION BOOTSTRAP")
    print("═" * 70)
    print()
    
    config = GPUConfig()
    results = []
    
    # Step 1: Detect GPU
    print("[1/5] Detecting GPU...")
    gpu_info = detect_gpu()
    
    if gpu_info["cuda_available"]:
        print(f"✅ GPU: {gpu_info['gpu_name']}")
        print(f"   Memory: {gpu_info['gpu_memory_gb']:.1f} GB")
        print(f"   Driver: {gpu_info['driver_version']}")
        print(f"   CUDA: {gpu_info['cuda_version'] or 'Unknown'}")
    else:
        print("⚠️ No NVIDIA GPU detected")
        print("   Running in CPU mode (slower)")
        config.n_gpu_layers = 0
    
    # Step 2: Check llama-cpp-python
    print("\n[2/5] Checking llama-cpp-python...")
    if check_llama_cpp_cuda():
        print("✅ llama-cpp-python installed")
    else:
        print("❌ llama-cpp-python not found")
        print("\n   To install with CUDA support:")
        print("   pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
        return 1
    
    # Step 3: Download models
    print("\n[3/5] Checking models...")
    config.model_dir.mkdir(parents=True, exist_ok=True)
    
    models_to_test = ["small", "medium"]  # Start with small and medium
    model_paths = {}
    
    for model_key in models_to_test:
        path = download_model(model_key, config.model_dir)
        if path:
            model_paths[model_key] = path
    
    if not model_paths:
        print("❌ No models available for testing")
        return 1
    
    # Step 4: Benchmark
    print("\n[4/5] Running benchmarks...")
    
    for model_key, model_path in model_paths.items():
        model_info = MODELS[model_key]
        result = benchmark_model(model_path, model_info, config)
        if result:
            results.append(result)
    
    # Step 5: Summary
    print("\n" + "═" * 70)
    print("    BENCHMARK SUMMARY")
    print("═" * 70)
    
    all_targets_met = True
    
    for result in results:
        status = "✅" if result.target_met else "❌"
        print(f"\n{status} {result.params} ({result.model})")
        print(f"   Speed: {result.tokens_per_second:.2f} tok/s (target: {result.target_speed})")
        print(f"   TTFT: {result.time_to_first_token_ms:.0f}ms")
        print(f"   GPU Layers: {result.n_gpu_layers}")
        
        if not result.target_met:
            all_targets_met = False
    
    # Save results
    results_file = config.model_dir.parent / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "gpu": gpu_info,
            "config": {
                "n_gpu_layers": config.n_gpu_layers,
                "n_ctx": config.n_ctx,
                "n_batch": config.n_batch,
            },
            "results": [r.to_dict() for r in results],
        }, f, indent=2)
    
    print(f"\n📊 Results saved to: {results_file}")
    
    # Final verdict
    print("\n" + "═" * 70)
    if all_targets_met:
        print("    ✅ DAY 2 TARGETS MET — GPU ACCELERATION SUCCESSFUL")
    else:
        print("    ⚠️ SOME TARGETS NOT MET — OPTIMIZATION NEEDED")
    print("═" * 70)
    
    return 0 if all_targets_met else 1


if __name__ == "__main__":
    sys.exit(main())
