#!/usr/bin/env python3
"""
BIZRA INFERENCE INTEGRATION TEST
═══════════════════════════════════════════════════════════════════════════════

Full inference pipeline test with a real model.

Requires:
- llama-cpp-python installed
- GGUF model in /mnt/c/BIZRA-DATA-LAKE/models/

Run: 
  source .venv/bin/activate
  python test_inference_integration.py

Created: 2026-01-29 | BIZRA Sovereignty
"""

import asyncio
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from core.inference.gateway import (
    InferenceGateway,
    InferenceConfig,
    ComputeTier,
    InferenceStatus,
)

# Configuration
MODEL_DIR = Path(os.environ.get("BIZRA_MODEL_DIR", "/mnt/c/BIZRA-DATA-LAKE/models"))
MODEL_PATH = os.environ.get("BIZRA_MODEL_PATH", None)


def find_model() -> Path | None:
    """Find a GGUF model file."""
    if MODEL_PATH:
        path = Path(MODEL_PATH)
        if path.exists():
            return path
    
    if MODEL_DIR.exists():
        gguf_files = list(MODEL_DIR.glob("*.gguf"))
        if gguf_files:
            return gguf_files[0]
    
    return None


async def test_inference_pipeline():
    """Test the full inference pipeline."""
    print("═" * 70)
    print("    BIZRA INFERENCE INTEGRATION TEST")
    print("═" * 70)
    print()
    
    # Find model
    model_path = find_model()
    if not model_path:
        print("❌ No model found!")
        print(f"   Checked: {MODEL_DIR}")
        print()
        print("   To fix:")
        print("   1. Run: scripts/bootstrap_inference.sh")
        print("   2. Or download manually:")
        print("      huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF \\")
        print("        qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir models/")
        return False
    
    print(f"✅ Found model: {model_path.name}")
    print(f"   Size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
    print()
    
    # Create gateway
    config = InferenceConfig(
        model_path=str(model_path),
        n_gpu_layers=0,  # CPU for WSL2 testing
        context_length=4096,
        max_tokens=256,
    )
    
    gateway = InferenceGateway(config)
    
    # Initialize
    print("[1/4] Initializing gateway...")
    start = time.time()
    success = await gateway.initialize()
    init_time = time.time() - start
    
    if not success:
        print(f"❌ Gateway failed to initialize")
        print(f"   Status: {gateway.status.value}")
        return False
    
    print(f"✅ Gateway ready in {init_time:.2f}s")
    print(f"   Status: {gateway.status.value}")
    print()
    
    # Test 1: Simple completion
    print("[2/4] Simple completion test...")
    prompt = "What is 2 + 2? Answer with just the number."
    
    start = time.time()
    result = await gateway.infer(prompt, max_tokens=10)
    latency = time.time() - start
    
    print(f"✅ Response: {result.content.strip()}")
    print(f"   Latency: {latency:.2f}s")
    print(f"   Speed: {result.tokens_per_second} tok/s")
    print()
    
    # Test 2: BIZRA-specific knowledge
    print("[3/4] BIZRA context test...")
    prompt = """You are Maestro, an AI assistant for BIZRA - a sovereignty infrastructure project.
    
User: What does BIZRA stand for?

Maestro: Based on my knowledge, BIZRA"""
    
    start = time.time()
    result = await gateway.infer(prompt, max_tokens=100)
    latency = time.time() - start
    
    print(f"✅ Response: {result.content.strip()[:200]}...")
    print(f"   Latency: {latency:.2f}s")
    print(f"   Speed: {result.tokens_per_second} tok/s")
    print()
    
    # Test 3: Complexity routing
    print("[4/4] Complexity routing test...")
    
    test_cases = [
        ("Hi", "EDGE"),
        ("What is Python?", "EDGE/LOCAL"),
        ("Explain the mathematical proof of the halting problem.", "LOCAL/POOL"),
    ]
    
    for prompt, expected in test_cases:
        complexity = gateway.estimate_complexity(prompt)
        tier = gateway.route(complexity)
        print(f"   '{prompt[:40]}...' → {tier.value} (score: {complexity.score:.2f})")
    
    print()
    
    # Summary
    print("═" * 70)
    print("    INTEGRATION TEST COMPLETE")
    print("═" * 70)
    
    health = await gateway.health()
    print(f"   Status: {health['status']}")
    print(f"   Backend: {health['active_backend']}")
    print(f"   Model: {Path(health['active_model']).name if health['active_model'] else 'N/A'}")
    print(f"   Requests: {health['stats']['total_requests']}")
    print(f"   Avg Latency: {health['stats']['avg_latency_ms']:.0f}ms")
    print("═" * 70)
    
    return True


async def main():
    success = await test_inference_pipeline()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
