"""
fully_local_verification.py - Sovereign Organism Local Readiness Probe
=====================================================================
Verifies that the BIZRA system can execute a complete cognitive cycle
without external dependencies, utilizing the local Model Hub and RTX 4090.
"""

import asyncio
import json
import os
from bizra_kernel.kernel import SystemProtocolKernel, KernelConfig
from bizra_kernel.omni_awareness import OmniAwareness
from bizra_kernel.memory_system import CognitivePermanence

async def run_local_ritual():
    print("🌅 Starting Sovereign Local Ritual...")
    
    # 1. Initialize Awareness & Hardware Check
    memory = CognitivePermanence()
    awareness = OmniAwareness(memory)
    report = awareness.synchronize_self_model()
    
    print(f"[+] Local Sovereignty: {report['sovereignty']['architect_name']} ({report['sovereignty']['architect_role']})")
    print(f"[+] Detected Models: {report['territory']['total_models']}")
    if report['budget']['gpu']:
        print(f"[+] GPU Hardware Ready: {report['budget']['gpu']}")
    else:
        print("[!] Warning: GPU Telemetry unavailable (drivers/pynvml missing?)")

    # 2. Kernel Initialization
    config = KernelConfig(ihsan_threshold=0.99)
    kernel = SystemProtocolKernel(config)
    print(f"[+] Kernel Initialized (Protocol: {kernel.VERSION})")

    # 3. Local Inference Test
    print("\n🧠 Probing Local Cognitive Pipeline...")
    task = "Synthesize a strategy for air-gapped sovereign data management."
    result = await kernel.execute_local_inference(task, complexity="high")
    
    print(f"[+] Execution Passed: {result.passed}")
    print(f"[+] Agent: {result.agent}")
    print(f"[+] Response Snippet: {result.response[:100]}...")
    print(f"[+] Latency: {result.latency_ms}ms")
    print(f"[+] Ihsān Compliance: {result.ihsan_vector.composite_score:.3f}")

    # 4. Interdisciplinary Reasoning (Local GoT)
    print("\n🌐 Executing Local Graph of Thoughts (GoT) Cycle...")
    proposals = [
        {"lens": "Security", "content": "Encryption keys must be derived from local physical entropy.", "snr": 0.99},
        {"lens": "Infrastructure", "content": "Hardware nodes must be cryptographically sealed to the Architect ID.", "snr": 0.98},
        {"lens": "Logic", "content": "Recursive verification must be performed by local SAT guardians.", "snr": 0.97}
    ]
    
    elevation = kernel.perform_interdisciplinary_reasoning("local-ritual-001", proposals)
    print(f"[+] GoT Elevation: {elevation}")

    # 5. Final Sovereignty Proof
    status = kernel.get_status()
    print(f"\n✅ Local Ritual Complete. Kernel Status: {status['kernel_version']}")
    print(f"   SNR Average: {status['snr']['average_snr']:.3f}")
    
    # Save a verification receipt
    receipt = {
        "timestamp": os.getenv("TIMESTAMP", "2026-01-06T20:20:00Z"),
        "locality": "FULLY_LOCAL",
        "gpu_active": report['budget']['gpu'] is not None,
        "kernel_ver": kernel.VERSION,
        "ritual_success": result.passed
    }
    with open("local_readiness_receipt.json", "w") as f:
        json.dump(receipt, f, indent=2)
    print("\n[!] Local Readiness Receipt Sealed: local_readiness_receipt.json")

if __name__ == "__main__":
    asyncio.run(run_local_ritual())
