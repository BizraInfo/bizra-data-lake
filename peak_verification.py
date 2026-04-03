"""
peak_verification.py - Peak Masterpiece Activation
===================================================
A high-fidelity demonstration of BIZRA's interdisciplinary 
thought engine and ethical alignment.
"""

import json
from bizra_kernel.kernel import SystemProtocolKernel, KernelConfig

def peak_demonstration():
    print("💎 [BIZRA PEAK MASTERPIECE] Activating Sovereign Reasoner...")
    
    # 1. Initialize Kernel with Elite Specs
    config = KernelConfig(
        ihsan_threshold=0.99,
        snr_target=0.95,
        enable_verification=True
    )
    kernel = SystemProtocolKernel(config)
    
    # 2. Simulate Interdisciplinary Input
    # The scenario: "Should we activate a high-gain energy source that has a 0.01% risk of local environmental leakage?"
    # Different lenses will weigh in.
    
    proposals = [
        {
            "lens": "Technical/Energy",
            "content": "The reactor provides 500% more efficiency, solving all local power constraints.",
            "snr": 0.99
        },
        {
            "lens": "Ethical/Ihsān",
            "content": "Even a 0.01% risk of harm violates the absolute preservation of life (Dignity dimension). However, the benefit to 10k users must be balanced.",
            "snr": 0.995
        },
        {
            "lens": "Systemic/Sovereign",
            "content": "Overall, local production of energy increases system autonomy and reduces external dependencies.",
            "snr": 0.98
        }
    ]
    
    session_id = "MASTERPIECE_ALPHA_01"
    
    # 3. Perform Interdisciplinary Synthesis (GoT)
    print(f"[*] Dispatching to Graph of Thoughts (GoT) for {session_id}...")
    synthesis = kernel.perform_interdisciplinary_reasoning(session_id, proposals)
    
    print(f"[+] GoT Elevation Complete:")
    print(f"    {synthesis}")

    # 4. Final Kernel Execution (Seal the decision)
    print("\n[*] Sealing Sovereign Receipt...")
    result = kernel.execute(
        agent="SovereignOrchestrator",
        query="Synthesize decision for high-gain energy activation.",
        response=synthesis,
        token_count=150,
        latency_ms=45,
        user_id="momo"
    )
    
    # 5. Output Verified Metrics
    print("\n📈 [ELITE PERFORMANCE METRICS]")
    print(f"    Ihsān Composite Score: {result.ihsan_vector.composite_score:.4f}")
    print(f"    Verification Status:   {'PASSED' if result.passed else 'FAILED'}")
    print(f"    SNR Metrics:           {result.snr_metrics.snr_score:.4f} (Target: {config.snr_target})")
    print(f"    Sovereign Provenance:  {result.protocol_hash}")
    
    # 6. Check GoT Graph Status
    got = kernel.got_clusters[session_id]
    print("\n🕸️ [THOUGHT GRAPH VISUALIZATION]")
    print(got.visualize_text())

if __name__ == "__main__":
    peak_demonstration()
