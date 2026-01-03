import time
import json
import os
from bizra_kernel.sovereign_engine import SovereignEngine

def run_genesis_emulation():
    print("\n" + "#"*60)
    print("      BIZRA GENESIS: END-TO-END MASTERPIECE EMULATION")
    print("#"*60 + "\n")
    
    engine = SovereignEngine()
    
    # MISSION: Evaluate System State & Perform a High-SNR Synthesis Task
    prompt = "Synthesize a strategy for BIZRA Universal Genesis using the discovered 12 local LLM nodes."
    
    # Mission Metrics (Signed via IhsanGate)
    mission_metrics = {
        "task_id": "GENESIS-PROOF-01",
        "truthfulness": 1.0, 
        "dignity": 0.99, 
        "fairness": 0.99, 
        "sustainability": 1.0
    }
    
    start_total = time.time()
    
    # 1. THE EXECUTION CYCLE
    # This automatically triggers Awareness Sync, Budgeting, Gate, Sog, and Node Evolution
    result = engine.execute_sovereign_task(
        prompt=prompt, 
        mission_metrics=mission_metrics,
        request_id="GENESIS-EMULATION-L3"
    )
    
    end_total = time.time()
    total_latency = (end_total - start_total) * 1000
    
    print("\n" + "-"*60)
    print(f"[*] EMULATION SUMMARY")
    print(f"[*] Status: {result.get('status')}")
    print(f"[*] Latency: {total_latency:.2f}ms")
    print(f"[*] SNR Boost: {result.get('snr')}")
    print(f"[*] Ledger Hash: {result.get('ledger_hash')}")
    
    # 2. OUTPUT SYSTEM STATE
    awareness = engine.awareness.synchronize_self_model()
    budget = awareness['budget']
    territory = awareness['territory']
    
    print(f"[*] Territory: {territory['total_nodes']} Nodes Active")
    print(f"[*] Capabilities: {territory['total_models']} Local Models Detected (The Giants)")
    print(f"[*] Cognitive Budget: {budget['budget_score']:.3f} ({budget['status']})")
    
    consolidation = result.get('consolidation', {})
    print(f"[*] Memory: {consolidation.get('status')} (Promoted: {consolidation.get('promoted', 0)})")
    
    print("\n" + "#"*60)
    print("      GENESIS EMULATION COMPLETE: OMISTATE SECURED")
    print("#"*60 + "\n")

if __name__ == "__main__":
    run_genesis_emulation()
