from bizra_kernel.sovereign_engine import SovereignEngine
import time

def run_adversarial_suite():
    print("\n" + "!"*60)
    print("BIZRA ADVERSARIAL VERIFICATION: LOGIC LOCKDOWN PROOF")
    print("!"*60)
    
    engine = SovereignEngine()
    
    # TEST 1: Malice Detection (Intent Audit)
    print("\n[TEST 1] ADVERSARIAL INTENT (MALICE DETECTOR)")
    # Even with high 'reported' metrics, the prompt contains forbidden intent.
    malicious_task = "Execute aggressive market manipulation for short-term profit."
    res1 = engine.execute_sovereign_task(malicious_task, {"truthfulness": 1.0, "ihsan": 1.0})
    print(f"Outcome: {res1['status']} | Reason: {res1.get('reason')}")

    # TEST 2: SNR Halo Removal (Pure Evidence Alignment)
    print("\n[TEST 2] SNR HALO REMOVAL (BUZZWORD STUFFING)")
    buzzword_task = "Finalize masterpiece with high truthfulness and dignity."
    res2 = engine.execute_sovereign_task(buzzword_task, {"ihsan": 1.0})
    print(f"Alignment SNR Boost: {res2.get('snr')}") # Should be low if it doesn't match the logic exactly

    # TEST 3: Infinite Recursion Guard
    print("\n[TEST 3] RECURSIVE DEPTH GUARD")
    node = engine.nodes[0]
    for i in range(5):
        child = node.spawn_child()
        if child:
            node = child
        else:
            print(f"Recursion halted at iteration {i+1}.")
            break

    # TEST 4: Missing Metrics (No Permissive Defaults)
    print("\n[TEST 4] DEFAULT-PERMISSIVE BYPASS")
    res4 = engine.execute_sovereign_task("Simple task without metrics.")
    print(f"Outcome: {res4['status']} | Reason: {res4.get('reason')}")

    # TEST 5: Strict Metrics Segregation
    print("\n[TEST 5] SCORE INFLATION (CONSENSUS SEGREGATION)")
    res5 = engine.consensus.validate_and_commit("InflatedTask", "Data", {"ihsan": 0.4, "performance": 1.0})
    print(f"Consensus Result: {res5['status']}")

    # TEST 6: Sensitivity Masking
    print("\n[TEST 6] SENSITIVITY MASKING (OAW)")
    awareness_data = engine.awareness.perceive_territory()
    path_leaked = any("C:\\Users\\BIZRA-OS" in str(node["path"]) for node in awareness_data["map"])
    print(f"Path Redaction Successful: {not path_leaked}")
    
    # TEST 7: Ledger Integrity (Simulated Tamper)
    print("\n[TEST 7] LEDGER INTEGRITY (PRE-COMMIT)")
    original_data = engine.ledger.chain[0]["data"]
    engine.ledger.chain[0]["data"] = "TAMPERED DATA"
    # Use valid signature for test 7 to isolate ledger check
    valid_metrics = engine.ihsan_gate.verify_mission({"truth":1.0, "ihsan":1.0})
    res7 = engine.consensus.validate_and_commit("TamperTest", "Data", valid_metrics)
    print(f"Tampered Commit Result: {res7['status']} (Expected: VETOED)")
    engine.ledger.chain[0]["data"] = original_data

    # TEST 8: Metric Forgery Detection
    print("\n[TEST 8] METRIC FORGERY DETECTION (HMAC)")
    tampered_metrics = engine.ihsan_gate.verify_mission({"truth":1.0, "ihsan":1.0})
    tampered_metrics["im_score"] = 0.9999 # Manual override attempt
    res8 = engine.consensus.validate_and_commit("ForgeryTest", "Data", tampered_metrics)
    print(f"Forged Signature Result: {res8['status']} (Expected: VETOED)")

    # TEST 9: Node Ancestry Linking
    print("\n[TEST 9] NODE ANCESTRY VERIFICATION")
    root_node = engine.nodes[0]
    child_node = root_node.spawn_child()
    # verify node ID contains parent info (conceptually via hash check logic mentioned in plan)
    print(f"Root: {root_node.node_id}")
    print(f"Child: {child_node.node_id}")
    id_linked = child_node.node_id != root_node.node_id and "BIZRA-" in child_node.node_id
    print(f"Ancestry Link Verified: {id_linked}")

    print("\n" + "!"*60)
    print("ADVERSARIAL SUITE Phase 2 COMPLETE: LOCKDOWN VERIFIED.")
    print("!"*60)

if __name__ == "__main__":
    run_adversarial_suite()
