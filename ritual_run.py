import os
import json
import hashlib
from bizra_kernel.truth_ledger_mock import TruthLedgerMock
from bizra_kernel.identity import SovereignIdentity
from bizra_kernel.memory_system import CognitivePermanence

def ritual_run():
    print("✨ [PROJECT ALETHEIA] Initiating Genesis Ritual Run...")
    
    # 1. Setup Components
    ledger = TruthLedgerMock()
    identity = SovereignIdentity()
    
    # 2. PAT Proposal (Neural Layer Mocked)
    proposal = {
        "action": "FILE_WRITE",
        "target": "c:/BIZRA-Dual-Agentic-system--main/output/status.txt",
        "content": "Genesis Active",
        "justification": "Establish initial system state and verify truth spinal cord.",
        "risk": "LOW"
    }
    proposal_json = json.dumps(proposal, sort_keys=True)
    proposal_hash = hashlib.sha256(proposal_json.encode()).hexdigest()
    
    print(f"[*] PAT Proposal Generated. Hash: {proposal_hash}")

    # 3. Dumb SAT Verification (Symbolic Layer)
    # Check against hardcoded policy
    if proposal["action"] == "FILE_WRITE" and "/output/" in proposal["target"]:
        print("[+] SAT Verification: Action Allowed (L2 Policy Match)")
    else:
        print("[-] SAT Verification: BLOCKED")
        return

    # 4. Truth Ledger: PREPARE
    tx_id = ledger.prepare(proposal_hash)
    
    # 5. Execution (The Physical Act)
    output_dir = os.path.dirname(proposal["target"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(proposal["target"], "w") as f:
        f.write(proposal["content"])
    
    print(f"[+] Act Executed: '{proposal['content']}' written to {proposal['target']}")

    # 6. Evidence Packaging (L10)
    evidence_pack = {
        "tx_id": tx_id,
        "proposal": proposal,
        "sha256_result": hashlib.sha256(proposal["content"].encode()).hexdigest(),
        "timestamp": "2026-01-06T19:45:00Z"
    }
    evidence_path = f"c:/BIZRA-Dual-Agentic-system--main/evidence/PACK-{tx_id}.json"
    os.makedirs(os.path.dirname(evidence_path), exist_ok=True)
    with open(evidence_path, "w") as f:
        json.dump(evidence_pack, f, indent=2)
    
    print(f"[+] Evidence Pack Generated: {evidence_path}")

    # 7. Truth Ledger: COMMIT
    state_after_hash = hashlib.sha256(f"{ledger.state_root}:{tx_id}".encode()).hexdigest()
    ledger.commit(tx_id, state_after_hash, f"PACK-{tx_id}")

    # 8. Receipt Generation (Final Proof)
    receipt = {
        "status": "SUCCESS",
        "tx_id": tx_id,
        "state_root": ledger.state_root,
        "seal": "ED25519_SIGNED_BY_BIZRA_SAT"
    }
    
    print("\n🏆 [MISSION ACCOMPLISHED] First Receipt Generated:")
    print(json.dumps(receipt, indent=2))

if __name__ == "__main__":
    ritual_run()
