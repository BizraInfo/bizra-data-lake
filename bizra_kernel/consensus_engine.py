import hmac
import hashlib
from typing import Dict, Any
try:
    from .state_ledger import StateLedger
except ImportError:
    from .state_ledger import StateLedger

# Internal Shared Secret (Must match IhsanGate)
SHARED_SECRET = b"BIZRA-OMEGA-PROTOCOL-SECRET-2026"

class ConsensusEngine:
    """
    BIZRA Consensus Engine — Proof-of-Impact (PoI) Protocol.
    Ensures that only Ihsan-compliant states are persisted.
    """
    
    def __init__(self, ledger: StateLedger):
        self.ledger = ledger
        
    def validate_and_commit(self, action_name: str, action_data: Any, metrics: Any):
        """
        The PoI Validation logic.
        Hardened: Pre-commitment Integrity Audit.
        """
        print(f"[*] Validating Impact for: {action_name}...")
        
        # Pre-commitment Audit: Verify the Chain of Truth is intact
        if not self.ledger.verify_integrity():
            print("[!] CRITICAL: State Ledger Integrity compromised! Commitment VETOED.")
            return {"status": "VETOED", "reason": "Ledger Integrity Failure"}

        # Phase 4: Signature Verification (Anti-Forgery)
        if not isinstance(metrics, dict) or "signature" not in metrics:
            print("[!] VETOED: Unsigned mission metrics. Cryptographic Integrity Missing.")
            return {"status": "VETOED", "reason": "Unsigned Metrics"}
            
        # Re-calculate and verify signature
        # Use locally imported hashlib to avoid any name issues
        import hashlib as local_hl
        import hmac as local_hmac
        
        msg = f"{metrics.get('im_score', metrics.get('ihsan'))}:{metrics.get('status', 'APPROVED')}:{metrics['timestamp']}"
        expected_sig = local_hmac.new(SHARED_SECRET, msg.encode(), local_hl.sha256).hexdigest()
        
        if not local_hmac.compare_digest(metrics["signature"], expected_sig):
            print("[!] VETOED: Metric Signature Mismatch! Tampering Detected.")
            return {"status": "VETOED", "reason": "Signature Mismatch"}

        ihsan_score = metrics.get("im_score", metrics.get("ihsan", 0.0))
        
        if ihsan_score >= 0.99:
            print(f"[+] Proof-of-Impact Verified: Ihsan {ihsan_score:.4f}. Committing to Ledger.")
            proof_hash = self.ledger.append_state(action_name, {
                "data": action_data,
                "ihsan_score": ihsan_score,
                "metrics": metrics
            })
            return {"status": "COMMITTED", "hash": proof_hash, "score": ihsan_score}
        else:
            print(f"[-] Impact Vetoed: Ihsan {ihsan_score:.4f} below threshold.")
            return {"status": "VETOED", "score": ihsan_score}

if __name__ == "__main__":
    ledger = StateLedger()
    engine = ConsensusEngine(ledger)
    
    # Test Success (Needs signature in Phase 4)
    # We don't run tests here in module mode usually.
    print("[+] Consensus Engine Unit Test Logic Ready.")
