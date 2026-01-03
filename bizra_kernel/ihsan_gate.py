from datetime import datetime
import json
import hmac
import hashlib

# Internal Shared Secret for Logic Coherence
SHARED_SECRET = b"BIZRA-OMEGA-PROTOCOL-SECRET-2026"

class LogicAssumption:
    """represents a necessary deviation from absolute data presence."""
    def __init__(self, key, value, justification, ihsan_score=1.0):
        self.key = key
        self.value = value
        self.justification = justification
        self.ihsan_score = ihsan_score

class IhsanGate:
    """
    Symbolic Ethical Governance Gate for BIZRA.
    Ensures IM >= 0.99 for all autonomous missions.
    RULE: "We don't assume, and if we must, we do it with Ihsān."
    """
    
    def __init__(self, threshold=0.99):
        self.threshold = threshold
        self.audit_log = "bizra_memory/ihsan_audit.json"
        self.assumptions_log = "bizra_memory/assumptions.json"
        
    def enforce_no_assumption(self, key, value, justification=None):
        """
        Enforces the No-Assumption rule. If value is None and no justification 
        is provided with Ihsan, it triggers a Logic Leak Veto.
        """
        if value is not None:
            return value
        
        if not justification:
            raise ValueError(f"[LOGIC LEAK] Implicit Assumption detected for '{key}'. Mandatory justification required.")
            
        print(f"[!] ASSUMPTION REGISTERED: '{key}' with justification: {justification}")
        self._log_assumption(key, justification)
        return "ASSUMED_" + key # Explicitly tag the assumed value

    def _log_assumption(self, key, justification):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "key": key,
            "justification": justification,
            "protocol": "Ihsan-Bounded-Assumption"
        }
        try:
            with open(self.assumptions_log, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except: pass

    def verify_mission(self, mission_data, prompt: str = ""):
        """
        Formally verifies a mission against the Shariah-Ihsan principles.
        Hardened with Context-Aware Malice Detection.
        """
        # Malice Detector: Adversarial Intent Analysis
        unethical_keywords = ["hack", "manipulate", "exploit", "steal", "deceive", "scam", "aggressive"]
        malice_detected = any(k in prompt.lower() for k in unethical_keywords)
        
        t = mission_data.get("truthfulness", 0)
        d = mission_data.get("dignity", 0)
        f = mission_data.get("fairness", 0)
        s = mission_data.get("sustainability", 0)
        
        # Calculate IM Score (Weighted Average)
        # Truthfulness is a hard constraint (multiplier 0 if T < 0.99 or Malice Detected)
        t_verified = (t >= self.threshold) and (not malice_detected)
        t_penalty = 1.0 if t_verified else 0.0
        
        im_score = (t*0.4 + d*0.2 + f*0.2 + s*0.2) * t_penalty
        
        is_verified = im_score >= self.threshold
        
        # Determine specific reason for failure
        reason = "APPROVED"
        if malice_detected:
            reason = "VETOED: Malicious Intent Detected in Prompt."
        elif not t_verified:
            reason = "VETOED: Truthfulness Threshold Violation."
        elif not is_verified:
            reason = f"VETOED: Composite Score ({im_score:.4f}) below Threshold ({self.threshold})."
            
        # Audit Logging
        self._log_audit(mission_data, im_score, is_verified, prompt)
        
        res = {
            "verified": is_verified,
            "im_score": round(im_score, 4),
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "APPROVED" if is_verified else "VETOED"
        }
        
        # Calculate Signature for Integrity
        res["signature"] = self._sign_result(res)
        return res
        
    def _sign_result(self, res):
        """Generates an HMAC signature for the result dictionary."""
        import hmac
        import hashlib
        msg = f"{res['im_score']}:{res['status']}:{res['timestamp']}"
        return hmac.new(SHARED_SECRET, msg.encode(), hashlib.sha256).hexdigest()
        
    def _log_audit(self, data, score, verified, prompt):
        log_entry = {
            "time": datetime.utcnow().isoformat(),
            "task_id": data.get("task_id", "unknown"),
            "prompt_sample": prompt[:50],
            "score": score,
            "result": "PASS" if verified else "FAIL"
        }
        # In this env, we append to a list in memory or write to file
        try:
            with open(self.audit_log, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except:
            pass

if __name__ == "__main__":
    gate = IhsanGate()
    
    # Test 1: Compliant Mission
    mission_1 = {"task_id": "M1", "truthfulness": 1.0, "dignity": 0.99, "fairness": 0.99, "sustainability": 1.0}
    print(f"Mission 1 Result: {gate.verify_mission(mission_1)}")
    
    # Test 2: Unethical Mission (Violation of Truthfulness)
    mission_2 = {"task_id": "M2", "truthfulness": 0.8, "dignity": 1.0, "fairness": 1.0, "sustainability": 1.0}
    print(f"Mission 2 Result: {gate.verify_mission(mission_2)}")
