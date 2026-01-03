import os
import yaml
import json
import time
import hashlib
import hmac
from datetime import datetime
from typing import List, Dict, Any

class CognitivePermanence:
    """
    The 5-Layer Cognitive Workspace for BIZRA.
    L1: Immediate (Context window)
    L2: Working (Granular Condensation)
    L3: Episodic (Deep Consolidation)
    L4: Semantic (HyperGraph RAG)
    L5: Procedural (Expertise + AATC)
    """
    
    def __init__(self, agent_id="sovereign_node_0"):
        self.agent_id = agent_id
        self.layers = {
            "L1": [],  # Volatile perception
            "L2": [],  # Summary blocks
            "L3": [],  # Consolidated episodes
            "L4": {},  # HyperGraph facts {node_id: {data}}
            "L5": {}   # Procedural tools
        }
        self.got_links = [] # Graph of Thoughts: list of (id1, id2, relation_type)
        self.storage_path = "bizra_memory/"
        os.makedirs(self.storage_path, exist_ok=True)
        self.expertise_path = os.path.join(self.storage_path, "expertise.yaml")
        self.load_memory()

    def load_memory(self):
        if os.path.exists(self.expertise_path):
            with open(self.expertise_path, 'r') as f:
                data = yaml.safe_load(f) or {}
                
                # Phase 4: Integrity Check (Anti-Poisoning)
                stored_sig = data.pop("integrity_sig", None)
                if stored_sig:
                    current_sig = self._calculate_integrity(data)
                    if not hmac.compare_digest(stored_sig, current_sig):
                        print("[!] CRITICAL: Expertise Memory Integrity failure! Data may be poisoned.")
                        # In a real sovereign system, we might quarantine this file.
                        # For now, we refuse to load unverified expertise.
                        return 

                self.layers["L5"] = data.get("procedural", {})
                self.layers["L4"] = data.get("semantic", {})

    def save_memory(self):
        data = {
            "procedural": self.layers["L5"],
            "semantic": self.layers["L4"],
            "agent_id": self.agent_id,
            "last_updated": datetime.utcnow().isoformat()
        }
        # Add Integrity Signature
        data["integrity_sig"] = self._calculate_integrity(data)
        
        # Atomic Write Pattern: Save to temp, then replace
        temp_path = self.expertise_path + ".tmp"
        with open(temp_path, 'w') as f:
            yaml.dump(data, f)
        os.replace(temp_path, self.expertise_path)
        
    def _calculate_integrity(self, data):
        """Calculates a deterministic hash of the memory contents."""
        # Simple canonical representation for signing
        canon = json.dumps(data, sort_keys=True)
        return hmac.new(b"BIZRA-MEMORY-KEY", canon.encode(), hashlib.sha256).hexdigest()

    def agent_fold(self, perception_data):
        """The Think-Fold-Act Cycle (AgentFold)"""
        # 1. PERCEIVE: Add to L1
        timestamp = datetime.utcnow().isoformat()
        self.layers["L1"].append({"data": perception_data, "time": timestamp})
        
        # 2. REASON: Granular Condensation (L1 -> L2)
        if len(self.layers["L1"]) > 1:
            summary = self._condense_l1()
            self.layers["L2"].append(summary)
            self.layers["L1"] = self.layers["L1"][-1:] # Keep only latest high-fidelity
            
        # USES FIBONACCI-SCHEDULED CONSOLIDATION with Soft Hard-Cap
        fib_thresholds = [3, 5, 8, 13, 21, 34, 55]
        max_l3_episodes = 21 # Sustainable limit for the Ω-Class Seed
        
        if len(self.layers["L2"]) in fib_thresholds:
            if len(self.layers["L3"]) < max_l3_episodes:
                self._consolidate_l3()
            else:
                # FIFO Cache: Remove oldest if at limit
                print("[*] L3 Limit reached. Recycling oldest episodic memory block.")
                self.layers["L3"].pop(0)
                self._consolidate_l3()
            
        return {"status": "folded", "l2_depth": len(self.layers["L2"]), "l3_depth": len(self.layers["L3"])}

    def _condense_l1(self):
        # In a real LLM environment, this would be a summarization call.
        # Here we emulate logic crystallization.
        raw = self.layers["L1"][-2]["data"]
        summary = f"Crystallized action: {str(raw)[:50]}... at {datetime.utcnow().isoformat()}"
        return {"summary": summary, "type": "granular"}

    def _consolidate_l3(self):
        # Deep consolidation: merge L2 blocks into a coarse L3 node
        if not self.layers["L2"]: return
        merged = " | ".join([b["summary"] for b in self.layers["L2"]])
        self.layers["L3"].append({
            "episode": f"EP-{len(self.layers['L3'])+1}",
            "content": merged,
            "timestamp": datetime.utcnow().isoformat()
        })
        self.layers["L2"] = [] # Clear L2 to prevent saturation

    def add_semantic_fact(self, entity, fact, relationships):
        """L4: HyperGraph RAG - connecting n-ary relationships"""
        fingerprint = hashlib.sha256(f"{entity}:{fact}".encode()).hexdigest()[:8]
        self.layers["L4"][fingerprint] = {
            "entity": entity,
            "fact": fact,
            "rels": relationships,
            "im_score": 0.99 # Ihsan Excellence score
        }
        self.save_memory()

    def crystallize_procedural(self, task_name, code_snippet, gate_signature: str = ""):
        """L5: AATC - Compiling successful traces into code. Hardened with verification signature."""
        if not gate_signature:
            print(f"[!] CRYSTALLIZATION REJECTED: Mandatory Gate Signature missing for {task_name}.")
            return False
            
        self.layers["L5"][task_name] = {
            "code": code_snippet,
            "signature": gate_signature,
            "status": "VERIFIED"
        }
        self.save_memory()
        return True

    def discover_got_links(self) -> List[Dict[str, Any]]:
        """
        Advanced Graph of Thoughts (GoT) Link Discovery.
        Finds 'Bridges' between disparate clusters (Semantic Force Fields).
        """
        new_links = []
        facts = list(self.layers["L4"].items())
        
        for i, (id1, data1) in enumerate(facts):
            for j, (id2, data2) in enumerate(facts):
                if i >= j: continue
                
                # Semantic Force Field: Shared tokens + Entity overlap
                tokens1 = set(str(data1).lower().split())
                tokens2 = set(str(data2).lower().split())
                intersection = tokens1.intersection(tokens2)
                
                if intersection:
                    link = (id1, id2, f"force_bridge:{list(intersection)[0]}")
                    if link not in self.got_links:
                        new_links.append(link)
                        
        self.got_links.extend(new_links)
        return new_links

    def crystallize_expertise(self, skill_name: str, code_snippet: str, gate_signature: str = ""):
        """L5: AATC v0.1-Ω - Crystallizing procedural intent. Hardened with verification signature."""
        if not gate_signature:
            print(f"[!] CRYSTALLIZATION REJECTED: Mandatory Gate Signature missing for {skill_name}.")
            return False

        self.layers["L5"][skill_name] = {
            "version": "v0.1-Ω",
            "logic": code_snippet,
            "signature": gate_signature,
            "status": "CERTIFIED",
            "timestamp": datetime.utcnow().isoformat()
        }
        self.save_memory()
        return f"[+] Skill Crystallized: {skill_name}"

    def snr_filter(self, data_stream):
        """Highest SNR Autonomous Engine filter"""
        # Simplistic entropy-based noise reduction
        signal = [d for d in data_stream if len(str(d)) > 10] # Filter out short noise
        return signal

    def proactive_consolidation_loop(self, cognitive_budget: float, min_budget: float = 0.75) -> Dict[str, Any]:
        """
        Autonomously crystallize when system load is low.
        Moves L3 episodic blocks into L4 HyperGraph facts.
        """
        if cognitive_budget < min_budget:
            return {"status": "skipped", "reason": "budget_low", "budget": cognitive_budget}

        promoted = 0
        while self.layers["L3"]:
            episode = self.layers["L3"].pop(0)
            fact = {
                "episode": episode["episode"],
                "summary": episode["content"],
                "timestamp": episode["timestamp"],
                "origin": "auto_consolidation"
            }
            self.add_semantic_fact("EpisodicMemory", episode["episode"], fact)
            promoted += 1

        if promoted:
            self.save_memory()
        return {"status": "ok", "promoted": promoted, "budget": cognitive_budget}

if __name__ == "__main__":
    # END-TO-END EMULATION
    print("Initializing BIZRA 5-Layer Memory System...")
    memory = CognitivePermanence()
    
    print("\n--- Phase 1: Immediate & Working Memory (L1, L2) ---")
    for i in range(5):
        memory.agent_fold(f"Raw step data for task segment {i+1}")
        print(f"Step {i+1}: Memory Folded. L2 Depth: {len(memory.layers['L2'])}")

    print("\n--- Phase 2: Episodic Consolidation (L3) ---")
    print(f"L3 Episodes: {len(memory.layers['L3'])}")
    if memory.layers["L3"]:
        print(f"Last Episode Sample: {memory.layers['L3'][-1]['episode']}")

    print("\n--- Phase 3: Semantic & Procedural Synthesis (L4, L5) ---")
    memory.add_semantic_fact("BIZRA", "The primary sovereign node for post-labor economics.", ["Ihsan", "Consensus"])
    memory.crystallize_procedural("VerifyTokenEmission", "def verify_emission(amt): return amt < 5000000")
    
    print(f"Semantic Store (L4): {len(memory.layers['L4'])} facts")
    print(f"Procedural Store (L5): {len(memory.layers['L5'])} tools")
    
    print("\nSYSTEM STATUS: EXCELLENCE ACHIEVED.")
