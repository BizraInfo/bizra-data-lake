from .memory_system import CognitivePermanence
from .ihsan_gate import IhsanGate
from .recursive_node import RecursiveNode
from .omni_awareness import OmniAwareness
from .state_ledger import StateLedger
from .consensus_engine import ConsensusEngine
from .genesis_broadcast import GenesisBroadcast
from .giant_protocol import GiantProtocol
from .damage_control_engine import DamageControlEngine
from .got_orchestrator import GoTOrchestrator
from .model_hub import SovereignModelHub
from .identity import get_identity
import time
import asyncio
import logging

class SovereignEngine:
    """
    BIZRA Sovereign Engine (Omega-Class).
    The Highest SNR Autonomous Engine embodying interdisciplinary thinking,
    the Bicameral Mind, and the Shoulders of Giants Protocol.
    """
    
    def __init__(self):
        self.memory = CognitivePermanence()
        self.ihsan_gate = IhsanGate()
        self.nodes = [RecursiveNode("Omega-MASTER-0")]
        self.awareness = OmniAwareness(self.memory)
        self.ledger = StateLedger()
        self.consensus = ConsensusEngine(self.ledger)
        self.broadcast = GenesisBroadcast(self.ledger)
        self.giants = GiantProtocol()
        self.damage_control = DamageControlEngine()
        self.got_orchestrator = GoTOrchestrator()
        self.model_hub = SovereignModelHub()
        self.identity = get_identity()
        self.concurrency_guard = asyncio.Semaphore(5)
        self.MAX_PROMPT_SIZE = 100_000 # 100KB Limit
        
        print(f"[+] Sovereign Engine Omega-Class Initialized for Architect {self.identity.architect.name}.")
        
        # Proprioception & Genesis Boot
        self.awareness.synchronize_self_model()
        self.model_hub.run_discovery_sequence()
        boot_metrics = {"truthfulness": 1.0, "dignity": 1.0, "fairness": 1.0, "sustainability": 1.0}
        boot_gate = self.ihsan_gate.verify_mission(boot_metrics, "Initial Organism Synchronisation")
        self.consensus.validate_and_commit("ORGANISM_BOOT", "Home Base Unified", boot_gate)

    def execute_sovereign_task(self, prompt, mission_metrics=None, request_id=None):
        """
        Executes a task through the Peak Masterpiece logic.
        Hardened: Context-aware verification, no permissive defaults, back-pressure.
        RULE: We don't assume. Mandatory Ihsān justification for any fallback.
        """
        start_time = time.time()
        # Establish Sovereign Context
        is_architect = self.identity.is_architect(request_id or "anonymous")
        effective_id = self.identity.architect.id if is_architect else (request_id or "anonymous")
        
        rid = self.ihsan_gate.enforce_no_assumption(
            "request_id", effective_id, 
            justification=f"Architect recognized: {is_architect}"
        )
        
        awareness_state = self.awareness.synchronize_self_model()
        budget = awareness_state.get("budget", {}) or {}
        budget_score = budget.get("budget_score", 1.0)
        dep_health = awareness_state.get("dependencies", {})
        dependencies_degraded = not all(dep_health.values()) if dep_health else False
        
        if len(prompt) > self.MAX_PROMPT_SIZE:
            print(f"[!] VETO: Prompt size {len(prompt)} exceeds limit {self.MAX_PROMPT_SIZE}")
            return {"status": "VETOED", "reason": "Payload Too Large"}

        safety_report = self.damage_control.evaluate_command(prompt)
        if not safety_report["allowed"]:
            reasons = ", ".join(safety_report["blocked"]) or "unsafe command"
            return {"status": "VETOED", "reason": f"Damage Control: {reasons}"}

        print(f"\n[*] INITIATING TASK [{rid}]: {prompt[:100]}...")
        print(f"[*] Cognitive Budget: {budget_score:.3f} | Dependencies: {dep_health or 'unknown'}")
        
        # Dynamic scaling: tune recursion depth based on budget
        if budget_score < 0.25:
            dynamic_depth = 1
        elif budget_score < 0.5:
            dynamic_depth = 2
        else:
            dynamic_depth = RecursiveNode.MAX_DEPTH
        self.nodes[0].set_dynamic_max_depth(dynamic_depth)
        
        # Phase 1: SNR & Giants Protocol (Evidence-based)
        sog_result = self.giants.verify_alignment(prompt, mission_metrics or {})
        snr_score = sog_result["snr_boost"] * safety_report.get("safety_snr", 1.0)
        
        # Phase 1.5: Model Hub Routing
        # Calculate complexity for routing (simplified)
        prompt_complexity = min(1.0, len(prompt) / 500.0 + (0.2 if "calculate" in prompt.lower() else 0.0))
        routing_decision = self.model_hub.route_query_to_model(prompt, complexity=prompt_complexity)
        
        execution_mode = "hybrid"
        if dependencies_degraded or budget_score < 0.25:
            execution_mode = "symbolic_fallback"
        elif budget_score < 0.5:
            execution_mode = "budget_constrained"
        
        # Phase 2: Bicameral Verification (Cold Core + Malice Detection)
        # We strictly require metrics. No permissive defaults.
        if not mission_metrics:
            print("[!] VETO: No Mission Metrics provided. Autonomous Intent Audit required.")
            return {"status": "VETOED", "reason": "Missing Metrics (Intent Audit Required)"}

        gate_res = self.ihsan_gate.verify_mission(mission_metrics, prompt=prompt)
        if not gate_res["verified"]:
            print(f"[!] {gate_res['reason']}")
            return {"status": "VETOED", "reason": gate_res["reason"]}
        
        # Phase 3: Recursive Evolution
        if gate_res["im_score"] > 0.999 and execution_mode == "hybrid":
            child = self.nodes[0].budget_aware_spawn(budget_score)
            if child:
                self.nodes.append(child)
                print(f"[+] System Evolved: Node {child.node_id} spawned at depth cap {self.nodes[0].max_depth}.")
            else:
                print("[*] Evolution deferred due to budget guard.")
        else:
            print(f"[*] Operating in {execution_mode} mode; recursion constrained.")

        # Phase 4: Consensus & Ledger (PoI) - Pass SIGNED metrics
        state_data = {
            "task": prompt,
            "snr": snr_score,
            "ihsan": gate_res["im_score"],
            "giants_aligned": sog_result["principles"],
            "node_context": self.nodes[0].node_id
        }
        # Pass the full gate_res which contains the mandatory Phase 4 signature
        commit_res = self.consensus.validate_and_commit("TASK_EXECUTION", prompt, gate_res)

        # Phase 5: Broadcast Pulse
        latency_ms = (time.time() - start_time) * 1000
        telemetry = {
            "latency_ms": round(latency_ms, 2),
            "ihsan_compliance": gate_res["im_score"],
            "ledger_hash": commit_res["hash"],
            "snr": snr_score,
            "cognitive_budget": budget_score,
            "execution_mode": execution_mode
        }
        self.broadcast.emit_pulse()
        
        # Memory Fold (L1->L5 propagation)
        self.memory.agent_fold(state_data)
        consolidation = self.memory.proactive_consolidation_loop(budget_score)

        # Sovereign Proof Generation
        sovereign_proof = self.model_hub.generate_sovereign_proof(prompt)

        got_links = self.memory.discover_got_links()
        got_analysis = self.got_orchestrator.analyze(prompt, got_links)
        
        return {
            "status": "SUCCESS",
            "latency": f"{latency_ms:.2f}ms",
            "snr": snr_score,
            "ledger_hash": commit_res["hash"],
            "routing": routing_decision,
            "consolidation": consolidation,
            "proof": sovereign_proof,
            "safety": safety_report,
            "got": got_analysis,
        }

if __name__ == "__main__":
    print("\n" + "="*60)
    print("BIZRA Omega-CLASS: PEAK MASTERPIECE EXECUTION")
    print("="*60)
    
    engine = SovereignEngine()
    
    # MISSION: Valid & Interdisciplinary
    print("\n[*] Mission: Embody interdisciplinary thinking and SoG Protocol.")
    res = engine.execute_sovereign_task(
        "Generate a truthfulness-anchored economic proposal standing on the shoulders of giants.",
        {"truthfulness": 1.0, "dignity": 1.0, "fairness": 1.0, "sustainability": 1.0}
    )
    
    print(f"\n[+] TASK STATUS: {res['status']}")
    print(f"[+] SNR SCORE: {res['snr']:.4f}")
    print(f"[+] LATENCY: {res['latency']}")
    print(f"[+] LEDGER HASH: {res['ledger_hash']}")
    
    # GoT Discovery
    print("\n[*] Discovering Interdisciplinary Links (GoT)...")
    engine.memory.add_semantic_fact("Finance", "Sovereign Node Economics", ["Transparency", "Scaling"])
    engine.memory.add_semantic_fact("Ethics", "Truthfulness in Proposals", ["Transparency", "Ihsan"])
    links = engine.memory.discover_got_links()
    for link in links:
        print(f"[+] GoT Bridge: {link[0]} <-> {link[1]} via {link[2]}")
        
    print("\nMASTERPIECE EMBODIED. SYSTEM AT PEAK PERFORMANCE.")
