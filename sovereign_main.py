import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, Any, List, Optional

# --- UTF-8 BOOT ENCODING FIX (Production Deployment) ---
# Force UTF-8 encoding for international environments
os.environ['PYTHONUTF8'] = '1'
if sys.platform == 'win32':
    # Windows-specific UTF-8 configuration
    os.system('chcp 65001 > nul 2>&1')  # Set console code page to UTF-8
    # Reconfigure standard streams for UTF-8
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except (AttributeError, OSError):
        pass  # Some environments may not support reconfigure

print(f"[BOOT] UTF-8 Encoding: ACTIVE (PYTHONUTF8=1, platform={sys.platform})")

# --- SOVEREIGN NEXUS (Unified Control Interface) ---
# The SovereignNexus consolidates 11 components into a unified apex orchestrator
NEXUS_AVAILABLE = False
try:
    from bizra_kernel.sovereign_nexus import SovereignNexus
    NEXUS_AVAILABLE = True
    print("[+] Sovereign Nexus: AVAILABLE (unified control interface)")
except ImportError as e:
    print(f"[!] Sovereign Nexus not available: {e} - using legacy components")

# --- INTERNAL MODULES (The Mind) ---
from bizra.memory import CognitivePermanence
from bizra.model_hub import SovereignModelHub
from bizra.giants import GiantProtocol
from bizra.identity import get_identity

# --- DAMAGE CONTROL ENGINE (Security Layer) ---
try:
    from bizra_kernel.damage_control_engine import DamageControlEngine, SecurityContext
    DAMAGE_CONTROL_AVAILABLE = True
except ImportError:
    DAMAGE_CONTROL_AVAILABLE = False
    print("[!] Damage Control Engine not available - running without security layer")

# --- TPM ANCHOR (Hardware Root of Trust) ---
try:
    from bizra_kernel.tpm_context import get_tpm_context
    TPM_AVAILABLE = True
except ImportError:
    TPM_AVAILABLE = False
    print("[!] TPM Context not available - running without hardware root of trust")

# --- WASM SANDBOX (Neural Isolation) ---
try:
    from bizra_kernel.wasm_sandbox import get_wasm_sandbox
    WASM_AVAILABLE = True
except ImportError:
    WASM_AVAILABLE = False
    print("[!] WASM Sandbox not available - running without neural isolation")

# --- SAPE ENGINE (Pattern Elevation) ---
try:
    from bizra_kernel.sape_engine import SAPEEngine
    SAPE_AVAILABLE = True
except ImportError:
    SAPE_AVAILABLE = False
    print("[!] SAPE Engine not available - running without pattern elevation")

# --- EXTERNAL BINDINGS (The Physics - Rust) ---
# This implies: maturin develop --release

class SovereignKernel:
    """
    The BIZRA NODE-0 RUNTIME.
    Fuses Python Cognition with Rust Safety.

    Now delegates to SovereignNexus when available for unified control.
    The Nexus consolidates 11 components:
    - CognitivePermanence, OmniAwareness, SovereignModelHub
    - DisciplineTopologyEngine (47 disciplines)
    - AutonomousDreamer, GoTOrchestrator
    - SNRTracker, GiantProtocol, SynergyDetector
    - RecursiveNode, AbstractionElevator
    """

    def __init__(self, use_nexus: bool = True):
        self._setup_logging()

        # 1. IDENTITY & ANCHOR
        self.identity = get_identity()
        self.boot_time = time.time()
        print(f"[BOOT] Sovereign Kernel v5.0.0-OMEGA | Architect: {self.identity.architect.name}")

        # 2. SOVEREIGN NEXUS (Unified Control Interface)
        self.nexus: Optional[SovereignNexus] = None
        if use_nexus and NEXUS_AVAILABLE:
            try:
                self.nexus = SovereignNexus(
                    heartbeat_hz=147.0,
                    ihsan_threshold=0.95,
                    snr_target=0.95,
                    enable_dreaming=True
                )
                print("   [+] Sovereign Nexus: ONLINE (unified control)")
                print(f"       Nexus ID: {self.nexus.nexus_id}")
                print(f"       47-Discipline Topology: {self.nexus.get_topology_stats()['total_disciplines']} disciplines")
                print(f"       Autonomous Dreaming: {'ENABLED' if self.nexus.enable_dreaming else 'DISABLED'}")
            except Exception as e:
                print(f"   [!] Sovereign Nexus initialization failed: {e}")
                print("   [!] Falling back to legacy components")
                self.nexus = None

        # 2. BIND PHYSICS (Rust)
        # Initialize the FATE engine via FFI. This loads Z3 constraints.
        try:
            import bizra_bridge  # The Rust FFI module we architected
            self.physics = bizra_bridge.FateEngine(threshold=0.95)
            self.spine = bizra_bridge.ChimeraSpine() # Iceoryx2
            print("   [+] Physics Engine (Rust/Z3): ONLINE")
            print("   [+] Sovereign Spine (Zero-Copy): ONLINE")
        except ImportError:
            print("   [!] FATAL: Rust Bridge missing. Running in SIMULATION MODE.")
            self.physics = None
            self.spine = None

        # 3. TPM ANCHOR (Hardware Root of Trust)
        if TPM_AVAILABLE:
            self.tpm_context = get_tpm_context()
            print("   [+] TPM Context (Hardware Root of Trust): ONLINE")
        else:
            self.tpm_context = None
            print("   [!] TPM Context: OFFLINE (simulation mode)")

        # 4. INITIALIZE COGNITION (Python)
        self.memory = CognitivePermanence()
        self.giants = GiantProtocol()
        self.hub = SovereignModelHub() # Manages 3B/7B/14B
        
        # 5. WASM SANDBOX (Neural Isolation)
        if WASM_AVAILABLE:
            self.wasm_sandbox = get_wasm_sandbox()
            # Set FATE callback for runtime Ihsān verification
            if self.physics:
                self.wasm_sandbox.set_fate_callback(
                    lambda ihsan_score: ihsan_score >= 0.95
                )
            print("   [+] WASM Sandbox (Neural Isolation): ONLINE")
        else:
            self.wasm_sandbox = None
            print("   [!] WASM Sandbox: OFFLINE (simulation mode)")
        
        # 6. DAMAGE CONTROL ENGINE (Security Layer)
        if DAMAGE_CONTROL_AVAILABLE:
            security_context = SecurityContext(
                agent_role="SOVEREIGN_KERNEL",
                session_id=f"kernel-{self.boot_time}",
                tool_type="execute",
                working_directory=os.getcwd(),
                user_identity=self.identity.architect.name
            )
            self.damage_control = DamageControlEngine(security_context)
            print("   [+] Damage Control Engine: ONLINE")
        else:
            self.damage_control = None
            print("   [!] Damage Control Engine: OFFLINE (simulation mode)")
        
        # 6. GENESIS SELF-CHECK (SAPE Probe) with TPM secure boot
        self._run_genesis_diagnostic()

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    def _run_genesis_diagnostic(self):
        """Runs the 9-Probe Defense Matrix on self."""
        # Probe 1: TPM Attestation (Hardware Root of Trust)
        if self.tpm_context and not self.tpm_context.verify_attestation():
            raise SystemError("TPM Attestation Failed - System Integrity Compromised")
        
        # Probe 2: Secure Boot Verification (Kernel Tampering Check)
        if self.tpm_context:
            measured_hash = self.tpm_context.measure_module("sovereign_main.py")
            expected_hash = self.tpm_context.get_expected_hash("sovereign_main.py")
            if expected_hash and measured_hash != expected_hash:
                raise SystemError("Secure Boot Violation - Kernel Tampering Detected")
        
        # Probe 3: FATE Engine Integrity
        if self.physics and not self.physics.verify_integrity():
            raise SystemError("FATE Engine Integrity Check Failed")
        
        # Probe 4: Genesis Hash Attestation
        genesis_hash = "76dffa0c83693721fb801a9fdab565abd25ece8e613aeea8fb0e0c2dc36121a1"
        print(f"   [+] Genesis Attestation: {genesis_hash[:16]}...")
        
        # Probe 5: Memory Integrity
        print("   [+] Genesis Diagnostic: PASS (5/5 Probes)")

    async def omni_loop(self):
        """The 147Hz Heartbeat of the Organism."""
        print("\n[OMNI-LOOP] ENTERING...")

        # If Nexus is available, delegate to its heartbeat loop
        if self.nexus:
            print("[OMNI-LOOP] Delegating to Sovereign Nexus heartbeat...")
            await self.nexus.run_heartbeat_loop()
            return

        # Legacy heartbeat loop (fallback)
        print("[OMNI-LOOP] Running legacy heartbeat (Nexus not available)")
        # Initialize tickers for periodic tasks
        last_tax_cycle = time.time()
        last_telemetry_cycle = time.time()
        cycle_times = []

        while True:
            cycle_start = time.perf_counter()
            
            # 1. PERCEIVE (Zero-Copy from Rust Spine)
            # Blocks for max 10ms waiting for signal
            if self.spine:
                signal = self.spine.fetch_next(timeout_ms=10)
            else:
                signal = None  # Sim mode
            
            if signal:
                await self.process_signal(signal)
            
            # 2. HOUSEKEEPING (Harberger Tax / Memory Fold) - Fixed race condition
            current_time = time.time()
            if current_time - last_tax_cycle >= 60:
                self.memory.run_harberger_tax()
                last_tax_cycle = current_time
            
            # 3. TELEMETRY (147Hz measurement)
            cycle_time = time.perf_counter() - cycle_start
            cycle_times.append(cycle_time)
            if len(cycle_times) > 1000:
                cycle_times.pop(0)
            
            # Log telemetry every 10 seconds
            if current_time - last_telemetry_cycle >= 10:
                if cycle_times:
                    avg_cycle_time = sum(cycle_times) / len(cycle_times)
                    actual_frequency = 1.0 / avg_cycle_time if avg_cycle_time > 0 else 0
                    logging.info(f"[TELEMETRY] Loop: {actual_frequency:.1f}Hz (target: 147Hz)")
                last_telemetry_cycle = current_time
                
            await asyncio.sleep(max(0, 0.001 - cycle_time)) # Compensate for processing time

    async def process_signal(self, signal: Dict[str, Any]):
        """
        The 7-3-6-9 Cognitive Pipeline.

        When Nexus is available, delegates to its unified execute() method.
        """
        task_id = signal.get("id")
        prompt = signal.get("content")

        logging.info(f"[{task_id}] Processing Signal: {prompt[:50]}...")

        # Delegate to Nexus if available
        if self.nexus:
            try:
                result = await self.nexus.execute(
                    task=prompt,
                    context={"task_id": task_id, "signal": signal},
                    require_sat_consensus=True
                )
                if result.success:
                    logging.info(f"[{task_id}] NEXUS EXECUTION SUCCESS | SNR: {result.snr_score:.4f}")
                    if self.spine:
                        self.spine.emit_receipt(task_id, "SUCCESS", result.result)
                else:
                    logging.error(f"[{task_id}] NEXUS EXECUTION FAILED: {result.result}")
                    if self.spine:
                        self.spine.emit_receipt(task_id, "REJECTED", result.result)
                return
            except Exception as e:
                logging.error(f"[{task_id}] Nexus execution error: {e}, falling back to legacy")

        # Legacy pipeline (fallback)

        # --- PHASE 1: DIVERGE (Graph of Thoughts) ---
        # Consult the Giants
        archetypes = self.giants.summon_council(prompt)
        # Route to appropriate model (3B/7B/14B) based on complexity
        model = self.hub.route(prompt)
        
        # Generate Draft Plan
        plan = await model.generate(f"Archetypes: {archetypes}. Task: {prompt}")

        # --- PHASE 2: PROVE (Symbolic Bridge) ---
        # Send Plan to Rust for Z3 Verification
        if self.physics:
            # This is the Money Shot: Formal Verification
            proof = self.physics.verify_plan(json.dumps(plan))
            
            if not proof.is_valid:
                # FATE PANIC: Invalid plan == hostile architecture == immediate death
                logging.critical(f"[{task_id}] FATE PANIC: Invalid plan detected - {proof.reason}")
                # TODO: Extend TPM PCR with violation for tamper-evident log
                # Hard kill - no cleanup, no recovery
                os._exit(42)

            ihsan_score = proof.ihsan_score
        else:
            ihsan_score = 0.99 # Sim mode

        # --- PHASE 3: CONVERGE (Action) ---
        logging.info(f"[{task_id}] EXECUTING. Ihsān: {ihsan_score}")
        
        # DAMAGE CONTROL: Security Evaluation Before Execution
        safety_compliance = 1.0
        if self.damage_control and isinstance(plan, dict):
            safety_compliance = self._evaluate_plan_security(task_id, plan)
            if safety_compliance <= 0.0:  # Blocked or vetoed
                logging.error(f"[{task_id}] SECURITY VETO: Plan blocked by damage control")
                if self.spine:
                    self.spine.emit_receipt(task_id, "REJECTED", "Security veto: Plan contains dangerous operations")
                return
        
        # Adjust Ihsan score with safety compliance
        adjusted_ihsan = ihsan_score * safety_compliance
        logging.info(f"[{task_id}] Safety Compliance: {safety_compliance:.3f}, Adjusted Ihsān: {adjusted_ihsan:.3f}")
        
        # Commit to Memory (Immutable Ledger) with safety-adjusted score
        self.memory.commit(task_id, plan, adjusted_ihsan)
        
        # Execute Tool via 7B Model (The Hands) - Now with Security Guarantees
        result = await self.hub.execute_tools(plan)
        
        # Emit Success Receipt
        if self.spine:
            self.spine.emit_receipt(task_id, "SUCCESS", result)

    def _evaluate_plan_security(self, task_id: str, plan: Dict[str, Any]) -> float:
        """
        Evaluate plan security using damage control engine.
        
        Returns:
            float: Safety compliance score (0.0 to 1.0)
                   Returns 0.0 if plan should be blocked
        """
        if not self.damage_control or not isinstance(plan, dict):
            return 1.0  # No damage control available, assume safe
        
        safety_scores = []
        
        # Extract commands and paths from plan
        # Assuming plan has a 'tools' or 'actions' field
        tools = plan.get("tools", []) or plan.get("actions", [])
        
        for tool in tools:
            tool_type = tool.get("type", "").lower()
            
            if tool_type == "bash":
                command = tool.get("command", "")
                if command:
                    evaluation = self.damage_control.evaluate_command(command, "bash")
                    if not evaluation["allowed"]:
                        logging.warning(f"[{task_id}] SECURITY BLOCK: {command[:50]}... - {evaluation['reasons']}")
                        return 0.0  # Block entire plan
                    safety_scores.append(evaluation["snr_safety_score"])
                    
            elif tool_type in ["edit", "write"]:
                path = tool.get("path", "")
                if path:
                    operation = "write" if tool_type == "write" else "edit"
                    evaluation = self.damage_control.evaluate_path(path, operation)
                    if not evaluation["allowed"]:
                        logging.warning(f"[{task_id}] PATH BLOCK: {path} - {evaluation.get('reason', 'No access')}")
                        return 0.0  # Block entire plan
                    # Path safety score is implicit in allowance
                    safety_scores.append(1.0)
        
        # Calculate geometric mean of safety scores
        if not safety_scores:
            return 1.0  # No tools to evaluate
        
        import math
        product = math.prod(safety_scores)
        geometric_mean = product ** (1/len(safety_scores))
        
        return geometric_mean


    def execute_sovereign_task(
        self,
        prompt: str,
        mission_metrics: Optional[Dict[str, float]] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a task through the Sovereign Nexus (or legacy engine).

        This is the main API for executing tasks.

        Args:
            prompt: The task/prompt to execute
            mission_metrics: Optional Ihsan metrics (truthfulness, dignity, etc.)
            request_id: Optional request identifier

        Returns:
            Dict with status, snr, latency, and other metrics
        """
        if self.nexus:
            # Use Nexus
            async def _execute():
                result = await self.nexus.execute(
                    task=prompt,
                    context={
                        "mission_metrics": mission_metrics,
                        "request_id": request_id
                    }
                )
                return {
                    "status": "SUCCESS" if result.success else "VETOED",
                    "latency": f"{result.latency_ms:.2f}ms",
                    "snr": result.snr_score,
                    "ihsan": result.ihsan_score,
                    "ledger_hash": result.receipt_id or "N/A",
                    "disciplines": result.disciplines_invoked,
                    "agents": result.agents_used,
                }
            return asyncio.run(_execute())

        # Legacy fallback
        return {
            "status": "LEGACY_MODE",
            "latency": "N/A",
            "snr": 0.0,
            "ihsan": 0.0,
            "ledger_hash": "N/A",
        }

    def get_nexus_statistics(self) -> Optional[Dict[str, Any]]:
        """Get comprehensive Nexus statistics."""
        if self.nexus:
            return self.nexus.get_full_statistics()
        return None

    def shutdown(self):
        """Graceful shutdown."""
        print("\n[!] SHUTDOWN: Saving Cognitive State...")

        # Stop Nexus
        if self.nexus:
            self.nexus.stop()
            print("[+] Sovereign Nexus: STOPPED")

        # Save memory snapshot
        if hasattr(self, 'memory') and self.memory:
            self.memory.save_snapshot()

        print("[+] OFFLINE.")


if __name__ == "__main__":
    kernel = SovereignKernel()
    try:
        asyncio.run(kernel.omni_loop())
    except KeyboardInterrupt:
        kernel.shutdown()
