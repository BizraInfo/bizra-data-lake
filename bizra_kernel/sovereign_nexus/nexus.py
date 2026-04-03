"""
BIZRA Sovereign Nexus - Unified Control Interface for BIZRA Sovereign Intelligence

This is the apex orchestrator that consolidates 11 existing components
into a unified control interface with:
- 47-Discipline Topology Engine
- Autonomous Dreaming capability
- SNR self-healing optimization (0.95 Ihsān threshold)
"""

import asyncio
import threading
from typing import Dict, Any, Optional, List
from datetime import datetime
import time

from bizra_kernel.sovereign_engine import SovereignEngine
from bizra_kernel.snr_tracker import SNRTracker
from bizra_kernel.got_orchestrator import GoTOrchestrator
from bizra_kernel.model_hub import SovereignModelHub
from bizra_kernel.omni_awareness import OmniAwareness
from bizra_kernel.memory_system import CognitivePermanence
from bizra_kernel.ihsan_gate import IhsanGate

from .topology_engine import DisciplineTopologyEngine
from .dreamer import AutonomousDreamer
from .subsystems.neural_subsystem import NeuralSubsystem
from .subsystems.symbolic_subsystem import SymbolicSubsystem
from .subsystems.agentic_subsystem import AgenticSubsystem
from .subsystems.optimization_subsystem import OptimizationSubsystem
from .adapters.rust_bridge_adapter import RustBridgeAdapter
from .adapters.synapse_adapter import SynapseAdapter
from .adapters.data_lake_adapter import DataLakeAdapter
from .adapters.dual_agentic_adapter import DualAgenticAdapter


class SovereignNexus:
    """
    Unified Control Interface for BIZRA Sovereign Intelligence.
    
    Orchestrates all subsystems and maintains the 147Hz heartbeat loop
    with integrated autonomous dreaming capabilities.
    """
    
    def __init__(
        self,
        *,
        heartbeat_hz: float = 147.0,
        ihsan_threshold: float = 0.95,
        snr_target: float = 0.95,
        enable_dreaming: bool = True
    ):
        """
        Initialize the SovereignNexus.
        
        Args:
            heartbeat_hz: Frequency of the main heartbeat loop in Hz
            ihsan_threshold: Minimum Ihsan compliance threshold
            snr_target: Target SNR score for optimization
            enable_dreaming: Whether to enable autonomous dreaming
        """
        # Core subsystems - initialize in dependency order
        self.memory: CognitivePermanence = CognitivePermanence()
        self.awareness: OmniAwareness = OmniAwareness(self.memory)  # Pass memory as required
        self.model_hub: SovereignModelHub = SovereignModelHub()
        self.topology: DisciplineTopologyEngine = DisciplineTopologyEngine()
        self.dreamer: Optional[AutonomousDreamer] = None
        
        # Orchestration
        self.got: GoTOrchestrator = GoTOrchestrator()
        self.snr_tracker: SNRTracker = SNRTracker()
        
        # Governance
        self.ihsan_gate: IhsanGate = IhsanGate(ihsan_threshold)
        
        # Initialize remaining components
        self.heartbeat_hz = heartbeat_hz
        self.ihsan_threshold = ihsan_threshold
        self.snr_target = snr_target
        self.enable_dreaming = enable_dreaming
        self.running = False
        
        # Subsystems
        self.neural_subsystem: Optional[NeuralSubsystem] = None
        self.symbolic_subsystem: Optional[SymbolicSubsystem] = None
        self.agentic_subsystem: Optional[AgenticSubsystem] = None
        self.optimization_subsystem: Optional[OptimizationSubsystem] = None
        
        # Adapters
        self.rust_bridge: Optional[RustBridgeAdapter] = None
        self.synapse_adapter: Optional[SynapseAdapter] = None
        self.data_lake_adapter: Optional[DataLakeAdapter] = None
        
        # Stats
        self.heartbeat_count = 0
        self.dream_count = 0
        self.operation_log = []
    
    async def initialize(self):
        """Initialize all components of the SovereignNexus."""
        print("Initializing BIZRA Sovereign Nexus...")
        
        # Initialize components that have async initialization methods
        # For now, we'll initialize the subsystems and adapters that have async initialization
        
        # Initialize the dreamer if enabled
        if self.enable_dreaming:
            self.dreamer = AutonomousDreamer(
                memory=self.memory,
                got_orchestrator=self.got,
                snr_tracker=self.snr_tracker,
                snr_threshold=self.snr_target
            )
        
        # Initialize subsystems
        self.neural_subsystem = NeuralSubsystem()
        self.symbolic_subsystem = SymbolicSubsystem(topology_engine=self.topology)
        self.agentic_subsystem = AgenticSubsystem()
        self.optimization_subsystem = OptimizationSubsystem(
            snr_tracker=self.snr_tracker,
            got_orchestrator=self.got,
            ihsan_threshold=self.ihsan_threshold,
            snr_target=self.snr_target
        )
        
        # Initialize adapters
        self.rust_bridge = RustBridgeAdapter()
        self.synapse_adapter = SynapseAdapter()
        self.data_lake_adapter = DataLakeAdapter()
        self.dual_agentic_adapter = DualAgenticAdapter()
        
        # Initialize all subsystems and adapters that have async initialization
        await self.neural_subsystem.initialize()
        await self.symbolic_subsystem.initialize()
        await self.agentic_subsystem.initialize()
        await self.optimization_subsystem.initialize()
        
        # Connect to synapse
        await self.synapse_adapter.connect()
        
        # Connect to data lake
        await self.data_lake_adapter.connect()
        
        # Connect to dual agentic server
        await self.dual_agentic_adapter.connect()
        
        print("BIZRA Sovereign Nexus initialization complete!")
        
    async def heartbeat(self):
        """Execute a single heartbeat cycle."""
        self.heartbeat_count += 1
        
        # Log the heartbeat
        heartbeat_info = {
            'timestamp': datetime.now().isoformat(),
            'cycle': self.heartbeat_count,
            'type': 'heartbeat'
        }
        self.operation_log.append(heartbeat_info)
        
        # Perform basic health checks
        await self._perform_health_checks()
        
        # Run optimization checks
        await self._run_optimization_checks()
        
        # Potentially run dreaming cycle
        if self.enable_dreaming and self.dreamer and self.heartbeat_count % 10 == 0:
            await self._run_dream_cycle()
    
    async def _perform_health_checks(self):
        """Perform system health checks."""
        # Check memory integrity
        memory_health = await self.memory.health_check()
        
        # Check SNR metrics
        snr_metrics = await self.snr_tracker.get_current_metrics()
        
        # Check Ihsan compliance
        ihsan_score = await self.ihsan_gate.get_current_ihsan_score()
        
        # Log any issues
        if not memory_health.get('ok', True):
            self.operation_log.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'alert',
                'message': f'Memory health issue: {memory_health.get("details", "")}'
            })
        
        if ihsan_score < self.ihsan_threshold:
            self.operation_log.append({
                'timestamp': datetime.now().isoformat(),
                'type': 'alert',
                'message': f'Ihsan score below threshold: {ihsan_score} < {self.ihsan_threshold}'
            })
    
    async def _run_optimization_checks(self):
        """Run optimization checks and adjustments."""
        # Get optimization recommendations
        recommendations = await self.optimization_subsystem.get_optimization_recommendations()
        
        # Apply high-priority recommendations
        for rec in recommendations:
            if rec.get('priority') in ['critical', 'high']:
                print(f"Applying high-priority optimization: {rec.get('suggestion')}")
                await self.optimization_subsystem.optimize_general_performance()
    
    async def _run_dream_cycle(self):
        """Run a dream cycle if conditions are favorable."""
        if not self.dreamer:
            return
        
        # Get system budget to determine if dreaming is appropriate
        budget_score = await self.awareness.get_cognitive_budget()
        
        # Run dream cycle
        dream_result = await self.dreamer.dream_cycle(budget_score)
        
        if dream_result and dream_result.crystallized:
            self.dream_count += 1
            print(f"Dream crystallized: SNR={dream_result.snr_score:.2f}")
            
            # Submit insight to data lake if available
            if self.data_lake_adapter:
                insight = {
                    'type': 'crystallized_hypothesis',
                    'content': dream_result.hypothesis,
                    'snr_score': dream_result.snr_score,
                    'origin_seed': dream_result.origin_seed,
                    'patterns_discovered': dream_result.patterns_discovered,
                    'timestamp': dream_result.timestamp.isoformat()
                }
                await self.data_lake_adapter.submit_insight(insight)
    
    async def run_heartbeat_loop(self):
        """Run the continuous heartbeat loop."""
        if not self.running:
            await self.initialize()
            self.running = True
        
        print(f"Starting BIZRA Sovereign Nexus heartbeat at {self.heartbeat_hz}Hz...")
        
        heartbeat_interval = 1.0 / self.heartbeat_hz
        
        while self.running:
            start_time = time.time()
            
            try:
                await self.heartbeat()
            except Exception as e:
                print(f"Error in heartbeat: {e}")
                self.operation_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'error',
                    'message': str(e)
                })
            
            # Calculate sleep time to maintain target frequency
            elapsed = time.time() - start_time
            sleep_time = max(0, heartbeat_interval - elapsed)
            
            await asyncio.sleep(sleep_time)
    
    def stop_heartbeat(self):
        """Stop the heartbeat loop."""
        self.running = False
        print("BIZRA Sovereign Nexus heartbeat stopped.")
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a request through the Nexus.
        
        Args:
            request: Request dictionary with 'query' and optional 'context'
            
        Returns:
            Response dictionary
        """
        query = request.get('query', '')
        context = request.get('context', {})
        
        # Determine the appropriate processing path based on the request
        if 'discipline' in context:
            # Use symbolic subsystem for discipline-specific requests
            disciplines = [context['discipline']] if isinstance(context['discipline'], str) else context['discipline']
            result = await self.symbolic_subsystem.reason_about_disciplines(disciplines, query)
            return {
                'success': True,
                'result': result.content,
                'confidence': result.confidence,
                'reasoning_path': result.reasoning_path,
                'metadata': {
                    'processing_path': 'symbolic',
                    'disciplines_applied': result.applied_disciplines
                }
            }
        else:
            # Use neural subsystem for general queries
            result = await self.neural_subsystem.process_query(query, context)
            return {
                'success': True,
                'result': result.content,
                'confidence': result.confidence,
                'sources': result.sources,
                'metadata': {
                    'processing_path': 'neural',
                    'embedding_generated': result.embeddings is not None
                }
            }
    
    async def execute_agentic_task(self, task_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an agentic task through the Nexus.
        
        Args:
            task_spec: Specification of the task to execute
            
        Returns:
            Result of the task execution
        """
        if not self.agentic_subsystem:
            return {
                'success': False,
                'error': 'Agentic subsystem not initialized'
            }
        
        from .subsystems.agentic_subsystem import AgenticTask
        
        task = AgenticTask(
            task_id=task_spec.get('task_id', f'task_{int(datetime.now().timestamp())}'),
            agent_type=task_spec.get('agent_type', 'general'),
            goal=task_spec.get('goal', ''),
            context=task_spec.get('context', {}),
            priority=task_spec.get('priority', 1)
        )
        
        # Assign the task
        task_id = await self.agentic_subsystem.assign_task(task)
        
        # Wait for result (with timeout)
        timeout = task_spec.get('timeout', 30)
        start_time = asyncio.get_event_loop().time()
        
        result = None
        while asyncio.get_event_loop().time() - start_time < timeout:
            result = await self.agentic_subsystem.get_result(task_id)
            if result:
                break
            await asyncio.sleep(0.1)
        
        if result:
            return {
                'success': True,
                'result': result.result,
                'agent_id': result.agent_id,
                'execution_time': result.execution_time,
                'metadata': result.metadata
            }
        else:
            return {
                'success': False,
                'error': 'Task timed out',
                'task_id': task_id
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status.
        
        Returns:
            Dictionary with system status information
        """
        snr_metrics = await self.snr_tracker.get_current_metrics()
        ihsan_score = await self.ihsan_gate.get_current_ihsan_score()
        memory_stats = await self.memory.get_stats()
        
        return {
            'nexus_status': 'running' if self.running else 'stopped',
            'heartbeat_frequency_hz': self.heartbeat_hz,
            'heartbeat_count': self.heartbeat_count,
            'dream_count': self.dream_count,
            'snr_metrics': snr_metrics,
            'ihsan_score': ihsan_score,
            'memory_stats': memory_stats,
            'subsystem_status': {
                'neural': 'initialized' if self.neural_subsystem else 'not_initialized',
                'symbolic': 'initialized' if self.symbolic_subsystem else 'not_initialized',
                'agentic': 'initialized' if self.agentic_subsystem else 'not_initialized',
                'optimization': 'initialized' if self.optimization_subsystem else 'not_initialized'
            },
            'adapter_status': {
                'rust_bridge': 'connected' if self.rust_bridge and self.rust_bridge.connected else 'disconnected',
                'synapse': 'connected' if self.synapse_adapter and self.synapse_adapter.connected else 'disconnected',
                'data_lake': 'connected' if self.data_lake_adapter and self.data_lake_adapter.connected else 'disconnected'
            }
        }
    
    async def shutdown(self):
        """Shutdown the SovereignNexus and all its components."""
        print("Shutting down BIZRA Sovereign Nexus...")
        
        # Stop heartbeat
        self.stop_heartbeat()
        
        # Shutdown subsystems
        if self.agentic_subsystem:
            await self.agentic_subsystem.shutdown()
        
        if self.optimization_subsystem:
            await self.optimization_subsystem.stop_self_healing()
        
        # Disconnect adapters
        if self.synapse_adapter:
            await self.synapse_adapter.disconnect()
        
        if self.data_lake_adapter:
            await self.data_lake_adapter.disconnect()
        
        print("BIZRA Sovereign Nexus shutdown complete.")