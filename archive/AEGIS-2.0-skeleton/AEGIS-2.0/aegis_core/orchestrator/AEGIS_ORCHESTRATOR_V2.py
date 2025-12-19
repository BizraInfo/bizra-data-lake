from __future__ import annotations
from typing import Dict, Any
from aegis_core.trd.TRD_PROCESSOR_V2 import TRDProcessor
from aegis_core.tdlc.TDLC_MANAGER_V2 import TDLCManager, HierarchicalMemoryManager
from aegis_core.thinktank.AGENTIC_THINKTANK_V2 import AgenticThinktank
from aegis_core.optimization.SELF_OPTIMIZATION_NEXUS_V2 import SelfOptimizationNexus
from aegis_core.knowledge.ALPHA_DISCOVERY_ENGINE_V2 import AlphaDiscoveryEngine

class AEGISOrchestrator:
    def __init__(self):
        self.trd_processor = TRDProcessor()
        self.tdlc_manager = TDLCManager()
        self.thinktank = AgenticThinktank()
        self.optimization_nexus = SelfOptimizationNexus()
        self.discovery_engine = AlphaDiscoveryEngine()
        self.memory_manager = HierarchicalMemoryManager()
        self.system_state = {}

    async def process_user_request(self, user_input: str, context: Dict[str, Any]|None=None) -> Dict[str, Any]:
        opt = await self.memory_manager.manage_context(self.system_state, context or {})
        trd = await self.trd_processor.process_user_input(user_input, opt)
        tdlc_result = await self.tdlc_manager.execute_tdlc(trd)
        return {"result": tdlc_result.get("solution", {}), "trd": trd}
