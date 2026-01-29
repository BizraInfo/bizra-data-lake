"""
GOLDEN GEM #6: COLIMIT INTERFACE
═════════════════════════════════

The Universal Adapter — Category Theory Made Practical.

One interface for all subsystems. The "colimit" in practical terms
is a universal adapter that:
- Preserves all operations
- Unifies all protocols  
- Requires no translation at runtime

SNR Score: 0.88
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, TypeVar, Generic
from dataclasses import dataclass

T = TypeVar('T')


@dataclass
class UniversalOp:
    """
    The universal operation type.
    
    Every subsystem operation can be expressed as:
    - intent: What we want to do
    - payload: The data
    - domain: Which subsystem handles it
    """
    intent: str
    payload: Dict[str, Any]
    domain: str


class SubsystemAdapter(ABC, Generic[T]):
    """
    Adapter from universal operations to subsystem-specific calls.
    
    The "injection" in categorical terms — how each subsystem
    maps into the universal structure.
    """
    
    @property
    @abstractmethod
    def domain(self) -> str:
        """Which domain this adapter handles."""
        pass
    
    @abstractmethod
    async def execute(self, op: UniversalOp) -> T:
        """Execute operation in this subsystem."""
        pass
    
    def can_handle(self, op: UniversalOp) -> bool:
        """Check if this adapter handles the operation."""
        return op.domain == self.domain


class AccumulatorAdapter(SubsystemAdapter[Dict]):
    """Adapter for the Accumulator subsystem."""
    
    @property
    def domain(self) -> str:
        return "accumulator"
    
    def can_handle(self, op: UniversalOp) -> bool:
        return op.domain == "accumulator" or op.intent in [
            "record_impact", "get_bloom", "harvest", "stake"
        ]
    
    async def execute(self, op: UniversalOp) -> Dict:
        intent = op.intent
        if intent == "record_impact":
            return {"bloom_earned": 1.23, "poi_hash": "abc123"}
        elif intent == "get_bloom":
            return {"bloom": 100.08, "seeds": 1}
        else:
            return {"status": "unknown_intent"}


class FlywheelAdapter(SubsystemAdapter[Dict]):
    """Adapter for the Flywheel (LLM) subsystem."""
    
    @property
    def domain(self) -> str:
        return "flywheel"
    
    def can_handle(self, op: UniversalOp) -> bool:
        return op.domain == "flywheel" or op.intent in [
            "infer", "embed", "generate", "reason"
        ]
    
    async def execute(self, op: UniversalOp) -> Dict:
        intent = op.intent
        if intent == "infer":
            return {"response": "LLM response", "latency_ms": 4899}
        elif intent == "embed":
            return {"embedding": [0.1] * 64}
        else:
            return {"status": "unknown_intent"}


class KnowledgeAdapter(SubsystemAdapter[List]):
    """Adapter for the Knowledge Graph subsystem."""
    
    @property
    def domain(self) -> str:
        return "knowledge"
    
    def can_handle(self, op: UniversalOp) -> bool:
        return op.domain == "knowledge" or op.intent in [
            "query_graph", "add_node", "add_edge", "traverse"
        ]
    
    async def execute(self, op: UniversalOp) -> List:
        if op.intent == "query_graph":
            return [{"node": "concept_1"}, {"node": "concept_2"}]
        return []


class ColimitDispatcher:
    """
    The Colimit — Universal dispatcher that routes to subsystems.
    
    In practical terms:
    - One interface handles everything
    - Subsystems are hot-swappable
    - No tight coupling
    """
    
    def __init__(self):
        self.adapters: List[SubsystemAdapter] = []
    
    def register(self, adapter: SubsystemAdapter):
        """Register a subsystem adapter."""
        self.adapters.append(adapter)
    
    async def dispatch(self, op: UniversalOp) -> Any:
        """Dispatch operation to the appropriate subsystem."""
        for adapter in self.adapters:
            if adapter.can_handle(op):
                return await adapter.execute(op)
        
        raise ValueError(f"No adapter for: {op.intent} in {op.domain}")
    
    async def multi_dispatch(self, op: UniversalOp) -> List[Any]:
        """Dispatch to ALL adapters that can handle (fan-out)."""
        results = []
        for adapter in self.adapters:
            if adapter.can_handle(op):
                result = await adapter.execute(op)
                results.append({"domain": adapter.domain, "result": result})
        return results
    
    # === CONVENIENCE METHODS ===
    
    async def accumulate(self, contributor: str, action: str, impact: float) -> Dict:
        """Record impact to accumulator."""
        op = UniversalOp(
            intent="record_impact",
            payload={"contributor": contributor, "action": action, "impact": impact},
            domain="accumulator",
        )
        return await self.dispatch(op)
    
    async def infer(self, prompt: str, model: str = "default") -> Dict:
        """Run LLM inference."""
        op = UniversalOp(
            intent="infer",
            payload={"prompt": prompt, "model": model},
            domain="flywheel",
        )
        return await self.dispatch(op)
    
    async def query_knowledge(self, query: str) -> List:
        """Query knowledge graph."""
        op = UniversalOp(
            intent="query_graph",
            payload={"query": query},
            domain="knowledge",
        )
        return await self.dispatch(op)


async def demo():
    """Show the universal interface in action."""
    
    dispatcher = ColimitDispatcher()
    dispatcher.register(AccumulatorAdapter())
    dispatcher.register(FlywheelAdapter())
    dispatcher.register(KnowledgeAdapter())
    
    print("=== COLIMIT DISPATCHER DEMO ===\n")
    
    # High-level API
    print("High-level API:")
    
    result = await dispatcher.accumulate("mumo", "genesis", 100.0)
    print(f"  accumulate() → {result}")
    
    result = await dispatcher.infer("What is BIZRA?")
    print(f"  infer() → {result}")
    
    result = await dispatcher.query_knowledge("MATCH (n) RETURN n")
    print(f"  query_knowledge() → {result}")
    
    # Low-level API
    print("\nLow-level API:")
    
    op = UniversalOp(
        intent="get_bloom",
        payload={"contributor": "mumo"},
        domain="accumulator",
    )
    result = await dispatcher.dispatch(op)
    print(f"  dispatch(get_bloom) → {result}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo())
