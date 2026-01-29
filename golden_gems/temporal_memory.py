"""
GOLDEN GEM #2: TEMPORAL MEMORY DECAY
════════════════════════════════════

Memory as a living system with γ-indexed decay.

| Layer | γ (Decay) | Half-Life | Purpose |
|-------|-----------|-----------|---------|
| L1 Perception | 0.5 | ~1 cycle | Immediate sensory buffer |
| L2 Working | 0.99 | ~70 cycles | Active reasoning context |
| L3 Episodic | 0.999 | ~700 cycles | Recent conversations |
| L4 Semantic | 0.9999 | ~7000 cycles | Knowledge graph |
| L5 Expertise | 1.0 | ∞ | Compiled skills (never decay) |

SNR Score: 0.92
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MemoryItem:
    """A single memory with decay tracking."""
    content: str
    hash: str
    created_ns: int
    last_accessed_ns: int
    access_count: int = 1
    salience: float = 0.5
    strength: float = 1.0
    
    def decay(self, gamma: float, cycles_elapsed: int) -> float:
        """Apply decay and return new strength."""
        self.strength *= (gamma ** cycles_elapsed)
        return self.strength
    
    def reinforce(self, boost: float = 0.1) -> float:
        """Reinforce memory (accessed again)."""
        self.strength = min(1.0, self.strength + boost)
        self.access_count += 1
        self.last_accessed_ns = time.time_ns()
        return self.strength


class TemporalMemoryLayer:
    """A single layer of the temporal memory hierarchy."""
    
    def __init__(self, name: str, gamma: float, max_items: int):
        self.name = name
        self.gamma = gamma
        self.max_items = max_items
        self.items: Dict[str, MemoryItem] = {}
        self.cycle_count = 0
    
    @property
    def half_life(self) -> float:
        """Cycles until memory strength halves."""
        if self.gamma >= 1.0:
            return float('inf')
        return math.log(0.5) / math.log(self.gamma)
    
    def add(self, content: str, hash: str, salience: float = 0.5) -> MemoryItem:
        """Add item to layer."""
        item = MemoryItem(
            content=content,
            hash=hash,
            created_ns=time.time_ns(),
            last_accessed_ns=time.time_ns(),
            salience=salience,
        )
        self.items[hash] = item
        
        if len(self.items) > self.max_items:
            self._prune()
        
        return item
    
    def get(self, hash: str) -> Optional[MemoryItem]:
        """Retrieve and reinforce memory."""
        if hash in self.items:
            item = self.items[hash]
            item.reinforce()
            return item
        return None
    
    def tick(self) -> int:
        """Advance one cycle, apply decay, return items forgotten."""
        self.cycle_count += 1
        forgotten = 0
        
        to_forget = []
        for hash, item in self.items.items():
            item.decay(self.gamma, 1)
            if item.strength < 0.01:
                to_forget.append(hash)
        
        for hash in to_forget:
            del self.items[hash]
            forgotten += 1
        
        return forgotten
    
    def _prune(self):
        """Remove weakest items to stay under capacity."""
        if len(self.items) <= self.max_items:
            return
        
        ranked = sorted(
            self.items.items(),
            key=lambda x: x[1].strength * x[1].salience,
            reverse=True
        )
        self.items = dict(ranked[:self.max_items])
    
    def promote_candidates(self, threshold: float = 0.8) -> List[MemoryItem]:
        """Find items worthy of promotion to higher layer."""
        candidates = []
        for item in self.items.values():
            if item.access_count > 5 and item.salience > threshold:
                candidates.append(item)
        return candidates


class TemporalMemoryHierarchy:
    """
    The 5-Layer Temporal Memory System.
    
    Each layer has different decay characteristics matching
    cognitive science research on memory consolidation.
    """
    
    def __init__(self):
        self.layers = {
            "L1_perception": TemporalMemoryLayer("perception", gamma=0.5, max_items=10),
            "L2_working": TemporalMemoryLayer("working", gamma=0.99, max_items=7),
            "L3_episodic": TemporalMemoryLayer("episodic", gamma=0.999, max_items=100),
            "L4_semantic": TemporalMemoryLayer("semantic", gamma=0.9999, max_items=10000),
            "L5_expertise": TemporalMemoryLayer("expertise", gamma=1.0, max_items=1000),
        }
        self.layer_order = ["L1_perception", "L2_working", "L3_episodic", "L4_semantic", "L5_expertise"]
    
    def perceive(self, content: str, hash: str) -> MemoryItem:
        """Entry point: new perception."""
        return self.layers["L1_perception"].add(content, hash, salience=0.3)
    
    def focus(self, content: str, hash: str, salience: float = 0.7) -> MemoryItem:
        """Deliberate focus: add to working memory."""
        return self.layers["L2_working"].add(content, hash, salience=salience)
    
    def remember(self, content: str, hash: str) -> MemoryItem:
        """Store episodic memory."""
        return self.layers["L3_episodic"].add(content, hash, salience=0.6)
    
    def know(self, content: str, hash: str) -> MemoryItem:
        """Store semantic knowledge."""
        return self.layers["L4_semantic"].add(content, hash, salience=0.8)
    
    def master(self, content: str, hash: str) -> MemoryItem:
        """Store expertise (permanent)."""
        return self.layers["L5_expertise"].add(content, hash, salience=1.0)
    
    def tick(self) -> Dict[str, int]:
        """Advance all layers, handle promotions."""
        forgotten = {}
        
        for name, layer in self.layers.items():
            forgotten[name] = layer.tick()
        
        self._handle_promotions()
        
        return forgotten
    
    def _handle_promotions(self):
        """Promote memories that have proven their worth."""
        for i in range(len(self.layer_order) - 1):
            lower_name = self.layer_order[i]
            upper_name = self.layer_order[i + 1]
            
            lower = self.layers[lower_name]
            upper = self.layers[upper_name]
            
            for candidate in lower.promote_candidates():
                upper.add(candidate.content, candidate.hash, candidate.salience)
    
    def search(self, query_hash: str) -> Optional[MemoryItem]:
        """Search all layers for a memory."""
        for name in reversed(self.layer_order):
            result = self.layers[name].get(query_hash)
            if result:
                return result
        return None
    
    def status(self) -> Dict:
        """Get hierarchy status."""
        return {
            name: {
                "items": len(layer.items),
                "gamma": layer.gamma,
                "half_life": layer.half_life,
                "cycles": layer.cycle_count,
            }
            for name, layer in self.layers.items()
        }


def demo():
    """Show the decay system in action."""
    
    mem = TemporalMemoryHierarchy()
    
    print("=== TEMPORAL MEMORY DEMO ===")
    print(f"Half-lives: L1={mem.layers['L1_perception'].half_life:.1f}, L2={mem.layers['L2_working'].half_life:.1f}")
    
    # Add some perceptions
    mem.perceive("saw a bird", "bird1")
    mem.perceive("heard a sound", "sound1")
    mem.focus("important meeting", "meeting1")
    mem.know("BIZRA is transformative", "bizra_core")
    mem.master("لا نفترض", "principle_1")
    
    print(f"\nInitial: {mem.status()}")
    
    # Simulate 10 cycles
    print("\nSimulating 10 cycles...")
    for i in range(10):
        forgotten = mem.tick()
        if any(v > 0 for v in forgotten.values()):
            print(f"Cycle {i+1}: Forgotten {forgotten}")
    
    print(f"\nFinal: {mem.status()}")
    
    # Check what survived
    print("\nSurvival check:")
    for test_hash in ["bird1", "meeting1", "bizra_core", "principle_1"]:
        result = mem.search(test_hash)
        if result:
            print(f"  {test_hash}: strength={result.strength:.4f}")
        else:
            print(f"  {test_hash}: FORGOTTEN")


if __name__ == "__main__":
    demo()
