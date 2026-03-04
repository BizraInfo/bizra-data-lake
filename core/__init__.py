"""
BIZRA Core Package — The Stem Cell Kernel

Sovereignty Infrastructure (Thermodynamic Framing):
- pci: Proof-Carrying Inference Protocol (Genome + Epigenome)
- vault: Encryption at Rest (Membrane)
- federation: P2P Network (Mycorrhizal Connections)
- inference: Embedded LLM Gateway (Metabolism)
- a2a: Agent-to-Agent Protocol (Orchestration)
- integration: Unified Bridge (Cohesion)
- ntu: NeuroTemporal Unit (Pattern Detection)
- protocols: Interface Contracts (Structural Typing)

Decomposed from sovereign module (SAPE v2.3.1):
- governance: Constitutional Gates & Autonomy
- reasoning: Graph-of-Thoughts & Quality Validation
- orchestration: Event Bus & Agent Coordination
- treasury: Resource Management & Justice Enforcement
- bridges: Cross-System Integration

"Entropy reduction as a service."

Created: 2026-01-27
Updated: 2026-01-30 — Added integration bridge for module cohesion
Updated: 2026-02-03 — Added NTU (NeuroTemporal Unit) for pattern detection
Updated: 2026-02-05 — Added protocols + decomposed sovereign (SAPE Elite Analysis)
Updated: 2026-02-17 — Phase 31: HyperGraph + Cognitive Fusion + Memory Coder
Updated: 2026-02-17 — Phase 25-28: Genesis + Guild + Quest + HRM + NorthStar + Memory
"""

# Lazy subpackage loading — Standing on Giants: Knuth (1974) "optimize after profiling"
# Eager import of 27 subpackages caused 7.2s cold boot (210 transitive modules).
# Lazy __getattr__ defers loading until first access: 7,228ms → <50ms.
_SUBPACKAGES = frozenset(
    {
        "a2a",
        "agentic",
        "apex",
        "auth",
        "autonomous",
        "autopoiesis",
        "benchmark",
        "bounty",
        "bridges",
        "cognitive_fusion",
        "command",
        "constitutional",
        "elite",
        "embedding",
        "federation",
        "genesis",
        "governance",
        "graph",
        "guild",
        "hashtable",
        "hrm",
        "hypergraph",
        "iaas",
        "identity",
        "inference",
        "integration",
        "living_memory",
        "marketplace",
        "memory",
        "memory_coder",
        "nexus",
        "northstar",
        "ntu",
        "orchestration",
        "pat",
        "pci",
        "pek",
        "personaplex",
        "prediction",
        "proof_engine",
        "protocols",
        "quest",
        "rdve",
        "reasoning",
        "rollout",
        "sdpo",
        "search",
        "skills",
        "sovereign",
        "spearpoint",
        "token",
        "treasury",
        "uers",
        "vault",
        "voice",
        "zpk",
    }
)


def __getattr__(name: str):
    """Lazy import for subpackages — loaded on first access, not at boot."""
    if name in _SUBPACKAGES:
        import importlib

        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module  # Cache for subsequent access
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = sorted(_SUBPACKAGES)
__version__ = "2.5.0"
