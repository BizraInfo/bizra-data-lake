"""
BIZRA Graph Module — Semantic Layer Separation & Topology Analysis
═══════════════════════════════════════════════════════════════════

Ω-1 Implementation: Separate semantic edges from structural PART_OF edges
to lift pattern confidence from 0.46 → 0.85+.

Standing on Giants:
- Watts & Strogatz (1998): Small-world networks
- Barabási & Albert (1999): Scale-free networks
- Newman (2003): Modularity and community structure
- Shannon (1948): Information-theoretic graph measures

Genesis Strict Synthesis v2.2.2
"""

__version__ = "1.0.0"

__all__ = [
    "SemanticLayerSeparator",
    "GraphTopologyReport",
    "EdgeClassification",
    "DualOverlayGraph",
    "create_semantic_separator",
]


def __getattr__(name):
    """Lazy import."""
    if name in __all__:
        from .semantic_layer import (
            SemanticLayerSeparator,
            GraphTopologyReport,
            EdgeClassification,
            DualOverlayGraph,
            create_semantic_separator,
        )
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
