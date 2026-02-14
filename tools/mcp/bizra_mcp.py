#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════════════════════
    BIZRA MCP SERVER — Unified Ecosystem Interface
═══════════════════════════════════════════════════════════════════════════════════════════════
    
    Exposes the BIZRA DDAGI OS v2.0.0 via the Model Context Protocol (MCP).
    
    CAPABILITIES:
      - Deep Cognitive Query (via UltimateEngine + Orchestrator + Peak)
      - System Health & Invariance Checks
      - Constitutional & Daughter Test Verification
    
    This server integrates the full "Ecosystem Bridge" created in step 1.
    
    Author: BIZRA Genesis NODE0 | v2.0.0
═══════════════════════════════════════════════════════════════════════════════════════════════
"""

import asyncio
import logging
from typing import Dict, Any, List

# MCP Framework
from fastmcp import FastMCP

# BIZRA Ecosystem
from ecosystem_bridge import (
    initialize_ecosystem, 
    UnifiedQuery, 
    EcosystemBridge,
    UnifiedResponse
)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] MCP | %(message)s'
)
logger = logging.getLogger("BIZRA-MCP")

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# SERVER DEFINITION
# ═══════════════════════════════════════════════════════════════════════════════════════════════

mcp = FastMCP("BIZRA DDAGI OS")

# Global Ecosystem Instance
_ecosystem_instance: EcosystemBridge | None = None

async def get_instance() -> EcosystemBridge:
    """Singleton accessor for the ecosystem."""
    global _ecosystem_instance
    if _ecosystem_instance is None:
        logger.info("🚀 Initializing BIZRA Ecosystem...")
        _ecosystem_instance = await initialize_ecosystem()
        logger.info("✅ Ecosystem initialized")
    return _ecosystem_instance


# ═══════════════════════════════════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════════════════════════════════

@mcp.tool(description="Execute a cognitive query against the BIZRA DDAGI ecosystem")
async def query_bizra(
    query: str, 
    require_deep_scan: bool = True,
    snr_threshold: float = 0.85
) -> str:
    """
    Process a natural language query through the BIZRA Unified Ecosystem.
    
    Args:
        query: The user's question or instruction.
        require_deep_scan: If True, engages the Orchestrator and Apex (slower, deeper).
        snr_threshold: Minimum Signal-to-Noise Ratio acceptable (0.0-1.0).
        
    Returns:
        The synthesized answer with provenance and confidence scores.
    """
    ecosystem = await get_instance()
    
    # Construct unified query context
    uq = UnifiedQuery(
        text=query,
        require_constitution_check=True,
        require_daughter_test=True,
        snr_threshold=snr_threshold,
        use_orchestrator=require_deep_scan,
        use_apex=require_deep_scan,
        metadata={"source": "mcp_client"}
    )
    
    # Execute
    start = asyncio.get_event_loop().time()
    result: UnifiedResponse = await ecosystem.query(uq)
    elapsed = asyncio.get_event_loop().time() - start
    
    # Format Response for LLM consumption
    return f"""
# BIZRA System Response
-----------------------
{result.synthesis}

# Meta-Analysis
- SNR Score: {result.snr_score:.3f}
- Ihsan Score: {result.ihsan_score:.3f}
- Execution: {elapsed:.2f}s
- Constitution: {'PASSED' if result.constitution_check else 'FAILED'}
- Daughter Test: {'PASSED' if result.daughter_test_check else 'FAILED'}
- Component Trace: {', '.join(result.components_used)}
- Sources: {len(result.sources)}
    """.strip()


@mcp.tool(description="Get the current health and status of the BIZRA OS Kernel")
async def get_system_health() -> Dict[str, Any]:
    """
    Returns diagnostic information about the ecosystem engines.
    Verifies Kernel Invariants (RIBA_ZERO, ZANN_ZERO, IHSAN_FLOOR).
    """
    ecosystem = await get_instance()
    health = ecosystem.get_health()
    status = ecosystem.get_status()
    
    return {
        "status": "operational" if health.overall_health > 0.8 else "degraded",
        "health_score": f"{health.overall_health * 100:.1f}%",
        "invariants_secure": health.kernel_invariants_ok,
        "uptime_hours": status["uptime_hours"],
        "components": health.to_dict()
    }

# ═══════════════════════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Stdout transport is default for fastmcp, making it compatible with agentic tools
    mcp.run()
