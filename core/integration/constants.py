"""
BIZRA Integration Constants

Unified constants across all core modules to ensure consistency.
These values override module-specific constants when using the
IntegrationBridge.

Sovereignty: Single source of truth for quality thresholds.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════

# Unified Ihsan threshold - excellence constraint across all modules
# Resolves inconsistency between consensus (0.99) and others (0.95)
# Decision: Use 0.95 as the standard, with optional stricter mode
UNIFIED_IHSAN_THRESHOLD = 0.95
STRICT_IHSAN_THRESHOLD = 0.99  # For consensus-critical operations

# Signal-to-Noise Ratio threshold
UNIFIED_SNR_THRESHOLD = 0.85

# ═══════════════════════════════════════════════════════════════════════════════
# TIMING CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Maximum allowed clock skew for timestamp validation
UNIFIED_CLOCK_SKEW_SECONDS = 120

# Nonce TTL for replay protection
UNIFIED_NONCE_TTL_SECONDS = 300

# Pattern sync interval
UNIFIED_SYNC_INTERVAL_SECONDS = 60

# Consensus check interval
UNIFIED_CONSENSUS_INTERVAL_SECONDS = 30

# Agent timeout
UNIFIED_AGENT_TIMEOUT_MS = 30000

# ═══════════════════════════════════════════════════════════════════════════════
# NETWORK CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Default bind address for federation
DEFAULT_FEDERATION_BIND = "0.0.0.0:7654"

# Default A2A port offset from federation port
A2A_PORT_OFFSET = 100

# Maximum retry attempts for A2A operations
MAX_RETRY_ATTEMPTS = 3

# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Primary LLM backend
LMSTUDIO_URL = "http://192.168.56.1:1234"

# Fallback LLM backend
OLLAMA_URL = "http://localhost:11434"

# Model directory (unified path)
MODEL_DIR = "/mnt/c/BIZRA-DATA-LAKE/models"
