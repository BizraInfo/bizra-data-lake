/**
 * BIZRA Sovereign LLM - Core Exports
 *
 * "Every model is welcome if they accept the rules of BIZRA."
 * بذرة — The Seed accepts all who honor the Constitution.
 */

// Network Mode
export {
  NetworkMode,
  NetworkModeConfig,
  NetworkStatus,
  DEFAULT_NETWORK_CONFIG,
  allowsExternalConnections,
  allowsInternetAccess,
  getFallbackMode,
  createInitialNetworkStatus,
} from './network-mode';

// Capability Card
export {
  CapabilityCard,
  ModelTier,
  TaskType,
  Quantization,
  IHSAN_THRESHOLD,
  SNR_THRESHOLD,
  createCapabilityCard,
  getCardCanonicalBytes,
  signCapabilityCard,
  verifyCapabilityCardSignature,
  isCapabilityCardValid,
  getCardRemainingDays,
  getCardFingerprint,
} from './capability-card';

// Model Registry
export {
  ModelRegistry,
  RegisteredModel,
  ModelSelectionCriteria,
  RegistryStats,
} from './model-registry';

// Constitution Challenge
export {
  ChallengeDefinition,
  ChallengeResult,
  ChallengeOutcome,
  DEFAULT_CHALLENGES,
  InferenceFunction,
  runConstitutionChallenge,
  formatChallengeOutcome,
  scoreIhsanResponse,
  scoreSnrResponse,
  checkSovereigntyResponse,
} from './constitution-challenge';

// Model Router
export {
  ModelRouter,
  RouterConfig,
  RoutingDecision,
  RoutingRequest,
  TaskComplexity,
  estimateComplexity,
} from './model-router';

// FATE Binding
export {
  FateValidator,
  GateResult,
  FateChallengeResult,
  InferenceOutput,
  createFateValidator,
  getFateVersion,
} from './fate-binding';

// Runtime
export {
  SovereignRuntime,
  RuntimeConfig,
  DEFAULT_RUNTIME_CONFIG,
  InferenceRequest,
  InferenceResult,
  RuntimeStatus,
  createSovereignRuntime,
  printBanner as printRuntimeBanner,
} from './runtime';

/**
 * Constitutional thresholds as constants
 */
export const CONSTITUTION = {
  /** إحسان (Ihsān) - Excellence threshold */
  IHSAN_THRESHOLD: 0.95,

  /** Shannon SNR threshold */
  SNR_THRESHOLD: 0.85,

  /** Card validity period in days */
  CARD_VALIDITY_DAYS: 90,

  /** Maximum retry attempts for federation */
  MAX_FEDERATION_RETRIES: 3,

  /** Target IPC latency in ms */
  TARGET_IPC_LATENCY_MS: 55,
} as const;

/**
 * Print BIZRA banner
 */
export function printBanner(): void {
  console.log(`
╔══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║    "بذرة — Every seed is welcome that bears good fruit."         ║
║                                                                   ║
║    Small models (0.5B) ✓   via EDGE tier                         ║
║    Medium models (7B) ✓    via LOCAL tier                        ║
║    Large models (70B) ✓    via POOL tier (federated)             ║
║                                                                   ║
║    IF AND ONLY IF:                                               ║
║    ┌────────────────────────────────────────────────────────┐    ║
║    │ Ihsān (Excellence) ≥ 0.95  — Z3 SMT verified           │    ║
║    │ SNR (Signal Quality) ≥ 0.85 — Shannon enforced         │    ║
║    │ Sovereignty — Dilithium-5 signed CapabilityCard        │    ║
║    └────────────────────────────────────────────────────────┘    ║
║                                                                   ║
║    Offline ✓  |  Online ✓  |  Federated ✓                        ║
║                                                                   ║
║    "We do not assume. We verify with formal proofs."             ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝
`);
}
