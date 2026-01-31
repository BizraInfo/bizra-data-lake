/**
 * FATE Binding TypeScript Definitions
 *
 * Native Rust bindings for Z3 SMT verification, Dilithium-5 signatures,
 * and Ed25519 CapabilityCard management.
 */

/** Constitutional thresholds */
export const IHSAN_THRESHOLD: number;
export const SNR_THRESHOLD: number;

/** Model capability tiers */
export const enum ModelTier {
  EDGE = 'EDGE',
  LOCAL = 'LOCAL',
  POOL = 'POOL',
}

/** Task types for capability matching */
export const enum TaskType {
  REASONING = 'reasoning',
  CHAT = 'chat',
  SUMMARIZATION = 'summarization',
  CODE_GENERATION = 'code_generation',
  TRANSLATION = 'translation',
  CLASSIFICATION = 'classification',
  EMBEDDING = 'embedding',
}

/** Gate validation result */
export interface GateResult {
  passed: boolean;
  gateName: string;
  score: number;
  reason?: string;
}

/** Capability card structure */
export interface CapabilityCard {
  modelId: string;
  modelName: string;
  tier: ModelTier;
  parameterCount: number | null;
  quantization: string;
  ihsanScore: number;
  snrScore: number;
  maxContext: number;
  tasksSupported: TaskType[];
  signature: string;
  issuerPublicKey: string;
  issuedAt: string;
  expiresAt: string;
}

/** Inference output for gate validation */
export interface InferenceOutput {
  content: string;
  modelId: string;
  ihsanScore: number;
  snrScore: number;
  capabilityCardValid: boolean;
}

/** FATE Validator class */
export class FateValidator {
  constructor();

  /** Validate inference output through gate chain */
  validateOutput(output: InferenceOutput): GateResult;

  /** Validate detailed gate-by-gate results */
  validateDetailed(output: InferenceOutput): GateResult[];

  /** Verify Z3 SMT Ihsan constraint */
  verifyIhsan(score: number): boolean;

  /** Verify SNR threshold */
  verifySNR(score: number): boolean;

  /** Get version information */
  static getVersion(): string;
}

/** Card Issuer for signing capability cards */
export class CardIssuer {
  constructor();

  /** Issue a signed capability card */
  issue(card: Omit<CapabilityCard, 'signature' | 'issuerPublicKey'>): CapabilityCard;

  /** Verify a capability card signature */
  verify(card: CapabilityCard): boolean;

  /** Get the issuer's public key */
  getPublicKey(): string;
}

/** Challenge result from Constitution Challenge */
export interface ChallengeResult {
  passed: boolean;
  ihsanScore: number;
  snrScore: number;
  sovereigntyAcknowledged: boolean;
  capabilityCard?: CapabilityCard;
  failureReason?: string;
}

/** Run Constitution Challenge for a model */
export function runChallenge(
  modelId: string,
  modelPath: string,
  inferenceFunction: (modelId: string, prompt: string) => Promise<string>,
  options?: {
    tier?: ModelTier;
    tasksSupported?: TaskType[];
  }
): Promise<ChallengeResult>;

/** Create a new FATE validator instance */
export function createFateValidator(): Promise<FateValidator>;
