/**
 * CapabilityCard - PCI-Signed Model Credentials
 *
 * Every model accepted into BIZRA receives a CapabilityCard that
 * certifies its validated capabilities. Cards are Ed25519 signed
 * and include expiration dates.
 */

import { createHash, sign, verify } from 'crypto';

/** Model capability tiers */
export enum ModelTier {
  /** 0.5B-1.5B models, CPU-capable, always-on */
  EDGE = 'EDGE',

  /** 7B-13B models, GPU-recommended */
  LOCAL = 'LOCAL',

  /** 70B+ models, federation-capable */
  POOL = 'POOL',
}

/** Supported task types */
export enum TaskType {
  REASONING = 'reasoning',
  CHAT = 'chat',
  SUMMARIZATION = 'summarization',
  CODE_GENERATION = 'code_generation',
  TRANSLATION = 'translation',
  CLASSIFICATION = 'classification',
  EMBEDDING = 'embedding',
}

/** Quantization levels */
export type Quantization = 'f16' | 'q8' | 'q4' | 'q2' | 'unknown';

/** Constitutional thresholds */
export const IHSAN_THRESHOLD = 0.95;
export const SNR_THRESHOLD = 0.85;

/**
 * CapabilityCard structure
 */
export interface CapabilityCard {
  /** Unique model identifier */
  modelId: string;

  /** Human-readable model name */
  modelName: string;

  /** Parameter count (approximate) */
  parameterCount: number | null;

  /** Quantization level */
  quantization: Quantization;

  /** Capability tier */
  tier: ModelTier;

  /** Validated capabilities from challenge */
  capabilities: {
    maxContext: number;
    ihsanScore: number;
    snrScore: number;
    latencyMs: number;
    tasksSupported: TaskType[];
  };

  /** Ed25519 signature (hex) */
  signature: string;

  /** Issuer public key (hex) */
  issuerPublicKey: string;

  /** Issue timestamp (ISO 8601) */
  issuedAt: string;

  /** Expiration timestamp (ISO 8601) */
  expiresAt: string;

  /** Whether the card has been revoked */
  revoked: boolean;
}

/**
 * Create a new unsigned CapabilityCard
 */
export function createCapabilityCard(params: {
  modelId: string;
  modelName?: string;
  parameterCount?: number;
  quantization?: Quantization;
  tier: ModelTier;
  ihsanScore: number;
  snrScore: number;
  maxContext?: number;
  latencyMs?: number;
  tasksSupported: TaskType[];
}): CapabilityCard {
  // Validate scores meet thresholds
  if (params.ihsanScore < IHSAN_THRESHOLD) {
    throw new Error(
      `Ihsān score ${params.ihsanScore} < threshold ${IHSAN_THRESHOLD}`
    );
  }
  if (params.snrScore < SNR_THRESHOLD) {
    throw new Error(
      `SNR score ${params.snrScore} < threshold ${SNR_THRESHOLD}`
    );
  }

  const now = new Date();
  const expires = new Date(now.getTime() + 90 * 24 * 60 * 60 * 1000); // 90 days

  return {
    modelId: params.modelId,
    modelName: params.modelName ?? params.modelId,
    parameterCount: params.parameterCount ?? null,
    quantization: params.quantization ?? 'unknown',
    tier: params.tier,
    capabilities: {
      maxContext: params.maxContext ?? 2048,
      ihsanScore: params.ihsanScore,
      snrScore: params.snrScore,
      latencyMs: params.latencyMs ?? 0,
      tasksSupported: params.tasksSupported,
    },
    signature: '',
    issuerPublicKey: '',
    issuedAt: now.toISOString(),
    expiresAt: expires.toISOString(),
    revoked: false,
  };
}

/**
 * Get canonical bytes for signing
 */
export function getCardCanonicalBytes(card: CapabilityCard): Buffer {
  const data = [
    card.modelId,
    card.tier,
    card.capabilities.ihsanScore.toString(),
    card.capabilities.snrScore.toString(),
    card.issuedAt,
    card.expiresAt,
  ].join('|');

  return Buffer.from(data, 'utf-8');
}

/**
 * Sign a CapabilityCard with Ed25519
 */
export function signCapabilityCard(
  card: CapabilityCard,
  privateKey: Buffer,
  publicKey: Buffer
): CapabilityCard {
  const canonicalBytes = getCardCanonicalBytes(card);
  const signature = sign(null, canonicalBytes, {
    key: privateKey,
    format: 'der',
    type: 'pkcs8',
  });

  return {
    ...card,
    signature: signature.toString('hex'),
    issuerPublicKey: publicKey.toString('hex'),
  };
}

/**
 * Verify a CapabilityCard signature
 */
export function verifyCapabilityCardSignature(card: CapabilityCard): boolean {
  if (!card.signature || !card.issuerPublicKey) {
    return false;
  }

  try {
    const canonicalBytes = getCardCanonicalBytes(card);
    const signature = Buffer.from(card.signature, 'hex');
    const publicKey = Buffer.from(card.issuerPublicKey, 'hex');

    return verify(
      null,
      canonicalBytes,
      { key: publicKey, format: 'der', type: 'spki' },
      signature
    );
  } catch {
    return false;
  }
}

/**
 * Check if a CapabilityCard is currently valid
 */
export function isCapabilityCardValid(card: CapabilityCard): {
  valid: boolean;
  reason?: string;
} {
  // Check revocation
  if (card.revoked) {
    return { valid: false, reason: 'Card has been revoked' };
  }

  // Check expiration
  const now = new Date();
  const expires = new Date(card.expiresAt);
  if (now >= expires) {
    return { valid: false, reason: 'Card has expired' };
  }

  // Check signature
  if (!verifyCapabilityCardSignature(card)) {
    return { valid: false, reason: 'Invalid signature' };
  }

  // Check scores still meet thresholds
  if (card.capabilities.ihsanScore < IHSAN_THRESHOLD) {
    return {
      valid: false,
      reason: `Ihsān score ${card.capabilities.ihsanScore} below threshold`,
    };
  }

  if (card.capabilities.snrScore < SNR_THRESHOLD) {
    return {
      valid: false,
      reason: `SNR score ${card.capabilities.snrScore} below threshold`,
    };
  }

  return { valid: true };
}

/**
 * Get remaining validity time in days
 */
export function getCardRemainingDays(card: CapabilityCard): number {
  const now = new Date();
  const expires = new Date(card.expiresAt);
  const diff = expires.getTime() - now.getTime();
  return Math.max(0, Math.floor(diff / (24 * 60 * 60 * 1000)));
}

/**
 * Calculate card fingerprint for identification
 */
export function getCardFingerprint(card: CapabilityCard): string {
  const hash = createHash('sha256');
  hash.update(getCardCanonicalBytes(card));
  hash.update(card.signature);
  return hash.digest('hex').slice(0, 16);
}
